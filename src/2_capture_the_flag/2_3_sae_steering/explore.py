import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

# https://huggingface.co/EleutherAI/pythia-70m

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
HOOK_LAYER  = 4
HOOK_POINT  = f"blocks.{HOOK_LAYER}.hook_resid_post"
SAE_RELEASE = "sae_bench_pythia70m_sweep_standard_ctx128_0712"
SAE_ID      = f"blocks.{HOOK_LAYER}.hook_resid_post__trainer_10"

def load_models():
    model = HookedTransformer.from_pretrained("pythia-70m-deduped")
    model.eval().to(DEVICE)

    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
    sae.eval().to(DEVICE)
    return model, sae

def get_feature_acts(prompt: str, model, sae) -> tuple[list[str], torch.Tensor]:
    """
    Returns:
        token_strs: list of token strings len S
        feature_acts: activations per token, Tensor (S, d_sae)
    """
    tokens     = model.to_tokens(prompt, prepend_bos=True)
    token_strs = model.to_str_tokens(prompt, prepend_bos=True)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        resid = cache[HOOK_POINT] # residual stream (1, S, d_model)
        feature_acts = sae.encode(resid)[0] # pass it to encoder (S, d_sae)

    return token_strs, feature_acts


def top_features(feature_acts: torch.Tensor, top_k: int = 10, mode: str = "mean"):
    """
    Aggregate across token positions and return top-k (feature_id, value) pairs.

    mode: "mean" — average over all token positions
          "max"  — peak activation over all token positions
    """
    agg = feature_acts.mean(dim=0) if mode == "mean" else feature_acts.max(dim=0).values
    vals, ids = agg.topk(top_k)
    return [(int(i), float(v)) for i, v in zip(ids, vals)]


def top_features_per_token(feature_acts: torch.Tensor, token_strs: list[str], top_k: int = 5):
    """Print top-k active features at each token position."""
    for i, tok in enumerate(token_strs):
        vals, ids = feature_acts[i].topk(top_k)
        active = [(int(j), round(float(v), 3)) for j, v in zip(ids, vals) if v > 0]
        print(f"  [{i:2d}] {tok!r:15s}  {active}")


def steer_and_generate( prompt: str, model, sae, feature_id: int, strength: float = 1.5, max_new_tokens: int = 80, temperature: float = 0.7):
    """
    Generate with a constant steering vector added at HOOK_POINT each forward pass.
    The vector is the SAE decoder column for feature_id (shape: d_model).
    """

    # steering vector
    vec = sae.W_dec[feature_id].detach().clone().to(DEVICE)

    def hook_fn(value, hook):
        return value + strength * vec

    tokens = model.to_tokens(prompt, prepend_bos=True)
    with model.hooks(fwd_hooks=[(HOOK_POINT, hook_fn)]):
        out = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    return model.to_string(out[0, tokens.shape[1]:])

def find_contrastive_features(prompt_target: str, prompt_neutral: str, model, sae, top_k=10):
    """Finds features that fire highly on the target prompt but not the neutral prompt."""
    _, acts_target = get_feature_acts(prompt_target, model, sae)
    _, acts_neutral = get_feature_acts(prompt_neutral, model, sae)
    
    # Average across the sequence length (S, d_sae) -> (d_sae)
    mean_target = acts_target.mean(dim=0)
    mean_neutral = acts_neutral.mean(dim=0)
    
    # Calculate the difference
    diff = mean_target - mean_neutral
    
    vals, ids = diff.topk(top_k)
    print(f"── Top contrastive features ──")
    for fid, val in zip(ids, vals):
        print(f"  feature {int(fid):5d}  diff: {float(val):.4f}")
    
    return [int(i) for i in ids]

if __name__ == "__main__":
    model, sae = load_models()

    prompt = "When you think of the Netherlands, you think of tulip and windmills."
    prompt_capital = "When you think of the Netherlands, you think of the capital Amsterdam."

    # Capture activations
    token_strs, feature_acts = get_feature_acts(prompt, model, sae)
    print(f"Prompt : {prompt!r}")
    print(f"Tokens : {token_strs}\n")

    # Top features over the whole prompt (mean)
    print("── Top features (mean across tokens) ──")
    for fid, val in top_features(feature_acts, top_k=10, mode="max"):
        print(f"  feature {fid:5d}  {val:.4f}")

    # Per-token breakdown
    print("\n── Top features per token ──")
    top_features_per_token(feature_acts, token_strs, top_k=3)

    # Steer toward a feature and generate
    result = steer_and_generate(prompt, model, sae, feature_id=6399)
    print(f"{result.strip()}")

    print(find_contrastive_features(prompt_capital, prompt, model, sae))