{
  description = "Dev shell with required deps";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      system = system;
      config.allowUnfree = true;
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        (python313.withPackages (ps: with ps; [
          torch-bin
          numpy
          tqdm
          transformers
          datasets
          ipykernel
          jupyter
        ]))
        cudaPackages.cudnn
        gcc13
      ];

      shellHook = ''
        export CUDA_PATH=${pkgs.cudatoolkit}
        export CC=${pkgs.gcc13}/bin/gcc
        export CXX=${pkgs.gcc13}/bin/g++
      '';
    };
  };
}
