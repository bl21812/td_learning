{
  description = "Python development environment";

  inputs = {
    # nixpkgs.url = "github:nixos/nixpkgs/dd5621df6dcb90122b50da5ec31c411a0de3e538";
    nixpkgs.url = "github:nixos/nixpkgs/24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = pkgs.python39.withPackages (ps: [
          ps.matplotlib
          ps.numpy
          ps.scipy
          ps.pandas
          ps.tkinter
          # ps.jaxtyping
          # ps.debugpy
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
            ruff
            pandoc
          ];
          shellHook = ''
            echo "Python development environment activated"
            python --version
          '';
        };
      });
}