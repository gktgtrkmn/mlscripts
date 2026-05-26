{
  description = "macOS Python dev environment with uv and pyright";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # Support both Apple Silicon (aarch64) and Intel (x86_64) Macs
      supportedSystems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            # Core tools
            python313
            uv
            llvmPackages.openmp

            # Language Server
            pyright
            ruff
          ];

          env = {
            # Tells uv to create and use the .venv in your project directory
            UV_PROJECT_ENVIRONMENT = ".venv";
          };

          shellHook = ''
            # Create the virtual environment if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating virtual environment with uv..."
              uv venv
            fi

            # Automatically activate the virtual environment
            source .venv/bin/activate
            uv sync
              
            export DYLD_LIBRARY_PATH="${pkgs.llvmPackages.openmp}/lib:$DYLD_LIBRARY_PATH"

            echo "🍎 macOS Python environment ready!"
            echo "Use 'uv add <package>' or 'uv pip install <package>'"
          '';
        };
      });
    };
}
