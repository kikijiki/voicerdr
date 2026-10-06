{
  description = "voicerdr — configurable local Herdr voice assistant";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python312
              uv
              just
              ruff
              jq
              git
              nixfmt
              # LocalAudio / PortAudio bindings
              portaudio
              libsndfile
              tbb
              zlib
              pkg-config
            ];

            # Native PyPI wheels in the voice stack need these at runtime too.
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.portaudio
              pkgs.libsndfile
              pkgs.stdenv.cc.cc
              pkgs.tbb
              pkgs.zlib
            ];

            shellHook = ''
              export UV_LINK_MODE="''${UV_LINK_MODE:-copy}"
              export UV_PYTHON="''${UV_PYTHON:-$(command -v python3)}"
              export VOICERDR_NIX_RUNTIME="$PWD"
              export VOICERDR_REPO_ROOT="$PWD"
              if [[ -z "''${VOICERDR_NIX_REEXEC:-}" ]]; then
                echo "voicerdr nix shell — python $(python3 --version 2>/dev/null | awk '{print $2}'), uv $(uv --version 2>/dev/null | awk '{print $2}')"
                echo "  just            # list tasks"
                echo "  just sync       # uv sync"
                echo "  just ensure     # start/reconnect daemon"
              fi
            '';
          };
        }
      );
    };
}
