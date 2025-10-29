{
  description = "A very basic flake";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, flake-utils, nixpkgs }: flake-utils.lib.eachDefaultSystem (system:
    let pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      lib = pkgs.lib;

      ott-jax = (pypkg: with pypkg; buildPythonPackage rec {
        pname = "ott_jax";
        version = "0.5.0";
        src = fetchPypi {
          inherit pname version;
          sha256 = "sha256-Cq6kFnXUxiEG5morJBq4px/2l37+FTm7hl5Tf/B2XlI=";
        };
        propagatedBuildInputs = [
            jax jaxopt lineax numpy optax
        ];
      });
      # Custom shell enter script for fish
      # sets up a nice prompt
      shell-enter = pkgs.writeText "shell-enter.fish" (''
        set fish_color_normal normal
        set fish_color_autosuggestion brblack
        set fish_color_cancel -r
        set fish_color_command blue
        set fish_color_comment red
        set fish_color_cwd green
        set fish_color_cwd_root red
        set fish_color_end green
        set fish_color_error brred
        set fish_color_escape brcyan
        set fish_color_history_current --bold
        set fish_color_host normal
        set fish_color_host_remote yellow
        set fish_color_keyword normal
        set fish_color_operator brcyan
        set fish_color_option cyan
        set fish_color_param cyan
        set fish_color_quote yellow
        set fish_color_redirection cyan --bold
        set fish_color_search_match white --background=brblack --bold
        set fish_color_selection white --background=brblack --bold
        set fish_color_status red
        set fish_color_user brgreen
        set fish_color_valid_path --underline
        set fish_pager_color_completion normal
        set fish_pager_color_description yellow -i
        set fish_pager_color_prefix normal --bold --underline
        set fish_pager_color_progress brwhite --background=cyan --bold
        set fish_pager_color_selected_background -r
        set fish_history fish

        function fish_vcs_prompt
          echo -n -s (set_color normal) " (" (set_color blue) $ENV_NAME (set_color normal) ")"
        end
      '');
      # Remove all non-nix store paths from PATH
      # To make the dev shell more hermetically clean
      clean-path = pkgs.writeShellScript "clean-path" ''
        IFS=':' read -ra DIRS <<< "$PATH"
        newpath=""
        for dir in "''${DIRS[@]}"; do
          [[ $dir == $NIX_STORE* ]] && {
            [[ -n $newpath ]] && newpath+=":"
            newpath+="$dir"
          }
        done
        export PATH="$newpath"
      '';
      # If on linux, set up CUDA libcuda symlink for drivers
      setup-cuda = pkgs.writeShellScript "setup-cuda" ''
        # If on linux, set up CUDA libcuda symlink for drivers
        TEMP_DIR=$(${pkgs.mktemp}/bin/mktemp -d)
        TEMP_LIB_DIR="''${TEMP_DIR}/lib"
        mkdir -p $TEMP_LIB_DIR
        export LD_LIBRARY_PATH=$TEMP_LIB_DIR
        ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 $TEMP_LIB_DIR/libcuda.so.1
      '';
      make-shell = {cuda ? false}:
      let enableCuda = cuda && pkgs.stdenv.isLinux;
        in
        pkgs.mkShell {
            env.ENV_NAME = "generalized-diffusion";
            shellHook = (
            ''
              source ${clean-path}
              export PYTHONPATH=$(pwd)/src
	      export NIX_BUILD_SHELL=fish
            '' +
            (if enableCuda then ''
              source ${setup-cuda}
            '' else ''
              export JAX_PLATFORMS=cpu
            '')
	    );
            packages = with pkgs; [
              fish which git tmux ncurses neovim gnupg openssh_hpnWithKerberos
	    ] ++ [(pkgs.python312.withPackages (py-pkgs: with py-pkgs; [
                  jax flax matplotlib optax
                  (ott-jax py-pkgs) pillow plotly polars rich wandb
                  marimo pytest
              ] ++ (lib.optionals enableCuda [
                  jax-cuda12-plugin
              ])))
            ];
        };
    in {
      devShells.default = make-shell {};
      devShells.cuda = make-shell { cuda = true; };
    }
  );
}
