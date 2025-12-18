{
  description = "A dev environment flake";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let 
    lib = nixpkgs.lib;
    platforms = lib.systems.flakeExposed;
    eachPlatform = lib.genAttrs platforms;
  in
  {
    devShells = eachPlatform (platform:
    let
      pkgs = import nixpkgs { system = platform; };
    in {
      default = pkgs.mkShell {
        packages = with pkgs; [
          lld
          python313
          rustc
          cargo
          openssl.dev
          dioxus-cli
          pkg-config
        ];
      };
    });
  };
}
