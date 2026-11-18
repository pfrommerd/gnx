{ pkgs, jail, ... }:
jail.mkJail {
  # Optional input declarations are pinned in .jail.lock:
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  # Local configuration imports are included, watched, and protected:
  jailFiles = [ ];
  sandbox = true;
  packages = with pkgs; [
    uv nodejs rustc cargo rustfmt gnumake cmake clang clang-tools
    zip tree file jujutsu git-lfs fd less
  ];
  network = true;
  nix = true;
  gpu = true;
}
