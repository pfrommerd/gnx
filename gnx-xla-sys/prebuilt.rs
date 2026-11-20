pub use gnx_cuda_build::download;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const VERSION: &str = "0.11.0";

// SHA-256 digests from the elixir-nx/xla v0.11.0 GitHub release assets.
fn archive(target: &str, variant: &str) -> (&'static str, &'static str) {
    match (target, variant) {
        ("x86_64-unknown-linux-gnu", "cpu") => ("x86_64-linux-gnu-cpu", "f58a65af94e7058a82af7783b657dd4c0b29dc710cb5b8e8e6a0c1dc8f5d3f39"),
        ("x86_64-unknown-linux-gnu", "cuda12") => ("x86_64-linux-gnu-cuda12", "6392d6f69098b27f5d150902503358be670ff76f07f0811447b02878439a072a"),
        ("x86_64-unknown-linux-gnu", "cuda13") => ("x86_64-linux-gnu-cuda13", "769d79db97d8dbeb714a62303ccc7bb95136c184c857ff8026b1e3c50f3eaa52"),
        ("aarch64-unknown-linux-gnu", "cpu") => ("aarch64-linux-gnu-cpu", "60c81319ea1c6c9a87c8ad2f8c683e95244c3f124a7e0b34efe0697a154f157c"),
        ("aarch64-unknown-linux-gnu", "cuda12") => ("aarch64-linux-gnu-cuda12", "19daebb5f55b93485cda6c8ed8f192e29ef08ca9e07b5b5963b6656601451a22"),
        ("aarch64-unknown-linux-gnu", "cuda13") => ("aarch64-linux-gnu-cuda13", "21d03c468a85e425981036823ecd148c3e51b6d4d95ef79627d45e142f38ea9f"),
        ("x86_64-apple-darwin", "cpu") => ("x86_64-darwin-cpu", "89e09a09eb571d80bacbe9fdfdf346e6d32ea5e771a35484040bc8a5d7c33969"),
        ("aarch64-apple-darwin", "cpu") => ("aarch64-darwin-cpu", "f050a91b0819ed895305f4a9a2ec10f35fa01cb77e50930e16d0e0a22ca43958"),
        _ => panic!("No {variant:?} prebuilt XLA extension for {target}; select a supported XLA_EXTENSION_VARIANT or set XLA_EXTENSION_DIR"),
    }
}

fn run(cmd: &mut Command) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("Could not run {cmd:?}: {e}"));
    assert!(status.success(), "{cmd:?} failed: {status}");
}

pub fn variant(target: &str) -> String {
    println!("cargo:rerun-if-env-changed=XLA_EXTENSION_VARIANT");
    let default_variant = if cfg!(feature = "cuda")
        && matches!(
            target,
            "x86_64-unknown-linux-gnu" | "aarch64-unknown-linux-gnu"
        ) {
        "cuda13"
    } else {
        "cpu"
    };
    env::var("XLA_EXTENSION_VARIANT").unwrap_or_else(|_| default_variant.into())
}

pub fn install(out_dir: &Path, target: &str, variant: &str) -> PathBuf {
    let (platform, expected) = archive(target, variant);
    let root = out_dir.join(format!("xla-{VERSION}-{platform}-{expected}"));
    if root.is_dir() {
        return root;
    }

    // Publish only a complete installation; failed downloads/extractions are retryable.
    let stage = tempfile::tempdir_in(out_dir).expect("create XLA staging directory");
    let archive = stage.path().join("archive.tar.gz");
    let url = format!("https://github.com/elixir-nx/xla/releases/download/v{VERSION}/xla_extension-{VERSION}-{platform}.tar.gz");
    download(&url, &archive, expected);

    run(Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(stage.path()));
    fs::rename(stage.path().join("xla_extension"), &root).expect("install verified XLA extension");
    root
}
