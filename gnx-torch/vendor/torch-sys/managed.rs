//! GNX's pinned LibTorch distribution. This runs in torch-sys, the *first*
//! consumer of the native headers. All downstream build scripts use its metadata.
use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

struct Archive {
    file: &'static str,
    variant: &'static str,
    digest: &'static str,
}
fn archive(target: &str) -> Result<Archive> {
    Ok(match (target, cfg!(feature="gnx-cuda")) {
        ("x86_64-unknown-linux-gnu", false) => Archive { variant:"cpu", file:"libtorch-shared-with-deps-2.9.0%2Bcpu.zip", digest:"22de42abde933be46ce843467930bd0190b72271bfa2c11f84db95591a9834f1" },
        ("x86_64-unknown-linux-gnu", true) => Archive { variant:"cu130", file:"torch-2.9.0%2Bcu130-cp311-cp311-manylinux_2_28_x86_64.whl", digest:"e748b84e700634fbf08f62319e18e228c861895fd41ea7c73043c81d6d0968c4" },
        ("aarch64-apple-darwin", _) => Archive { variant:"cpu", file:"libtorch-macos-arm64-2.9.0.zip", digest:"6d6af87cab301fa25cb4909697a03c65ed234e784cd96c8743a9ad6586238d0e" },
        _ => bail!("no pinned LibTorch {} archive for {target}; set LIBTORCH to a matching installation or use LIBTORCH_USE_PYTORCH=1",super::TORCH_VERSION),
    })
}
fn run(cmd: &mut Command) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("could not start {cmd:?}"))?;
    ensure!(status.success(), "{cmd:?} failed with {status}");
    Ok(())
}
fn verify(path: &Path, expected: &str) -> Result<()> {
    let mut hash = Sha256::new();
    io::copy(&mut fs::File::open(path)?, &mut hash)?;
    let actual = format!("{:x}", hash.finalize());
    ensure!(
        actual == expected,
        "LibTorch SHA-256 mismatch: expected {expected}, got {actual}"
    );
    Ok(())
}
pub(super) fn install() -> Result<PathBuf> {
    let target = env::var("TARGET")?;
    let a = archive(&target)?;
    for key in ["GNX_LIBTORCH_CACHE", "XDG_CACHE_HOME", "HOME"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let cache = env::var_os("GNX_LIBTORCH_CACHE")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache")))
                .map(|p| p.join("gnx/libtorch"))
        })
        .context("cannot determine cache directory; set GNX_LIBTORCH_CACHE")?;
    fs::create_dir_all(&cache)?;
    let cache = cache.canonicalize()?;
    let key = format!(
        "{}-{target}-{}-{}",
        super::TORCH_VERSION,
        a.variant,
        a.digest
    );
    #[cfg(feature = "gnx-cuda")]
    let shared = (a.variant == "cu130").then(gnx_cuda_build::shared_cuda13);
    #[cfg(feature = "gnx-cuda")]
    let key = match &shared {
        Some(dir) => format!("{key}-{}", gnx_cuda_build::torch_layout_key(dir)),
        None => key,
    };
    let root = cache.join(&key);
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(cache.join(format!("{key}.lock")))?;
    lock.lock().context("lock LibTorch installation")?;
    let marker = root.join(".gnx-verified");
    println!("cargo:rerun-if-changed={}", marker.display());
    if root.exists() {
        ensure!(
            fs::read_to_string(&marker).unwrap_or_default() == a.digest
                && root.join("build-version").is_file(),
            "incomplete LibTorch cache entry {}; remove it and rebuild",
            root.display()
        );
        return Ok(root);
    }
    let staging = tempfile::tempdir_in(&cache)?;
    let zip = staging.path().join("archive.zip");
    let url = if a.variant == "cu130" {
        format!("https://download.pytorch.org/whl/cu130/{}", a.file)
    } else {
        format!(
            "https://download.pytorch.org/libtorch/{}/{}",
            a.variant, a.file
        )
    };
    println!(
        "cargo:warning=Downloading LibTorch {} ({}) to {}",
        super::TORCH_VERSION,
        a.variant,
        root.display()
    );
    run(Command::new("curl")
        .args([
            "-q",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--retry",
            "3",
            "--output",
        ])
        .arg(&zip)
        .arg(&url))?;
    verify(&zip, a.digest).with_context(|| format!("verifying {url}"))?;
    let mut unzip = Command::new("unzip");
    unzip.arg("-q").arg(&zip);
    if a.variant == "cu130" {
        unzip.args(["torch/include/*", "torch/lib/*", "*.dist-info/licenses/*"]);
    }
    run(unzip.arg("-d").arg(staging.path()))?;
    let extracted = staging.path().join("libtorch");
    #[cfg(feature = "gnx-cuda")]
    let extracted = if let Some(common) = shared {
        let torch = staging.path().join("torch");
        fs::rename(
            staging.path().join("torch-2.9.0+cu130.dist-info/licenses"),
            torch.join("licenses"),
        )?;
        fs::write(torch.join("build-version"), super::TORCH_VERSION)?;
        gnx_cuda_build::install_torch_dependencies(&common, &torch, staging.path());
        torch
    } else {
        extracted
    };
    ensure!(
        extracted.join("build-version").is_file()
            && extracted.join("include/ATen/ATen.h").is_file(),
        "invalid LibTorch archive layout"
    );
    fs::write(extracted.join(".gnx-verified"), a.digest)?;
    fs::rename(extracted, &root)?;
    Ok(root)
}
