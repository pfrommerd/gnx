//! Build-time installation of the shared CUDA runtime and backend-specific
//! libraries and compiler tools. Packages are verified before extraction.
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

const PACKAGES: &str = include_str!("cuda-runtime.lock");

const TORCH: &str = include_str!("torch-runtime.lock");

pub fn download(url: &str, archive: &Path, expected: &str) {
    let status = Command::new("curl")
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
        .arg(archive)
        .arg(url)
        .status()
        .expect("start curl");
    assert!(status.success(), "download failed for {url}: {status}");
    let mut hash = Sha256::new();
    io::copy(&mut fs::File::open(archive).unwrap(), &mut hash).expect("hash download");
    assert_eq!(
        format!("{:x}", hash.finalize()),
        expected,
        "SHA-256 mismatch for {url}"
    );
}

fn packages<'a>(
    variant: &'a str,
    arch: &'a str,
) -> impl Iterator<Item = (&'static str, &'static str)> + 'a {
    PACKAGES.lines().filter_map(move |line| {
        let mut fields = line.split_whitespace();
        (fields.next()? == variant && fields.next()? == arch)
            .then(|| (fields.next().unwrap(), fields.next().unwrap()))
    })
}
fn xla_only(url: &str) -> bool {
    ["/jax_cuda", "/nvidia_cuda_nvcc-", "/nvidia_nvvm-"]
        .iter()
        .any(|p| url.contains(p))
}
fn wheel(hash: &str, url: &str, archive: &Path, bundle: &Path) {
    download(url, archive, hash);
    extract_wheel(archive, bundle).unwrap_or_else(|err| panic!("extract {url}: {err}"));
}

// Both build scripts resolve the same immutable files, not two copies with
// matching SONAMEs. This also covers components cuDNN loads via dlopen.
pub fn shared_cuda13() -> PathBuf {
    use std::env;
    for key in ["GNX_CUDA_CACHE", "XDG_CACHE_HOME", "HOME"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let cache = env::var_os("GNX_CUDA_CACHE")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache")))
                .map(|p| p.join("gnx/cuda"))
        })
        .expect("cannot determine CUDA cache directory; set GNX_CUDA_CACHE");
    fs::create_dir_all(&cache).unwrap();
    let cache = cache.canonicalize().unwrap();
    let digest = Sha256::digest(PACKAGES.as_bytes());
    let root = cache.join(format!("cuda13-x86_64-{digest:x}"));
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.with_extension("lock"))
        .unwrap();
    lock.lock().expect("lock shared CUDA installation");
    println!(
        "cargo:rerun-if-changed={}",
        root.join("lib/libcudnn.so.9").display()
    );
    if !root.is_dir() {
        let tmp = tempfile::tempdir_in(&cache).unwrap();
        let bundle = tmp.path().join("bundle");
        let archive = tmp.path().join("download");
        for (hash, url) in packages("cuda13", "x86_64").filter(|(_, u)| !xla_only(u)) {
            wheel(hash, url, &archive, &bundle);
        }
        install_zlib("x86_64", &archive, &bundle);
        fs::rename(bundle, &root).expect("publish shared CUDA runtime");
    }
    root
}

fn link_libraries(common: &Path, dest: &Path) {
    fs::create_dir_all(dest).unwrap();
    for entry in fs::read_dir(common.join("lib")).unwrap() {
        let entry = entry.unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(entry.path(), dest.join(entry.file_name())).unwrap();
        #[cfg(not(unix))]
        std::os::windows::fs::symlink_file(entry.path(), dest.join(entry.file_name())).unwrap();
    }
}

pub fn torch_layout_key(common: &Path) -> String {
    format!(
        "{:x}",
        Sha256::digest(format!("{}:{TORCH}", common.display()))
    )
}

pub fn install_torch_dependencies(common: &Path, torch: &Path, tmp: &Path) {
    link_libraries(common, &torch.join("lib"));
    for line in TORCH.lines().filter(|l| !l.starts_with('#')) {
        let (hash, url) = line.split_once(' ').unwrap();
        wheel(hash, url, &tmp.join("dependency.whl"), torch);
    }
}

pub fn install(out: &Path, target: &str, variant: &str) -> PathBuf {
    let arch = target.split('-').next().unwrap();
    let common = (variant == "cuda13" && arch == "x86_64").then(shared_cuda13);
    let digest = Sha256::digest(format!("{PACKAGES}{common:?}"));
    let root = out.join(format!("{variant}-{arch}-{digest:x}"));
    if root.is_dir() {
        return root;
    }
    let stage = tempfile::tempdir_in(out).expect("stage CUDA runtime");
    let bundle = stage.path().join("bundle");
    let archive = stage.path().join("download");
    for (hash, url) in packages(variant, arch).filter(|(_, url)| common.is_none() || xla_only(url))
    {
        wheel(hash, url, &archive, &bundle);
    }
    if let Some(common) = common {
        link_libraries(&common, &bundle.join("lib"));
    } else {
        install_zlib(arch, &archive, &bundle);
    }
    for file in [
        "lib/xla_cuda_plugin.so",
        "bin/ptxas",
        "nvvm/libdevice/libdevice.10.bc",
    ] {
        assert!(
            bundle.join(file).is_file(),
            "missing {file} in {variant} bundle"
        );
    }
    fs::rename(bundle, &root).expect("publish verified CUDA runtime");
    root
}

fn install_zlib(arch: &str, archive: &Path, bundle: &Path) {
    let hash = match arch {
        "x86_64" => "b9b82fcdbb893ca7470e326b9fa305ece5bf5f462004a622cfdbf54c5d8de4e5",
        "aarch64" => "5cffc1e0520c7e5d281650483a7118d3a3d2a8c9edc9222000ffdfdc3101e343",
        _ => panic!("no zlib for {arch}"),
    };
    let url = format!("https://github.com/JuliaBinaryWrappers/Zlib_jll.jl/releases/download/Zlib-v1.3.2%2B0/Zlib.v1.3.2.{arch}-linux-gnu.tar.gz");
    download(&url, archive, hash);
    assert!(Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(bundle)
        .args([
            "lib/libz.so.1",
            "lib/libz.so.1.3.2",
            "share/doc/zlib/zlib/LICENSE"
        ])
        .status()
        .expect("extract zlib")
        .success());
}

fn extract_wheel(archive: &Path, bundle: &Path) -> io::Result<()> {
    let mut zip = zip::ZipArchive::new(fs::File::open(archive)?)?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        let path = entry
            .enclosed_name()
            .ok_or_else(|| io::Error::other("invalid wheel path"))?;
        let name = path.file_name().unwrap().to_string_lossy();
        let parent = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|s| s.to_str());
        let dest = if (parent == Some("lib") && name.contains(".so."))
            || name == "xla_cuda_plugin.so"
        {
            bundle.join("lib").join(name.as_ref())
        } else if parent == Some("bin") && matches!(name.as_ref(), "ptxas" | "nvlink" | "fatbinary")
        {
            bundle.join("bin").join(name.as_ref())
        } else if name == "libdevice.10.bc" {
            bundle.join("nvvm/libdevice").join(name.as_ref())
        } else if path.to_string_lossy().contains(".dist-info/")
            && (name.to_ascii_lowercase().contains("license")
                || name == "METADATA"
                || parent == Some("licenses"))
        {
            bundle.join("licenses").join(&path)
        } else {
            continue;
        };
        fs::create_dir_all(dest.parent().unwrap())?;
        let mut file = fs::File::create_new(&dest)?;
        io::copy(&mut entry, &mut file)?;
        #[cfg(unix)]
        if parent == Some("bin") {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o755))?;
        }
    }
    Ok(())
}
