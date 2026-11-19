//! Stage the managed PJRT plugin and its userspace dependencies. No Python is invoked.
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

const PACKAGES: &str = include_str!("cuda-runtime.lock");

pub fn install(out: &Path, target: &str, variant: &str) -> PathBuf {
    println!("cargo:rerun-if-changed=cuda-runtime.lock");
    let arch = target.split('-').next().unwrap();
    let zlib = match arch {
        "x86_64" => "b9b82fcdbb893ca7470e326b9fa305ece5bf5f462004a622cfdbf54c5d8de4e5",
        "aarch64" => "5cffc1e0520c7e5d281650483a7118d3a3d2a8c9edc9222000ffdfdc3101e343",
        _ => panic!("no managed CUDA runtime for {target}"),
    };
    let digest = Sha256::digest(PACKAGES.as_bytes());
    let root = out.join(format!("{variant}-{arch}-{digest:x}"));
    if root.is_dir() {
        return root;
    }
    let stage = tempfile::tempdir_in(out).expect("stage CUDA runtime");
    let bundle = stage.path().join("bundle");
    let archive = stage.path().join("download");
    for line in PACKAGES.lines().filter(|l| !l.starts_with('#')) {
        let mut fields = line.split_whitespace();
        if fields.next() != Some(variant) || fields.next() != Some(arch) {
            continue;
        }
        let hash = fields.next().unwrap();
        let url = fields.next().unwrap();
        super::prebuilt::download(url, &archive, hash);
        extract_wheel(&archive, &bundle).unwrap_or_else(|err| panic!("extract {url}: {err}"));
    }
    // cuDNN also needs libz.so.1 on minimal hosts such as NixOS. Preserve its license.
    let url = format!("https://github.com/JuliaBinaryWrappers/Zlib_jll.jl/releases/download/Zlib-v1.3.2%2B0/Zlib.v1.3.2.{arch}-linux-gnu.tar.gz");
    super::prebuilt::download(&url, &archive, zlib);
    assert!(Command::new("tar")
        .arg("-xzf")
        .arg(&archive)
        .arg("-C")
        .arg(&bundle)
        .args([
            "lib/libz.so.1",
            "lib/libz.so.1.3.2",
            "share/doc/zlib/zlib/LICENSE"
        ])
        .status()
        .expect("extract zlib")
        .success());
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
