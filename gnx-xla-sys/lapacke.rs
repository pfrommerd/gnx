use std::{
    env, fs,
    path::{Path, PathBuf},
};

// scipy-openblas32 ships LP64 LAPACKE headers, shared library and its Fortran
// runtime together. Choose by Cargo TARGET, not by the build host.
const VERSION: &str = "0.3.34.481.3";
fn wheel(target: &str) -> Option<(&'static str, &'static str, &'static str)> {
    Some(match target {
        "x86_64-unknown-linux-gnu" => (
            "75/93/56ce22e41d59a45b885ac67f53425cd012329cb1e4de3285bebc4f8ad4bc",
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64",
            "0595f94cd58c1f8a0d90c975621f4587337a003916b1979e834ebdf2cb5d9504",
        ),
        "aarch64-unknown-linux-gnu" => (
            "04/3e/19a35f6b50552df2fa9464b526d0a4f98f4f9e7789b2adbb813d598640f7",
            "manylinux_2_27_aarch64.manylinux_2_28_aarch64",
            "e085dddbc5b9aa130f83def515d31c440e69f387a90c7e63b154a5575b4466bc",
        ),
        _ => return None,
    })
}

fn install(out: &Path, target: &str) -> PathBuf {
    let (asset, tag, digest) = wheel(target).unwrap_or_else(|| panic!("No system LAPACKE or prebuilt fallback for {target}; install LAPACKE or set LAPACKE_INCLUDE_DIR and LAPACKE_LIB_DIR"));
    let root = out.join(format!("openblas32-{digest}"));
    if !root.is_dir() {
        let stage = tempfile::tempdir_in(out).expect("stage LAPACKE package");
        let whl = stage.path().join("openblas.whl");
        let url = format!(
            "https://files.pythonhosted.org/packages/{asset}/scipy_openblas32-{VERSION}-py3-none-{tag}.whl"
        );
        super::prebuilt::download(&url, &whl, digest);
        let unpacked = stage.path().join("unpacked");
        zip::ZipArchive::new(fs::File::open(whl).unwrap())
            .unwrap()
            .extract(&unpacked)
            .expect("unpack LAPACKE wheel");
        fs::rename(unpacked, &root).expect("cache verified LAPACKE package");
    }
    root.join("scipy_openblas32")
}

pub fn configure(bridge: &mut cc::Build) -> (&'static str, Option<PathBuf>) {
    println!("cargo:rerun-if-env-changed=LAPACKE_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=LAPACKE_LIB_DIR");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let include = env::var_os("LAPACKE_INCLUDE_DIR");
    let lib = env::var_os("LAPACKE_LIB_DIR").map(PathBuf::from);
    if let Some(ref dir) = include {
        bridge.include(dir);
    }

    // Compile AND link: a header-only or runtime-only system installation is
    // not sufficient. Never execute target code, including when cross-building.
    let source = out.join("probe-lapacke.cc");
    fs::write(&source, "#include <lapacke.h>\nstatic_assert(sizeof(lapack_int) == 4);\nint main() { double x = 1; lapack_int p; return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, 1, 1, &x, 1, &p); }\n").unwrap();
    let mut cmd = bridge.get_compiler().to_command();
    if let Some(ref dir) = lib {
        cmd.arg("-L").arg(dir);
    }
    let probe = cmd
        .arg(&source)
        .arg("-llapacke")
        .arg("-o")
        .arg(out.join("probe-lapacke"))
        .output()
        .expect("probe system LAPACKE");
    if probe.status.success() {
        return ("lapacke", lib);
    }
    if include.is_some() || lib.is_some() {
        panic!(
            "Invalid LAPACKE override: {}",
            String::from_utf8_lossy(&probe.stderr)
        );
    }

    let root = install(&out, &env::var("TARGET").unwrap());
    bridge.include(root.join("include"));
    bridge.define("GNX_SCIPY_OPENBLAS", None);
    ("scipy_openblas", Some(root.join("lib")))
}
