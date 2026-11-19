use std::{env, fs::File, io, path::Path};

// PyPI libclang 18.1.1 wheels contain standalone shared libraries. Bindgen
// runs on HOST even when the C++ compiler is building XLA bindings for TARGET.
fn wheel(host: &str) -> Option<(&'static str, &'static str)> {
    Some(match host {
        "x86_64-unknown-linux-gnu" => ("1d/fc/716c1e62e512ef1c160e7984a73a5fc7df45166f2ff3f254e71c58076f7c/libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl", "c533091d8a3bbf7460a00cb6c1a71da93bffe148f172c7d03b1c31fbf8aa2a0b"),
        "aarch64-unknown-linux-gnu" => ("3c/3d/f0ac1150280d8d20d059608cf2d5ff61b7c3b7f7bcf9c0f425ab92df769a/libclang-18.1.1-py2.py3-none-manylinux2014_aarch64.whl", "54dda940a4a0491a9d1532bf071ea3ef26e6dbaf03b5000ed94dd7174e8f9592"),
        "x86_64-apple-darwin" => ("e2/e5/fc61bbded91a8830ccce94c5294ecd6e88e496cc85f6704bf350c0634b70/libclang-18.1.1-py2.py3-none-macosx_10_9_x86_64.whl", "6f14c3f194704e5d09769108f03185fce7acaf1d1ae4bbb2f30a72c2400cb7c5"),
        "aarch64-apple-darwin" => ("4b/49/f5e3e7e1419872b69f6f5e82ba56e33955a74bd537d8a1f5f1eff2f3668a/libclang-18.1.1-1-py2.py3-none-macosx_11_0_arm64.whl", "0b2e143f0fac830156feb56f9231ff8338c20aecfe72b4ffe96f19e5a1dbb69a"),
        _ => return None,
    })
}

pub fn load(out: &Path) {
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    let Err(error) = clang_sys::load() else {
        return;
    };
    if let Some(path) = env::var_os("LIBCLANG_PATH") {
        panic!("Cannot load libclang with LIBCLANG_PATH={path:?}: {error}");
    }
    let host = env::var("HOST").unwrap();
    let (asset, digest) = wheel(&host).unwrap_or_else(|| {
        panic!("No prebuilt libclang for build host {host}; install libclang and set LIBCLANG_PATH: {error}")
    });
    let dir = out.join(format!("libclang-18.1.1-{digest}"));
    let name = if cfg!(target_os = "macos") {
        "libclang.dylib"
    } else {
        "libclang.so"
    };
    let library = dir.join(name);
    if !library.is_file() {
        let stage = tempfile::tempdir_in(out).expect("create libclang staging directory");
        let archive = stage.path().join("libclang.whl");
        super::prebuilt::download(
            &format!("https://files.pythonhosted.org/packages/{asset}"),
            &archive,
            digest,
        );
        let mut zip =
            zip::ZipArchive::new(File::open(archive).unwrap()).expect("read libclang wheel");
        let mut member = zip
            .by_name(&format!("libclang-18.1.1.data/platlib/clang/native/{name}"))
            .expect("find shared library in libclang wheel");
        let staged = stage.path().join(name);
        io::copy(&mut member, &mut File::create(&staged).unwrap()).expect("extract libclang");
        std::fs::create_dir_all(&dir).expect("create libclang cache");
        std::fs::rename(staged, &library).expect("install verified libclang");
    }
    env::set_var("LIBCLANG_PATH", &dir);
    clang_sys::load()
        .unwrap_or_else(|e| panic!("Cannot load downloaded {}: {e}", library.display()));
}
