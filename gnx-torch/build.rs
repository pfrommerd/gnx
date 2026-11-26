use std::{env, path::PathBuf, process::Stdio};

// Query the *target linker* via the same C++ driver that builds the bridge.
// In particular, Nix wrappers can inject -L paths at link time that do not
// appear in the compiler's -print-file-name output or in ld.so's defaults.
fn linux_cxx_runpaths(compiler: &cc::Tool) -> Vec<PathBuf> {
    let probe = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("cxx-runtime-probe.so");
    let output = compiler
        .to_command()
        .args([
            "-shared",
            "-x",
            "c++",
            "-",
            "-Wl,-t,-l:libstdc++.so.6,-l:libgcc_s.so.1",
            "-o",
        ])
        .arg(probe)
        .stdin(Stdio::null())
        .output()
        .expect("query C++ runtime libraries");
    assert!(
        output.status.success(),
        "C++ runtime link probe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let trace = String::from_utf8_lossy(&output.stdout);
    let mut dirs = Vec::new();
    for soname in ["libstdc++.so.6", "libgcc_s.so.1"] {
        let path = trace
            .lines()
            .map(str::trim)
            .map(std::path::Path::new)
            .find(|p| p.ends_with(soname) && p.is_file())
            .unwrap_or_else(|| panic!("target linker did not report a path for {soname}: {trace}"));
        let dir = path
            .parent()
            .unwrap()
            .canonicalize()
            .expect("C++ runtime directory");
        if !dirs.contains(&dir) {
            dirs.push(dir);
        }
    }
    dirs
}

fn main() {
    println!("cargo:rerun-if-changed=native/control_flow.cpp");
    println!("cargo:rerun-if-changed=native/cuda.cpp");
    println!("cargo:rerun-if-changed=native/save.cpp");
    // torch-sys resolves the runtime before compiling libtch. Reuse its
    // exact configuration, including for explicit/Python installations.
    let includes = env::var_os("DEP_TCH_GNX_INCLUDES").expect("torch-sys header directories");
    let abi = env::var("DEP_TCH_GNX_ABI").expect("torch-sys C++ ABI");
    let lib = env::var("DEP_TCH_GNX_LIBDIR").expect("torch-sys library directory");
    let mut bridge = cc::Build::new();
    // NVRTC dlopens its builtins by SONAME. Its wheel has no RUNPATH
    // pointing at the managed lib directory; the CUDA bridge loads this
    // dependency on first GPU use, just like libtorch_cuda below.
    let builtins = PathBuf::from(&lib).join("libnvrtc-builtins.so.13.0");
    if builtins.is_file() {
        bridge.define("GNX_NVRTC_BUILTINS", format!("{builtins:?}").as_str());
    }
    bridge
        .cpp(true)
        .std("c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .includes(env::split_paths(&includes))
        .define("_GLIBCXX_USE_CXX11_ABI", abi.as_str())
        .define("GNX_TORCH_LIB_DIR", format!("{lib:?}").as_str())
        .file("native/control_flow.cpp")
        .file("native/cuda.cpp")
        .file("native/save.cpp");
    let Some(archive) = env::var_os("DEP_TCH_GNX_ARCHIVE") else {
        bridge.compile("gnx_torch_control_flow");
        return;
    };
    let mac = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let shim = out.join(if mac {
        "libgnx_torch_native.dylib"
    } else {
        "libgnx_torch_native.so"
    });
    let cxx = bridge.get_compiler();
    let mut link = cxx.to_command();
    link.arg("-o").arg(&shim).args(
        bridge
            .pic(true)
            .cargo_metadata(false)
            .compile_intermediates(),
    );
    // Like libxla_rs, make the backend's native bridge self-locating. Its
    // RUNPATH applies to its direct dependencies, independent of the Rust binary.
    if mac {
        link.args(["-dynamiclib", "-install_name"])
            .arg(&shim)
            .args(["-Xlinker", "-force_load", "-Xlinker"])
            .arg(archive)
            .arg("-Wl,-undefined,dynamic_lookup");
    } else {
        link.args(["-shared", "-Xlinker", "-soname", "-Xlinker"])
            .arg(&shim)
            .arg("-Wl,--whole-archive")
            .arg(archive)
            .arg("-Wl,--no-whole-archive");
        if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu") {
            for dir in linux_cxx_runpaths(&cxx) {
                link.args(["-Xlinker", "-rpath", "-Xlinker"]).arg(dir);
            }
        }
    }
    link.arg("-L")
        .arg(&lib)
        .args(["-ltorch_cpu", "-lc10", "-Xlinker", "-rpath", "-Xlinker"])
        .arg(&lib);
    assert!(
        link.status().expect("link Torch native bridge").success(),
        "linking {shim:?} failed"
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=gnx_torch_native");
}
