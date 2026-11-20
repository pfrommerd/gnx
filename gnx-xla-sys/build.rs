use gnx_cuda_build as cuda_runtime;
#[cfg(feature = "lapack")]
mod lapacke;
mod libclang;
mod prebuilt;

use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Eq, PartialEq)]
enum OS {
    Linux,
    #[allow(clippy::enum_variant_names)]
    MacOS,
    Windows,
}

impl OS {
    fn get() -> Self {
        let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
        match os.as_str() {
            "linux" => Self::Linux,
            "macos" => Self::MacOS,
            "windows" => Self::Windows,
            os => panic!("Unsupported system {os}"),
        }
    }
}

fn make_shared_lib<P: AsRef<Path>>(
    os: OS,
    xla_dir: P,
    out_dir: &Path,
    cuda: Option<(&Path, &str)>,
) {
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    let mut build = cc::Build::new();
    println!("cargo:rerun-if-changed=xla_rs/gpu_client.h");
    if let Some((root, variant)) = cuda {
        build.define("XLA_RS_CUDA_DIR", format!("{:?}", root).as_str());
        build.define("XLA_RS_CUDA_MAJOR", variant.strip_prefix("cuda").unwrap());
    }
    if let Some(overlay) = patched_headers(xla_dir.as_ref(), out_dir) {
        build.include(overlay);
    }
    match os {
        OS::Linux | OS::MacOS => {
            build
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .flag("-std=c++17")
                .flag("-Wno-deprecated-declarations")
                .flag("-DLLVM_ON_UNIX=1")
                .flag("-DLLVM_VERSION_STRING=")
                // The pre-built xla extension library is compiled in release
                // mode. Some headers, e.g. tsl/concurrency/async_value.h,
                // declare additional struct members in debug builds so NDEBUG
                // has to be set to avoid ABI mismatches (which result in
                // segfaults when running computations).
                .flag("-DNDEBUG")
                .file("xla_rs/xla_rs.cc");
        }
        OS::Windows => {
            build
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .flag("/DNDEBUG")
                .file("xla_rs/xla_rs.cc");
        }
    };
    #[cfg(feature = "pjrt")]
    {
        println!("cargo:rerun-if-changed=xla_rs/pjrt.cc");
        build.file("xla_rs/pjrt.cc");
    }
    #[cfg(feature = "lapack")]
    let library = {
        println!("cargo:rerun-if-changed=xla_rs/lapack.cc");
        let library = lapacke::configure(&mut build);
        build.file("xla_rs/lapack.cc");
        library
    };
    let extension_lib = xla_dir.as_ref().join("lib");
    if os == OS::Linux {
        let objects = build.compile_intermediates();
        let cxx = build.get_compiler();
        let shim = out_dir.join("libxla_rs.so");
        let mut link = cxx.to_command();
        // The absolute SONAME lets downstream executables locate this local build
        // artifact without their own build.rs, RPATH or loader environment.
        link.arg("-shared")
            .arg("-o")
            .arg(&shim)
            .args(["-Xlinker", "-soname", "-Xlinker"])
            .arg(&shim)
            .arg("-Wl,-z,defs")
            .args(objects)
            .arg("-L")
            .arg(&extension_lib)
            .arg("-lxla_extension");
        let mut runpaths = Vec::new();
        // RUNPATH applies only to direct dependencies. If XLA needs libcuda,
        // make it a direct dependency too so the loader resolves it using our
        // RUNPATH. Keep the stable driver profile path across NixOS upgrades.
        let driver_dir = Path::new("/run/opengl-driver/lib");
        let driver = driver_dir.join("libcuda.so.1");
        if cuda.is_none() && env::var_os("HOST") == env::var_os("TARGET") && driver.is_file() {
            let tool = cxx
                .to_command()
                .arg("-print-prog-name=readelf")
                .output()
                .expect("locate readelf");
            let readelf = String::from_utf8_lossy(&tool.stdout);
            let deps = std::process::Command::new(readelf.trim())
                .env("LC_ALL", "C")
                .arg("-d")
                .arg(extension_lib.join("libxla_extension.so"))
                .output()
                .expect("inspect XLA dependencies with readelf");
            assert!(
                deps.status.success(),
                "readelf failed: {}",
                String::from_utf8_lossy(&deps.stderr)
            );
            if String::from_utf8_lossy(&deps.stdout)
                .lines()
                .any(|line| line.contains("(NEEDED)") && line.contains("[libcuda.so.1]"))
            {
                link.arg("-Wl,--push-state,--no-as-needed").arg(driver);
                // PJRT also dlopens NVML to enumerate visible GPUs.
                let nvml = driver_dir.join("libnvidia-ml.so.1");
                if nvml.is_file() {
                    link.arg(nvml);
                }
                link.arg("-Wl,--pop-state");
                runpaths.push(driver_dir.to_path_buf());
            }
        }
        runpaths.push(extension_lib);
        #[cfg(feature = "lapack")]
        {
            let (name, dir) = library;
            if let Some(dir) = dir {
                link.arg("-L").arg(&dir);
                runpaths.push(dir);
            }
            link.arg(format!("-l{name}"));
        }
        for soname in ["libstdc++.so.6", "libgcc_s.so.1"] {
            let output = cxx
                .to_command()
                .arg(format!("-print-file-name={soname}"))
                .output()
                .expect("query target C++ runtime");
            let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
            if output.status.success() && path.is_absolute() && path.is_file() {
                // ELF retains the library's SONAME, even for an absolute link input.
                // Record its actual directory on the object that needs it.
                link.arg(&path);
                let dir = path.parent().unwrap().to_owned();
                if !runpaths.contains(&dir) {
                    runpaths.push(dir);
                }
            }
        }
        for dir in runpaths {
            link.args(["-Xlinker", "-rpath", "-Xlinker"]).arg(dir);
        }
        assert!(
            link.status().expect("link XLA C++ bindings").success(),
            "failed to link {shim:?}"
        );
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=dylib=xla_rs");
    } else {
        build.compile("xla_rs");
        #[cfg(feature = "lapack")]
        {
            let (name, dir) = library;
            if let Some(dir) = dir {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
            println!("cargo:rustc-link-lib={name}");
        }
        println!("cargo:rustc-link-search=native={}", extension_lib.display());
        if os == OS::MacOS {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                extension_lib.display()
            );
        }
        println!("cargo:rustc-link-lib=xla_extension");
    }
}

/// The pre-built headers are compiled with clang upstream and some of them do
/// not go through gcc. Rather than requiring clang, the offending headers are
/// copied with a minimal fix into an include directory that is searched before
/// the xla_extension one. Returns that directory when a header was patched.
///
/// xla_extension 0.11.0, `xla/tsl/concurrency/future.h`: the const& overload
/// of `Future<T>::Flatten` is declared with `!is_move_only` in its enable_if
/// but defined with `!Future<T>::is_move_only`; gcc does not consider the two
/// expressions equivalent and rejects the definition ("no declaration
/// matches"). Both are spelled as the underlying trait, which is what
/// `Future<T>::is_move_only` evaluates to.
fn patched_headers(xla_dir: &Path, out_dir: &Path) -> Option<PathBuf> {
    let rel = Path::new("xla/tsl/concurrency/future.h");
    let src = xla_dir.join("include").join(rel);
    println!("cargo:rerun-if-changed={}", src.display());
    let content = std::fs::read_to_string(&src).ok()?;
    let decl = "std::enable_if_t<internal::IsFuture<U>::value &&\n                                             !is_move_only>* = nullptr>";
    let def = "std::enable_if_t<internal::IsFuture<U>::value &&\n                                       !Future<T>::is_move_only>*>";
    if !content.contains(decl) || !content.contains(def) {
        return None;
    }
    let trait_ = "internal::IsMoveOnly<T>::value";
    let patched = content
        .replace(decl, &decl.replace("!is_move_only", &format!("!{trait_}")))
        .replace(
            def,
            &def.replace("!Future<T>::is_move_only", &format!("!{trait_}")),
        );
    let overlay = out_dir.join("include");
    let dst = overlay.join(rel);
    std::fs::create_dir_all(dst.parent().unwrap()).ok()?;
    std::fs::write(&dst, patched).ok()?;
    Some(overlay)
}

fn main() {
    let os = OS::get();
    println!("cargo:rerun-if-env-changed=XLA_EXTENSION_DIR");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    libclang::load(&out_path);
    let bindings = bindgen::Builder::default()
        .header("xla_rs/xla_rs.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    bindings
        .write_to_file(out_path.join("c_xla.rs"))
        .expect("Couldn't write bindings!");

    // Exit early on docs.rs as the C++ library would not be available.
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }
    let target = env::var("TARGET").unwrap();
    let variant = prebuilt::variant(&target);
    let (xla_dir, cuda_dir) = match env::var_os("XLA_EXTENSION_DIR") {
        Some(dir) => (
            std::fs::canonicalize(&dir)
                .unwrap_or_else(|e| panic!("Cannot resolve XLA_EXTENSION_DIR {dir:?}: {e}")),
            None,
        ),
        None if matches!(variant.as_str(), "cuda12" | "cuda13") => {
            assert!(
                matches!(
                    target.as_str(),
                    "x86_64-unknown-linux-gnu" | "aarch64-unknown-linux-gnu"
                ),
                "no managed CUDA plugin for {target}"
            );
            (
                prebuilt::install(&out_path, &target, "cpu"),
                Some(cuda_runtime::install(&out_path, &target, &variant)),
            )
        }
        None => (prebuilt::install(&out_path, &target, &variant), None),
    };
    println!("cargo:root={}", xla_dir.display());
    make_shared_lib(
        os,
        &xla_dir,
        &out_path,
        cuda_dir.as_deref().map(|root| (root, variant.as_str())),
    );
}
