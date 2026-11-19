fn main() {
    #[cfg(feature = "native")]
    {
        println!("cargo:rerun-if-env-changed=XLA_EXTENSION_DIR");
        println!("cargo:rerun-if-changed=src/native/cpu.cc");
        let root = std::env::var_os("XLA_EXTENSION_DIR")
            .map(std::path::PathBuf::from)
            .expect("the native feature requires XLA_EXTENSION_DIR");
        let mut bridge = cc::Build::new();
        bridge
            .cpp(true)
            // Match the release/C++17 ABI of the pinned XLA binary.
            .std("c++17")
            .define("NDEBUG", None)
            .flag_if_supported("-DLLVM_ON_UNIX=1")
            .flag_if_supported("-DLLVM_VERSION_STRING=")
            .warnings(false)
            .include(root.join("include"))
            .file("src/native/cpu.cc");
        println!("cargo:rerun-if-changed=src/native/lapack.cc");
        println!("cargo:rerun-if-env-changed=LAPACKE_INCLUDE_DIR");
        println!("cargo:rerun-if-env-changed=LAPACKE_LIB_DIR");
        if let Some(dir) = std::env::var_os("LAPACKE_INCLUDE_DIR") {
            bridge.include(dir);
        }
        if let Some(dir) = std::env::var_os("LAPACKE_LIB_DIR") {
            println!(
                "cargo:rustc-link-search=native={}",
                std::path::Path::new(&dir).display()
            );
        }
        bridge
            .file("src/native/lapack.cc")
            .compile("gnx_xla_bridge");
        println!("cargo:rustc-link-lib=lapacke");
    }
}
