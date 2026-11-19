fn main() {
    #[cfg(feature = "mesh")]
    {
        println!("cargo:rerun-if-env-changed=XLA_EXTENSION_DIR");
        println!("cargo:rerun-if-changed=src/mesh/shardy.cc");
        let root = std::env::var_os("XLA_EXTENSION_DIR")
            .map(std::path::PathBuf::from)
            .expect("the mesh feature requires XLA_EXTENSION_DIR");
        cc::Build::new()
            .cpp(true)
            // Match the release/C++17 ABI of the pinned XLA binary.
            .std("c++17")
            .define("NDEBUG", None)
            .flag_if_supported("-DLLVM_ON_UNIX=1")
            .flag_if_supported("-DLLVM_VERSION_STRING=")
            .warnings(false)
            .include(root.join("include"))
            .file("src/mesh/shardy.cc")
            .compile("gnx_shardy");
    }
}
