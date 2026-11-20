use tch::CModule;

/// Exercise the serialized method, not the in-memory module that made it.
pub fn reload(module: &CModule) -> CModule {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "gnx-torch-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("module.pt");
    module.save(&path).unwrap();
    let loaded = CModule::load(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(dir).unwrap();
    loaded
}
