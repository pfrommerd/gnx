use std::sync::Once;

pub(crate) fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe { crate::c_lib::gnx_install_log_sink(Some(forward)) });
}

unsafe extern "C" fn forward(
    severity: i32,
    file: *const u8,
    file_len: usize,
    line: i32,
    text: *const u8,
    text_len: usize,
) {
    // User-provided loggers can panic. Never unwind through the native sink.
    let _ = std::panic::catch_unwind(|| {
        let level = match severity {
            0 => log::Level::Info,
            1 => log::Level::Warn,
            _ => log::Level::Error, // Includes FATAL; Abseil handles termination.
        };
        if !log::log_enabled!(target: "xla", level) {
            return;
        }
        // Abseil owns both string views until Send returns. Empty views may be null.
        let bytes = |ptr, len| {
            if len == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(ptr, len) }
            }
        };
        let file = String::from_utf8_lossy(bytes(file, file_len));
        let text = String::from_utf8_lossy(bytes(text, text_len));
        log::logger().log(
            &log::Record::builder()
                .target("xla")
                .level(level)
                .file(Some(&file))
                .line(line.try_into().ok())
                .args(format_args!("{text}"))
                .build(),
        );
    });
}
