//! Keep Rust and native XLA log records above the examples' progress bars.
use indicatif::{MultiProgress, ProgressDrawTarget};
use std::io;

pub struct LogGuard {
    #[cfg(target_os = "linux")]
    _capture: Option<linux::Capture>,
}

/// Keep the returned guard alive for as long as native logging is needed.
pub fn init(spec: &str, default_filter: &str) -> io::Result<(MultiProgress, LogGuard)> {
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter));
    #[cfg(target_os = "linux")]
    let capture = if spec == "default" || spec.starts_with("xla/cuda:") {
        Some(linux::Output::new(&mut builder)?)
    } else {
        None
    };
    #[cfg(target_os = "linux")]
    let target = match &capture {
        Some(output) => output.draw_target()?,
        None => ProgressDrawTarget::stderr(),
    };
    #[cfg(not(target_os = "linux"))]
    let target = {
        let _ = spec;
        ProgressDrawTarget::stderr()
    };
    let progress = MultiProgress::with_draw_target(target);
    let logger = builder.build();
    let level = logger.filter();
    indicatif_log_bridge::LogWrapper::new(progress.clone(), logger)
        .try_init()
        .map_err(io::Error::other)?;
    log::set_max_level(level);
    #[cfg(target_os = "linux")]
    let _capture = capture
        .map(|output| output.capture(progress.clone()))
        .transpose()?;
    Ok((
        progress,
        LogGuard {
            #[cfg(target_os = "linux")]
            _capture,
        },
    ))
}

#[cfg(target_os = "linux")]
mod linux {
    use indicatif::{MultiProgress, ProgressDrawTarget};
    use regex::bytes::Regex;
    use std::{
        fs::File,
        io::{self, BufRead, BufReader, Write},
        os::fd::AsFd,
        sync::LazyLock,
        thread,
    };

    // The managed CUDA plugin doesn't export an Abseil log sink. Keep stderr
    // redirection in the example process, and give the logger and UI their own
    // handle to the original terminal so neither writes back into the pipe.
    pub struct Output(File);
    impl Output {
        pub fn new(builder: &mut env_logger::Builder) -> io::Result<Self> {
            let stderr = File::from(io::stderr().as_fd().try_clone_to_owned()?);
            let color = console::colors_enabled_stderr();
            if std::env::var("RUST_LOG_STYLE").map_or(true, |s| s == "auto") {
                builder.write_style(if color {
                    env_logger::WriteStyle::Always
                } else {
                    env_logger::WriteStyle::Never
                });
            }
            builder.target(env_logger::Target::Pipe(Box::new(stderr.try_clone()?)));
            Ok(Self(stderr))
        }
        pub fn draw_target(&self) -> io::Result<ProgressDrawTarget> {
            // indicatif treats custom terminals as stdout for style detection.
            console::set_colors_enabled(console::colors_enabled_stderr());
            Ok(ProgressDrawTarget::term(
                console::Term::read_write_pair(io::stdin(), self.0.try_clone()?),
                20,
            ))
        }
        pub fn capture(self, progress: MultiProgress) -> io::Result<Capture> {
            let (reader, writer) = io::pipe()?;
            let mut output = self.0.try_clone()?;
            let reader = thread::Builder::new().name("xla-stderr".into()).spawn(
                move || -> io::Result<()> {
                    let mut reader = BufReader::new(reader);
                    let mut line = Vec::new();
                    while reader.read_until(b'\n', &mut line)? != 0 {
                        if !forward(&line) {
                            progress.suspend(|| output.write_all(&line))?;
                        }
                        line.clear();
                    }
                    Ok(())
                },
            )?;
            rustix::stdio::dup2_stderr(writer)?;
            Ok(Capture {
                stderr: self.0,
                reader: Some(reader),
            })
        }
    }

    pub struct Capture {
        stderr: File,
        reader: Option<thread::JoinHandle<io::Result<()>>>,
    }
    impl Drop for Capture {
        fn drop(&mut self) {
            if rustix::stdio::dup2_stderr(&self.stderr).is_ok() {
                let _ = self.reader.take().unwrap().join();
            }
        }
    }

    fn forward(line: &[u8]) -> bool {
        static PREFIX: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(concat!(
                r"^([IWEF])[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2,}\.[0-9]{6} +",
                r"[0-9]+ ([^\r\n\]]+):([0-9]+)\] "
            ))
            .unwrap()
        });
        let Some(fields) = PREFIX.captures(line) else {
            return false;
        };
        let level = match &fields[1] {
            b"I" => log::Level::Info,
            b"W" => log::Level::Warn,
            _ => log::Level::Error,
        };
        if log::log_enabled!(target: "xla", level) {
            let file = String::from_utf8_lossy(&fields[2]);
            let number = std::str::from_utf8(&fields[3])
                .ok()
                .and_then(|s| s.parse().ok());
            let text = &line[fields.get(0).unwrap().end()..];
            log::logger().log(
                &log::Record::builder()
                    .target("xla")
                    .level(level)
                    .file(Some(&file))
                    .line(number)
                    .args(format_args!(
                        "{}",
                        String::from_utf8_lossy(text.strip_suffix(b"\n").unwrap_or(text))
                    ))
                    .build(),
            );
        }
        true
    }
}
