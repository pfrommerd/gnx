//! Host LAPACKE kernels for XLA typed FFI calls.
use crate::{Result, Shape, XlaOp};

/// Kernel IDs used by the CPU handler. Keep in sync with xla_rs/lapack.cc.
#[derive(Clone, Copy)]
#[repr(i64)]
pub enum Kernel {
    Lu,
    Hessenberg,
    Tridiagonal,
    Schur,
    Eig,
    Svd,
    PivotedQr,
    TridiagonalSolve,
    PivotsToPermutation,
}

impl Kernel {
    /// Emit a CPU LAPACK call with the supplied result shapes and kernel flags.
    #[cfg(feature = "lapack")]
    pub fn call(self, flags: i64, inputs: &[XlaOp], outputs: &[Shape]) -> Result<XlaOp> {
        use std::sync::OnceLock;
        extern "C" {
            fn gnx_lapack_execute(frame: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
        }
        let name = c"gnx_lapack";
        static REGISTRATION: OnceLock<std::result::Result<(), String>> = OnceLock::new();
        REGISTRATION
            .get_or_init(|| {
                // This statically linked XLA FFI handler is valid for the process lifetime.
                unsafe {
                    crate::ffi::register_foreign_handler(
                        name,
                        c"Host",
                        gnx_lapack_execute as *const _,
                    )
                }
                .map_err(|err| err.to_string())
            })
            .as_ref()
            .map_err(|err| std::io::Error::other(err.clone()))?;
        let attrs = format!(
            "{{operation = {} : i64, flags = {flags} : i64}}",
            self as i64
        );
        XlaOp::ffi_call(inputs, name, &attrs, outputs)
    }

    #[cfg(not(feature = "lapack"))]
    pub fn call(self, _: i64, _: &[XlaOp], _: &[Shape]) -> Result<XlaOp> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "CPU LAPACK kernels require the xla/lapack feature",
        )
        .into())
    }
}
