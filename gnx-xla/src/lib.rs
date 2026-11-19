//! GNX's Python-free XLA backend (enable the `native` feature).

#[cfg(feature = "native")]
mod native;
#[cfg(feature = "native")]
pub use native::*;
