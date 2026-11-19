//! GNX's Python-free XLA backend (enable the `native` feature).

#[cfg(feature = "native")]
mod native;
#[cfg(feature = "native")]
pub use native::*;

#[cfg(feature = "mesh")]
pub mod mesh;
#[cfg(feature = "mesh")]
pub use mesh::{Mesh, MeshExecutable, Worker, WorkerInfo};
