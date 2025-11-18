pub mod callable;
pub mod transforms;

pub use callable::Callable;
pub use transforms::{jit, transform};

// Re-export gnx-graph as graph
// and gnx-io as io
pub use gnx_graph as graph;
pub use gnx_expr as expr;

pub use gnx_expr::backend;
pub use expr::Array;

// Include all util modules from gnx-graph and gnx-io
pub mod util {
    pub use gnx_graph::util::*;
    pub use gnx_io::util::*;
}

extern crate self as gnx;