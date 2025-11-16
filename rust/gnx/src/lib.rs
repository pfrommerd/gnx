pub mod array;
pub mod backend;
pub mod expr;

// Re-export gnx-graph as graph
// and gnx-io as io
pub use gnx_graph as graph;
pub use gnx_io as io;

// Include all util modules from gnx-graph and gnx-io
pub mod util {
    pub use gnx_graph::util::*;
}

pub mod transforms;

pub use transforms::jit;

extern crate self as gnx;
