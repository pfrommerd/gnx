
pub use gnx_transforms as transforms;
pub use gnx_transforms::callable as callable;

// Re-export gnx-graph as graph
// and gnx-io as io
pub use gnx_graph as graph;

pub use gnx_expr::{
    backend,
    device,
    array
};
// expr module contains the expr, trace, and value modules from gnx-expr
pub mod expr {
    pub use gnx_expr::expr::*;
    pub use gnx_expr::value::*;
}

// Include all util modules from gnx-graph and gnx-io
pub mod util {
    pub use gnx_graph::util::*;
    pub use gnx_io::util::*;
}

// for convenience, re-export a handful of top-level namespace items
// - Array, Shape, Device, and devices() from gnx-expr
// - Graph and GraphId from gnx-graph
pub use gnx_expr::array::{Array, Shape};
pub use gnx_expr::device::Device;
pub use gnx_expr::backend::devices;
pub use gnx_graph::{Graph, GraphId};

pub use gnx_transforms::{jit, transform, Callable};

extern crate self as gnx;