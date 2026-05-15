mod util;

pub mod expr;
pub use expr::trace;
pub use expr::value;

pub mod backend;
pub mod device;
pub mod array;