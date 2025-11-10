mod deserialize;
mod impls;
mod serialize;
mod visitor;

pub mod fs;
pub use fs::{Origin, Resource};

pub mod json;

pub use deserialize::*;
pub use serialize::*;
pub use visitor::*;