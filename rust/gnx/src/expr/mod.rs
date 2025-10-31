use std::borrow::Cow;

use crate::array::{Array, ArrayRef};
use crate::graph::LeafUnion;

mod builtins;
mod trace;
mod util;

pub use trace::{ArrayRefTracer, ArrayTracer};
pub use util::{TypeInfo, TypeValue};

pub struct Op {
    dialect: Cow<'static, str>,
    name: Cow<'static, str>,
}

pub struct Expr {}

pub enum Value {
    Array(Array),
    ArrayRef(ArrayRef),
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dialect.as_ref() {
            "builtin" => write!(f, "{}", self.name),
            d => write!(f, "{}::{}", d, self.name),
        }
    }
}
