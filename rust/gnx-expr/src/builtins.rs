use crate::Op;

// Built-in basic
// operation types

pub enum Math {
    Sum { axis: usize },
    Mul { axis: usize },
}

// Built-in transformations
pub trait Vectorize {
    fn vectorized(&self) -> Option<Self>;
}

impl Vectorize for Op {
    fn vectorized(&self) -> Option<Self> {
        None
    }
}