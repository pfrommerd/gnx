use gnx::{Array, lax};

pub fn dims(x: &Array) -> Vec<usize> {
    x.fixed_dims().expect("fixed checkpoint shapes")
}

/// Decoder contractions use leading batch axes and one contracting axis per operand.
pub fn dot(a: &Array, b: &Array, ac: usize, bc: usize, batch: usize) -> Array {
    lax::dot_general(
        a,
        b,
        lax::DotDimensions {
            lhs_contracting: vec![ac],
            rhs_contracting: vec![bc],
            lhs_batch: (0..batch).collect(),
            rhs_batch: (0..batch).collect(),
        },
    )
}
