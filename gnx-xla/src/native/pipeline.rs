use std::sync::LazyLock;

use gnx_expr::Expr;
use gnx_expr::pass::{CheckSupported, Dce, FnPass, Pipeline};
use gnx_transforms::ExpandTransforms;
use xla::XlaComputation;

use super::lower::{self, XlaLowering};

/// The passes `XlaDevice::compile` runs: expand `ad` transformations, remove
/// dead code, check that every operation has an XLA lowering, and lower to HLO.
pub fn pipeline() -> &'static Pipeline<XlaComputation> {
    static PIPELINE: LazyLock<Pipeline<XlaComputation>> = LazyLock::new(|| {
        Pipeline::new()
            .with(ExpandTransforms)
            .with(Dce)
            .with(CheckSupported::<dyn XlaLowering>::new())
            .finalize(FnPass::new("xla.lower", |expr: Expr| lower::lower(&expr)))
    });
    &PIPELINE
}
