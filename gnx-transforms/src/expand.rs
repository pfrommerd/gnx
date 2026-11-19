//! Expand function-transformation operations (`map.map`, `ad.jvp`, `ad.vjp`,
//! and any other operation implementing [`Transform`]) into ordinary ones.

use gnx_expr::pass::{Builder, Pass, PassError, rewrite};
use gnx_expr::{Eqn, Expr, InterfaceError, Var};

use crate::{TransformError, TransformResult};

/// An operation that a backend without native support expands away. This is
/// an op interface: register with `register_op_interface!(dyn Transform: Op)`.
pub trait Transform: Sync {
    /// Emit the expansion of `eqn` (already translated into `b`, with its
    /// body expanded) and return one variable per result.
    fn expand(&self, b: &mut Builder, eqn: &Eqn) -> TransformResult<Vec<Var>>;
}

/// Rewrite every [`Transform`] operation, innermost first, into ordinary
/// operations. Bottom-up order means each expansion only sees bodies without
/// transformations, so `map`, `jvp`, and `vjp` nest in any order.
pub struct ExpandTransforms;

impl Pass for ExpandTransforms {
    fn name(&self) -> &'static str {
        "transforms.expand"
    }

    fn run(&self, expr: Expr) -> Result<Expr, PassError> {
        expand_transforms(&expr).map_err(|error| PassError::new(self.name(), error))
    }
}

pub fn expand_transforms(expr: &Expr) -> TransformResult<Expr> {
    rewrite(expr, &mut |b: &mut Builder, eqn: &Eqn| {
        let transform = match eqn.op().interface::<dyn Transform>() {
            Ok(transform) => transform,
            Err(InterfaceError::Missing { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        transform
            .expand(b, eqn)
            .map(Some)
            .map_err(|error| TransformError::Expanding {
                op: eqn.op().to_string(),
                source: Box::new(error),
            })
    })
}
