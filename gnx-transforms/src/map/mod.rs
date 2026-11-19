//! Vectorizing map (JAX's `vmap`).
//!
//! [`map`] wraps a function so that it runs once per slice of its mapped
//! inputs, using JAX-style `in_axes`/`out_axes` specified as [`Dag`] graphs:
//!
//! ```ignore
//! let per_row = map(|x: Array, w: Array| f(x, w)).in_axes(dag!(0, *));
//! ```
//!
//! Calls trace the function once per input signature (through `jit`) and
//! record a `map.map` region operation:
//!
//! | op | operands | results | attributes |
//! |---|---|---|---|
//! | `map.map` | mapped inputs | mapped results | `size`, `in_axes`, `out_axes` |
//!
//! Unmapped values are captures of the body. Backends with native support
//! (JAX) lower `map.map` directly; others run
//! [`ExpandTransforms`](crate::ExpandTransforms), which batches the body with
//! per-operation [`BatchRule`]s.
//!
//! [`Dag`]: gnx_graph::util::Dag

mod api;
pub mod batch;
pub mod def;
pub mod region;
mod rules;

pub use api::{Map, map};
pub use batch::{BatchRule, Batched};

use gnx_expr::lower::LoweringContext;
use gnx_expr::pass::Builder;
use gnx_expr::{
    Attr, Dialect, Eqn, HEqn, LoweringError, Op, Operation, ValueInfo, Var, register_op_interface,
};
use gnx_math::ops::{index_attr, index_list};

use crate::body::{BodySignature, check_expr_with, exact_attrs, lower_transform, signature};
use crate::expand::Transform;
use crate::{TransformResult, util};

pub struct MapDialect;
pub struct MapOperation;

static MAP: MapDialect = MapDialect;
gnx_expr::register_dialect!(&MAP);

impl Dialect for MapDialect {
    fn name(&self) -> &'static str {
        "map"
    }

    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        match name {
            "map" => Some(&MapOperation),
            _ => None,
        }
    }
}

/// Parameters of `map.map`: every operand is mapped over axis `in_axes[i]`
/// and every result is stacked along `out_axes[i]`, with `size` instances.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapParameters {
    pub size: usize,
    pub in_axes: Vec<usize>,
    pub out_axes: Vec<usize>,
}

impl MapParameters {
    pub fn from_op(op: &Op) -> Result<Self, LoweringError> {
        exact_attrs(op, &["size", "in_axes", "out_axes"])?;
        let Some(&Attr::I64(size)) = op.attrs().get("size") else {
            return Err(signature("map size must be an i64"));
        };
        Ok(Self {
            size: usize::try_from(size).map_err(|_| signature("negative map size"))?,
            in_axes: index_list(op, "in_axes")?,
            out_axes: index_list(op, "out_axes")?,
        })
    }

    pub fn to_op(&self) -> Result<Op, LoweringError> {
        let size = i64::try_from(self.size).map_err(|_| signature("map size exceeds i64"))?;
        Ok(Op::new_with_attrs(
            &MapOperation,
            [
                ("size".into(), Attr::I64(size)),
                ("in_axes".into(), index_attr(&self.in_axes)),
                ("out_axes".into(), index_attr(&self.out_axes)),
            ]
            .into(),
        ))
    }

    /// Metadata with `size` inserted at each axis.
    pub fn mapped(
        &self,
        infos: &[ValueInfo],
        axes: &[usize],
    ) -> Result<Vec<ValueInfo>, LoweringError> {
        infos
            .iter()
            .zip(axes)
            .map(|(info, &axis)| {
                util::with_axis(info, axis, self.size)
                    .map_err(|_| signature("map axis out of range"))
            })
            .collect()
    }

    /// Check the operand (if known) and result metadata against the body.
    pub fn check(
        &self,
        body: &BodySignature,
        operands: Option<&[ValueInfo]>,
        operand_count: usize,
        outputs: &[ValueInfo],
    ) -> Result<(), LoweringError> {
        if self.in_axes.len() != body.inputs.len()
            || operand_count != body.inputs.len()
            || self.out_axes.len() != body.outputs.len()
        {
            return Err(signature("map arity"));
        }
        let expected = self.mapped(&body.inputs, &self.in_axes)?;
        if operands.is_some_and(|operands| operands != expected) {
            return Err(signature("map operand metadata"));
        }
        if self.mapped(&body.outputs, &self.out_axes)? != outputs {
            return Err(signature("map result metadata"));
        }
        Ok(())
    }
}

impl Operation for MapOperation {
    fn name(&self) -> &'static str {
        "map"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &MAP
    }
    fn lower(&self, eqn: &HEqn, ctx: &mut LoweringContext<'_>) -> Result<(), LoweringError> {
        let parameters = MapParameters::from_op(eqn.op())?;
        lower_transform(eqn, ctx, |body, operands, outputs| {
            parameters.check(body, Some(operands), operands.len(), outputs)
        })
    }
    fn check_expr(&self, eqn: &Eqn) -> Result<(), LoweringError> {
        let parameters = MapParameters::from_op(eqn.op())?;
        check_expr_with(eqn, |body, count, outputs| {
            parameters.check(body, None, count, outputs)
        })
    }
}

impl Transform for MapOperation {
    fn expand(&self, b: &mut Builder, eqn: &Eqn) -> TransformResult<Vec<Var>> {
        batch::expand_map(b, eqn)
    }
}
register_op_interface!(dyn Transform: MapOperation);
