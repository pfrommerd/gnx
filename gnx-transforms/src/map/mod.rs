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
//! | op | operands | results | parameters |
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
pub(crate) mod rules;

pub use api::{Map, map};
pub use batch::{BatchRule, Batched};

use std::fmt;

use gnx_expr::lower::LoweringContext;
use gnx_expr::pass::Builder;
use gnx_expr::{
    Dialect, Eqn, HEqn, LoweringError, Op, Operation, ValueInfo, Var, register_op_interface,
};

use crate::body::{BodySignature, check_expr_with, lower_transform, signature};
use crate::expand::Transform;
use crate::{TransformResult, util};

pub struct MapDialect;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct MapOperation {
    parameters: MapParameters,
}

impl MapOperation {
    pub fn parameters(&self) -> &MapParameters {
        &self.parameters
    }
}

static MAP: MapDialect = MapDialect;

impl Dialect for MapDialect {
    fn name(&self) -> &'static str {
        "map"
    }
}

/// Parameters of `map.map`: every operand is mapped over axis `in_axes[i]`
/// and every result is stacked along `out_axes[i]`, with `size` instances.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MapParameters {
    pub size: usize,
    pub in_axes: Vec<usize>,
    pub out_axes: Vec<usize>,
}

impl MapParameters {
    pub fn from_op(op: &Op) -> Result<Self, LoweringError> {
        op.downcast_ref::<MapOperation>()
            .map(|op| op.parameters().clone())
            .ok_or_else(|| signature("expected a map operation"))
    }

    pub fn to_op(&self) -> Op {
        Op::new(MapOperation {
            parameters: self.clone(),
        })
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

impl fmt::Display for MapOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parameters = self.parameters();
        write!(
            f,
            "map::map<in_axes: {:?}, out_axes: {:?}, size: {}>",
            parameters.in_axes, parameters.out_axes, parameters.size
        )
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
        lower_transform(eqn, ctx, |body, operands, outputs| {
            self.parameters
                .check(body, Some(operands), operands.len(), outputs)
        })
    }
    fn check_expr(&self, eqn: &Eqn) -> Result<(), LoweringError> {
        check_expr_with(eqn, |body, count, outputs| {
            self.parameters.check(body, None, count, outputs)
        })
    }
}

impl Transform for MapOperation {
    fn expand(&self, b: &mut Builder, eqn: &Eqn) -> TransformResult<Vec<Var>> {
        batch::expand_map(b, eqn, self.parameters())
    }
}
register_op_interface!(dyn Transform: MapOperation);
