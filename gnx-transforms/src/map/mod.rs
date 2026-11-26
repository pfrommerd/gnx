//! Vectorizing map (JAX's `vmap`).
//!
//! [`map`] wraps a function so that it runs once per slice of its mapped
//! inputs. Both setters accept a positional [`Dag`] or a graph [`Filter`](gnx_graph::Filter)
//! whose annotation is `usize`. Matches cover the tracer leaves below them; the
//! annotation supplies the axis. Output filters run on the per-instance result at
//! trace time. An unmatched input is shared; every dynamic output needs an axis.
//!
//! ```ignore
//! let per_row = map(|x: Array, w: Array| f(x, w)).in_axes(dag!(0, *));
//! let per_column = map(f).in_axes(on_axis(1, Of::<Array>::filter()));
//! // An empty-only DAG in a setter needs a payload type for inference.
//! let repeated = map(f).in_axes(dag!(as usize; *)).axis_size(4);
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

use gnx_base::ops::{indices, stored_indices};
use gnx_base::pass::Builder;
use gnx_expr::lower::LoweringContext;
use gnx_expr::{
    Eqn, EqnSource, HEqn, LoweringError, Op, OpRef, Operation, Var, VarInfo, register_op_interface,
};
use rkyv::{Archive, Serialize};

use crate::expand::Transform;
use crate::{TransformResult, util};
use gnx_base::ops::signature;

/// Builds a `map::map`, stored as a [`MapOperation`].
#[derive(Archive, Serialize)]
#[rkyv(archived = MapOperation)]
pub struct MapBuilder {
    size: u64,
    in_axes: Vec<u32>,
    out_axes: Vec<u32>,
}

impl MapOperation {
    pub fn parameters(&self) -> MapParameters {
        MapParameters {
            size: usize::try_from(self.size.to_native()).expect("map size exceeds usize"),
            in_axes: indices(&self.in_axes),
            out_axes: indices(&self.out_axes),
        }
    }
}

gnx_expr::dialect! {
    dyn Operation, "map" => { "map" => MapOperation }
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
        Self::from_operation(op.get())
    }

    /// Like [`MapParameters::from_op`], for an operation read from an expression.
    pub fn from_op_ref(op: OpRef<'_>) -> Result<Self, LoweringError> {
        Self::from_operation(op.get())
    }

    fn from_operation(op: &dyn Operation) -> Result<Self, LoweringError> {
        op.downcast_ref::<MapOperation>()
            .map(|op| op.parameters())
            .ok_or_else(|| signature("expected a map operation"))
    }

    pub fn to_op(&self) -> Op {
        Op::new(MapBuilder {
            size: self.size as u64,
            in_axes: stored_indices(&self.in_axes),
            out_axes: stored_indices(&self.out_axes),
        })
    }

    /// Metadata with `size` inserted at each axis.
    pub fn mapped(&self, infos: &[VarInfo], axes: &[usize]) -> Result<Vec<VarInfo>, LoweringError> {
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
        body_inputs: &[VarInfo],
        body_outputs: &[VarInfo],
        operands: Option<&[VarInfo]>,
        operand_count: usize,
        outputs: &[VarInfo],
    ) -> Result<(), LoweringError> {
        if self.in_axes.len() != body_inputs.len()
            || operand_count != body_inputs.len()
            || self.out_axes.len() != body_outputs.len()
        {
            return Err(signature("map arity"));
        }
        let expected = self.mapped(body_inputs, &self.in_axes)?;
        if operands.is_some_and(|operands| operands != expected) {
            return Err(signature("map operand metadata"));
        }
        if self.mapped(body_outputs, &self.out_axes)? != outputs {
            return Err(signature("map result metadata"));
        }
        Ok(())
    }
}

impl Operation for MapOperation {
    fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parameters = self.parameters();
        write!(
            f,
            "map::map<in_axes: {:?}, out_axes: {:?}, size: {}>",
            parameters.in_axes, parameters.out_axes, parameters.size
        )
    }

    fn lower(&self, eqn: HEqn<'_>, ctx: &mut LoweringContext<'_>) -> Result<(), LoweringError> {
        ctx.lower_value_region_apply(eqn, |inputs, results, operands, outputs| {
            self.parameters()
                .check(inputs, results, Some(operands), operands.len(), outputs)
        })
    }
    fn check_expr(&self, eqn: Eqn<'_>) -> Result<(), LoweringError> {
        let (inputs, results) = util::region_signature(eqn)?;
        self.parameters().check(
            &inputs,
            &results,
            None,
            eqn.inputs().len(),
            &util::output_infos(eqn),
        )
    }
}

impl Transform for MapOperation {
    fn expand(&self, b: &mut Builder, eqn: &EqnSource) -> TransformResult<Vec<Var>> {
        batch::expand_map(b, eqn, &self.parameters())
    }
}
register_op_interface!(dyn Transform: MapOperation);
