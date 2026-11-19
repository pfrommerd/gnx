use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt, result_rank, signature};

pub use super::select_op as select;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SliceOperation {
    ranges: Vec<(usize, usize, usize)>,
}
impl SliceOperation {
    pub fn ranges(&self) -> &[(usize, usize, usize)] {
        &self.ranges
    }
}
gnx_math_base::array_operation!(
    SliceOperation,
    "slice",
    |op, eqn| {
        arity(eqn, 1)?;
        if op.ranges.len() != result_rank(eqn)?
            || op
                .ranges
                .iter()
                .any(|&(start, end, stride)| start > end || stride == 0)
        {
            return Err(signature("invalid slice ranges"));
        }
        Ok(())
    },
    ranges
);
pub fn slice(ranges: &[(usize, usize, usize)]) -> Op {
    Op::new(SliceOperation {
        ranges: ranges.to_vec(),
    })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DynamicSliceOperation;
gnx_math_base::array_operation!(DynamicSliceOperation, "dynamic_slice", |eqn| arity(
    eqn,
    1 + result_rank(eqn)?
));
pub fn dynamic_slice() -> Op {
    Op::new(DynamicSliceOperation)
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DynamicUpdateSliceOperation;
gnx_math_base::array_operation!(DynamicUpdateSliceOperation, "dynamic_update_slice", |eqn| {
    arity(eqn, 2 + result_rank(eqn)?)
});
pub fn dynamic_update_slice() -> Op {
    Op::new(DynamicUpdateSliceOperation)
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SelectNOperation;
gnx_math_base::array_operation!(SelectNOperation, "select_n", |eqn| {
    if eqn.inputs().len() < 2 {
        return Err(signature("select_n needs a selector and at least one case"));
    }
    arity(eqn, eqn.inputs().len())
});
pub fn select_n() -> Op {
    Op::new(SelectNOperation)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundsMode {
    PromiseInBounds,
    Clip,
    FillOrDrop,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatherDimensions {
    pub offset: Vec<usize>,
    pub collapsed: Vec<usize>,
    pub start_index_map: Vec<usize>,
    pub operand_batch: Vec<usize>,
    pub index_batch: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatherParameters {
    pub dimensions: GatherDimensions,
    pub slice_sizes: Vec<usize>,
    pub mode: BoundsMode,
    pub sorted: bool,
    pub unique: bool,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GatherOperation {
    parameters: GatherParameters,
}
impl GatherOperation {
    pub fn parameters(&self) -> &GatherParameters {
        &self.parameters
    }
}
gnx_math_base::array_operation!(
    GatherOperation,
    "gather",
    |op, eqn| {
        arity(
            eqn,
            if op.parameters.mode == BoundsMode::FillOrDrop {
                3
            } else {
                2
            },
        )
    },
    parameters
);

/// The operands are the array, the indices, and a scalar fill value in fill mode.
pub fn gather(parameters: GatherParameters) -> Op {
    Op::new(GatherOperation { parameters })
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScatterDimensions {
    pub update_window: Vec<usize>,
    pub inserted_window: Vec<usize>,
    pub index_map: Vec<usize>,
    pub operand_batch: Vec<usize>,
    pub index_batch: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScatterParameters {
    pub dimensions: ScatterDimensions,
    pub mode: BoundsMode,
    pub sorted: bool,
    pub unique: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScatterUpdate {
    Replace,
    Add,
    Subtract,
    Multiply,
    Min,
    Max,
    Apply,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ScatterOperation {
    parameters: ScatterParameters,
    update: ScatterUpdate,
}

impl ScatterOperation {
    pub fn parameters(&self) -> &ScatterParameters {
        &self.parameters
    }
    pub fn update(&self) -> ScatterUpdate {
        self.update
    }
}

gnx_math_base::array_operation!(ScatterOperation, "scatter", |op, eqn| {
    if op.update == ScatterUpdate::Apply {
        if eqn.inputs().len() != 3 || eqn.outputs().len() != 1
            || eqn.subexprs().len() != 1 || !eqn.subexprs()[0].closure().is_empty()
        {
            return Err(signature("scatter_apply needs one closed update region"));
        }
        Ok(())
    } else {
        arity(eqn, 3)
    }
}, parameters;
    fn lower(&self, eqn: &gnx_expr::HEqn, ctx: &mut gnx_expr::lower::LoweringContext<'_>) -> Result<()> {
        ctx.lower_closed_regions_apply(eqn)
    }
);

macro_rules! scatter_constructors {
    ($($name:ident => $update:ident),* $(,)?) => {$(
        pub fn $name(parameters: ScatterParameters) -> Op {
            Op::new(ScatterOperation { parameters, update: ScatterUpdate::$update })
        }
    )*};
}
scatter_constructors!(
    scatter => Replace, scatter_add => Add, scatter_sub => Subtract,
    scatter_mul => Multiply, scatter_min => Min, scatter_max => Max, scatter_apply => Apply,
);
