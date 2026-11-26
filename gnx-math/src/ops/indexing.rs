use super::{Op, Result, arity, count, native, result_rank, signature, stored_exactly};

pub use super::select_op as select;

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = SliceOperation)]
pub struct Slice {
    ranges: Vec<[u64; 3]>,
}
impl SliceOperation {
    /// The `(start, end, stride)` of every axis.
    pub fn ranges(&self) -> Vec<(usize, usize, usize)> {
        (self.ranges.iter())
            .map(|&[start, end, stride]| (count(start), count(end), count(stride)))
            .collect()
    }
}
gnx_base::array_operation!(
    SliceOperation,
    "slice",
    |op, eqn| {
        arity(eqn, 1)?;
        let ranges = op.ranges();
        if ranges.len() != result_rank(eqn)?
            || ranges
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
    Op::new(Slice {
        ranges: (ranges.iter())
            .map(|&(start, end, stride)| [start as u64, end as u64, stride as u64])
            .collect(),
    })
}

gnx_base::unit_operations! {
    pub struct DynamicSliceOperation;
    pub struct DynamicUpdateSliceOperation;
    pub struct SelectNOperation;
}

gnx_base::array_operation!(DynamicSliceOperation, "dynamic_slice", |eqn| arity(
    eqn,
    1 + result_rank(eqn)?
));
pub fn dynamic_slice() -> Op {
    Op::new(DynamicSliceOperation)
}

gnx_base::array_operation!(DynamicUpdateSliceOperation, "dynamic_update_slice", |eqn| {
    arity(eqn, 2 + result_rank(eqn)?)
});
pub fn dynamic_update_slice() -> Op {
    Op::new(DynamicUpdateSliceOperation)
}

gnx_base::array_operation!(SelectNOperation, "select_n", |eqn| {
    if eqn.inputs().len() < 2 {
        return Err(signature("select_n needs a selector and at least one case"));
    }
    arity(eqn, eqn.inputs().len())
});
pub fn select_n() -> Op {
    Op::new(SelectNOperation)
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Portable,
)]
#[rkyv(as = Self)]
#[repr(u8)]
pub enum BoundsMode {
    PromiseInBounds,
    Clip,
    FillOrDrop,
}

impl<D: rkyv::rancor::Fallible + ?Sized> rkyv::Deserialize<BoundsMode, D> for BoundsMode {
    fn deserialize(&self, _: &mut D) -> std::result::Result<BoundsMode, D::Error> {
        Ok(*self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct GatherDimensions {
    pub offset: Vec<usize>,
    pub collapsed: Vec<usize>,
    pub start_index_map: Vec<usize>,
    pub operand_batch: Vec<usize>,
    pub index_batch: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct GatherParameters {
    pub dimensions: GatherDimensions,
    pub slice_sizes: Vec<usize>,
    pub mode: BoundsMode,
    pub sorted: bool,
    pub unique: bool,
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = GatherOperation)]
pub struct Gather {
    parameters: GatherParameters,
}
impl GatherOperation {
    pub fn parameters(&self) -> GatherParameters {
        native(&self.parameters)
    }
}
gnx_base::array_operation!(
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
    let op = Op::new(Gather {
        parameters: parameters.clone(),
    });
    stored_exactly(
        (op.downcast_ref::<GatherOperation>()).is_some_and(|op| op.parameters() == parameters),
    );
    op
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ScatterDimensions {
    pub update_window: Vec<usize>,
    pub inserted_window: Vec<usize>,
    pub index_map: Vec<usize>,
    pub operand_batch: Vec<usize>,
    pub index_batch: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ScatterParameters {
    pub dimensions: ScatterDimensions,
    pub mode: BoundsMode,
    pub sorted: bool,
    pub unique: bool,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Portable,
)]
#[rkyv(as = Self)]
#[repr(u8)]
pub enum ScatterUpdate {
    Replace,
    Add,
    Subtract,
    Multiply,
    Min,
    Max,
    Apply,
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = ScatterOperation)]
pub struct Scatter {
    parameters: ScatterParameters,
    update: ScatterUpdate,
}

impl ScatterOperation {
    pub fn parameters(&self) -> ScatterParameters {
        native(&self.parameters)
    }
    pub fn update(&self) -> ScatterUpdate {
        self.update
    }
}

gnx_base::array_operation!(ScatterOperation, "scatter", |op, eqn| {
    if op.update == ScatterUpdate::Apply {
        let mut regions = eqn.subexprs();
        let closed = matches!(
            (regions.next(), regions.next()),
            (Some(region), None) if region.provided_vars().is_empty()
        );
        if eqn.inputs().len() != 3 || eqn.outputs().len() != 1 || !closed {
            return Err(signature("scatter_apply needs one closed update region"));
        }
        Ok(())
    } else {
        arity(eqn, 3)
    }
}, parameters;
    fn lower(&self, eqn: gnx_expr::HEqn<'_>, ctx: &mut gnx_expr::lower::LoweringContext<'_>) -> Result<()> {
        ctx.lower_closed_regions_apply(eqn)
    }
);

fn scatter_with(parameters: ScatterParameters, update: ScatterUpdate) -> Op {
    let op = Op::new(Scatter {
        parameters: parameters.clone(),
        update,
    });
    stored_exactly(
        (op.downcast_ref::<ScatterOperation>()).is_some_and(|op| op.parameters() == parameters),
    );
    op
}

macro_rules! scatter_constructors {
    ($($name:ident => $update:ident),* $(,)?) => {$(
        pub fn $name(parameters: ScatterParameters) -> Op {
            scatter_with(parameters, ScatterUpdate::$update)
        }
    )*};
}
scatter_constructors!(
    scatter => Replace, scatter_add => Add, scatter_sub => Subtract,
    scatter_mul => Multiply, scatter_min => Min, scatter_max => Max, scatter_apply => Apply,
);
