pub use super::select_op as select;
use super::{Op, Result, arity, native, result_rank, signature, stored_exactly};
pub use gnx_base::ops::indexing::*;
gnx_base::unit_operations! {
    pub struct DynamicSliceOperation;
    pub struct SelectNOperation;
}

gnx_base::array_operation!(DynamicSliceOperation, "dynamic_slice", |eqn| arity(
    eqn,
    1 + result_rank(eqn)?
));
pub fn dynamic_slice() -> Op {
    Op::new(DynamicSliceOperation)
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
