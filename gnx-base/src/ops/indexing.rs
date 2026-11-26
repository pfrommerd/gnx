//! Primitive operations used by the indexing frontend.
use super::{Op, arity, result_rank, signature};
use rkyv::rancor::Panic;
#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = SliceOperation)]
pub struct Slice {
    ranges: Vec<[u64; 3]>,
}
impl SliceOperation {
    /// The `(start, end, stride)` of every axis.
    pub fn ranges(&self) -> Vec<(usize, usize, usize)> {
        (self.ranges.iter())
            .map(|&[start, end, stride]| {
                (
                    start.to_native() as usize,
                    end.to_native() as usize,
                    stride.to_native() as usize,
                )
            })
            .collect()
    }
}
crate::array_operation!(
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

crate::unit_operations! {
    pub struct DynamicUpdateSliceOperation;
}
crate::array_operation!(DynamicUpdateSliceOperation, "dynamic_update_slice", |eqn| {
    arity(eqn, 2 + result_rank(eqn)?)
});
pub fn dynamic_update_slice() -> Op {
    Op::new(DynamicUpdateSliceOperation)
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
        rkyv::deserialize::<GatherParameters, Panic>(&self.parameters).unwrap()
    }
}
crate::array_operation!(
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
    assert!((op.downcast_ref::<GatherOperation>()).is_some_and(|op| op.parameters() == parameters),);
    op
}
