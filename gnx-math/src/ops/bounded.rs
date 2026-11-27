//! Operations on uniform, symbol-bound axes. The execution boundary supplies
//! actual extents as scalar operands before backend lowering.
use super::*;
gnx_base::unit_operations! { pub struct PadToBoundOperation; }
gnx_base::array_operation!(PadToBoundOperation, "pad_to_bound", |eqn| arity(eqn, 1));

#[derive(Archive, rkyv::Serialize)]
#[rkyv(archived=AxisLengthOperation)]
pub struct AxisLength {
    axis: u32,
}
impl AxisLengthOperation {
    pub fn axis(&self) -> usize {
        self.axis.to_native() as usize
    }
}
gnx_base::array_operation!(AxisLengthOperation, "axis_length", |eqn| arity(eqn, 1));

fn invoke(op: Op, args: &[&crate::Array], info: crate::ArrayInfo) -> crate::Array {
    gnx_expr::Invocation::invoke(
        op,
        vec![],
        args.iter().map(|x| x.tracer().clone()).collect(),
        vec![gnx_expr::VarInfo::new(info)],
    )
    .remove(0)
    .try_cast::<crate::Array>()
    .unwrap()
    .into()
}

/// Return the physical-capacity rectangle, with every invalid lane set to zero.
pub fn pad_to_bound(x: &crate::Array) -> crate::Array {
    invoke(
        Op::new(PadToBoundOperation),
        &[x],
        crate::ArrayInfo::new(x.shape(), x.dtype()).capacity(),
    )
}
/// Runtime length of a bounded axis as an i32 scalar.
pub fn axis_length(x: &crate::Array, axis: usize) -> crate::Array {
    invoke(
        Op::new(AxisLength {
            axis: axis.try_into().expect("axis fits u32"),
        }),
        &[x],
        crate::ArrayInfo::new(crate::Shape::fixed([]), crate::DType::I32),
    )
}
