//! Operations on uniform, symbol-bound axes. The execution boundary supplies
//! actual extents as scalar operands before backend lowering.
use super::*;
gnx_common::unit_operations! { pub struct PadToBoundOperation; }
gnx_common::array_operation!(PadToBoundOperation, "pad_to_bound", |eqn| arity(eqn, 1));

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
gnx_common::array_operation!(AxisLengthOperation, "axis_length", |eqn| arity(eqn, 1));

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
/// A local bounded axis whose length is an SSA scalar (for example a loop
/// cursor). Views sharing this declaration share its runtime length. Unlike
/// a compact input, the backing allocation is already at the bound.
pub struct BoundedAxis {
    symbol: gnx_expr::Symbol,
    max: usize,
    length: crate::Array,
}
impl BoundedAxis {
    /// `length` must be in `0..=max` at execution. In a bounded loop this is
    /// established by its trip count; constructing a view does not read back
    /// the scalar. The declaration belongs to the enclosing trace.
    pub fn new(max: usize, length: &crate::Array) -> Self {
        assert!(
            i32::try_from(max).is_ok()
                && length.dtype() == crate::DType::I32
                && length.shape().dims().is_empty()
        );
        let ctx = gnx_expr::TraceContext::current().expect("declare a bounded axis inside a trace");
        Self {
            symbol: ctx.create_symbol(),
            max,
            length: length.clone(),
        }
    }
    pub fn prefix<'a>(&'a self, storage: &'a crate::Array, axis: usize) -> Prefix<'a> {
        let info = crate::ArrayInfo::new(storage.shape(), storage.dtype());
        assert!(
            info.shape()
                .dims()
                .iter()
                .all(|d| matches!(d, crate::Dim::Fixed(_))),
            "prefix storage must have a fixed capacity shape"
        );
        assert_eq!(info.shape().dims()[axis], crate::Dim::Fixed(self.max));
        Prefix {
            storage,
            axis: self,
            logical: info
                .with_bounded_axis(axis, self.symbol, self.max)
                .expect("checked bound"),
        }
    }
}
/// Logical view of capacity storage. Padding is materialized only on request.
pub struct Prefix<'a> {
    storage: &'a crate::Array,
    axis: &'a BoundedAxis,
    logical: crate::ArrayInfo,
}
impl Prefix<'_> {
    pub fn shape(&self) -> &crate::Shape {
        self.logical.shape()
    }
    pub fn padded(&self, fill: f64) -> crate::Array {
        use crate::{ArrayInfo, DType, Dim, Shape};
        let capacity = self.logical.capacity();
        let axis = self
            .shape()
            .dims()
            .iter()
            .position(|d| matches!(d,Dim::Bounded{symbol,..} if *symbol==self.axis.symbol))
            .expect("bound axis");
        let shape = capacity.shape();
        let meta = |dtype| ArrayInfo::new(shape.clone(), dtype);
        let indices = invoke(creation::iota(axis), &[], meta(DType::I32));
        let limit = invoke(
            broadcast_in_dim_op(&[]),
            &[&self.axis.length],
            meta(DType::I32),
        );
        let valid = invoke(elementwise::lt(), &[&indices, &limit], meta(DType::Bool));
        let fill = invoke(
            creation::constant(fill),
            &[],
            ArrayInfo::new(Shape::fixed([]), capacity.dtype()),
        );
        let padding = invoke(broadcast_in_dim_op(&[]), &[&fill], capacity.clone());
        invoke(select_op(), &[&valid, self.storage, &padding], capacity)
    }
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
