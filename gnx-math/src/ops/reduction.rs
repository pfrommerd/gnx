//! Reductions with a scalar combiner region.

use gnx_common::ops::reduction::{array, check_region, emit};
use gnx_common::{ArrayInfo, Shape, fixed_dims};
use gnx_expr::pass::Builder;
use gnx_expr::{ExprHandle, Var, VarInfo};

use super::{Op, Result, native, signature, stored_exactly};

pub use gnx_common::ops::reduction::{ReduceOperation, reduce, reduce_add_op, reduce_op};

#[derive(Clone, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Window {
    pub dimensions: Vec<usize>,
    pub strides: Vec<usize>,
    pub padding: Vec<(i64, i64)>,
    pub base_dilation: Vec<usize>,
    pub window_dilation: Vec<usize>,
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = ReduceWindowOperation)]
pub struct ReduceWindow {
    window: Window,
}

impl ReduceWindowOperation {
    pub fn window(&self) -> Window {
        native(&self.window)
    }
}

gnx_common::array_operation!(ReduceWindowOperation, "reduce_window", |op, eqn| {
    op.window().check(check_region(eqn)?)
}, window;
    fn lower(&self, eqn: gnx_expr::HEqn<'_>, ctx: &mut gnx_expr::lower::LoweringContext<'_>) -> Result<()> {
        ctx.lower_closed_regions_apply(eqn)
    }
);

impl Window {
    fn check(&self, rank: usize) -> Result<()> {
        for parameter in [
            &self.dimensions,
            &self.strides,
            &self.base_dilation,
            &self.window_dilation,
        ] {
            if parameter.len() != rank || parameter.contains(&0) {
                return Err(signature(
                    "window parameters must be positive and have operand rank",
                ));
            }
        }
        if self.padding.len() != rank {
            return Err(signature("window padding must have operand rank"));
        }
        Ok(())
    }

    fn output_shape(&self, input: &[usize]) -> Result<Vec<usize>> {
        self.check(input.len())?;
        input
            .iter()
            .enumerate()
            .map(|(axis, &size)| {
                let span = |size: usize, dilation: usize| -> Result<i128> {
                    if size == 0 {
                        return Ok(0);
                    }
                    ((size - 1) as i128)
                        .checked_mul(dilation as i128)
                        .and_then(|n| n.checked_add(1))
                        .ok_or_else(|| signature("window extent overflow"))
                };
                let (lo, hi) = self.padding[axis];
                let extent = span(size, self.base_dilation[axis])?
                    .checked_add(lo as i128 + hi as i128)
                    .ok_or_else(|| signature("padded extent overflow"))?;
                let window = span(self.dimensions[axis], self.window_dilation[axis])?;
                if extent < window {
                    return Ok(0);
                }
                usize::try_from((extent - window) / self.strides[axis] as i128 + 1)
                    .map_err(|_| signature("window result size overflow"))
            })
            .collect()
    }
}

pub fn reduce_window_op(window: Window) -> Op {
    let op = Op::new(ReduceWindow {
        window: window.clone(),
    });
    stored_exactly(
        (op.downcast_ref::<ReduceWindowOperation>()).is_some_and(|op| op.window() == window),
    );
    op
}

pub fn reduce_window(
    builder: &mut Builder,
    operands: &[Var],
    initial: &[Var],
    window: Window,
    reducer: ExprHandle,
) -> Result<Vec<Var>> {
    emit(
        builder,
        operands,
        initial,
        reducer,
        reduce_window_op(window.clone()),
        |shape| window.output_shape(shape),
    )
}

fn arg_extreme(
    b: &mut Builder,
    input: Var,
    axis: usize,
    index_dtype: gnx_common::DType,
    max: bool,
) -> Result<Var> {
    use super::{
        bitwise::or_op,
        creation::{iota, scalar_bound},
        elementwise::{eq_op, gt_op, lt_op, ne_op},
        select_op,
    };
    use gnx_common::DType;
    let x = array(b.info(input))?;
    let shape = fixed_dims(&x).map_err(|_| signature("arg reduction requires fixed dimensions"))?;
    if shape.get(axis).is_none_or(|&dim| dim == 0) {
        return Err(signature("arg reduction requires a nonempty axis"));
    }
    let index_type = VarInfo::new(ArrayInfo::new(Shape::fixed([]), index_dtype));
    let value_type = VarInfo::new(ArrayInfo::new(Shape::fixed([]), x.dtype()));
    let flag = VarInfo::new(ArrayInfo::new(Shape::fixed([]), DType::Bool));
    let index = b.emit1(
        iota(axis),
        &[],
        VarInfo::new(ArrayInfo::new(x.shape().clone(), index_dtype)),
    );
    let value_identity = b.emit1(scalar_bound(max), &[], value_type.clone());
    let index_identity = b.emit1(scalar_bound(false), &[], index_type.clone());

    let mut reducer = Builder::new();
    let a = reducer.input(value_type.clone());
    let ai = reducer.input(index_type.clone());
    let c = reducer.input(value_type.clone());
    let ci = reducer.input(index_type.clone());
    let strict = if max { gt_op() } else { lt_op() };
    let better = reducer.emit1(strict, &[a, c], flag.clone());
    let equal = reducer.emit1(eq_op(), &[a, c], flag.clone());
    let first = reducer.emit1(lt_op(), &[ai, ci], flag.clone());
    let choose_a = reducer.emit1(select_op(), &[equal, first, better], flag.clone());
    let choose_a = if matches!(
        x.dtype(),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64
    ) {
        let a_nan = reducer.emit1(ne_op(), &[a, a], flag.clone());
        let c_nan = reducer.emit1(ne_op(), &[c, c], flag.clone());
        let nan = reducer.emit1(or_op(), &[a_nan, c_nan], flag.clone());
        let both = reducer.emit1(super::bitwise::and_op(), &[a_nan, c_nan], flag.clone());
        let preferred = reducer.emit1(select_op(), &[both, first, a_nan], flag.clone());
        reducer.emit1(select_op(), &[nan, preferred, choose_a], flag)
    } else {
        choose_a
    };
    let val = reducer.emit1(select_op(), &[choose_a, a, c], value_type);
    let ix = reducer.emit1(select_op(), &[choose_a, ai, ci], index_type);
    Ok(reduce(
        b,
        &[input, index],
        &[value_identity, index_identity],
        &[axis],
        reducer.finish(vec![val, ix]).build()?.handle(),
    )?[1])
}

pub fn argmin(
    b: &mut Builder,
    input: Var,
    axis: usize,
    index_dtype: gnx_common::DType,
) -> Result<Var> {
    arg_extreme(b, input, axis, index_dtype, false)
}
pub fn argmax(
    b: &mut Builder,
    input: Var,
    axis: usize,
    index_dtype: gnx_common::DType,
) -> Result<Var> {
    arg_extreme(b, input, axis, index_dtype, true)
}
