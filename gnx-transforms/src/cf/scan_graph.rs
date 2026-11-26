//! Graph-level selection and reconstruction around the axis-zero scan primitive.
use crate::cache::{rebuild_graph, split_graph};
use crate::filters::{AxisType, DefaultAxes, IntoAxisFilter};
use crate::on_axis;
use crate::spec::output_axes;
use crate::traced::{Leaves, trace_region, value};
use crate::{TransformError, TransformResult, util};
use gnx_expr::{Invocation, ObjectTracer, Trace};
use gnx_graph::{All, Graph, GraphEq};

/// Scan a graph function `(carry, x) -> (carry, y)`. By default all `xs` and
/// `ys` use axis zero. Apply `in_axes` to `xs` only; unmatched leaves are captures.
pub fn scan_graph<F>(body: F) -> GraphScan<F> {
    GraphScan {
        body,
        inputs: on_axis(0, All),
        outputs: on_axis(0, All),
    }
}

pub struct GraphScan<F, I: AxisType = DefaultAxes, O: AxisType = DefaultAxes> {
    body: F,
    inputs: I::Filter,
    outputs: O::Filter,
}

impl<F, I: AxisType, O: AxisType> GraphScan<F, I, O> {
    pub fn in_axes<K, S: IntoAxisFilter<K>>(self, s: S) -> GraphScan<F, S::Binding, O> {
        GraphScan {
            body: self.body,
            inputs: s.into_axis_filter(),
            outputs: self.outputs,
        }
    }
    pub fn out_axes<K, S: IntoAxisFilter<K>>(self, s: S) -> GraphScan<F, I, S::Binding> {
        GraphScan {
            body: self.body,
            inputs: self.inputs,
            outputs: s.into_axis_filter(),
        }
    }
}

impl<F, I: AxisType, O: AxisType> GraphScan<F, I, O> {
    /// Run the scan, inferring length from selected inputs when possible.
    pub fn run<C, X, Y>(&self, carry: &C, xs: &X, length: Option<usize>) -> TransformResult<(C, Y)>
    where
        C: Graph<Owned = C> + GraphEq,
        X: Graph,
        Y: Graph<Owned = Y>,
        F: Fn(C, X::Owned) -> (C, Y),
    {
        let (carry_builder, carry_objects) = split_graph(carry);
        let carry_values = carry_objects
            .iter()
            .map(|v| value(v, "scan carry"))
            .collect::<TransformResult<Vec<_>>>()?;
        let mut infos: Vec<_> = carry_values.iter().map(|v| v.info().clone()).collect();
        let leaves = Leaves::<X>::of(xs);
        let axes = crate::spec::axes::<I>(&self.inputs, xs, &leaves.paths, "in_axes")?;
        let mut inputs = Vec::new();
        let mut length = length;
        for ((object, axis), path) in leaves.objects.iter().zip(&axes).zip(&leaves.paths) {
            if let Some(axis) = axis.value {
                let locate = |e| crate::spec::at_path("in_axes", path, e);
                let x = value(object, "a scanned input").map_err(locate)?;
                let x = move_axis(x, axis, 0).map_err(locate)?;
                let (n, item) = super::scan::slice(x.info())
                    .map_err(TransformError::from)
                    .map_err(locate)?;
                if *length.get_or_insert(n) != n {
                    return Err(TransformError::invalid("scan input lengths must match"));
                }
                inputs.push(x);
                infos.push(item);
            }
        }
        let length = length
            .ok_or_else(|| TransformError::invalid("scan without inputs requires a length"))?;
        let mut ys = None;
        let capture = trace_region(&infos, |args| {
            let (c, x) = args.split_at(carry_values.len());
            let c = rebuild_graph(&carry_builder, c.iter().cloned().map(ObjectTracer::from));
            let mut x = x.iter();
            let x = leaves.rebuild(axes.iter().zip(&leaves.objects).map(|(axis, original)| {
                match axis.value {
                    Some(_) => ObjectTracer::from(
                        x.next().expect("one placeholder per scanned input").clone(),
                    ),
                    None => original.clone(),
                }
            }));
            let (c, y) = (self.body)(c, x);
            let (next, objects) = split_graph(&c);
            if !C::builders_eq::<ObjectTracer>(&carry_builder, &next, &mut Default::default()) {
                return Err(TransformError::invalid("scan carry structure changed"));
            }
            let yl = Leaves::<Y>::of(&y);
            let out = output_axes::<O>(&self.outputs, &y, &yl.paths)?;
            for ((y, axis), path) in yl.objects.iter().zip(&out).zip(&yl.paths) {
                let y = value(y, "a scan result")?;
                if *axis > util::array(y.info())?.shape().dims().len() {
                    return Err(crate::spec::at_path(
                        "out_axes",
                        path,
                        TransformError::invalid("scan axis out of range"),
                    ));
                }
            }
            ys = Some((yl.builder, out));
            objects
                .iter()
                .chain(&yl.objects)
                .map(|v| value(v, "a scan result"))
                .collect()
        })?;
        let (cs, stacked) = super::scan(&capture, &carry_values, &inputs, Some(length))?;
        let (builder, out) = ys.expect("scan body traced");
        let stacked = stacked
            .into_iter()
            .zip(out)
            .map(|(y, axis)| move_axis(y, 0, axis).map(ObjectTracer::from))
            .collect::<TransformResult<Vec<_>>>()?;
        Ok((
            rebuild_graph(&carry_builder, cs.into_iter().map(ObjectTracer::from)),
            rebuild_graph(&builder, stacked),
        ))
    }
}

fn move_axis(trace: Trace, from: usize, to: usize) -> TransformResult<Trace> {
    let array = util::array(trace.info())?;
    let shape = array.shape().dims();
    if from >= shape.len() || to >= shape.len() {
        return Err(TransformError::invalid("scan axis out of range"));
    }
    if from == to {
        return Ok(trace);
    }
    let mut perm: Vec<_> = (0..shape.len()).collect();
    perm.remove(from);
    perm.insert(to, from);
    let shape = perm.iter().map(|&i| shape[i]).collect();
    let info = gnx_expr::VarInfo::new(gnx_common::ArrayInfo::new(shape, array.dtype()));
    Ok(Invocation::invoke(
        gnx_common::ops::transpose_op(&perm),
        vec![],
        vec![trace],
        vec![info],
    )
    .remove(0)
    .into_trace_ref())
}
