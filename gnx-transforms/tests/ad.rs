use gnx_common::pass::{Builder, Dce, Pipeline};
use gnx_common::{ArrayInfo, DType, Shape};
use std::sync::Arc;

use gnx_common::ops::reduction::{ReduceAddOperation, reduce_add};
use gnx_common::ops::{add_op, mul_op};
use gnx_expr::util::TypedIn;
use gnx_expr::{
    Capture, CellTrace, Expr, InputSpec, Invocation, LoweringError, ObjectTracer, Op, OpRef,
    Operation, RootExpr, Trace, TraceContext, VarInfo,
};
use gnx_transforms::ExpandTransforms;
use gnx_transforms::ad::region::{jvp, vjp};
use gnx_transforms::ad::{DiffParameters, JvpOperation, VjpOperation};
use gnx_transforms::cf::CallOperation;
use gnx_transforms::map::{MapOperation, MapParameters};
/// Record `map.map` at the region level with an inferred size.
fn map(
    body: &Capture,
    inputs: &[Trace],
    in_axes: &[usize],
    out_axes: &[usize],
) -> Result<Vec<Trace>, gnx_expr::LoweringError> {
    gnx_transforms::map::region::map(body, inputs, in_axes, out_axes, None)
}

fn info(dims: &[usize]) -> VarInfo {
    VarInfo::new(ArrayInfo::new(
        Shape::fixed(dims.iter().copied()),
        DType::F32,
    ))
}
fn apply(op: Op, a: &Trace, b: &Trace) -> Trace {
    Invocation::invoke(
        op,
        vec![],
        vec![a.clone(), b.clone()],
        vec![a.info().clone()],
    )
    .pop()
    .unwrap()
    .into_trace_ref()
}
fn region(inputs: &[VarInfo], body: impl FnOnce(&[Trace]) -> Vec<Trace>) -> Capture {
    let guard = TraceContext::enter();
    let args: Vec<_> = inputs.iter().cloned().map(Trace::placeholder).collect();
    let outputs = body(&args);
    Capture::from_context(
        guard.context(),
        &args.into_iter().map(ObjectTracer::from).collect::<Vec<_>>(),
        &outputs
            .into_iter()
            .map(ObjectTracer::from)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

/// Capture `build` over placeholders and lower it to an Expr.
fn lowered(inputs: &[VarInfo], build: impl FnOnce(&[Trace]) -> Vec<Trace>) -> Arc<RootExpr> {
    let guard = TraceContext::enter();
    let args: Vec<ObjectTracer> = inputs
        .iter()
        .cloned()
        .map(|i| Trace::placeholder(i).into())
        .collect();
    let traces: Vec<_> = args.iter().map(ObjectTracer::resolve).collect();
    let outputs: Vec<ObjectTracer> = build(&traces).into_iter().map(Into::into).collect();
    let capture = Capture::from_context(guard.context(), &args, &outputs).unwrap();
    let program = capture
        .expr()
        .lower(&InputSpec::from_args(&[], &args))
        .unwrap();
    program.expr.view().check().unwrap();
    program.expr.clone()
}

fn is<T: TypedIn<dyn Operation>>(op: OpRef<'_>) -> bool {
    op.downcast::<T>().is_some()
}

fn contains_ad(expr: Expr<'_>) -> bool {
    expr.eqns().any(|eqn| {
        is::<MapOperation>(eqn.op())
            || is::<JvpOperation>(eqn.op())
            || is::<VjpOperation>(eqn.op())
            || eqn.subexprs().any(|sub| contains_ad(sub.expr()))
    })
}

#[test]
fn additive_reduction_expands_without_a_math_rule_provider() {
    use gnx_expr::SubExprBuilder;
    use gnx_transforms::ops::{ad, map};

    let mut body = Builder::new();
    let x = body.input(info(&[3]));
    let sum = reduce_add(&mut body, x, &[0]).unwrap();
    let body = SubExprBuilder::new(body.finish(vec![sum]));

    let mut program = Builder::new();
    let x = program.input(info(&[3]));
    let scalar = program.input(info(&[]));
    let batch = program.input(info(&[2, 3]));
    let mut outputs = program.emit(
        ad::vjp(&[0], &[0]),
        &[x, scalar],
        [info(&[]), info(&[3])],
        vec![body.clone()],
    );
    outputs.extend(program.emit(
        ad::jvp(&[0], &[0]),
        &[x, x],
        [info(&[]), info(&[])],
        vec![body.clone()],
    ));
    outputs.extend(program.emit(map::map(2, &[0], &[0]), &[batch], [info(&[2])], vec![body]));
    let program = program.finish(outputs).build().unwrap();
    let expanded = gnx_transforms::expand_transforms(program.view())
        .unwrap()
        .build()
        .unwrap();
    let expanded = expanded.view();
    expanded.check().unwrap();
    assert!(!contains_ad(expanded));
    let reductions: Vec<_> = expanded
        .eqns()
        .filter_map(|eqn| {
            eqn.op()
                .downcast::<ReduceAddOperation>()
                .map(|op| (eqn, op))
        })
        .collect();
    assert!(reductions.iter().any(|(_, op)| op.axes() == [0]));
    assert!(reductions.iter().any(|(_, op)| op.axes() == [1]));
    assert!(reductions.iter().all(|(eqn, _)| eqn.subexprs().len() == 0));
}

#[test]
fn transformations_survive_lowering_and_expand_away() {
    let square = region(&[info(&[3])], |a| vec![apply(mul_op(), &a[0], &a[0])]);
    let expr = lowered(&[info(&[2, 3]), info(&[3]), info(&[3])], |a| {
        let mapped = map(&square, &[a[0].clone()], &[0], &[1]).unwrap();
        let (_, tangent) = jvp(&square, &[a[1].clone()], &[a[2].clone()], &[0], &[0]).unwrap();
        let (_, grad) = vjp(&square, &[a[1].clone()], &[a[2].clone()], &[0], &[0]).unwrap();
        vec![mapped[0].clone(), tangent[0].clone(), grad[0].clone()]
    });
    let view = expr.view();
    assert_eq!(view.eqns().len(), 3);
    let find = |pred: fn(OpRef<'_>) -> bool| view.eqns().find(|eqn| pred(eqn.op())).unwrap();
    let mapped = find(is::<MapOperation>);
    assert_eq!(
        MapParameters::from_op_ref(mapped.op()).unwrap(),
        MapParameters {
            size: 2,
            in_axes: vec![0],
            out_axes: vec![1]
        }
    );
    assert_eq!(mapped.outputs().next().unwrap().info_ref(), info(&[3, 2]));
    find(is::<JvpOperation>);
    assert_eq!(
        DiffParameters::from_op_ref(find(is::<VjpOperation>).op()).unwrap(),
        DiffParameters {
            wrt: vec![0],
            of: vec![0]
        }
    );

    let expanded = futures_executor::block_on(
        Pipeline::new()
            .with(ExpandTransforms)
            .with(Dce)
            .run(expr, &gnx_common::cache::Cache::in_memory()),
    )
    .unwrap();
    assert!(!contains_ad(expanded.view()));
}

#[test]
fn nested_transformations_expand_innermost_first() {
    let square = region(&[info(&[])], |a| vec![apply(mul_op(), &a[0], &a[0])]);
    let derivative = region(&[info(&[])], |a| {
        jvp(&square, &[a[0].clone()], &[a[0].clone()], &[0], &[0])
            .unwrap()
            .1
    });
    let expr = lowered(&[info(&[4])], |a| {
        map(&derivative, &[a[0].clone()], &[0], &[0]).unwrap()
    });
    let first = expr.view().eqns().next().unwrap();
    assert!(is::<MapOperation>(first.op()));
    let body = first.subexprs().next().unwrap().expr();
    assert!(is::<JvpOperation>(body.eqns().next().unwrap().op()));
    let expanded = futures_executor::block_on(
        Pipeline::new()
            .with(ExpandTransforms)
            .run(expr, &gnx_common::cache::Cache::in_memory()),
    )
    .unwrap();
    assert!(!contains_ad(expanded.view()));
}

#[test]
fn transformations_reject_invalid_signatures() {
    let _guard = TraceContext::enter();
    let body = region(&[info(&[3])], |a| vec![apply(add_op(), &a[0], &a[0])]);
    let xs = Trace::placeholder(info(&[2, 3]));
    assert!(matches!(
        map(&body, &[xs.clone()], &[2], &[0]),
        Err(LoweringError::RegionSignature(_))
    ));
    assert!(matches!(
        map(&body, &[xs], &[0], &[]),
        Err(LoweringError::RegionSignature(_))
    ));
    let x = Trace::placeholder(info(&[3]));
    // Tangent metadata must match the differentiated input.
    assert!(
        jvp(
            &body,
            &[x.clone()],
            &[Trace::placeholder(info(&[2]))],
            &[0],
            &[0]
        )
        .is_err()
    );
    // Indices must be in range.
    assert!(vjp(&body, &[x.clone()], &[x.clone()], &[1], &[0]).is_err());
    // Integer inputs are not differentiable.
    let int = VarInfo::new(ArrayInfo::new(Shape::fixed([3]), DType::I32));
    let int_body = region(std::slice::from_ref(&int), |a| a.to_vec());
    let n = Trace::placeholder(int);
    assert!(jvp(&int_body, &[n.clone()], &[n], &[0], &[0]).is_err());

    // Transformed functions may read captured cells but not write them.
    let cell = CellTrace::placeholder(info(&[3]));
    let reads = region(&[info(&[3])], |a| vec![apply(add_op(), &a[0], &cell.get())]);
    assert!(jvp(&reads, &[x.clone()], &[x.clone()], &[0], &[0]).is_ok());
    let writes = region(&[info(&[3])], |a| {
        cell.set(a[0].clone());
        a.to_vec()
    });
    assert!(matches!(
        jvp(&writes, &[x.clone()], &[x], &[0], &[0]),
        Err(LoweringError::RegionSignature(message)) if message.contains("cannot write")
    ));
}

/// Like JAX's `pjit`, `cf.call` keeps its boundary under every transformation:
/// forward mode emits a primal call and a linear call, reverse mode evaluates
/// the primal call and transposes the linear one into a call, and batching
/// emits a call to the batched body.
#[test]
fn call_boundaries_survive_transformations() {
    let square = region(&[info(&[3])], |a| vec![apply(mul_op(), &a[0], &a[0])]);
    // f(x) = square(x) + x, with the square behind a call.
    let f = region(&[info(&[3])], |a| {
        let y = gnx_transforms::call(&square, &[a[0].clone().into()])[0].resolve();
        vec![apply(add_op(), &y, &a[0])]
    });
    let calls = |expr: Expr<'_>| {
        expr.eqns()
            .filter(|eqn| is::<CallOperation>(eqn.op()))
            .count()
    };
    let expand = |expr: Arc<RootExpr>| {
        futures_executor::block_on(
            Pipeline::new()
                .with(ExpandTransforms)
                .with(Dce)
                .run(expr, &gnx_common::cache::Cache::in_memory()),
        )
        .unwrap()
    };
    let no_top_level_mul = |expr: Expr<'_>| !expr.eqns().any(|eqn| eqn.op() == mul_op());

    let jvp_expr = expand(lowered(&[info(&[3]), info(&[3])], |a| {
        let (out, tangent) = jvp(&f, &[a[0].clone()], &[a[1].clone()], &[0], &[0]).unwrap();
        vec![out[0].clone(), tangent[0].clone()]
    }));
    let jvp_expr = jvp_expr.view();
    assert_eq!(calls(jvp_expr), 2, "primal and linear calls:\n{jvp_expr}");
    assert!(no_top_level_mul(jvp_expr), "{jvp_expr}");

    let vjp_expr = expand(lowered(&[info(&[3]), info(&[3])], |a| {
        let (out, grad) = vjp(&f, &[a[0].clone()], &[a[1].clone()], &[0], &[0]).unwrap();
        vec![out[0].clone(), grad[0].clone()]
    }));
    let vjp_expr = vjp_expr.view();
    assert_eq!(
        calls(vjp_expr),
        2,
        "primal and transposed calls:\n{vjp_expr}"
    );
    assert!(no_top_level_mul(vjp_expr), "{vjp_expr}");
    // The transposed call consumes the cotangent and returns one value.
    let transposed = vjp_expr
        .eqns()
        .filter(|eqn| is::<CallOperation>(eqn.op()))
        .last()
        .unwrap();
    assert_eq!(transposed.outputs().len(), 1);
    assert!(
        transposed
            .inputs()
            .contains(&vjp_expr.inputs().nth(1).unwrap().var())
    );

    let map_expr = expand(lowered(&[info(&[2, 3])], |a| {
        map(&f, &[a[0].clone()], &[0], &[0]).unwrap()
    }));
    let map_expr = map_expr.view();
    assert_eq!(calls(map_expr), 1, "{map_expr}");
    assert!(no_top_level_mul(map_expr), "{map_expr}");
    let batched = map_expr.eqns().next().unwrap();
    let body = batched.subexprs().next().unwrap().expr();
    assert_eq!(body.inputs().next().unwrap().info_ref(), info(&[2, 3]));
}
