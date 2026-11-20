use gnx_common::{ArrayInfo, DType, Shape};
use gnx_expr::pass::{Builder, Dce, Pipeline};
use gnx_expr::{
    Capture, CellTrace, Expr, InputSpec, Invocation, LoweringError, ObjectTracer, Op, Operation,
    Trace, TraceContext, ValueInfo,
};
use gnx_common::ops::reduction::{ReduceAddOperation, reduce_add};
use gnx_common::ops::{add_op, mul_op};
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

fn info(dims: &[usize]) -> ValueInfo {
    ValueInfo::new(ArrayInfo::new(
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
fn region(inputs: &[ValueInfo], body: impl FnOnce(&[Trace]) -> Vec<Trace>) -> Capture {
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
fn lowered(inputs: &[ValueInfo], build: impl FnOnce(&[Trace]) -> Vec<Trace>) -> Expr {
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
    program.expr.check().unwrap();
    program.expr.as_ref().clone()
}

fn is<T: Operation>(op: &Op) -> bool {
    op.downcast_ref::<T>().is_some()
}

fn contains_ad(expr: &Expr) -> bool {
    expr.eqns().iter().any(|eqn| {
        is::<MapOperation>(eqn.op())
            || is::<JvpOperation>(eqn.op())
            || is::<VjpOperation>(eqn.op())
            || eqn.subexprs().iter().any(|sub| contains_ad(sub.expr()))
    })
}

#[test]
fn additive_reduction_expands_without_a_math_rule_provider() {
    use gnx_expr::SubExpr;
    use gnx_transforms::ops::{ad, map};

    let mut body = Builder::new();
    let x = body.input(info(&[3]));
    let sum = reduce_add(&mut body, x, &[0]).unwrap();
    let body = SubExpr::new(body.finish(vec![sum]).into(), vec![]);

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
    let expanded = gnx_transforms::expand_transforms(&program.finish(outputs)).unwrap();
    expanded.check().unwrap();
    assert!(!contains_ad(&expanded));
    let reductions: Vec<_> = expanded
        .eqns()
        .iter()
        .filter_map(|eqn| eqn.op().downcast_ref::<ReduceAddOperation>().map(|op| (eqn, op)))
        .collect();
    assert!(reductions.iter().any(|(_, op)| op.axes() == [0]));
    assert!(reductions.iter().any(|(_, op)| op.axes() == [1]));
    assert!(reductions.iter().all(|(eqn, _)| eqn.subexprs().is_empty()));
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
    assert_eq!(expr.eqns().len(), 3);
    let find = |pred: fn(&Op) -> bool| expr.eqns().iter().find(|eqn| pred(eqn.op())).unwrap();
    let mapped = find(is::<MapOperation>);
    assert_eq!(
        MapParameters::from_op(mapped.op()).unwrap(),
        MapParameters {
            size: 2,
            in_axes: vec![0],
            out_axes: vec![1]
        }
    );
    assert_eq!(mapped.outputs()[0].1, info(&[3, 2]));
    find(is::<JvpOperation>);
    assert_eq!(
        DiffParameters::from_op(find(is::<VjpOperation>).op()).unwrap(),
        DiffParameters {
            wrt: vec![0],
            of: vec![0]
        }
    );

    let expanded = Pipeline::new()
        .with(ExpandTransforms)
        .with(Dce)
        .run(expr)
        .unwrap();
    assert!(!contains_ad(&expanded));
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
    assert!(is::<MapOperation>(expr.eqns()[0].op()));
    assert!(is::<JvpOperation>(
        expr.eqns()[0].subexprs()[0].expr().eqns()[0].op()
    ));
    let expanded = Pipeline::new().with(ExpandTransforms).run(expr).unwrap();
    assert!(!contains_ad(&expanded));
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
    let int = ValueInfo::new(ArrayInfo::new(Shape::fixed([3]), DType::I32));
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
    let calls = |expr: &Expr| {
        expr.eqns()
            .iter()
            .filter(|eqn| is::<CallOperation>(eqn.op()))
            .count()
    };
    let expand = |expr: Expr| {
        Pipeline::new()
            .with(ExpandTransforms)
            .with(Dce)
            .run(expr)
            .unwrap()
    };
    let no_top_level_mul = |expr: &Expr| {
        !expr
            .eqns()
            .iter()
            .any(|eqn| eqn.op() == &mul_op())
    };

    let jvp_expr = expand(lowered(&[info(&[3]), info(&[3])], |a| {
        let (out, tangent) = jvp(&f, &[a[0].clone()], &[a[1].clone()], &[0], &[0]).unwrap();
        vec![out[0].clone(), tangent[0].clone()]
    }));
    assert_eq!(calls(&jvp_expr), 2, "primal and linear calls:\n{jvp_expr}");
    assert!(no_top_level_mul(&jvp_expr), "{jvp_expr}");

    let vjp_expr = expand(lowered(&[info(&[3]), info(&[3])], |a| {
        let (out, grad) = vjp(&f, &[a[0].clone()], &[a[1].clone()], &[0], &[0]).unwrap();
        vec![out[0].clone(), grad[0].clone()]
    }));
    assert_eq!(calls(&vjp_expr), 2, "primal and transposed calls:\n{vjp_expr}");
    assert!(no_top_level_mul(&vjp_expr), "{vjp_expr}");
    // The transposed call consumes the cotangent and returns one value.
    let transposed = vjp_expr
        .eqns()
        .iter()
        .filter(|eqn| is::<CallOperation>(eqn.op()))
        .last()
        .unwrap();
    assert_eq!(transposed.outputs().len(), 1);
    assert!(transposed.inputs().contains(&vjp_expr.inputs()[1].0));

    let map_expr = expand(lowered(&[info(&[2, 3])], |a| {
        map(&f, &[a[0].clone()], &[0], &[0]).unwrap()
    }));
    assert_eq!(calls(&map_expr), 1, "{map_expr}");
    assert!(no_top_level_mul(&map_expr), "{map_expr}");
    let batched = &map_expr.eqns()[0];
    assert_eq!(batched.subexprs()[0].expr().inputs()[0].1, info(&[2, 3]));
}
