//! Scope and substitution tests using only opaque metadata, not array types.
use gnx_expr::{
    Capture, Dialect, Generic, InputSpec, Invocation, Kind, Op, Operation, Symbol, SymbolInfo,
    TraceContext, Tracer, ValueInfo, VarScope,
    pass::{Builder, dce, rewrite},
};
use std::{fmt, sync::Arc};

struct Test;
impl Dialect for Test {
    fn name(&self) -> &'static str {
        "test"
    }
}
#[derive(Clone, Hash, PartialEq, Eq)]
struct Pair;
impl fmt::Display for Pair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("pair")
    }
}
impl Operation for Pair {
    fn name(&self) -> &'static str {
        "pair"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &Test
    }
}
fn sort() -> SymbolInfo {
    SymbolInfo::new("index")
}
fn info(symbols: &[Symbol]) -> ValueInfo {
    ValueInfo::new("formal").with_symbols(symbols.iter().copied())
}

#[test]
fn inline_substitutes_local_symbols_and_nested_attachments() {
    let mut nested = Builder::new();
    let cap = nested.symbol_closure(sort());
    let v = nested.emit1(Op::new(Pair), &[], info(&[cap]));
    let nested = Arc::new(nested.finish(vec![v]));
    let mut child = Builder::new();
    let a = child.symbol_closure(sort());
    let b = child.symbol_closure(sort());
    let c = child.symbol_input(sort());
    let x = child.input(info(&[c, c]));
    let region = gnx_expr::SubExpr::new(nested, vec![]).with_symbol_captures(vec![b]);
    let result = child.emit(Op::new(Pair), &[x], [info(&[a, b, c, a])], vec![region]);
    let child = child.finish(result);
    let mut parent = Builder::new();
    let first = parent.symbol_input(sort());
    let second = parent.symbol_input(sort());
    let wrong = parent.symbol_input(SymbolInfo::new("ring"));
    let x = parent.input(info(&[first, first]));
    let y = parent.input(info(&[first, second]));
    parent.input(info(&[wrong]));
    assert!(parent.inline(&child, &[second], &[], &[x]).is_err());
    assert!(parent.inline(&child, &[wrong, first], &[], &[x]).is_err());
    assert!(parent.inline(&child, &[second, first], &[], &[y]).is_err());
    let results = parent.inline(&child, &[second, first], &[], &[x]).unwrap();
    let parent = parent.finish(results);
    let copied = rewrite(&parent, &mut |_, _| Ok::<_, ()>(None)).unwrap();
    for e in [&parent, &copied, &dce(&copied)] {
        e.check().unwrap();
        assert_eq!(
            e.value_info(&e.outputs()[0]).unwrap().symbols(),
            &[second, first, first, second]
        );
        assert_eq!(e.eqns()[0].subexprs()[0].symbol_captures, [first]);
    }
}

#[test]
fn trace_captures_a_symbol_without_a_value_capture_and_lowers_call() {
    let outer = TraceContext::enter();
    let a = outer.context().create_symbol(sort());
    let b = outer.context().create_symbol(sort());
    let x = Tracer::<Generic>::placeholder(info(&[a, b]));
    // The child's two symbols are captures solely through its output metadata.
    let child = TraceContext::enter();
    let v = Invocation::invoke(
        Op::new(Pair),
        vec![],
        Vec::<gnx_expr::Trace>::new(),
        vec![info(&[a, b])],
    )
    .pop()
    .unwrap();
    let cap = Capture::from_context(child.context(), &[], &[v.into()]).unwrap();
    drop(child);
    assert!(cap.expr().closure().is_empty());
    assert_eq!(cap.expr().symbol_closure(), &[(a, sort()), (b, sort())]);
    let result = Invocation::invoke_scoped(
        gnx_transforms::cf::call_op(),
        vec![(Arc::new(cap.expr().clone()), vec![], vec![b, a])],
        Vec::<gnx_expr::Trace>::new(),
        vec![info(&[b, a])],
    )
    .pop()
    .unwrap();
    let cap = Capture::from_context(outer.context(), &[x.into()], &[result.into()]).unwrap();
    assert_eq!(cap.expr().inputs()[0].1, Kind::Value);
    assert_eq!(cap.expr().symbol_inputs(), &[(a, sort()), (b, sort())]);
    let lowered = cap
        .expr()
        .lower(&InputSpec::from_args(cap.closure(), cap.inputs()))
        .unwrap();
    lowered.expr.check().unwrap();
    assert_eq!(lowered.expr.symbol_inputs(), cap.expr().symbol_inputs());
    assert_eq!(lowered.output_infos[0].symbols(), &[b, a]);
    assert_eq!(
        lowered.trace_body().symbol_inputs(),
        cap.expr().symbol_inputs()
    );
}

#[test]
fn symbol_inputs_require_binding_occurrences_and_declarations_are_disjoint() {
    let mut scope = VarScope::new();
    let s = scope.create_symbol();
    let x = scope.create_var();
    let e = gnx_expr::Expr::new(vec![], vec![(x, info(&[]))], vec![], vec![x]);
    assert!(e.with_symbols(vec![], vec![(s, sort())]).check().is_err());
    let e = gnx_expr::Expr::new(vec![], vec![(x, info(&[s]))], vec![], vec![x]);
    assert!(e.check().is_err());
    assert!(
        e.clone()
            .with_symbols(vec![(s, sort())], vec![(s, sort())])
            .check()
            .is_err()
    );
    e.with_symbols(vec![], vec![(s, sort())]).check().unwrap();
}

#[test]
fn jit_cache_keys_use_sorts_and_sharing_not_caller_symbol_numbers() {
    use gnx_expr::ObjectTracer;
    use gnx_transforms::{Callable, jit};
    let traces = std::cell::Cell::new(0);
    let fun = jit(|x: ObjectTracer| {
        traces.set(traces.get() + 1);
        let ObjectTracer::Ref(v) = x else { panic!() };
        let out_info = v.info().clone();
        ObjectTracer::from(
            Invocation::invoke(Op::new(Pair), vec![], vec![v], vec![out_info])
                .pop()
                .unwrap(),
        )
    });
    let ctx = TraceContext::enter();
    let a = ctx.context().create_symbol(sort());
    let b = ctx.context().create_symbol(sort());
    let ring = ctx.context().create_symbol(SymbolInfo::new("ring"));
    for (slots, expected_traces) in [([a, a], 1), ([b, b], 1), ([a, b], 2), ([ring, ring], 3)] {
        let x = ObjectTracer::from(Tracer::<Generic>::placeholder(info(&slots)));
        let y = fun.invoke((x.clone(),));
        let ObjectTracer::Ref(v) = &y else { panic!() };
        assert_eq!(v.info().symbols(), &slots);
        let cap = Capture::from_context(ctx.context(), &[x], &[y]).unwrap();
        let lowered = cap
            .expr()
            .lower(&InputSpec::from_args(cap.closure(), cap.inputs()))
            .unwrap();
        assert_eq!(lowered.output_infos[0].symbols(), &slots);
        assert_eq!(traces.get(), expected_traces);
    }
}
