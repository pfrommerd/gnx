use gnx_base::{Array, ArrayInfo, DType, Shape};
use gnx_expr::{TraceContext, Tracer, VarInfo};
use gnx_graph::{Filter, Graph, KeyRef, Leaf, Of, util::dag};
use gnx_transforms::on_axis;
use gnx_transforms::{Callable, TryCallable, jvp, map, scan_graph, vjp};
use std::cell::Cell;

fn array(shape: &[usize]) -> Array {
    Tracer::<Array>::placeholder(VarInfo::new(ArrayInfo::new(
        Shape::fixed(shape.iter().copied()),
        DType::F32,
    )))
    .into()
}
fn shape(a: &Array) -> Vec<usize> {
    gnx_transforms::util::dims(a.tracer().trace_ref().info()).unwrap()
}

// A domain-specific match value; neither the matched type nor its projection is a tracer.
#[derive(Clone, Leaf)]
struct Weights(usize);
#[derive(Clone)]
struct WeightPairs;
impl Filter for WeightPairs {
    type Leaf = Weights;
    type Owned = Self;
    type ChildFilter<'s> = Self;
    type Annotation<'g> = usize;
    fn owned(&self) -> Self {
        self.clone()
    }
    fn child<'s>(&'s self, _: KeyRef<'s>) -> Self {
        self.clone()
    }
    fn matches_ref<'g, G: Graph>(&self, g: &'g G) -> Result<&'g Weights, &'g G> {
        gnx_graph::cast!(g, &(Array, Array)).map(|_| &Weights(1))
    }
    fn matches_value<G: Graph>(&self, g: G) -> Result<Weights, G> {
        if self.matches_ref(&g).is_ok() {
            Ok(Weights(1))
        } else {
            Err(g)
        }
    }
    fn annotate<'g>(&self, w: <Weights as Leaf>::Ref<'g>) -> Self::Annotation<'g> {
        w.0
    }
}

#[test]
fn domain_match_is_a_prefix_and_output_axes_belong_to_the_map_cache() {
    let _scope = TraceContext::enter();
    let args = ((array(&[3, 4]), array(&[5, 4])), array(&[7]));
    let count = Cell::new(0);
    let f = map(|w: (Array, Array), bias: Array| {
        count.set(count.get() + 1);
        assert_eq!(shape(&bias), [7]);
        w
    })
    .in_axes(WeightPairs)
    .out_axes(WeightPairs);
    let expected = map(|w: (Array, Array), _: Array| w)
        .in_axes(dag!(1,*))
        .out_axes(1);
    let (a, b) = expected.invoke(args.clone());
    for _ in 0..2 {
        let (x, y) = f.invoke(args.clone());
        assert_eq!((shape(&x), shape(&y)), (shape(&a), shape(&b)));
    }
    assert_eq!(count.get(), 1);
    let f = f.out_axes(on_axis(0, Of::<Array>::filter()));
    let (x, y) = f.invoke(args);
    assert_eq!((shape(&x), shape(&y)), (vec![4, 3], vec![4, 5]));
}

#[test]
fn map_preserves_fresh_and_passed_through_cell_aliases() {
    use gnx_base::ArrayRef;
    use gnx_expr::CellKey;
    let _scope = TraceContext::enter();
    let identity = |r: &ArrayRef| CellKey::from(r.tracer().cell_ref());
    let pair = map(|x: Array| {
        let r = ArrayRef::new(x);
        (r.clone(), r)
    });
    let (a, b) = pair.invoke((array(&[3, 2]),));
    assert_eq!(identity(&a), identity(&b));
    assert_eq!(shape(&a.get()), [3, 2]);
    let update = map(|dst: ArrayRef, alias: ArrayRef, x: Array| {
        dst.set(x);
        alias
    });
    let result = update.invoke((a.clone(), b, array(&[3, 2])));
    assert_eq!(identity(&result), identity(&a));
}

#[test]
fn map_rejects_writes_to_unmapped_cells() {
    use gnx_base::ArrayRef;
    let _scope = TraceContext::enter();
    let shared = ArrayRef::new(array(&[2]));
    let f = map(|x: Array, shared: ArrayRef| {
        shared.set(x.clone());
        x
    })
    .in_axes(dag!(0,*));
    let err = f.try_invoke((array(&[3, 2]), shared)).err().unwrap();
    assert!(
        err.to_string().contains("cannot write captured cells"),
        "{err}"
    );
}

#[test]
fn uncovered_map_result_is_an_error() {
    let _scope = TraceContext::enter();
    let f = map(|x: Array| x).out_axes(WeightPairs);
    assert!(
        f.try_invoke((array(&[2]),))
            .err()
            .expect("missing axis")
            .to_string()
            .contains("out_axes")
    );
}

#[test]
fn ad_uses_matches_not_axis_annotations() {
    let _scope = TraceContext::enter();
    let wrt = dag!(on_axis(100, Of::<Array>::filter()), *);
    let args = (array(&[2]), array(&[3]));
    let (y, dy) = jvp(|x: Array, _: Array| x)
        .wrt(&wrt)
        .invoke((args.clone(), args.clone()));
    assert_eq!(shape(&dy), shape(&y));
    let (_, dx) = vjp(|x: Array, _: Array| x).wrt(wrt).invoke((args, y));
    assert_eq!(shape(&dx.0), [2]);
    assert_eq!(shape(&dx.1), [3]);
}

#[test]
fn scan_with_only_captures_requires_length_and_stacks_results() {
    let _scope = TraceContext::enter();
    let f = || scan_graph(|carry: Array, x: Array| (carry, x)).in_axes(WeightPairs);
    let carry = array(&[2]);
    let xs = array(&[3]);
    let shared = map(|x: Array| x)
        .in_axes(dag!(as usize; *))
        .axis_size(4)
        .invoke((xs.clone(),));
    assert_eq!(shape(&shared), [4, 3]);
    assert!(
        f().run(&carry, &xs, None)
            .err()
            .expect("length required")
            .to_string()
            .contains("length")
    );
    let (_, ys) = f().run(&carry, &xs, Some(4)).unwrap();
    assert_eq!(shape(&ys), [4, 3]);
}

#[test]
fn macro_accepts_filters_and_ad_distinguishes_exact_from_prefix() {
    let _scope = TraceContext::enter();
    let rows = gnx_transforms::map!(
        |x: Array| x,
        in_axes = on_axis(1, Of::<Array>::filter()),
        out_axes = 0
    );
    assert_eq!(shape(&rows(array(&[2, 3]))), [3, 2]);
    let integer: Array =
        Tracer::<Array>::placeholder(VarInfo::new(ArrayInfo::new(Shape::fixed([2]), DType::I32)))
            .into();
    let args = (integer,);
    let f = jvp(|x: Array| x).wrt(Of::<Array>::filter());
    assert!(
        f.try_invoke((args.clone(), args.clone()))
            .err()
            .expect("direct integer selection")
            .to_string()
            .contains("not a floating-point array")
    );
    jvp(|x: Array| x)
        .wrt(gnx_graph::All)
        .try_invoke((args.clone(), args))
        .unwrap();
}

#[test]
fn grad_rejects_unresolved_cell_aliases_instead_of_assuming_distinct_inputs() {
    use gnx_base::ArrayRef;
    use gnx_expr::CellTracer;
    let _scope = TraceContext::enter();
    let input = || {
        ArrayRef::from(CellTracer::<Array>::placeholder(VarInfo::new(
            ArrayInfo::new(Shape::fixed([]), DType::F32),
        )))
    };
    let df = gnx_transforms::grad(|a: ArrayRef, _: ArrayRef| a.get()).wrt_arg(0);
    let err = df.try_invoke((input(), input())).err().unwrap().to_string();
    assert!(err.contains("aliases"), "{err}");
}

#[test]
fn wrt_indices_select_top_level_arguments_only() {
    let _scope = TraceContext::enter();
    let f = |x: Array, _: Array, _: Array| x;
    let integer: Array =
        Tracer::<Array>::placeholder(VarInfo::new(ArrayInfo::new(Shape::fixed([]), DType::I32)))
            .into();
    let args = (array(&[]), integer, array(&[]));
    let grads = gnx_transforms::grad(f)
        .wrt_args([0, 2])
        .invoke(args.clone());
    let visible =
        gnx_graph::util::GraphExt::collect_leaves_with_paths(&grads, Of::<Array>::filter());
    assert_eq!(
        visible
            .into_iter()
            .map(|(path, _)| path)
            .collect::<Vec<_>>(),
        [vec![gnx_graph::Key::from(0)], vec![gnx_graph::Key::from(2)]]
    );
    let (dx, dy, dz) = grads.graph;
    assert_eq!(
        (shape(&dx), shape(&dy), shape(&dz)),
        (vec![], vec![], vec![])
    );
    let err = gnx_transforms::grad(f)
        .wrt_arg(1)
        .try_invoke(args.clone())
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("not a floating-point array"), "{err}");
    let err = gnx_transforms::grad(f)
        .wrt_arg(3)
        .try_invoke(args.clone())
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("argument index 3 is out of range"), "{err}");
    let last = 2usize;
    let df = gnx_transforms::grad!(|x: Array, _y: Array, _z: Array| x, wrt_args = [0, last]);
    df(args.0.clone(), args.1.clone(), args.2.clone());
    let masked = gnx_transforms::grad!(
        |x: Array, _y: Array, _z: Array| x,
        wrt = dag!(gnx_graph::All, *, gnx_graph::All)
    );
    masked(args.0, args.1, args.2);
}

#[test]
fn scan_slices_only_xs_and_moves_stacked_axis() {
    let _scope = TraceContext::enter();
    let xs = (array(&[3, 4]), array(&[7]));
    let (c, ys) = scan_graph(|carry: Array, (x, captured): (Array, Array)| {
        assert_eq!(shape(&x), [3]);
        assert_eq!(shape(&captured), [7]);
        (carry, x)
    })
    .in_axes(dag!(on_axis(1, Of::<Array>::filter()), *))
    .out_axes(on_axis(1, Of::<Array>::filter()))
    .run(&array(&[2]), &xs, None)
    .unwrap();
    assert_eq!(shape(&c), [2]);
    assert_eq!(shape(&ys), [3, 4]);
}
