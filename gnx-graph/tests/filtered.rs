use gnx_graph::util::{GraphExt, IterSource, dag};
use gnx_graph::{All, Builder, Filtered, Graph, GraphContext, Of, SourceError};
use std::sync::Arc;

#[test]
fn subtree_view_rebuilds_only_visible_leaves_and_preserves_ties() {
    let cell = Arc::new(3u32);
    let full = ((cell.clone(), cell), 9u32);
    let selected = Filtered::new(full, dag!(All, *));
    let view = selected.as_view();
    let numbers = Of::<u32>::filter();
    assert_eq!(view.collect_leaves(&numbers), [3]);
    // The caller's positional filter is relative to the original root.
    assert!(view.collect_leaves(dag!(*, numbers.clone())).is_empty());
    let updated = view
        .builder::<_, _, SourceError>(numbers, &mut GraphContext::new())
        .unwrap()
        .build(IterSource::new([7u32]), &mut GraphContext::new())
        .unwrap();
    assert_eq!(updated.collect_leaves(Of::<u32>::filter()), [7]);
    let ((a, b), hidden) = updated.graph;
    assert_eq!((*a, hidden), (7, 9));
    assert!(Arc::ptr_eq(&a, &b));
}
