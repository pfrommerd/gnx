use gnx::graph::{Builder, Graph, GraphContext, Of, TypedGraph};

#[test]
fn test_graph() {
    let a: (u32, u32) = (1, 2);
    let b: Vec<u32> = vec![3, 4];
    let c = a
        .builder(Of::<u32>::filter())
        .build(b.as_source(), &mut GraphContext::new())
        .unwrap();
    assert_eq!(c, (3, 4))
}
