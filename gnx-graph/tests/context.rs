use gnx_graph::{Error, GraphContext, GraphId, SourceError};
use std::sync::Arc;

#[test]
fn completed_shared_nodes_are_reused() {
    let mut context = GraphContext::new();
    let id = GraphId::from(1u64);
    let first = context
        .create::<_, SourceError, _>(id, |_| Ok(Arc::new(7u32)))
        .unwrap();
    let second = context
        .create::<Arc<u32>, SourceError, _>(id, |_| panic!("must reuse completed node"))
        .unwrap();
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn construction_cycles_fail_and_errors_release_the_reservation() {
    let mut context = GraphContext::new();
    let id = GraphId::from(1u64);
    let cycle =
        context.create::<u32, SourceError, _>(id, |context| context.create(id, |_| Ok(7u32)));
    assert!(cycle.is_err());
    let failed = context.create::<u32, SourceError, _>(id, |_| Err(SourceError::custom("failure")));
    assert!(failed.is_err());
    assert_eq!(
        context
            .create::<u32, SourceError, _>(id, |_| Ok(7))
            .unwrap(),
        7
    );
}
