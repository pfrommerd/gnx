use gnx_expr::{EqnSource, ExprBuilder, VarInfo, VarScope};
use gnx_math::{
    ArrayInfo, DType, Shape,
    shard::{Sharding, constraint_op, replicate_op},
};

#[test]
fn sharding_is_a_checked_metadata_preserving_ssa_annotation() {
    let info = VarInfo::new(ArrayInfo::new(Shape::fixed([8, 4]), DType::F32));
    let mut scope = VarScope::new();
    let input = scope.create_var();
    let output = scope.create_var();
    let spec = Sharding::new(vec![vec!["data".into()], vec!["model".into()]]).unwrap();
    assert_eq!(
        Sharding::from_op(&constraint_op(spec.clone()), 2).unwrap(),
        spec
    );
    assert!(
        Sharding::from_op(&replicate_op(), 2)
            .unwrap()
            .is_replicated()
    );
    assert!(Sharding::new(vec![vec!["data".into()], vec!["data".into()]]).is_err());
    assert!(Sharding::new(vec![vec!["invalid\"axis".into()]]).is_err());

    let expr = |spec| {
        ExprBuilder::from_parts(
            vec![],
            vec![],
            vec![(input, info.clone())],
            vec![EqnSource::new(
                constraint_op(spec),
                vec![],
                vec![input],
                vec![(output, info.clone())],
            )],
            vec![output],
        )
        .build()
        .unwrap()
    };
    assert!(expr(spec).view().check().is_ok());
    assert!(expr(Sharding::replicated(1)).view().check().is_err());
}
