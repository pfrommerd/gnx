use gnx_math::{
    Array, ArrayInfo, DType, Dim, Shape, lax,
    random::{self, Algorithm::*},
};

#[test]
fn key_and_shape_contracts_are_checked_before_execution() {
    let k = random::key(7, Philox4x32);
    assert!(k.try_add(&k).is_err());
    assert!(lax::try_neg(&k).is_err());
    assert!(random::try_uniform(&k, (), DType::I64).is_err());
    assert!(random::try_normal(&random::split(&k, 3), (), DType::F64).is_err());
    assert!(random::try_bits(&k, (usize::MAX, 2), DType::U32).is_err());
    let symbolic: Shape = [Dim::Dynamic, Dim::Fixed(3)].into_iter().collect();
    let a = Array::placeholder(ArrayInfo::new(&symbolic, DType::F32));
    assert_eq!(a.shape(), symbolic);
    assert!(random::try_uniform(&k, &symbolic, DType::F32).is_err());
    let a = lax::zeros((2, 3), DType::F32).reshape((3, 2));
    assert_eq!(
        lax::broadcast_in_dim(&a, (4, 3, 2), &[1, 2]).shape(),
        Shape::fixed([4, 3, 2])
    );
}
