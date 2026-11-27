use gnx_math::{Array, ArrayInfo, DType, Shape, lax, np};

#[test]
fn signed_axes_are_distinct_after_normalization() {
    let x = Array::placeholder(ArrayInfo::new(Shape::fixed([2, 0, 3]), DType::I32));
    assert_eq!(np::sum(&x, [-1, 0]).shape(), Shape::fixed([0]));
    assert_eq!(np::sum(&x, []).shape(), x.shape());
    assert_eq!(np::sum(&x, np::ALL).shape(), Shape::fixed([]));
    assert!(np::try_sum(&x, [-1, 2]).is_err());
    assert!(np::try_sum(&x, isize::MIN).is_err());
    assert!(lax::try_reduce_sum(&x, &[2, 1]).is_err());
    assert!(lax::try_tanh(&x).is_err());
}

#[test]
fn dot_keeps_free_axes_and_handles_vectors_and_scalars() {
    let x = np::zeros([2, 3, 4], DType::F32);
    let y = np::zeros([5, 4, 6], DType::F32);
    let v = np::zeros([4], DType::F32);
    let scalar = Array::scalar(2f32);
    assert_eq!(np::dot(&x, &y).shape(), Shape::fixed([2, 3, 5, 6]));
    assert_eq!(np::dot(&x, &v).shape(), Shape::fixed([2, 3]));
    assert_eq!(np::dot(&v, &v).shape(), scalar.shape());
    assert_eq!(np::dot(&scalar, &y).shape(), y.shape());
    assert_eq!(np::dot(&x, &scalar).shape(), x.shape());
    assert!(np::try_dot(&v, &x).is_err());
}

#[test]
fn dot_general_orders_batch_then_free_axes() {
    let a = np::zeros([3, 2, 4], DType::F32);
    let b = np::zeros([4, 5, 2], DType::F32);
    let mut axes = lax::DotDimensions {
        lhs_batch: vec![1],
        rhs_batch: vec![2],
        lhs_contracting: vec![2],
        rhs_contracting: vec![0],
    };
    assert_eq!(
        lax::dot_general(&a, &b, axes.clone()).shape(),
        Shape::fixed([2, 3, 5])
    );
    axes.lhs_contracting = vec![1];
    assert!(lax::try_dot_general(&a, &b, axes).is_err());
}

#[test]
fn indexing_checks_extent_and_index_metadata() {
    let x = np::zeros([2, 3], DType::F32);
    let i = np::scalar(0., DType::I32);
    assert!(lax::try_dynamic_slice(&x, &[&i, &i], &[1, 4]).is_err());
    assert!(lax::try_dynamic_update_slice(&x, &x, &[&i, &x]).is_err());
    assert!(lax::try_top_k(&x, 4).is_err());
    let (v, ix) = lax::top_k(&x, 2);
    assert_eq!(v.shape(), Shape::fixed([2, 2]));
    assert_eq!(ix.shape(), v.shape());
    assert_eq!(ix.dtype(), DType::I32);
}

#[test]
fn explicit_broadcast_checks_the_mapping() {
    let arg = |shape| Array::placeholder(ArrayInfo::new(shape, DType::F32));
    let x = arg(Shape::fixed([2, 3]));
    let b = arg(Shape::fixed([3]));
    assert!(lax::try_add(&x, &b).is_err());
    assert!(lax::try_broadcast_in_dim(&b, x.shape(), &[0]).is_err());
    assert!(lax::try_broadcast_in_dim(&x, x.shape(), &[1, 0]).is_err());
    assert_eq!(np::add(&x, &b).shape(), x.shape());
}
