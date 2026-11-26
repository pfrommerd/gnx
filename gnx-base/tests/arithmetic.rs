use gnx_base::{Array, ArrayInfo, DType, DataHandle, HostBuffer, Shape};

fn array(shape: &[usize], dtype: DType, bytes: &[u8]) -> Array {
    let info = ArrayInfo::new(Shape::fixed(shape.iter().copied()), dtype);
    Array::from_data(DataHandle::new(
        HostBuffer::copy_from_slice(info, bytes).unwrap(),
    ))
}

#[test]
fn broadcasting_preserves_empty_axes_and_checks_compatibility() {
    let empty = array(&[0, 2], DType::F32, &[]);
    let row = array(&[1, 2], DType::F32, &[0; 8]);
    assert_eq!((&row + &empty).shape(), empty.shape());
    assert_eq!((&empty * &row).shape(), empty.shape());
    assert!(
        empty
            .try_sub(&array(&[2, 2], DType::F32, &[0; 16]))
            .is_err()
    );
    assert!(empty.try_div(&array(&[], DType::I32, &[0; 4])).is_err());
}
