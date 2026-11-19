use gnx_array::{ArrayInfo, DType, Dim, Shape};

#[test]
fn array_info_displays_the_compact_dtype_before_the_shape() {
    let scalar = ArrayInfo::new(Shape::fixed([]), DType::F32);
    assert_eq!(scalar.to_string(), "<f32>()");

    let matrix = ArrayInfo::new(Shape::fixed([2, 3]), DType::I32);
    assert_eq!(matrix.to_string(), "<i32>(2, 3)");

    let dynamic = ArrayInfo::new(
        [Dim::Dynamic, Dim::Fixed(4)].into_iter().collect(),
        DType::Bool,
    );
    assert_eq!(dynamic.to_string(), "<bool>(?, 4)");
}
