use gnx_expr::{ConcreteValue, Value};
use gnx_math::{
    Array, ArrayInfo, BufferLayout, DType, DataHandle, HostBuffer, MemoryLocation, Shape,
    TransferPolicy,
};

#[test]
fn host_buffer_separates_storage_metadata_from_array_info() {
    let info = ArrayInfo::new(Shape::fixed([2]), DType::F16);
    let bytes = vec![0, 60, 0, 64];
    let data = DataHandle::new(HostBuffer::new(info.clone(), bytes.clone()).unwrap());
    assert_eq!(data.buffer_info().location, MemoryLocation::Host);
    assert_eq!(data.buffer_info().layout, BufferLayout::Packed);
    let array = Array::from_data(data);
    assert_eq!(array.data().unwrap().info(), info);
    assert_eq!(array.data().unwrap().to_host().unwrap().bytes(), bytes);

    let data = DataHandle::new(HostBuffer::new(info.clone(), bytes.clone()).unwrap());
    let value = Value::new(data);
    assert_eq!(value.to_info().downcast::<ArrayInfo>(), Ok(&info));
    assert_eq!(
        value
            .downcast_into::<DataHandle>()
            .unwrap()
            .into_host(TransferPolicy::RequireZeroCopy)
            .unwrap()
            .into_bytes(),
        bytes
    );
}

#[test]
fn packed_host_buffer_rejects_invalid_storage() {
    let half = ArrayInfo::new(Shape::fixed([2]), DType::F16);
    assert!(HostBuffer::new(half, vec![0; 8]).is_err());
    let boolean = ArrayInfo::new(Shape::fixed([1]), DType::Bool);
    assert!(HostBuffer::new(boolean, vec![2]).is_err());
    let dynamic = ArrayInfo::new([gnx_math::Dim::Dynamic].into_iter().collect(), DType::U8);
    assert!(HostBuffer::new(dynamic, vec![]).is_err());
}
