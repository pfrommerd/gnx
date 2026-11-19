use gnx_expr::{ConcreteValue, Value};
use gnx_array::{
    AllocationId, AllocationRef, Array, ArrayInfo, ByteOrder, DType, DataHandle, DlpackRequest,
    HostBuffer, HostId, Lease, LocalAllocations, MemoryLocation, MemoryView, OwnerId, Shape, Shard,
    ShardStorage, StridedLayout, TransferPolicy,
};

#[test]
fn host_buffer_separates_storage_metadata_from_array_info() {
    let info = ArrayInfo::new(Shape::fixed([2]), DType::F16);
    let bytes = vec![0, 60, 0, 64];
    let data = DataHandle::new(HostBuffer::new(info.clone(), bytes.clone()).unwrap());
    let shard = data.single_shard().unwrap();
    assert_eq!(shard.location, MemoryLocation::host(HostId::local()));
    assert!(matches!(shard.storage, ShardStorage::Host(_)));
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
    let dynamic = ArrayInfo::new([gnx_array::Dim::Dynamic].into_iter().collect(), DType::U8);
    assert!(HostBuffer::new(dynamic, vec![]).is_err());
}

#[test]
fn host_sharing_and_dlpack_roundtrip_preserve_the_allocation() {
    let info = ArrayInfo::new(Shape::fixed([4]), DType::U8);
    let host = HostBuffer::new(info, vec![1, 2, 3, 4]).unwrap();
    let address = host.bytes().as_ptr();
    let data = DataHandle::new(host.clone());
    assert_eq!(
        data.clone()
            .into_host(TransferPolicy::RequireZeroCopy)
            .unwrap()
            .bytes()
            .as_ptr(),
        address
    );
    assert!(host.try_into_bytes().is_err());
    let tensor = data
        .into_dlpack()
        .ok()
        .unwrap()
        .export(DlpackRequest {
            stream: None,
            max_version: Some((1, 0)),
        })
        .unwrap();
    // SAFETY: the producer is GNX and no writer has been introduced.
    let data = unsafe { tensor.into_data(MemoryLocation::host(HostId::local()), None) }.unwrap();
    assert_eq!(data.to_host().unwrap().bytes().as_ptr(), address);
    assert_eq!(data.to_host().unwrap().bytes(), [1, 2, 3, 4]);
}

#[test]
fn negative_strides_require_explicit_packing() {
    let host = vec![1_u8, 2, 3];
    let ptr = host.as_ptr().cast_mut();
    // SAFETY: the boxed bytes are owned by the lease and never mutated.
    let info = ArrayInfo::new(Shape::fixed([3]), DType::U8);
    let view = unsafe {
        MemoryView::new(
            info.clone(),
            ptr,
            Some(3),
            StridedLayout {
                byte_offset: 2,
                byte_strides: vec![-1],
                byte_order: ByteOrder::Little,
            },
            true,
            None,
            Lease::on_drop(move || drop(host)),
        )
    }
    .unwrap();
    let storage = ShardStorage::Local {
        view,
        allocation: None,
    };
    let location = MemoryLocation::host(HostId::local());
    assert!(
        DataHandle::single(
            ArrayInfo::new(Shape::fixed([4]), DType::U8),
            location.clone(),
            storage.clone()
        )
        .is_err()
    );
    let array = DataHandle::single(info, location, storage).unwrap();
    assert_eq!(array.to_host().unwrap().bytes(), [3, 2, 1]);
    assert_eq!(
        array
            .into_host(TransferPolicy::RequireZeroCopy)
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::Unsupported
    );
}

#[test]
fn remote_and_mixed_shards_retain_references_without_fetching() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let releases = Arc::new(AtomicUsize::new(0));
    let drop_count = releases.clone();
    let remote = AllocationRef::new(
        AllocationId {
            owner: OwnerId::fresh(),
            id: uuid::Uuid::new_v4(),
        },
        Lease::on_drop(move || {
            drop_count.fetch_add(1, Ordering::Relaxed);
        }),
    );
    let remote_id = remote.id();
    let remote_host = HostId(uuid::Uuid::new_v4());
    let info = ArrayInfo::new(Shape::fixed([4]), DType::U8);
    let whole = DataHandle::from_shards(info.clone(), vec![], Some(remote.clone())).unwrap();
    assert_eq!(whole.single_allocation().unwrap().id(), remote_id);
    assert!(whole.to_host().is_err());
    let shard = Shard {
        bounds: std::iter::once(2..4).collect(),
        replica: 0,
        location: MemoryLocation::host(remote_host),
        storage: ShardStorage::Reference(remote),
    };
    assert!(DataHandle::from_shards(info.clone(), vec![shard.clone()], None).is_err());
    let local = Shard {
        bounds: std::iter::once(0..2).collect(),
        replica: 0,
        location: MemoryLocation::host(HostId::local()),
        storage: ShardStorage::Host(
            HostBuffer::new(ArrayInfo::new(Shape::fixed([2]), DType::U8), vec![1, 2]).unwrap(),
        ),
    };
    let array =
        Array::from_data(DataHandle::from_shards(info.clone(), vec![local, shard], None).unwrap());
    assert_eq!(array.data().unwrap().array_info(), &info);
    assert!(array.data().unwrap().clone().into_dlpack().is_err());
    assert!(array.data().unwrap().to_host().is_err());
    drop(whole);
    assert_eq!(releases.load(Ordering::Relaxed), 0);
    drop(array);
    assert_eq!(releases.load(Ordering::Relaxed), 1);
}

#[test]
fn allocation_ids_are_scoped_to_one_owner_incarnation() {
    let first = LocalAllocations::new();
    let second = LocalAllocations::<usize>::new();
    let reference = first.insert(42_usize);
    assert_eq!(*first.resolve(&reference).unwrap(), 42);
    assert!(second.resolve(&reference).is_none());
    let expired = reference.id();
    drop(reference);
    assert!(
        first
            .resolve(&AllocationRef::new(expired, Lease::on_drop(|| {})))
            .is_none()
    );
}
