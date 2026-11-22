//! Backend-independent references to immutable array storage.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::{self, Display};
use std::io::{self, ErrorKind};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use gnx_expr::{ConcreteValue, VarInfo};
use uuid::Uuid;

use crate::array::{ArrayInfo, DType, Dim};

pub mod dlpack;
pub use dlpack::{DlpackAbi, DlpackDevice, DlpackRequest, DlpackSource, DlpackTensor};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TransferPolicy {
    #[default]
    RequireZeroCopy,
    AllowCopy,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HostId(pub Uuid);

impl HostId {
    pub fn fresh() -> Self {
        Self(Uuid::new_v4())
    }

    /// Process-scoped by default, not an assertion that two processes share a host.
    pub fn local() -> Self {
        static ID: OnceLock<HostId> = OnceLock::new();
        *ID.get_or_init(|| Self(Uuid::new_v4()))
    }
}

/// Identifies one incarnation of an allocation owner, not a physical device.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OwnerId(pub Uuid);

impl OwnerId {
    pub fn fresh() -> Self {
        Self(Uuid::new_v4())
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum DeviceId {
    Physical {
        namespace: String,
        uuid: Uuid,
    },
    /// A logical ordinal scoped to the runtime that reported it.
    Runtime {
        owner: OwnerId,
        ordinal: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HostMemory {
    Ordinary,
    Pinned,
    Unknown,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MemoryLocation {
    Host {
        host: HostId,
        numa_node: Option<u32>,
        kind: HostMemory,
    },
    Device {
        host: HostId,
        device: DeviceId,
        kind: Option<String>,
    },
    /// Shared/managed memory has no guaranteed single device residency.
    Managed {
        host: HostId,
        device: Option<DeviceId>,
    },
    Unknown,
}

impl MemoryLocation {
    pub fn host(host: HostId) -> Self {
        Self::Host {
            host,
            numa_node: None,
            kind: HostMemory::Ordinary,
        }
    }
}

/// Portable identity of a live allocation within its owner's incarnation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AllocationId {
    pub owner: OwnerId,
    pub id: Uuid,
}

/// A shared lifetime token. The release function cannot inspect storage or
/// perform transfers. Its only job is to relinquish one retained reference.
#[derive(Clone)]
pub struct Lease {
    _release: Arc<Release>,
}

struct Release {
    context: NonNull<c_void>,
    release: unsafe fn(NonNull<c_void>),
}

// SAFETY: the constructor requires release to be callable on any thread and
// the context is never dereferenced except by that once-only release.
unsafe impl Send for Release {}
unsafe impl Sync for Release {}

impl Drop for Release {
    fn drop(&mut self) {
        unsafe { (self.release)(self.context) }
    }
}

impl Lease {
    pub fn on_drop<F: FnOnce() + Send + 'static>(release: F) -> Self {
        unsafe fn run<F: FnOnce()>(p: NonNull<c_void>) {
            // SAFETY: the pointer came from Box<F> and is consumed exactly once.
            let release = unsafe { Box::from_raw(p.cast::<F>().as_ptr()) };
            release();
        }
        Self {
            _release: Arc::new(Release {
                context: NonNull::from(Box::leak(Box::new(release))).cast(),
                release: run::<F>,
            }),
        }
    }
}

impl fmt::Debug for Lease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Lease(..)")
    }
}

#[derive(Clone, Debug)]
pub struct AllocationRef {
    id: AllocationId,
    _lease: Lease,
}

impl AllocationRef {
    pub fn new(id: AllocationId, lease: Lease) -> Self {
        Self { id, _lease: lease }
    }

    pub fn id(&self) -> AllocationId {
        self.id
    }
}

/// An optional typed allocation table for an in-process backend. The payload
/// stays in the table, never inside a `DataHandle`.
pub struct LocalAllocations<T> {
    owner: OwnerId,
    values: Arc<Mutex<HashMap<Uuid, Arc<T>>>>,
}

impl<T: Send + Sync + 'static> Default for LocalAllocations<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync + 'static> LocalAllocations<T> {
    pub fn new() -> Self {
        Self {
            owner: OwnerId::fresh(),
            values: Arc::default(),
        }
    }

    pub fn owner(&self) -> OwnerId {
        self.owner
    }

    pub fn insert(&self, value: T) -> AllocationRef {
        let id = Uuid::new_v4();
        self.values.lock().unwrap().insert(id, Arc::new(value));
        let values = self.values.clone();
        let lease = Lease::on_drop(move || {
            let value = values.lock().unwrap().remove(&id);
            drop(value); // Run backend destructors without holding the table lock.
        });
        AllocationRef::new(
            AllocationId {
                owner: self.owner,
                id,
            },
            lease,
        )
    }

    pub fn resolve(&self, allocation: &AllocationRef) -> Option<Arc<T>> {
        let id = allocation.id();
        if id.owner != self.owner {
            return None;
        }
        self.values.lock().unwrap().get(&id.id).cloned()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    pub const NATIVE: Self = if cfg!(target_endian = "little") {
        Self::Little
    } else {
        Self::Big
    };
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StridedLayout {
    pub byte_offset: usize,
    pub byte_strides: Vec<isize>,
    pub byte_order: ByteOrder,
}

/// Ready, affine memory in this process's address space or a device context.
/// An address is not assumed to be CPU-dereferenceable.
#[derive(Clone, Debug)]
pub struct MemoryView {
    info: ArrayInfo,
    address: Option<NonNull<u8>>,
    extent: Option<usize>,
    layout: StridedLayout,
    host_readable: bool,
    dlpack_device: Option<DlpackDevice>,
    _lease: Lease,
}

// SAFETY: construction requires ready, immutable bytes and a thread-safe lease.
unsafe impl Send for MemoryView {}
unsafe impl Sync for MemoryView {}

impl MemoryView {
    /// # Safety
    /// The addressed affine view (including negative strides) must stay valid,
    /// initialized and immutable for the lease. Its writes must already be
    /// complete. `host_readable` promises ordinary CPU loads are legal; an
    /// advertised DLPack device must describe this pointer in this process.
    /// Pass a null address only for an empty view. If `extent` is supplied, it
    /// bounds the allocation starting at `address`.
    pub unsafe fn new(
        info: ArrayInfo,
        address: *mut u8,
        extent: Option<usize>,
        layout: StridedLayout,
        host_readable: bool,
        dlpack_device: Option<DlpackDevice>,
        lease: Lease,
    ) -> io::Result<Self> {
        let view = Self {
            info,
            address: NonNull::new(address),
            extent,
            layout,
            host_readable,
            dlpack_device,
            _lease: lease,
        };
        validate_view(&view, &dimensions(&view.info)?, view.info.dtype())?;
        Ok(view)
    }

    pub fn info(&self) -> &ArrayInfo {
        &self.info
    }
    pub fn layout(&self) -> &StridedLayout {
        &self.layout
    }
    pub fn address(&self) -> Option<NonNull<u8>> {
        self.address
    }
    pub fn extent(&self) -> Option<usize> {
        self.extent
    }
    pub fn is_host_readable(&self) -> bool {
        self.host_readable
    }
}

/// A reference can describe either one allocation per shard or one allocation
/// for the whole array. A local view can coexist with its owning runtime ID.
#[derive(Clone, Debug)]
pub enum ShardStorage {
    Host(HostBuffer),
    Local {
        view: MemoryView,
        allocation: Option<AllocationRef>,
    },
    Reference(AllocationRef),
    /// The enclosing handle has a whole-array allocation reference.
    WholeArray,
}

#[derive(Clone, Debug)]
pub struct Shard {
    pub bounds: Vec<Range<usize>>,
    pub replica: u32,
    pub location: MemoryLocation,
    pub storage: ShardStorage,
}

#[derive(Clone, Debug)]
pub struct DataHandle(Arc<ArrayStorage>);

#[derive(Debug)]
struct ArrayStorage {
    info: ArrayInfo,
    shards: Vec<Shard>,
    whole: Option<AllocationRef>,
}

impl DataHandle {
    pub fn new(storage: HostBuffer) -> Self {
        Self::from_host_on(storage, HostId::local())
    }

    pub fn from_host_on(storage: HostBuffer, host: HostId) -> Self {
        let info = storage.info.clone();
        Self::single(
            info,
            MemoryLocation::host(host),
            ShardStorage::Host(storage),
        )
        .expect("validated host buffer")
    }

    pub fn single(
        info: ArrayInfo,
        location: MemoryLocation,
        storage: ShardStorage,
    ) -> io::Result<Self> {
        let bounds = dimensions(&info)?.into_iter().map(|dim| 0..dim).collect();
        Self::from_shards(
            info,
            vec![Shard {
                bounds,
                replica: 0,
                location,
                storage,
            }],
            None,
        )
    }

    /// `whole` can retain a distributed array for which only placement (and
    /// possibly some local views) is exposed. No transport is implied.
    pub fn from_shards(
        info: ArrayInfo,
        shards: Vec<Shard>,
        whole: Option<AllocationRef>,
    ) -> io::Result<Self> {
        let dims = dimensions(&info)?;
        let total = volume(&dims)?;
        if shards.is_empty() && whole.is_none() {
            return Err(invalid("array needs storage"));
        }
        let mut ranges: Vec<&[Range<usize>]> = Vec::new();
        let mut covered = 0usize;
        for (i, shard) in shards.iter().enumerate() {
            if shard.bounds.len() != dims.len()
                || shard
                    .bounds
                    .iter()
                    .zip(&dims)
                    .any(|(r, dim)| r.start > r.end || r.end > *dim)
            {
                return Err(invalid("shard bounds are outside the global shape"));
            }
            let local_dims = shard.bounds.iter().map(|r| r.len()).collect::<Vec<_>>();
            let size = volume(&local_dims)?;
            if shards[..i]
                .iter()
                .any(|prior| prior.bounds == shard.bounds && prior.replica == shard.replica)
            {
                return Err(invalid("duplicate shard replica"));
            }
            if !ranges.contains(&shard.bounds.as_slice()) {
                if size > 0
                    && ranges.iter().any(|range| {
                        range
                            .iter()
                            .zip(&shard.bounds)
                            .all(|(a, b)| a.start.max(b.start) < a.end.min(b.end))
                    })
                {
                    return Err(invalid("shards overlap without being exact replicas"));
                }
                covered = covered
                    .checked_add(size)
                    .ok_or_else(|| invalid("shard coverage overflow"))?;
                ranges.push(&shard.bounds);
            }
            match &shard.storage {
                ShardStorage::Host(host) => {
                    if !matches!(shard.location, MemoryLocation::Host { .. })
                        || host.info.dtype() != info.dtype()
                        || dimensions(&host.info)? != local_dims
                    {
                        return Err(invalid("packed host storage does not match its shard"));
                    }
                }
                ShardStorage::Local { view, .. } => {
                    if view.info.dtype() != info.dtype() || dimensions(&view.info)? != local_dims {
                        return Err(invalid("local memory view does not match its shard"));
                    }
                }
                ShardStorage::WholeArray if whole.is_none() => {
                    return Err(invalid("shard has no allocation reference"));
                }
                _ => {}
            }
        }
        if whole.is_none() && covered != total {
            return Err(invalid("shards do not cover the global array"));
        }
        Ok(Self(Arc::new(ArrayStorage {
            info,
            shards,
            whole,
        })))
    }

    pub fn info(&self) -> ArrayInfo {
        self.0.info.clone()
    }
    pub fn array_info(&self) -> &ArrayInfo {
        &self.0.info
    }
    pub fn shards(&self) -> &[Shard] {
        &self.0.shards
    }
    pub fn whole_allocation(&self) -> Option<&AllocationRef> {
        self.0.whole.as_ref()
    }

    /// Describe one shard as its own array without copying or contacting its
    /// owner. A whole-array reference alone cannot identify its shard storage.
    pub fn shard_data(&self, index: usize) -> io::Result<Self> {
        let shard = self
            .shards()
            .get(index)
            .ok_or_else(|| invalid("unknown shard index"))?;
        if matches!(shard.storage, ShardStorage::WholeArray) {
            return Err(unsupported(
                "the owner has not supplied a per-shard reference",
            ));
        }
        let info = ArrayInfo::new(
            crate::Shape::fixed(shard.bounds.iter().map(|r| r.len())),
            self.array_info().dtype(),
        );
        Self::single(info, shard.location.clone(), shard.storage.clone())
    }

    /// Returns a single full-size shard, not an arbitrary shard of a partition.
    pub fn single_shard(&self) -> Option<&Shard> {
        let [shard] = self.shards() else { return None };
        shard
            .bounds
            .iter()
            .zip(self.array_info().shape().dims())
            .all(|(r, dim)| r.start == 0 && *dim == Dim::Fixed(r.end))
            .then_some(shard)
    }

    pub fn single_allocation(&self) -> Option<&AllocationRef> {
        self.whole_allocation()
            .or_else(|| match &self.single_shard()?.storage {
                ShardStorage::Local { allocation, .. } => allocation.as_ref(),
                ShardStorage::Reference(allocation) => Some(allocation),
                _ => None,
            })
    }

    /// Access an existing packed host allocation or explicitly create a packed
    /// copy of an addressable CPU view. Never invokes a runtime or transport.
    pub fn to_host(&self) -> io::Result<HostBuffer> {
        self.host_buffer(TransferPolicy::AllowCopy)
    }

    pub fn into_host(self, policy: TransferPolicy) -> io::Result<HostBuffer> {
        self.host_buffer(policy)
    }

    fn host_buffer(&self, policy: TransferPolicy) -> io::Result<HostBuffer> {
        match self.single_shard().map(|s| &s.storage) {
            Some(ShardStorage::Host(host)) => Ok(host.clone()),
            Some(ShardStorage::Local { view, .. }) if view.host_readable => {
                let dims = dimensions(self.array_info())?;
                let width = scalar_width(self.array_info().dtype());
                let len = volume(&dims)?
                    .checked_mul(width)
                    .ok_or_else(|| invalid("packed byte count overflow"))?;
                if view.layout.byte_order == ByteOrder::Little
                    && is_packed(&dims, &view.layout.byte_strides, width)
                {
                    let address = if len == 0 {
                        NonNull::dangling()
                    } else {
                        // SAFETY: the view was checked and the lease keeps it alive.
                        unsafe { view.address.unwrap().add(view.layout.byte_offset) }
                    };
                    return Ok(HostBuffer {
                        info: self.info(),
                        bytes: HostBytes::Leased {
                            address,
                            len,
                            _lease: view._lease.clone(),
                        },
                    });
                }
                if policy == TransferPolicy::RequireZeroCopy {
                    return Err(unsupported("packing this host view requires a copy"));
                }
                let mut bytes = Vec::new();
                bytes.try_reserve_exact(len).map_err(io::Error::other)?;
                let lane = match self.array_info().dtype() {
                    DType::C64 | DType::C128 => width / 2,
                    _ => width,
                };
                for mut index in 0..(len / width) {
                    let mut offset = view.layout.byte_offset as i128;
                    for (&dim, &stride) in dims.iter().zip(&view.layout.byte_strides).rev() {
                        offset += (index % dim) as i128 * stride as i128;
                        index /= dim;
                    }
                    // SAFETY: validation bounds all reachable offsets; data is ready and immutable.
                    let scalar = unsafe {
                        std::slice::from_raw_parts(
                            view.address.unwrap().offset(offset as isize).as_ptr(),
                            width,
                        )
                    };
                    if view.layout.byte_order == ByteOrder::Little {
                        bytes.extend(scalar);
                    } else {
                        for chunk in scalar.chunks_exact(lane) {
                            bytes.extend(chunk.iter().rev());
                        }
                    }
                }
                HostBuffer::new(self.info(), bytes)
            }
            _ => Err(unsupported(
                "array has no directly accessible single host allocation",
            )),
        }
    }
}

impl Display for DataHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ArrayStorage({})", self.array_info())
    }
}

impl ConcreteValue for DataHandle {
    fn to_info(&self) -> VarInfo {
        VarInfo::new(self.info())
    }
}

/// Alignment for owned host snapshots that can be shared with CPU runtimes.
pub const HOST_BUFFER_ALIGNMENT: usize = 64;

#[derive(Debug)]
struct AlignedBytes {
    address: NonNull<u8>,
    layout: Layout,
    len: usize,
}

// SAFETY: the allocation is owned, and mutable access requires exclusive access.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    fn new(len: usize) -> io::Result<Self> {
        let layout = Layout::from_size_align(len.max(1), HOST_BUFFER_ALIGNMENT)
            .map_err(|_| invalid("host allocation size overflow"))?;
        // SAFETY: layout is nonzero and valid. The initialized allocation is owned here.
        let address = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or_else(|| {
            io::Error::new(ErrorKind::OutOfMemory, "aligned host allocation failed")
        })?;
        Ok(Self {
            address,
            layout,
            len,
        })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the allocation is initialized and stays live for this borrow.
        unsafe { std::slice::from_raw_parts(self.address.as_ptr(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: storage is initialized and this is the only mutable access, before sharing.
        unsafe { std::slice::from_raw_parts_mut(self.address.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // SAFETY: unique allocation, freed with its original layout.
        unsafe { dealloc(self.address.as_ptr(), self.layout) }
    }
}

#[derive(Clone, Debug)]
enum HostBytes {
    Owned(Arc<Vec<u8>>),
    Aligned(Arc<AlignedBytes>),
    Leased {
        address: NonNull<u8>,
        len: usize,
        _lease: Lease,
    },
}

// SAFETY: leased bytes are only constructed from validated immutable CPU views.
unsafe impl Send for HostBytes {}
unsafe impl Sync for HostBytes {}

/// A packed, little-endian host view. Cloning retains its allocation.
#[derive(Clone, Debug)]
pub struct HostBuffer {
    info: ArrayInfo,
    bytes: HostBytes,
}

impl HostBuffer {
    pub fn new(info: ArrayInfo, bytes: Vec<u8>) -> io::Result<Self> {
        if volume(&dimensions(&info)?)?.checked_mul(scalar_width(info.dtype())) != Some(bytes.len())
        {
            return Err(invalid(
                "packed byte count does not match fixed array metadata",
            ));
        }
        if info.dtype() == DType::Bool && bytes.iter().any(|byte| *byte > 1) {
            return Err(invalid("packed booleans must be 0 or 1"));
        }
        Ok(Self {
            info,
            bytes: HostBytes::Owned(Arc::new(bytes)),
        })
    }

    /// Make one snapshot in storage aligned for CPU runtime imports.
    pub fn copy_from_slice(info: ArrayInfo, source: &[u8]) -> io::Result<Self> {
        if source.len() != Self::packed_len(&info)? {
            return Err(invalid(
                "packed byte count does not match fixed array metadata",
            ));
        }
        Self::try_fill(info, |destination| {
            destination.copy_from_slice(source);
            Ok(())
        })
    }

    fn packed_len(info: &ArrayInfo) -> io::Result<usize> {
        volume(&dimensions(info)?)?
            .checked_mul(scalar_width(info.dtype()))
            .ok_or_else(|| invalid("packed byte count overflow"))
    }

    /// Fill aligned storage before publishing it as immutable. The destination
    /// is initialized to zero and may only be written during `fill`.
    pub fn try_fill<E: From<io::Error>>(
        info: ArrayInfo,
        fill: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E> {
        let mut bytes = AlignedBytes::new(Self::packed_len(&info)?)?;
        fill(bytes.bytes_mut())?;
        if info.dtype() == DType::Bool && bytes.bytes().iter().any(|byte| *byte > 1) {
            return Err(invalid("packed booleans must be 0 or 1").into());
        }
        Ok(Self {
            info,
            bytes: HostBytes::Aligned(Arc::new(bytes)),
        })
    }

    pub fn info(&self) -> &ArrayInfo {
        &self.info
    }

    pub fn bytes(&self) -> &[u8] {
        match &self.bytes {
            HostBytes::Owned(bytes) => bytes,
            HostBytes::Aligned(bytes) => bytes.bytes(),
            HostBytes::Leased { address, len, .. } => unsafe {
                std::slice::from_raw_parts(address.as_ptr(), *len)
            },
        }
    }

    /// Take a uniquely owned Vec without copying. Sharing a host view is a
    /// separate operation and never depends on unique ownership.
    pub fn try_into_bytes(self) -> Result<Vec<u8>, Self> {
        let Self { info, bytes } = self;
        match bytes {
            HostBytes::Owned(bytes) => Arc::try_unwrap(bytes).map_err(|bytes| Self {
                info,
                bytes: HostBytes::Owned(bytes),
            }),
            bytes => Err(Self { info, bytes }),
        }
    }

    /// Extract a Vec, copying unless the storage is a uniquely owned Vec.
    pub fn into_bytes(self) -> Vec<u8> {
        self.try_into_bytes()
            .unwrap_or_else(|host| host.bytes().to_vec())
    }
}

pub(crate) fn invalid(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}
pub(crate) fn unsupported(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::Unsupported, message)
}

pub(crate) fn dimensions(info: &ArrayInfo) -> io::Result<Vec<usize>> {
    info.shape()
        .dims()
        .iter()
        .map(|dim| match dim {
            Dim::Fixed(size) => Ok(*size),
            _ => Err(invalid("concrete arrays require fixed dimensions")),
        })
        .collect()
}

fn volume(dims: &[usize]) -> io::Result<usize> {
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter()
        .try_fold(1usize, |n, dim| n.checked_mul(*dim))
        .ok_or_else(|| invalid("array size overflow"))
}

pub(crate) fn scalar_width(dtype: DType) -> usize {
    match dtype {
        DType::Bool | DType::U8 | DType::I8 => 1,
        DType::U16 | DType::I16 | DType::F16 | DType::BF16 => 2,
        DType::F8E4M3FN | DType::F8E5M2 | DType::F8E8M0FNU => 1,
        DType::U32 | DType::I32 | DType::F32 => 4,
        DType::U64 | DType::I64 | DType::F64 | DType::C64 => 8,
        DType::C128 => 16,
    }
}

pub(crate) fn packed_strides(dims: &[usize], width: usize) -> io::Result<Vec<isize>> {
    let mut strides = vec![0; dims.len()];
    let mut stride = width;
    for (dim, slot) in dims.iter().zip(&mut strides).rev() {
        *slot = isize::try_from(stride).map_err(|_| invalid("stride overflow"))?;
        stride = stride
            .checked_mul(*dim)
            .ok_or_else(|| invalid("stride overflow"))?;
    }
    Ok(strides)
}

fn is_packed(dims: &[usize], strides: &[isize], width: usize) -> bool {
    dims.contains(&0)
        || packed_strides(dims, width).is_ok_and(|packed| {
            dims.iter()
                .zip(strides.iter().zip(packed))
                .all(|(&dim, (&actual, expected))| dim < 2 || actual == expected)
        })
}

fn validate_view(view: &MemoryView, dims: &[usize], dtype: DType) -> io::Result<()> {
    let StridedLayout {
        byte_offset,
        byte_strides,
        ..
    } = &view.layout;
    if byte_strides.len() != dims.len() {
        return Err(invalid("stride rank does not match shard rank"));
    }
    if dims.contains(&0) {
        return Ok(());
    }
    if view.address.is_none() {
        return Err(invalid("nonempty shard has no address"));
    }
    let mut low = *byte_offset as i128;
    let mut high = low;
    for (&dim, &stride) in dims.iter().zip(byte_strides) {
        let reach = ((dim - 1) as i128) * stride as i128;
        low = low
            .checked_add(reach.min(0))
            .ok_or_else(|| invalid("stride overflow"))?;
        high = high
            .checked_add(reach.max(0))
            .ok_or_else(|| invalid("stride overflow"))?;
    }
    let high = high
        .checked_add(scalar_width(dtype) as i128)
        .ok_or_else(|| invalid("stride overflow"))?;
    if low < isize::MIN as i128
        || high > isize::MAX as i128
        || view.extent.is_some_and(|len| low < 0 || high > len as i128)
    {
        return Err(invalid("strided view exceeds allocation bounds"));
    }
    Ok(())
}
