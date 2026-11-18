use std::any::Any;
use std::ffi::c_void;
use std::fmt::{Debug, Display, Formatter};
use std::io::{self, ErrorKind};
use std::ptr::NonNull;
use std::sync::Arc;

use gnx_expr::{ConcreteValue, ValueInfo};

use crate::array::{ArrayInfo, DType, Dim};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TransferPolicy {
    #[default]
    RequireZeroCopy,
    AllowCopy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryLocation {
    Host,
    /// A runtime-scoped identifier, not a cross-runtime hardware identity.
    Device {
        runtime: &'static str,
        device_id: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BufferLayout {
    /// Row-major with little-endian scalars and interleaved complex components.
    Packed,
    /// Native strides and offset measured in bytes.
    Strided {
        strides: Vec<isize>,
        offset: usize,
    },
    Opaque,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferInfo {
    pub location: MemoryLocation,
    pub memory_kind: Option<String>,
    pub layout: BufferLayout,
}

/// Concrete storage. An owned handle does not imply exclusive allocation access.
pub trait DataImpl: Any + Send + Sync + Debug + Display {
    fn info(&self) -> ArrayInfo;
    fn buffer_info(&self) -> BufferInfo;

    /// Materialize a packed host copy; may wait for outstanding writes.
    fn to_host(&self) -> io::Result<HostBuffer> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "no host export for this buffer",
        ))
    }

    fn dlpack_device(&self) -> Option<DlpackDevice> {
        None
    }

    /// Export an immutable, synchronized lease without copying.
    fn export_dlpack(&self, _request: DlpackRequest) -> io::Result<DlpackTensor> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "no zero-copy DLPack export for this buffer",
        ))
    }
}

#[derive(Clone, Debug)]
pub struct DataHandle(Arc<dyn DataImpl>);

impl DataHandle {
    pub fn new(storage: impl DataImpl) -> Self {
        Self(Arc::new(storage))
    }

    pub fn info(&self) -> ArrayInfo {
        self.0.info()
    }

    pub fn buffer_info(&self) -> BufferInfo {
        self.0.buffer_info()
    }

    pub fn downcast_ref<T: DataImpl>(&self) -> Option<&T> {
        (self.0.as_ref() as &dyn Any).downcast_ref()
    }

    pub fn to_host(&self) -> io::Result<HostBuffer> {
        self.0.to_host()
    }

    pub fn into_host(self, policy: TransferPolicy) -> io::Result<HostBuffer> {
        if self.downcast_ref::<HostBuffer>().is_some() {
            let storage: Arc<dyn Any + Send + Sync> = self.0;
            let storage = storage
                .downcast::<HostBuffer>()
                .expect("checked buffer type");
            return match Arc::try_unwrap(storage) {
                Ok(buffer) => Ok(buffer),
                Err(shared) if policy == TransferPolicy::AllowCopy => Ok((*shared).clone()),
                Err(_) => Err(io::Error::new(
                    ErrorKind::Unsupported,
                    "shared host storage cannot be taken without a copy",
                )),
            };
        }
        if policy == TransferPolicy::RequireZeroCopy {
            return Err(io::Error::new(
                ErrorKind::Unsupported,
                "host transfer would require a copy",
            ));
        }
        self.to_host()
    }

    /// Hand this storage to a destination that imports one DLPack lease.
    /// If unsupported, return the original storage for possible copy fallback.
    pub fn into_dlpack(self) -> Result<DlpackSource, Self> {
        match self.0.dlpack_device() {
            Some(device) => Ok(DlpackSource {
                source: self,
                device,
            }),
            None => Err(self),
        }
    }
}

impl Display for DataHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl ConcreteValue for DataHandle {
    fn to_info(&self) -> ValueInfo {
        ValueInfo::new(self.info())
    }
}

/// An owned packed host buffer; all scalar widths match their declared dtype.
#[derive(Clone, Debug)]
pub struct HostBuffer {
    info: ArrayInfo,
    bytes: Vec<u8>,
}

impl HostBuffer {
    pub fn new(info: ArrayInfo, bytes: Vec<u8>) -> io::Result<Self> {
        let count = info
            .shape()
            .dims()
            .iter()
            .try_fold(1usize, |n, dim| match dim {
                Dim::Fixed(size) => n.checked_mul(*size),
                _ => None,
            });
        let width = match info.dtype() {
            DType::Bool | DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 | DType::F16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 | DType::C64 => 8,
            DType::C128 => 16,
        };
        if count.and_then(|n| n.checked_mul(width)) != Some(bytes.len()) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "packed byte count does not match fixed array metadata",
            ));
        }
        if info.dtype() == DType::Bool && bytes.iter().any(|byte| *byte > 1) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "packed booleans must be 0 or 1",
            ));
        }
        Ok(Self { info, bytes })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl Display for HostBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HostBuffer({})", self.info)
    }
}

impl DataImpl for HostBuffer {
    fn info(&self) -> ArrayInfo {
        self.info.clone()
    }

    fn buffer_info(&self) -> BufferInfo {
        BufferInfo {
            location: MemoryLocation::Host,
            memory_kind: None,
            layout: BufferLayout::Packed,
        }
    }

    fn to_host(&self) -> io::Result<Self> {
        Ok(self.clone())
    }
}

/// The standard DLPack device type and runtime-local ordinal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DlpackDevice(pub i32, pub i32);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DlpackRequest {
    pub stream: Option<isize>,
    pub max_version: Option<(u32, u32)>,
}

pub struct DlpackSource {
    source: DataHandle,
    device: DlpackDevice,
}

impl DlpackSource {
    pub fn device(&self) -> DlpackDevice {
        self.device
    }

    pub fn export(self, request: DlpackRequest) -> io::Result<DlpackTensor> {
        self.source.0.export_dlpack(request)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DlpackAbi {
    Legacy,
    Versioned,
}

/// Owns one DLPack managed-tensor lease, not necessarily its allocation.
pub struct DlpackTensor {
    ptr: NonNull<c_void>,
    abi: DlpackAbi,
    release: unsafe fn(NonNull<c_void>),
}

// SAFETY: the uniquely owned lease may be released on any thread, as required
// by `from_raw`. It exposes no safe memory access or shared mutation.
unsafe impl Send for DlpackTensor {}

impl DlpackTensor {
    /// # Safety
    /// `ptr` must be a valid managed tensor of the indicated ABI. This token
    /// owns its lease, with no other party scheduled to release it. The tensor
    /// must remain valid and immutable until released, obey the negotiated
    /// stream, and allow `release` (equivalent to the tensor's embedded
    /// deleter) to be called once on any thread.
    pub unsafe fn from_raw(
        ptr: NonNull<c_void>,
        abi: DlpackAbi,
        release: unsafe fn(NonNull<c_void>),
    ) -> Self {
        Self { ptr, abi, release }
    }

    pub fn abi(&self) -> DlpackAbi {
        self.abi
    }

    pub fn as_ptr(&self) -> NonNull<c_void> {
        self.ptr
    }

    /// Transfer release responsibility to a compatible DLPack consumer.
    pub fn into_raw(self) -> NonNull<c_void> {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for DlpackTensor {
    fn drop(&mut self) {
        // SAFETY: the constructor binds this lease to exactly one release.
        unsafe { (self.release)(self.ptr) }
    }
}
