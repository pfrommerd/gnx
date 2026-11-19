//! DLPack translates one addressable shard, not a distributed allocation.

use std::{
    ffi::c_void,
    io,
    ptr::{self, NonNull},
};

use super::*;
use crate::Shape;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DlpackDevice(pub i32, pub i32);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DlpackRequest {
    pub stream: Option<isize>,
    pub max_version: Option<(u32, u32)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DlpackAbi {
    Legacy,
    Versioned,
}

/// Owns one managed-tensor lease, not necessarily its allocation.
pub struct DlpackTensor {
    ptr: NonNull<c_void>,
    abi: DlpackAbi,
    release: unsafe fn(NonNull<c_void>),
}

// SAFETY: `from_raw` requires that the lease may be released on any thread.
unsafe impl Send for DlpackTensor {}

#[repr(C)]
struct Tensor {
    data: *mut c_void,
    device: DlpackDevice,
    ndim: i32,
    code: u8,
    bits: u8,
    lanes: u16,
    shape: *const i64,
    strides: *const i64,
    offset: u64,
}

#[repr(C)]
struct Legacy {
    tensor: Tensor,
    context: *mut c_void,
    release: Option<unsafe extern "C" fn(*mut Legacy)>,
}

#[repr(C)]
struct Versioned {
    major: u32,
    minor: u32,
    context: *mut c_void,
    release: Option<unsafe extern "C" fn(*mut Versioned)>,
    flags: u64,
    tensor: Tensor,
}

const READ_ONLY: u64 = 1;
const IS_COPIED: u64 = 2;

impl DlpackTensor {
    /// # Safety
    /// `ptr` must be a valid tensor of the given ABI and `release` must be its
    /// once-only release, callable on any thread. The token is now its owner.
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

    /// Transfer release responsibility to the ABI consumer.
    pub fn into_raw(self) -> NonNull<c_void> {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }

    /// Turn an already synchronized DLPack lease into a shared tensor view.
    ///
    /// # Safety
    /// The producer must uphold the DLPack ABI, including its pointer/device
    /// claims. All writes must have finished and no alias may mutate or resize
    /// the allocation during the lease. A read-only flag alone does not prove
    /// that the producer has no writable aliases.
    pub unsafe fn into_data(
        self,
        location: MemoryLocation,
        allocation: Option<AllocationRef>,
    ) -> io::Result<DataHandle> {
        // SAFETY: guaranteed by the constructor and the caller.
        let t = unsafe {
            match self.abi {
                DlpackAbi::Legacy => &self.ptr.cast::<Legacy>().as_ref().tensor,
                DlpackAbi::Versioned => {
                    let t = self.ptr.cast::<Versioned>().as_ref();
                    if t.major != 1 || t.flags & IS_COPIED != 0 {
                        return Err(unsupported("unsupported or copied DLPack lease"));
                    }
                    &t.tensor
                }
            }
        };
        let host_readable = matches!(t.device.0, 1 | 3 | 11); // CPU, CUDAHost, ROCmHost
        if host_readable != matches!(location, MemoryLocation::Host { .. })
            && !matches!(
                location,
                MemoryLocation::Managed { .. } | MemoryLocation::Unknown
            )
        {
            return Err(invalid(
                "DLPack device disagrees with declared memory location",
            ));
        }
        let rank = usize::try_from(t.ndim).map_err(|_| invalid("negative DLPack rank"))?;
        if rank > (isize::MAX as usize) / size_of::<i64>() {
            return Err(invalid("DLPack rank overflow"));
        }
        let dtype = dtype(t.code, t.bits, t.lanes)?;
        let width = scalar_width(dtype);
        let dims = if rank == 0 {
            vec![]
        } else {
            if t.shape.is_null() {
                return Err(invalid("DLPack tensor has no shape"));
            }
            // SAFETY: DLPack supplies a rank-sized shape array.
            unsafe { std::slice::from_raw_parts(t.shape, rank) }
                .iter()
                .map(|&d| usize::try_from(d).map_err(|_| invalid("invalid DLPack dimension")))
                .collect::<io::Result<Vec<_>>>()?
        };
        let strides = if rank == 0 || t.strides.is_null() {
            packed_strides(&dims, width)?
        } else {
            // SAFETY: DLPack supplies a rank-sized stride array when non-null.
            unsafe { std::slice::from_raw_parts(t.strides, rank) }
                .iter()
                .map(|&s| {
                    isize::try_from(s)
                        .ok()
                        .and_then(|s| s.checked_mul(width as isize))
                        .ok_or_else(|| invalid("DLPack stride overflow"))
                })
                .collect::<io::Result<Vec<_>>>()?
        };
        let address = t.data.cast();
        let device = t.device;
        let offset = usize::try_from(t.offset).map_err(|_| invalid("DLPack offset overflow"))?;
        let layout = StridedLayout {
            byte_offset: offset,
            byte_strides: strides,
            byte_order: ByteOrder::NATIVE,
        };
        // SAFETY: the caller attests readiness, immutability and the validity of the affine view.
        let info = ArrayInfo::new(Shape::fixed(dims), dtype);
        let view = unsafe {
            MemoryView::new(
                info.clone(),
                address,
                None,
                layout,
                host_readable,
                Some(device),
                Lease::on_drop(move || drop(self)),
            )
        }?;
        DataHandle::single(info, location, ShardStorage::Local { view, allocation })
    }
}

impl Drop for DlpackTensor {
    fn drop(&mut self) {
        // SAFETY: the constructor binds the lease to exactly one release.
        unsafe { (self.release)(self.ptr) }
    }
}

pub struct DlpackSource {
    source: DataHandle,
    device: DlpackDevice,
}

impl DataHandle {
    pub fn into_dlpack(self) -> Result<DlpackSource, Self> {
        match self.single_shard().and_then(|s| match &s.storage {
            ShardStorage::Host(_) if ByteOrder::NATIVE == ByteOrder::Little => {
                Some(DlpackDevice(1, 0))
            }
            ShardStorage::Local { view, .. } if view.layout.byte_order == ByteOrder::NATIVE => {
                view.dlpack_device
            }
            _ => None,
        }) {
            Some(device) => Ok(DlpackSource {
                source: self,
                device,
            }),
            None => Err(self),
        }
    }
}

impl DlpackSource {
    pub fn device(&self) -> DlpackDevice {
        self.device
    }

    pub fn export(self, request: DlpackRequest) -> io::Result<DlpackTensor> {
        if request.max_version.is_none_or(|(major, _)| major < 1) {
            return Err(unsupported("read-only DLPack export requires version 1"));
        }
        self.export_abi(DlpackAbi::Versioned)
    }

    /// # Safety
    /// A legacy consumer has no read-only flag; the caller must ensure the
    /// consumer does not mutate the allocation for the life of its lease.
    pub unsafe fn export_legacy(self) -> io::Result<DlpackTensor> {
        self.export_abi(DlpackAbi::Legacy)
    }

    pub fn address(&self) -> Option<usize> {
        match &self.source.single_shard()?.storage {
            ShardStorage::Host(host) => Some(host.bytes().as_ptr().addr()),
            ShardStorage::Local { view, .. } => view
                .address?
                .as_ptr()
                .addr()
                .checked_add(view.layout.byte_offset),
            _ => None,
        }
    }

    fn export_abi(self, abi: DlpackAbi) -> io::Result<DlpackTensor> {
        let dims = dimensions(self.source.array_info())?;
        let width = scalar_width(self.source.array_info().dtype());
        let (code, bits) = encoding(self.source.array_info().dtype());
        let ndim = i32::try_from(dims.len()).map_err(|_| invalid("DLPack rank overflow"))?;
        let (data, offset, strides) = match &self.source.single_shard().unwrap().storage {
            ShardStorage::Host(host) => (
                host.bytes().as_ptr().cast_mut().cast(),
                0,
                packed_strides(&dims, width)?,
            ),
            ShardStorage::Local { view, .. } => (
                view.address.map_or(ptr::null_mut(), |p| p.as_ptr().cast()),
                view.layout.byte_offset,
                view.layout.byte_strides.clone(),
            ),
            _ => unreachable!("checked when constructing source"),
        };
        let shape = dims
            .into_iter()
            .map(|d| i64::try_from(d).map_err(|_| invalid("DLPack dimension overflow")))
            .collect::<io::Result<Vec<_>>>()?;
        let strides = strides
            .into_iter()
            .map(|s| {
                if s % width as isize != 0 {
                    return Err(unsupported("DLPack strides must be in whole elements"));
                }
                i64::try_from(s / width as isize).map_err(|_| invalid("DLPack stride overflow"))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let tensor = Tensor {
            data,
            device: self.device,
            ndim,
            code,
            bits,
            lanes: 1,
            shape: shape.as_ptr(),
            strides: strides.as_ptr(),
            offset: offset as u64,
        };
        let context = Box::into_raw(Box::new((self, shape, strides))).cast();
        let (ptr, release): (NonNull<c_void>, unsafe fn(NonNull<c_void>)) = match abi {
            DlpackAbi::Legacy => (
                NonNull::from(Box::leak(Box::new(Legacy {
                    tensor,
                    context,
                    release: Some(drop_legacy),
                })))
                .cast(),
                |p| unsafe { drop_legacy(p.cast().as_ptr()) },
            ),
            DlpackAbi::Versioned => (
                NonNull::from(Box::leak(Box::new(Versioned {
                    major: 1,
                    minor: 0,
                    context,
                    release: Some(drop_versioned),
                    flags: READ_ONLY,
                    tensor,
                })))
                .cast(),
                |p| unsafe { drop_versioned(p.cast().as_ptr()) },
            ),
        };
        // SAFETY: both allocation and release are defined above and ready on every thread.
        Ok(unsafe { DlpackTensor::from_raw(ptr, abi, release) })
    }
}

unsafe extern "C" fn drop_versioned(p: *mut Versioned) {
    // SAFETY: once-only DLPack release of the box and its owning context.
    unsafe {
        let value = Box::from_raw(p);
        drop(Box::from_raw(value.context.cast::<(
            DlpackSource,
            Vec<i64>,
            Vec<i64>,
        )>()));
    }
}

unsafe extern "C" fn drop_legacy(p: *mut Legacy) {
    // SAFETY: once-only DLPack release of the box and its owning context.
    unsafe {
        let value = Box::from_raw(p);
        drop(Box::from_raw(value.context.cast::<(
            DlpackSource,
            Vec<i64>,
            Vec<i64>,
        )>()));
    }
}

fn encoding(dtype: DType) -> (u8, u8) {
    let code = match dtype {
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => 0,
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => 1,
        DType::F16 | DType::F32 | DType::F64 => 2,
        DType::C64 | DType::C128 => 5,
        DType::Bool => 6,
        DType::BF16 => 4,
        DType::F8E4M3FN => 10,
        DType::F8E5M2 => 12,
        DType::F8E8M0FNU => 14,
    };
    (code, (scalar_width(dtype) * 8) as u8)
}

fn dtype(code: u8, bits: u8, lanes: u16) -> io::Result<DType> {
    [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::BF16,
        DType::F8E4M3FN,
        DType::F8E5M2,
        DType::F8E8M0FNU,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::C64,
        DType::C128,
    ]
    .into_iter()
    .find(|&d| lanes == 1 && encoding(d) == (code, bits))
    .ok_or_else(|| unsupported("unsupported DLPack dtype"))
}
