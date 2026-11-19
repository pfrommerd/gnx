//! Single-device CPU or CUDA PJRT execution with leased, immutable host buffers.

use std::ffi::{c_char, c_int, c_void};
use std::io::{self, ErrorKind};
use std::ptr::{self, NonNull};
use std::sync::Arc;

use crate::ElementType;

pub struct Client(NonNull<c_void>);

pub struct Buffer {
    raw: NonNull<c_void>,
    _client: Arc<Client>,
}

pub struct Executable {
    raw: NonNull<c_void>,
    client: Arc<Client>,
}

// SAFETY: PJRT clients, buffers and executables support concurrent use. These
// wrappers own each handle once, retain the client, and never donate buffers.
unsafe impl Send for Client {}
unsafe impl Sync for Client {}
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}
unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Client {
    pub fn new() -> io::Result<Arc<Self>> {
        output(|out, report| unsafe { gnx_pjrt_new(out, report) }).map(|raw| Arc::new(Self(raw)))
    }

    /// Create a client restricted to one CUDA-visible device, without preallocating VRAM.
    pub fn cuda(ordinal: usize) -> io::Result<Arc<Self>> {
        let ordinal = c_int::try_from(ordinal)
            .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "CUDA ordinal out of range"))?;
        output(|out, report| unsafe { gnx_pjrt_cuda_new(ordinal, out, report) })
            .map(|raw| Arc::new(Self(raw)))
    }

    pub fn device_info(&self) -> io::Result<(u64, String)> {
        let mut ordinal = 0;
        let mut name = String::new();
        // SAFETY: the client owns one addressable device. The callback is synchronous.
        unsafe { gnx_pjrt_info(self.0.as_ptr(), &mut ordinal, report_to(&mut name)) };
        let ordinal =
            u64::try_from(ordinal).map_err(|_| io::Error::other("invalid device ordinal"))?;
        Ok((ordinal, name))
    }

    pub fn compile(self: &Arc<Self>, computation: &[u8]) -> io::Result<Executable> {
        let size = c_int::try_from(computation.len())
            .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "HLO module is too large"))?;
        let raw = output(|out, report| unsafe {
            gnx_pjrt_compile(
                self.0.as_ptr(),
                computation.as_ptr().cast(),
                size,
                out,
                report,
            )
        })?;
        Ok(Executable {
            raw,
            client: self.clone(),
        })
    }

    /// Import contiguous, row-major bytes. PJRT retains the boxed owner until
    /// it releases the data, including when it aliases the caller's storage.
    pub fn import<T: Send + Sync + 'static>(
        self: &Arc<Self>,
        host: T,
        bytes: fn(&T) -> &[u8],
        dtype: ElementType,
        dims: &[i64],
        require_alias: bool,
    ) -> io::Result<Buffer> {
        let host = Box::new(host);
        let bytes = bytes(&host);
        let size = dims
            .iter()
            .try_fold(dtype.element_size_in_bytes(), |n, &d| {
                n.checked_mul(usize::try_from(d).ok()?)
            });
        if dtype == ElementType::F4E2M1FN || size != Some(bytes.len()) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "expected whole-byte elements matching host shape",
            ));
        }
        let primitive = dtype.primitive_type() as c_int;
        let data = bytes.as_ptr().cast();
        let lease = Box::into_raw(host).cast();
        // SAFETY: the host allocation is immutable and the bridge always adopts
        // its lease, including on error. The shape describes the entire buffer.
        let raw = output(|out, report| unsafe {
            gnx_pjrt_import(
                self.0.as_ptr(),
                data,
                primitive,
                dims.as_ptr(),
                dims.len(),
                require_alias,
                lease,
                release_host::<T>,
                out,
                report,
            )
        })?;
        Ok(Buffer {
            raw,
            _client: self.clone(),
        })
    }
}

impl Buffer {
    pub fn check(&self, dtype: ElementType, dims: &[i64]) -> io::Result<()> {
        let ty = dtype.primitive_type() as c_int;
        // SAFETY: live native buffer and a rank-sized slice.
        if !unsafe { gnx_pjrt_matches(self.raw.as_ptr(), ty, dims.as_ptr(), dims.len()) } {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "XLA returned mismatched array metadata",
            ));
        }
        Ok(())
    }

    pub fn download_into(&self, bytes: &mut [u8]) -> io::Result<()> {
        // The bridge checks the packed size before writing to the destination.
        checked(|report| unsafe {
            gnx_pjrt_download(
                self.raw.as_ptr(),
                bytes.as_mut_ptr().cast(),
                bytes.len(),
                report,
            )
        })
    }

    /// Address for alias comparisons, not a host dereference capability.
    pub fn address(&self) -> io::Result<*const u8> {
        let mut data = ptr::null();
        checked(|report| unsafe { gnx_pjrt_address(self.raw.as_ptr(), &mut data, report) })?;
        // Only used for comparing addresses, never for dereferencing the result.
        Ok(data.cast())
    }
}

impl Executable {
    pub fn run(&self, inputs: &[Arc<Buffer>], count: usize) -> io::Result<Vec<Buffer>> {
        if inputs
            .iter()
            .any(|buffer| !Arc::ptr_eq(&self.client, &buffer._client))
        {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "PJRT input belongs to a different client",
            ));
        }
        let input: Vec<_> = inputs.iter().map(|buffer| buffer.raw.as_ptr()).collect();
        let mut result = vec![ptr::null_mut(); count];
        // SAFETY: all inputs belong to this executable's client and are kept
        // alive during dispatch. The bridge fills outputs only on success.
        checked(|report| unsafe {
            gnx_pjrt_execute(
                self.raw.as_ptr(),
                input.as_ptr(),
                input.len(),
                result.as_mut_ptr(),
                count,
                report,
            )
        })?;
        Ok(result
            .into_iter()
            .map(|raw| Buffer {
                raw: NonNull::new(raw).expect("successful PJRT execution returned a null buffer"),
                _client: self.client.clone(),
            })
            .collect())
    }
}

unsafe extern "C" fn release_host<T>(lease: *mut c_void) {
    // SAFETY: this is the unique release of the Box transferred to PJRT.
    unsafe { drop(Box::from_raw(lease.cast::<T>())) };
}

impl Drop for Client {
    fn drop(&mut self) {
        unsafe { gnx_pjrt_free_client(self.0.as_ptr()) }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { gnx_pjrt_free_buffer(self.raw.as_ptr()) }
    }
}
impl Drop for Executable {
    fn drop(&mut self) {
        unsafe { gnx_pjrt_free_executable(self.raw.as_ptr()) }
    }
}

#[repr(C)]
struct Report {
    context: *mut c_void,
    write: unsafe extern "C" fn(*mut c_void, *const c_char, usize),
}

fn report_to(message: &mut String) -> Report {
    unsafe extern "C" fn write(context: *mut c_void, bytes: *const c_char, size: usize) {
        if size == 0 {
            return;
        }
        // SAFETY: both pointers are borrowed for this synchronous call.
        let bytes = unsafe { std::slice::from_raw_parts(bytes.cast(), size) };
        unsafe { (*context.cast::<String>()).push_str(&String::from_utf8_lossy(bytes)) };
    }
    Report {
        context: (message as *mut String).cast(),
        write,
    }
}

fn checked(f: impl FnOnce(Report) -> c_int) -> io::Result<()> {
    let mut message = String::new();
    let code = f(report_to(&mut message));
    if code == 0 {
        return Ok(());
    }
    let kind = match code {
        3 => ErrorKind::InvalidInput,
        12 => ErrorKind::Unsupported,
        _ => ErrorKind::Other,
    };
    Err(io::Error::new(kind, message))
}

fn output(f: impl FnOnce(*mut *mut c_void, Report) -> c_int) -> io::Result<NonNull<c_void>> {
    let mut output = ptr::null_mut();
    checked(|report| f(&mut output, report))?;
    NonNull::new(output)
        .ok_or_else(|| io::Error::other("successful PJRT call returned a null handle"))
}

unsafe extern "C" {
    fn gnx_pjrt_new(out: *mut *mut c_void, report: Report) -> c_int;
    fn gnx_pjrt_cuda_new(ordinal: c_int, out: *mut *mut c_void, report: Report) -> c_int;
    fn gnx_pjrt_info(client: *mut c_void, ordinal: *mut c_int, report: Report);
    fn gnx_pjrt_compile(
        client: *mut c_void,
        data: *const c_void,
        size: c_int,
        out: *mut *mut c_void,
        report: Report,
    ) -> c_int;
    fn gnx_pjrt_import(
        client: *mut c_void,
        data: *const c_void,
        dtype: c_int,
        dims: *const i64,
        rank: usize,
        require_alias: bool,
        lease: *mut c_void,
        release: unsafe extern "C" fn(*mut c_void),
        out: *mut *mut c_void,
        report: Report,
    ) -> c_int;
    fn gnx_pjrt_matches(buffer: *mut c_void, dtype: c_int, dims: *const i64, rank: usize) -> bool;
    fn gnx_pjrt_execute(
        executable: *mut c_void,
        inputs: *const *mut c_void,
        ninputs: usize,
        outputs: *mut *mut c_void,
        noutputs: usize,
        report: Report,
    ) -> c_int;
    fn gnx_pjrt_download(
        buffer: *mut c_void,
        destination: *mut c_char,
        size: usize,
        report: Report,
    ) -> c_int;
    fn gnx_pjrt_address(buffer: *mut c_void, out: *mut *const c_void, report: Report) -> c_int;
    fn gnx_pjrt_free_client(client: *mut c_void);
    fn gnx_pjrt_free_buffer(buffer: *mut c_void);
    fn gnx_pjrt_free_executable(executable: *mut c_void);
}
