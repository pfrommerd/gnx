//! Ownership wrappers for the CPU PJRT bridge. XLA computation construction
//! still uses xla-rs; no dependency-private Rust pointers cross this boundary.

use std::ffi::{c_char, c_int, c_void};
use std::io::{self, ErrorKind};
use std::ptr::{self, NonNull};
use std::sync::Arc;

use gnx_math::{ArrayInfo, HostBuffer, TransferPolicy};

use super::lower::{dimensions, element};

pub(crate) struct Client(NonNull<c_void>);

pub(crate) struct Buffer {
    raw: NonNull<c_void>,
    _client: Arc<Client>,
}

pub(crate) struct Executable {
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
        output(|out, report| unsafe { gnx_cpu_new(out, report) }).map(|raw| Arc::new(Self(raw)))
    }

    pub fn device_info(&self) -> io::Result<(u64, String)> {
        let mut ordinal = 0;
        let mut name = String::new();
        // SAFETY: the CPU client owns one addressable device. The callback is synchronous.
        unsafe { gnx_cpu_info(self.0.as_ptr(), &mut ordinal, report_to(&mut name)) };
        let ordinal =
            u64::try_from(ordinal).map_err(|_| io::Error::other("invalid CPU ordinal"))?;
        Ok((ordinal, name))
    }

    pub fn compile(self: &Arc<Self>, computation: &[u8]) -> io::Result<Executable> {
        let size = c_int::try_from(computation.len())
            .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "HLO module is too large"))?;
        let raw = output(|out, report| unsafe {
            gnx_cpu_compile(
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

    pub fn import(
        self: &Arc<Self>,
        host: HostBuffer,
        policy: TransferPolicy,
    ) -> io::Result<Buffer> {
        let dims = dimensions(host.info())?;
        let primitive = element(host.info().dtype()).primitive_type() as c_int;
        let data = host.bytes().as_ptr().cast();
        let lease = Box::into_raw(Box::new(host)).cast();
        // SAFETY: the host allocation is immutable and the bridge always adopts
        // its lease, including on error. The shape describes the entire buffer.
        let raw = output(|out, report| unsafe {
            gnx_cpu_import(
                self.0.as_ptr(),
                data,
                primitive,
                dims.as_ptr(),
                dims.len(),
                policy == TransferPolicy::RequireZeroCopy,
                lease,
                release_host,
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
    pub fn check(&self, info: &ArrayInfo) -> io::Result<()> {
        let dims = dimensions(info)?;
        let ty = element(info.dtype()).primitive_type() as c_int;
        // SAFETY: live native buffer and a rank-sized slice.
        if !unsafe { gnx_cpu_matches(self.raw.as_ptr(), ty, dims.as_ptr(), dims.len()) } {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "XLA returned mismatched array metadata",
            ));
        }
        Ok(())
    }

    pub fn download(&self, info: ArrayInfo) -> io::Result<HostBuffer> {
        self.check(&info)?;
        HostBuffer::try_fill(info, |bytes| {
            // SAFETY: the destination holds the entire checked buffer in packed form.
            checked(|report| unsafe {
                gnx_cpu_download(self.raw.as_ptr(), bytes.as_mut_ptr().cast(), report)
            })
        })
    }

    #[cfg(test)]
    pub fn address(&self) -> io::Result<*const u8> {
        let mut data = ptr::null();
        checked(|report| unsafe { gnx_cpu_address(self.raw.as_ptr(), &mut data, report) })?;
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
            gnx_cpu_execute(
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

unsafe extern "C" fn release_host(lease: *mut c_void) {
    // SAFETY: this is the unique release of the Box transferred to PJRT.
    unsafe { drop(Box::from_raw(lease.cast::<HostBuffer>())) };
}

impl Drop for Client {
    fn drop(&mut self) {
        unsafe { gnx_cpu_free_client(self.0.as_ptr()) }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { gnx_cpu_free_buffer(self.raw.as_ptr()) }
    }
}
impl Drop for Executable {
    fn drop(&mut self) {
        unsafe { gnx_cpu_free_executable(self.raw.as_ptr()) }
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
    fn gnx_cpu_new(out: *mut *mut c_void, report: Report) -> c_int;
    fn gnx_cpu_info(client: *mut c_void, ordinal: *mut c_int, report: Report);
    fn gnx_cpu_compile(
        client: *mut c_void,
        data: *const c_void,
        size: c_int,
        out: *mut *mut c_void,
        report: Report,
    ) -> c_int;
    fn gnx_cpu_import(
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
    fn gnx_cpu_matches(buffer: *mut c_void, dtype: c_int, dims: *const i64, rank: usize) -> bool;
    fn gnx_cpu_execute(
        executable: *mut c_void,
        inputs: *const *mut c_void,
        ninputs: usize,
        outputs: *mut *mut c_void,
        noutputs: usize,
        report: Report,
    ) -> c_int;
    fn gnx_cpu_download(buffer: *mut c_void, destination: *mut c_char, report: Report) -> c_int;
    #[cfg(test)]
    fn gnx_cpu_address(buffer: *mut c_void, out: *mut *const c_void, report: Report) -> c_int;
    fn gnx_cpu_free_client(client: *mut c_void);
    fn gnx_cpu_free_buffer(buffer: *mut c_void);
    fn gnx_cpu_free_executable(executable: *mut c_void);
}
