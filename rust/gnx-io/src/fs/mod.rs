pub use std::io::{Error, ErrorKind};
use std::io::Write as StdWrite;
use std::io::Read as StdRead;

use std::io::Result;

mod path;
mod buffer;

pub use path::*;
pub use buffer::*;

pub mod local;

pub trait Read {
    fn read_buf<'s, B: BorrowedBuf<'s>>(&mut self, cursor: &mut Cursor<B>) -> Result<()>;
    // Try and put back some number of bytes.
    fn try_put_back(&mut self, buf: &[u8]) -> Result<()>; 

    fn read<T: ReadTarget>(&mut self, target: T) -> Result<T::Output> {
        let mut sink = target.into_sink();
        while !sink.is_full() {
            let mut cursor = sink.cursor();
            let prev_filled = cursor.filled_len();
            self.read_buf(&mut cursor)?;
            let delta = cursor.filled_len() - prev_filled;
            // Commit any bytes read into the sink.
            cursor.commit()?;
            // EOF if no bytes were read.
            if delta == 0 { break }
        }
        // If the sink is full, check if any bytes were rejected
        // and try to put them back.
        if sink.is_full() && let Some(rejected) = sink.rejected() {
            self.try_put_back(rejected)?;
        }
        sink.finish()
    }

    fn as_std<'s>(&'s mut self) -> AsStdReader<'s, Self> { AsStdReader(self) }
}

impl<R: Read + ?Sized> Read for &mut R {
    fn read<T: ReadTarget>(&mut self, target: T) -> Result<T::Output> { R::read(self, target) }
}

pub trait ReadTarget: Sized {
    type Output;
    type Sink: DataSink<Output = Self::Output>;

    // Called to hint the size that we expect to be read.
    fn size_hint(&mut self, lower: usize, upper: Option<usize>) -> Result<()>;
    fn into_sink(self) -> Self::Sink;

    fn limit(self, max: usize) -> Limited<Self> {
        Limited(self, max)
    }
    fn count(self) -> Counted<Self> {
        Counted(self)
    }
}

pub struct Counted<T: ReadTarget>(T);
pub struct Limited<T: ReadTarget>(T, usize);

pub trait Write {
    fn write(&mut self, buf: &[u8]) -> Result<usize>;
    fn flush(&mut self) -> Result<()>;

    fn as_std<'s>(&'s mut self) -> AsStdWriter<'s, Self> { AsStdWriter(self) }
}

pub trait Origin {
    type Reader: Read;
    type Writer: Write;

    type Resource<'s>: Resource<'s, Origin = Self> + 's where Self: 's;

    fn get<'origin>(&'origin self, path: &Path) -> Result<Self::Resource<'origin>>;
}

pub trait Resource<'origin> {
    type Origin: Origin;

    fn origin(&self) -> &'origin Self::Origin;
    fn relative(&self, path: &Path) -> Result<<Self::Origin as Origin>::Resource<'origin>>;

    fn create(&self) -> Result<<Self::Origin as Origin>::Writer>;
    fn append(&self) -> Result<<Self::Origin as Origin>::Writer>;

    fn read(&self) -> Result<<Self::Origin as Origin>::Reader>;
}


// Default implementations and wrappers for Read and Write.
pub struct AsStdWriter<'w, W: Write + ?Sized>(&'w mut W);
pub struct AsStdReader<'r, R: Read + ?Sized>(&'r mut R);

impl<'w, W: Write> StdWrite for AsStdWriter<'w, W> {
    fn write(&mut self, buf: &[u8]) -> Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> Result<()> { self.0.flush() }
}
impl<'r, R: Read + ?Sized> StdRead for AsStdReader<'r, R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> { 
        todo!()
    }
}

pub struct StdReader<R: StdRead> {
    reader: R,
    rejected: Vec<u8>,
}
pub struct StdWriter<W: StdWrite>(W);


impl<R: StdRead> Read for StdReader<R> {
    fn read<D: DataSink>(&mut self, sink: D) -> Result<D::Output> {
        let mut buf = sink.into_buffer();
        // The portion of the target slice that we have initialized.
        let mut init: usize = 0;
        while !buf.finished() {
            if buf.target().len() == 0 {
                buf.try_expand(None)?;
                init = 0;
            }
            let target = buf.target();
            // If any data was previously rejected,
            // read from the rejected buffer into the target.
            if self.rejected.len() > 0 {
                // SAFETY: We only write into the unitialized data
                // and have written r.len() bytes into the target.
                unsafe {
                    let r = &self.rejected[..self.rejected.len().min(target.len())];
                    let target: &mut [u8] = core::slice::from_raw_parts_mut(
                        target.as_ptr() as *mut u8, r.len());
                    target.copy_from_slice(r);
                    buf.mark_filled(r.len())?;
                    init -= r.len();
                    // Drop the rejected bytes from the buffer.
                    self.rejected.drain(..r.len());
                }
                continue
            }
            if target.len() == 0 {
                panic!("Expand buffer succeeded but target has no capacity left!");
            }
            // We must initialize the target slice to 0 before
            // reading into it for safety.
            let target: &mut [u8] = unsafe {
                let target: &mut [u8] = core::slice::from_raw_parts_mut(target.as_ptr() as *mut u8, target.len());
                target[init..].fill(0);
                init = target.len();
                target
            };
            let res = self.reader.read(target)?.min(target.len());
            // SAFETY: We have initialized the entire target slice to 0,
            // and have ensured that res is at most the length of 
            // the target slice. Thus res bytes in target have been filled.
            unsafe {
                buf.mark_filled(res)?;
                init -= res;
            }
        }
        buf.done()
    }
}