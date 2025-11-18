use std::io::{Read, BufRead, Result, Error, ErrorKind};


pub trait PeekRead: BufRead {
    // Get access to the current
    // underlying buffer without readig it.
    fn buffer(&self) -> &[u8];
    // May return an error if we are unable to peek len bytes ahead.
    // May return less than len bytes if EOF is reached.
    fn peek(&mut self, len: usize) -> Result<&[u8]>;
}

impl<R: PeekRead> PeekRead for &mut R {
    fn buffer(&self) -> &[u8] {
        R::buffer(self)
    }
    fn peek(&mut self, len: usize) -> Result<&[u8]> {
        R::peek(self, len)
    }
}

// TODO: This is a placeholder implementation
// which is inefficient as it always uses a fully 
// initialized buffer. To do this efficiently, we require
// the unstable read_buf feature of the std library such
// that R can read into an uninitialized buffer.
pub struct BufReader<R: Read> {
    reader: R,
    buffer: Vec<u8>,
    // pos -- where we start reading from the buffer
    pos: usize,
    // filled -- where the buffer data lives
    filled: usize,
}

const DEFAULT_BUFFER_SIZE: usize = 8*1024;

impl<R: Read> BufReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: vec![0; DEFAULT_BUFFER_SIZE],
            pos: 0,
            filled: 0,
        }
    }

    pub fn with_capacity(capacity: usize, reader: R) -> Self {
        Self {
            reader,
            buffer: vec![0; capacity],
            pos: 0,
            filled: 0,
        }
    }
    pub fn discard_buffer(&mut self) {
        self.pos = 0;
        self.filled = 0;
    }
    pub fn compact_buffer(&mut self) {
        if self.pos == self.filled {
            self.discard_buffer();
        } else {
            self.buffer.copy_within(self.pos..self.filled, 0);
            self.filled = self.filled - self.pos;
            self.pos = 0;
        }
    }
}

impl<R: Read> Read for BufReader<R> {
    fn read(&mut self, mut buf: &mut [u8]) -> Result<usize> {
        // If we have bytes in the buffer, consume those first
        let src = self.buffer();
        let mut read = 0;
        if src.len() > 0 && buf.len() > 0 {
            let n = std::cmp::min(buf.len(), src.len());
            buf[..n].copy_from_slice(&src[..n]);
            self.consume(n);
            buf = &mut buf[n..];
            read += n;
        }
        if buf.len() > 0 {
            read += self.reader.read(buf)?;
        }
        Ok(read)
    }
}

impl<R: Read> BufRead for BufReader<R> {
    fn fill_buf(&mut self) -> Result<&[u8]> {
        if self.filled > 0 && self.pos == self.filled {
            self.discard_buffer();
        }
        // Read into the space at the end of the buffer.
        let n = self.reader.read(&mut self.buffer[self.filled..])?;
        self.filled += n;
        let buf = PeekRead::buffer(self);
        Ok(buf)
    }
    fn consume(&mut self, amt: usize) {
        self.pos = std::cmp::min(self.pos + amt, self.filled);
    }
}

impl<R: Read> PeekRead for BufReader<R> {
    fn buffer(&self) -> &[u8] {
        &self.buffer[self.pos..self.filled]
    }
    fn peek(&mut self, len: usize) -> Result<&[u8]> {
        if len > self.buffer.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "Cannot peek more than the buffer size"));
        }
        // Check if we need to shift the buffer back to the start.
        if len > self.buffer.len() - self.pos {
            self.compact_buffer();
        }
        // Check if we need to fill the buffer.
        if self.filled - self.pos < len {
            self.fill_buf()?;
        }
        let buf = PeekRead::buffer(self);
        // Get up to the first len bytes of the buffer.
        Ok(&buf[..len.min(buf.len())])
    }
}

impl<R: std::io::Read> PeekRead for std::io::BufReader<R> {
    fn buffer(&self) -> &[u8] {
        self.buffer()
    }
    // TODO: We may not fill the buffer fully,
    // so it is possible that we return less than len bytes.
    fn peek(&mut self, len: usize) -> Result<&[u8]> {
        self.fill_buf()?;
        let mut buf = PeekRead::buffer(self);
        // Truncate the buffer to at most the requested length.
        if buf.len() > len { buf = &buf[..len]; }
        Ok(buf)
    }
}