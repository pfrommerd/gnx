use std::borrow::{Borrow, BorrowMut};
use std::io::{Result, Error, ErrorKind};

use std::mem::MaybeUninit;



// A trait for a borrowed buffer that can be filled.
// Implementors must ensure that the contents of as_mut( are not de-initialized
// while the buffer is borrowed (e.g. the underlying buffer is not re-allocated).
pub unsafe trait BorrowedBuf<'s> {
    // SAFETY: len() must be equal to as_mut().len().
    fn len(&self) -> usize;

    unsafe fn as_mut<'r>(&'r mut self) -> &'r mut [MaybeUninit<u8>];
    unsafe fn into_mut(self) -> &'s mut [MaybeUninit<u8>];
    // Commit some number of filled bytes, initialized bytes to the underlying storage.
    // consumes the buffer and encapsulating cursor
    unsafe fn commit(self, filled: usize, init: usize) -> Result<()>;
}

fn _slice_len(slice: &[MaybeUninit<u8>]) -> usize {
    slice.len()
}

unsafe impl<'s> BorrowedBuf<'s> for &'s mut [MaybeUninit<u8>] {
    fn len(&self) -> usize { _slice_len(self) }
    unsafe fn as_mut<'r>(&'r mut self) -> &'r mut [MaybeUninit<u8>] { self }
    unsafe fn into_mut(self) -> &'s mut [MaybeUninit<u8>] { self }
    unsafe fn commit(self, _filled: usize, _init: usize) -> Result<()> { Ok(()) }
}

// Wraps a BorrowedBuf and provides a safe way to fill it
// and keep track of the filled and initialized bytes.
pub struct Cursor<B> {
    buf: B,
    filled: usize,
    init: usize,
}

impl<'s, B: BorrowedBuf<'s>> Cursor<B> {
    pub fn new(
        buf: B,
    ) -> Self {
        Self { buf, filled: 0, init: 0 }
    }

    pub unsafe fn with_filled(
        buf: B,
        filled: usize,
        init: usize,
    ) -> Self {
        let capacity = buf.len();
        let filled = filled.min(capacity);
        let init = init.min(capacity);
        Self { buf, filled, init }
    }

    pub fn capacity(&self) -> usize { self.buf.len() }
    pub fn filled_len(&self) -> usize { self.filled }
    pub fn unfilled_len(&self) -> usize { self.capacity() - self.filled }

    pub fn commit(self) -> Result<()> {
        unsafe {
            self.buf.commit(self.filled, self.init)
        }
    }

    pub fn clear(&mut self) -> &mut Self {
        self.filled = 0;
        self
    }

    pub fn init_mut<'r>(&'r mut self) -> &'r mut [u8] {
        // SAFETY: We can assume that buf[..init] has been initialized
        unsafe {
            let buf: &'r mut [MaybeUninit<u8>] = self.buf.as_mut();
            core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, self.init)
        }
    }
    pub fn filled_mut<'r>(&'r mut self) -> &'r mut [u8] {
        // SAFETY: We can assume that buf[..filled] has been filled and is initialized.
        // since filled <= init.
        unsafe {
            let buf: &'r mut [MaybeUninit<u8>] = self.buf.as_mut();
            core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, self.filled)
        }
    }

    // Initialize all uninitialized bytes to 0.
    pub fn ensure_init(&mut self) -> &mut Self {
        // SAFETY: We assume that init..capacity is uninitialized.
        let uninit = unsafe {
            let buf = self.buf.as_mut();
            buf.get_unchecked_mut(self.init..)
        };
        unsafe {
            std::ptr::write_bytes(uninit.as_mut_ptr(), 0, uninit.len());
        }
        self.init = self.buf.len();
        self
    }

    // SAFETY: Caller must ensure that init bytes are initialized.
    pub unsafe fn set_init(&mut self, init: usize) -> &mut Self {
        self.init = std::cmp::max(self.init, init);
        self
    }
    // SAFETY: Caller must ensure that filled bytes are initialized.
    pub unsafe fn set_filled(&mut self, filled: usize) -> &mut Self {
        self.init = std::cmp::max(self.init, filled);
        self.filled = std::cmp::max(self.filled, filled);
        self
    }
    // SAFETY: Caller cannot uninitialize the buffer before ..init
    pub unsafe fn as_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        unsafe {
            self.buf.as_mut()
        }
    }
}

pub trait DataSink : Sized {
    type Output;

    fn is_full(&self) -> bool;

    type BorrowedBuf<'s>: BorrowedBuf<'s> where Self: 's;
    fn cursor<'s>(&'s mut self) -> Cursor<Self::BorrowedBuf<'s>>;

    // Get any rejected bytes
    // If the data ends on a fragment of an
    // entity (e.g. a utf8 codepoint), this will
    // return the fragment. If this is not called, finish()
    // may throw an error.
    fn rejected(&mut self) -> Option<&[u8]>;

    // Called when done filling the buffer.
    // The implementation should ensure the resulting
    // buffer is valid as well as return an error if the buffer is invalid.
    fn finish(self) -> Result<Self::Output>;
}

// Basic buffer types

pub struct LimitedBuf<'s, B: BorrowedBuf<'s>> {
    buffer: B,
    total_filled: &'s mut usize,
    total_limit: usize,
}

unsafe impl<'s, B: BorrowedBuf<'s>> BorrowedBuf<'s> for LimitedBuf<'s, B> {
    fn len(&self) -> usize {
        std::cmp::min(self.buffer.len(), self.total_limit - *self.total_filled)
    }
    unsafe fn as_mut<'r>(&'r mut self) -> &'r mut [MaybeUninit<u8>] {
        unsafe {
            let len = self.len();
            let buf = self.buffer.as_mut();
            buf.get_unchecked_mut(..len)
        }
    }
    unsafe fn into_mut(self) -> &'s mut [MaybeUninit<u8>] {
        unsafe {
            let len = self.len();
            let buf = self.buffer.into_mut();
            buf.get_unchecked_mut(..len)
        }
    }

    unsafe fn commit(self, filled: usize, init: usize) -> Result<()> {
        unsafe {
            // Increment the filled count by the number of filled bytes.
            *self.total_filled = *self.total_filled + filled;
            self.buffer.commit(filled, init)
        }
    }
}

pub struct LimitedSink<S: DataSink> {
    sink: S,
    total_filled: usize,
    total_limit: usize,
}

impl<S: DataSink> DataSink for LimitedSink<S> {
    type Output = S::Output;
    type BorrowedBuf<'s> = LimitedBuf<'s, S::BorrowedBuf<'s>> where Self: 's;

    fn is_full(&self) -> bool {
        self.total_filled >= self.total_limit
    }
    fn cursor(&mut self) -> Cursor<Self::BorrowedBuf<'_>> {
        let cursor = self.sink.cursor();
        let buf = LimitedBuf { buffer: cursor.buf, total_filled: &mut self.total_filled, total_limit: self.total_limit };
        let init = cursor.init.min(buf.len());
        let filled = cursor.filled.min(buf.len());
        // SAFETY: LimitedBuf does not
        // change the bytes of the underlying buffer.
        unsafe { Cursor::with_filled(buf, init, filled) }
    }
    fn rejected(&mut self) -> Option<&[u8]> {
        self.sink.rejected()
    }
    fn finish(self) -> Result<Self::Output> {
        self.sink.finish()
    }
}

pub struct CountedBuf<'s, B: BorrowedBuf<'s>> {
    buffer: B,
    total_filled: &'s mut usize,
}

unsafe impl<'s, B: BorrowedBuf<'s>> BorrowedBuf<'s> for CountedBuf<'s, B> {
    fn len(&self) -> usize {
        self.buffer.len()
    }
    unsafe fn as_mut<'r>(&'r mut self) -> &'r mut [MaybeUninit<u8>] {
        unsafe { self.buffer.as_mut() }
    }
    unsafe fn into_mut(self) -> &'s mut [MaybeUninit<u8>] {
        unsafe { self.buffer.into_mut() }
    }

    unsafe fn commit(self, filled: usize, init: usize) -> Result<()> {
        unsafe {
            // Increment the filled count by the number of filled bytes.
            *self.total_filled = *self.total_filled + filled;
            self.buffer.commit(filled, init)
        }
    }
}

// The slice, vec, and string buffers

pub struct SliceBuf<'s> {
    slice: &'s mut [MaybeUninit<u8>],
    total_filled: &'s mut usize,
    total_init: &'s mut usize,
}

impl<'s> SliceBuf<'s> {
    fn new(slice: &'s mut [MaybeUninit<u8>], total_filled: &'s mut usize, total_init: &'s mut usize) -> Self {
        Self { slice, total_filled, total_init }
    }
}

unsafe impl<'s> BorrowedBuf<'s> for SliceBuf<'s> {
    fn len(&self) -> usize {
        self.slice.len()
    }
    unsafe fn as_mut<'r>(&'r mut self) -> &'r mut [MaybeUninit<u8>] {
        self.slice
    }
    unsafe fn into_mut(self) -> &'s mut [MaybeUninit<u8>] {
        self.slice
    }
    unsafe fn commit(self, filled: usize, init: usize) -> Result<()> {
        *self.total_filled = *self.total_filled + filled;
        *self.total_init = *self.total_init + init;
        Ok(())
    }
}

pub struct VecBuf<'v>(&'v mut Vec<u8>);

unsafe impl<'v> BorrowedBuf<'v> for VecBuf<'v> {
    fn len(&self) -> usize {
        self.0.capacity() - self.0.len()
    }
    unsafe fn as_mut<'r>(&'r mut self) -> &'r mut [MaybeUninit<u8>] {
        unsafe {
            let len = self.len();
            let buf = self.0.as_mut_ptr().add(self.0.len());
            core::slice::from_raw_parts_mut(buf as *mut MaybeUninit<u8>, len)
        }
    }
    unsafe fn into_mut(self) -> &'v mut [MaybeUninit<u8>] {
        unsafe {
            let len = self.len();
            let buf = self.0.as_mut_ptr().add(self.0.len());
            core::slice::from_raw_parts_mut(buf as *mut MaybeUninit<u8>, len)
        }
    }
    unsafe fn commit(self, filled: usize, _init: usize) -> Result<()> {
        unsafe {
            self.0.set_len(self.0.len() + filled);
            Ok(())
        }
    }
}

// Last but not least, the string buffer
// which ensures that the string is always valid utf-8.

pub struct StringBuf<'s> {
    string: &'s mut String,
    // A temporary storage for
    // partial utf-8 data
    remainder: &'s mut [u8; 4],
    remainder_len: &'s mut usize,
}

impl<'s> StringBuf<'s> {
    fn new(string: &'s mut String, remainder: &'s mut [u8; 4], remainder_len: &'s mut usize) -> Self {
        // We should have no remainder at this point.
        assert!(*remainder_len == 0);
        Self { string, remainder, remainder_len }
    }
}

unsafe impl<'s> BorrowedBuf<'s> for StringBuf<'s> {
    fn len(&self) -> usize {
        self.string.len()
    }
    unsafe fn as_mut<'r>(&'r mut self) -> &'r mut [MaybeUninit<u8>] {
        todo!()
    }
    unsafe fn into_mut(self) -> &'s mut [MaybeUninit<u8>] {
        todo!()
    }
    unsafe fn commit(self, filled: usize, init: usize) -> Result<()> {
        Ok(())
    }
}

pub struct StringSink<S: BorrowMut<String>> {
    string: S,
    remainder: [u8; 4],
    remainder_len: usize,
}

impl<S: BorrowMut<String>> DataSink for StringSink<S> {
    type Output = S;
    type BorrowedBuf<'s> = StringBuf<'s> where Self: 's;

    // We can always add more data to the string.
    fn is_full(&self) -> bool { false }
    fn cursor(&mut self) -> Cursor<Self::BorrowedBuf<'_>> {
        // Ensure that the string has excess capacity of at least 8 bytes (2 4 byte codepoints).
        let s = self.string.borrow_mut();
        s.reserve(8);
        // If we have a remainder, copy it into the string's excess capacity.
        let remainder = if self.remainder_len > 0 {
            unsafe {
                let v = s.as_mut_vec();
                let uninit = core::slice::from_raw_parts_mut(
                    v.as_mut_ptr().add(v.len()) as *mut u8,
                    v.capacity() - v.len()
                );
                uninit[..self.remainder_len].copy_from_slice(&self.remainder[..self.remainder_len]);
                let r = self.remainder_len;
                self.remainder_len = 0;
                r
            }
        } else { 0 };

        let buf = StringBuf::new(s, &mut self.remainder, &mut self.remainder_len);
        unsafe { Cursor::with_filled(buf, remainder, remainder) }
    }
    // Look at the remainder for rejected bytes.
    fn rejected(&mut self) -> Option<&[u8]> {
        let r = self.remainder_len;
        if r > 0 {
            self.remainder_len = 0;
            Some(&self.remainder[..r])
        } else {
            None
        }
    }
    fn finish(self) -> Result<Self::Output> {
        if self.remainder_len > 0 {
            return Err(Error::new(ErrorKind::InvalidData, "Partial utf-8 data"));
        }
        Ok(self.string)
    }
}