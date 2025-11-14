use std::cell::{Ref, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;
use std::ops::Deref;

// A scratch buffer is one that is "append-only"
// in that it can be appended to "immutably."
// T is the underlying data contained in the buffer
// and the type of reference that can be taken, either u8 or str
pub struct ScratchBuffer<T: ?Sized>  {
    data: RefCell<Vec<u8>>,
    // A token is used to track whether there are outstanding
    // Ranges referencing the existing data. If there are not,
    // When start() is called, if there are no outstanding ranges,
    // we can clear the data and reuse the capacity for new data
    token: Rc<()>,
    _marker: PhantomData<T>,
}

impl<T: ?Sized> ScratchBuffer<T> {
    pub fn new() -> Self {
        Self {
            data: RefCell::new(Vec::new()),
            token: Rc::new(()),
            _marker: PhantomData,
        }
    }
    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }

    pub fn start(&self) -> ScratchRangeBuilder<'_, T> {
        // If we are the only owner of the token on the buffer
        // clear the buffer before starting to build a new range
        if Rc::strong_count(&self.token) == 1 {
            self.data.borrow_mut().clear();
        }
        ScratchRangeBuilder {
            buffer: self,
            token: self.token.clone(),
            start: 0,
        }
    }
}

impl ScratchBuffer<str> {
    pub fn extend_from_str(&self, s: &str) {
        self.data.borrow_mut().extend_from_slice(s.as_bytes());
    }
    pub fn push_char(&self, c: char) {
        self.data.borrow_mut().push(c as u8);
    }
}

pub struct ScratchRangeBuilder<'buf, T: ?Sized> {
    buffer: &'buf ScratchBuffer<T>,
    token: Rc<()>,
    start: usize,
}

impl<'buf, T: ?Sized> ScratchRangeBuilder<'buf, T> {
    // Stop collecting and return a range of the buffer.
    // Since the scratch start was taken when the buffer was created,
    // and existing bytes are immutable, the range is valid!
    pub fn end(&self) -> ScratchRange<'buf, T> {
        ScratchRange {
            buffer: self.buffer,
            token: self.token.clone(),
            start: self.start,
            end: self.buffer.len(),
        }
    }
}


// A valid range of a ScratchBuffer.
// Note that this does not actually borrow the buffer,
// but merely references the range. This way the buffer
// can be re-allocated as needed, and only when we need the data (via ScratchSlice)
// do we actually borrow the buffer using the RefCell.
pub struct ScratchRange<'buf, T: ?Sized> {
    buffer: &'buf ScratchBuffer<T>,
    // ensure the range is valid as long as the token is valid
    token: Rc<()>,
    start: usize,
    end: usize,
}

impl<'buf, T: ?Sized> Clone for ScratchRange<'buf, T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer,
            token: self.token.clone(),
            start: self.start,
            end: self.end,
        }
    }
}

impl<'buf, T: ?Sized> ScratchRange<'buf, T> {
    pub fn borrow(&self) -> ScratchSlice<'buf, T> {
        ScratchSlice {
            range: self.clone(),
            data: self.buffer.data.borrow(),
        }
    }

    pub fn into_vec(self) -> Vec<u8> {
        // Consume our token on the buffer
        std::mem::drop(self.token);
        // If the only other person with a token is the buffer itself,
        // borrow the buffer mutably and swap it for a new buffer 
        // so we don't have to copy the data.
        if self.start == 0 && Rc::strong_count(&self.buffer.token) == 1 {
            let mut buf = self.buffer.data.borrow_mut();
            let mut data = Vec::new();
            std::mem::swap(&mut *buf, &mut data);
            data.truncate(self.end);
            data
        } else {
            let buf = self.buffer.data.borrow();
            let data = unsafe { buf.get_unchecked(self.start..self.end) };
            data.to_vec()
        }
    }
}

impl<'buf> ScratchRange<'buf, str> {
    pub fn into_string(self) -> String {
        let data = self.into_vec();
        unsafe { String::from_utf8_unchecked(data) }
    }

    pub fn pop(&mut self) -> Option<char> {
        let r = self.borrow();
        let s: &str = r.deref();
        let c = s.chars().next_back()?;
        let len = s.chars().as_str().len();
        self.end = self.start + len;
        Some(c)
    }
}

// A borrowed version of a ScratchRange that
// actually borrows the Vec<u8>.
pub struct ScratchSlice<'buf, T: ?Sized> {
    range: ScratchRange<'buf, T>,
    data: Ref<'buf, Vec<u8>>,
}

impl<'buf, T: ?Sized> Into<ScratchRange<'buf, T>> for ScratchSlice<'buf, T> {
    fn into(self) -> ScratchRange<'buf, T> {
        self.range
    }
}

impl<'buf> Deref for ScratchSlice<'buf, u8> {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        let buf: &Vec<u8> = self.data.deref();
        unsafe { buf.get_unchecked(self.range.start..self.range.end) }
    }
}

impl<'buf> Deref for ScratchSlice<'buf, str> {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        let buf: &Vec<u8> = self.data.deref();
        unsafe { std::str::from_utf8_unchecked(buf.get_unchecked(self.range.start..self.range.end)) }
    }
}

impl<'buf> ScratchSlice<'buf, str> {
}