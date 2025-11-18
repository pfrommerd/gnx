use std::io::{Error, ErrorKind, Result};
use std::io::{Seek, SeekFrom};

use std::hash::Hash;
use std::fmt::{Debug, Display};

use std::ops::Deref;

use super::scratch::{ScratchBuffer, ScratchRange, ScratchSlice, ScratchRangeBuilder};
use super::peek::PeekRead;

pub enum Text<'src, 'buf> {
    Source(&'src str),
    Scratch(ScratchRange<'buf, str>),
    Owned(String)
}

impl<'src, 'buf> Text<'src, 'buf> {
    pub fn borrow<'owned>(&'owned self) -> TextRef<'src, 'buf, 'owned> {
        match self {
            Text::Source(s) => TextRef::Source(s),
            Text::Scratch(s) => TextRef::Scratch(s.borrow()),
            Text::Owned(s) => TextRef::Owned(&s),
        }
    }

    pub fn visit<V: gnx_graph::DataVisitor<'src>, E: gnx_graph::Error>(self, visitor: V)
            -> std::result::Result<V::Value, E> {
        match self {
            Text::Source(s) => visitor.visit_borrowed_str(s),
            Text::Scratch(s) => {
                let s = s.borrow();
                visitor.visit_str(&*s)
            },
            Text::Owned(s) => visitor.visit_string(s),
        }
    }
    pub fn pop(self) -> (Self, Option<char>) {
        match self {
            Text::Source(s) => {
                let mut chars =s.chars();
                let c = chars.next_back();
                (Text::Source(chars.as_str()), c)
            }
            Text::Scratch(mut s) => {
                let c = s.pop(); // pop a char from the end of the range
                (Text::Scratch(s), c)
            }
            Text::Owned(mut s) => {
                let c = s.pop();
                (Text::Owned(s), c)
            }
        }
    }
    pub fn into_string(self) -> String {
        match self {
            Text::Source(s) => s.to_string(),
            Text::Scratch(s) => s.borrow().to_string(),
            Text::Owned(s) => s
        }
    }
}

pub enum TextRef<'src, 'buf, 'owned> {
    Source(&'src str),
    Scratch(ScratchSlice<'buf, str>),
    Owned(&'owned str)
}

impl<'src, 'buf, 'owned> Deref for TextRef<'src, 'buf, 'owned> {
    type Target = str;
    fn deref(&self) -> &str {
        match self {
            TextRef::Source(s) => s,
            TextRef::Scratch(s) => s,
            TextRef::Owned(s) => s,
        }
    }
}



// A utility for consuming characters from a UTF-8 source of text.
pub trait TextSource<'src> {
    type Position: Copy + Eq + Ord + Hash + Debug + Display;

    fn position(&mut self) -> Result<Self::Position>;
    fn goto(&mut self, position: Self::Position) -> Result<()>;

    fn next(&mut self) -> Result<Option<char>>;
    fn peek(&mut self) -> Result<Option<char>>;
    // Peek the next n *characters* (not bytes!). n should be a small constant (e.g. 2 or 3)
    // This may return a slice of up to 4*n bytes, as a UTF-8 character may be up to 4 bytes.
    // This may return fewer than n characters if there are fewer than n characters left.
    fn peek_chars(&mut self, n: usize) -> Result<&str>;

    // Skip the next n *bytes* worth of UTF-8 encoded text. Should error if 
    // (1) any of the skipped bytes are not valid UTF-8
    // (2) we hit an EOF before skipping n bytes.
    fn skip_bytes(&mut self, n: usize) -> Result<()>;

    // Unsafe version of skip_bytes that does not check for UTF-8 validity or EOF.
    // Useful if you have already peeked the bytes and know they are valid UTF-8,
    // so that the source can skip the bytes without checking for UTF-8 validity or EOF.
    // The default implementation falls back to the safe version.
    // 
    // Use peek_and or peek_chars_and to invoke this method safely.
    //
    // SAFETY: Caller must have peeked n bytes from the source, either via peek_chars or peek.
    // and has not since called next() or skip_bytes(). If this is not the case, Undefined Behavior may occur!
    unsafe fn skip_bytes_unchecked(&mut self, n: usize) {
        // Call the safe version by default.
        let _ = self.skip_bytes(n);
    }
    // Start buffering raw source bytes.
    type Buffering<'s, 'tmp> : BufferedSource<'src, 'tmp> where Self: 's + 'tmp;
    // Self must live as long as either the borrow on self or the borrow on the scratch buffer.
    // This way an implementation can use an internal buffer instead of the scratch buffer, if needed.
    fn buffering<'s, 'tmp>(&'s mut self, scratch: &'tmp ScratchBuffer<str>) -> Self::Buffering<'s, 'tmp> where Self: 's + 'tmp;

    // Peek a character and then call the provided function. If the function returns true
    // and the character is Some, we will also consume the character.
    fn peek_and<R, F: FnOnce(Option<char>) -> (bool, R)>(&mut self, f: F) -> Result<R> {
        let peek = self.peek()?;
        // SAFETY: Since we have captured self mutably, 
        // f cannot call self.next() or self.skip_bytes().
        let (consume, result) = f(peek);
        // if consume is true and peek is Some, consume the character.
        if consume {
            if let Some(c) = peek {
                // SAFETY: We know that the next character is valid and so can consume its length in UTF-8 bytes.
                unsafe { self.skip_bytes_unchecked(c.len_utf8()); }
            }
        }
        Ok(result)
    }


    fn peek_chars_and<R, F: FnOnce(&str) -> (usize, R)>(&mut self, n: usize, f: F) -> Result<R> {
        let peek = self.peek_chars(n)?;
        // SAFETY: Since we have captured self mutably, 
        // f cannot call self.next() or self.skip_bytes().
        let (mut consume, result) = f(peek);
        if consume > 0 {
            // Ensure we do not consume more than the amount peeked.
            consume = consume.min(peek.len());
            // Ensure that we do not slice in the middle of a UTF-8 character
            // if we are consuming less than the entire peeked amount.
            if consume < peek.len() {
                peek.get(0..consume).ok_or_else(|| Error::new(ErrorKind::InvalidData, "Invalid UTF-8"))?;
            }
            // SAFETY: We know that all of peek is valid UTF-8 and have ensured that 
            // we do not slice in the middle of a UTF-8 character or go off the end.
            unsafe { self.skip_bytes_unchecked(consume); }
        }
        Ok(result)
    }

}

pub trait BufferedSource<'src, 'buf> : TextSource<'src> + Into<Text<'src, 'buf>> {
    fn into_buffer(self) -> Text<'src, 'buf> {
        self.into()
    }
}

// Utility functions for consuming characters from a &[u8]
// TODO: These potentially valid the entire buffer, which is inefficient.
fn _next_char(data: &[u8]) -> Result<Option<(usize, char)>> {
    if data.is_empty() { return Ok(None); }
    // SAFETY data is not empty
    let first_byte = unsafe { *data.get_unchecked(0) };
    let width = match first_byte {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => return Err(Error::new(ErrorKind::InvalidData, "Invalid UTF-8")),
    };
    if data.len() < width {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid UTF-8"));
    }
    // We have checked the width, but not that payload is valid UTF-8.
    let string = std::str::from_utf8(&data[..width]).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    let char = string.chars().next().unwrap();
    Ok(Some((width, char)))
}
// Returns up to n characters from the data.
fn _next_chars(data: &[u8], n: usize) -> Result<&str> {
    if data.is_empty() { return Ok(&""); }
    // Figure out how many bytes to slice.
    let mut idx = 0;
    let mut count = 0;
    while count < n && idx < data.len() {
        // SAFETY: idx is a valid index into data
        let next_byte = unsafe { *data.get_unchecked(idx) };
        let width = match next_byte {
            0x00..=0x7F => 1,
            0xC2..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF4 => 4,
            _ => return Err(Error::new(ErrorKind::InvalidData, "Invalid UTF-8")),
        };
        // SAFETY: ensure that idx does not go past the end of the data
        idx = std::cmp::min(idx + width, data.len());
        count += 1;
    }
    // Parse as a UTF-8 string
    let string = std::str::from_utf8(&data[..idx]).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    Ok(string)
}


// Any peekable source of text.
pub struct IoSource<R: PeekRead + Seek> {
    reader: R,
}

impl<R: PeekRead + Seek> IoSource<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<'src, R: PeekRead + Seek> TextSource<'src> for IoSource<R> {
    type Position = u64;
    fn position(&mut self) -> Result<Self::Position> {
        self.reader.stream_position()
    }
    fn goto(&mut self, position: Self::Position) -> Result<()> {
        self.reader.seek(SeekFrom::Start(position))?;
        Ok(())
    }


    fn next(&mut self) -> Result<Option<char>> {
        let ahead = self.reader.peek(4)?;
        Ok(_next_char(ahead)?.map(|(len, char)| {
            self.reader.consume(len);
            char
        }))
    }
    fn skip_bytes(&mut self, n: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        let mut skipped = 0;
        while skipped < n {
            let buf = self.reader.fill_buf()?;
            if buf.len() == 0 {
                return Err(Error::new(ErrorKind::UnexpectedEof, "EOF while skipping bytes"));
            }
            // Consume up to buf bytes or n - skipped bytes, whichever is less.
            let amt = std::cmp::min(n - skipped, buf.len());
            // Validate that the skipped bytes are valid UTF-8.
            std::str::from_utf8(&buf[..amt]).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            self.reader.consume(amt);
            skipped += amt;
        }
        Ok(())
    }

    // Note that this particular implementation does cause UB
    // as we must always pessimicially assume that the Reader will return garbage data
    // upon the next call to peek() or fill_buf().
    //
    // However, it is a bit faster than the safe version. As we assume that the 
    // caller has called peek() or peek_chars(), we presume that the bytes we want to skip
    // are already in self.reader.buffer().
    // Even if they are not, it is always safe to consume bytes from the underlying Reader.
    unsafe fn skip_bytes_unchecked(&mut self, n: usize) {
        if n == 0 { return; }
        self.reader.consume(n);
    }

    fn peek(&mut self) -> Result<Option<char>> {
        let ahead = self.reader.peek(4)?;
        Ok(_next_char(ahead)?.map(|(_, char)| char ))
    }
    fn peek_chars(&mut self, n: usize) -> Result<&str> {
        let ahead = self.reader.peek(n*4)?;
        _next_chars(ahead, n)
    }

    type Buffering<'s,'tmp> = IoSourceBuf<'s, 'tmp, R> where Self: 's + 'tmp;
    fn buffering<'s, 'tmp>(&'s mut self, scratch: &'tmp ScratchBuffer<str>) -> Self::Buffering<'s, 'tmp> where Self: 's + 'tmp {
        IoSourceBuf { reader: &mut self.reader, scratch, consumer: scratch.start() }
    }
}

pub struct IoSourceBuf<'p, 'tmp, R: PeekRead + Seek> {
    reader: &'p mut R,
    scratch: &'tmp ScratchBuffer<str>,
    consumer: ScratchRangeBuilder<'tmp, str>,
}

impl<'p, 'buf, 'src, R: PeekRead + Seek> Into<Text<'src, 'buf>> for IoSourceBuf<'p, 'buf, R> {
    fn into(self) -> Text<'src, 'buf> {
        Text::Scratch(self.consumer.end())
    }
}

impl<'p, 'buf, 'src, R: PeekRead + Seek> TextSource<'src> for IoSourceBuf<'p, 'buf, R> {
    type Position = u64;
    fn position(&mut self) -> Result<Self::Position> {
        self.reader.stream_position()
    }
    fn goto(&mut self, position: Self::Position) -> Result<()> {
        self.reader.seek(SeekFrom::Start(position))?;
        Ok(())
    }
    // Buffering versions of next, skip for IoSourceBuf
    fn next(&mut self) -> Result<Option<char>> {
        let ahead = self.reader.peek(4)?;
        match _next_char(ahead)? {
            Some((len, char)) => {
                // copy len bytes to the buffer
                self.scratch.push_char(char);
                self.reader.consume(len);
                Ok(Some(char))
            }
            None => Ok(None),
        }
    }
    fn skip_bytes(&mut self, n: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        let mut skipped = 0;
        while skipped < n {
            let buf = self.reader.fill_buf()?;
            if buf.len() == 0 {
                return Err(Error::new(ErrorKind::UnexpectedEof, "EOF while skipping bytes"));
            }
            // Consume up to buf bytes or n - skipped bytes, whichever is less.
            let amt = std::cmp::min(n - skipped, buf.len());
            // copy amt bytes to the buffer
            let s = std::str::from_utf8(&buf[..amt]).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            self.scratch.extend_from_str(s);
            // mark the bytes as read
            self.reader.consume(amt);
            skipped += amt;
        }
        Ok(())
    }

    // These are the same as for IoSource
    fn peek(&mut self) -> Result<Option<char>> {
        let ahead = self.reader.peek(4)?;
        Ok(_next_char(ahead)?.map(|(_, char)| char ))
    }
    fn peek_chars(&mut self, n: usize) -> Result<&str> {
        // Try and look up to 4*N bytes ahead.
        let ahead = self.reader.peek(n*4)?;
        _next_chars(ahead, n)
    }
    // Ignore the scratch buffer, we use the same one so that the parent buffering 
    // operation captures the buffer correctly.
    type Buffering<'s, 'tmp> = IoSourceBuf<'s, 'tmp, R> where Self: 's + 'tmp;
    fn buffering<'s, 'tmp>(&'s mut self, _scratch: &'tmp ScratchBuffer<str>) -> Self::Buffering<'s, 'tmp> where Self: 's + 'tmp {
        IoSourceBuf { reader: &mut self.reader, scratch: self.scratch, consumer: self.scratch.start() }
    }
}

impl<'p, 'buf, 'src, R: PeekRead + Seek> BufferedSource<'src, 'buf>
    for IoSourceBuf<'p, 'buf, R> {}

pub struct RawSource<'src> {
    data: &'src [u8],
    pos: usize,
}

pub struct RawSourceBuf<'p, 'src> {
    parent: &'p mut RawSource<'src>,
    buffer_start: usize,
}


impl<'src> From<&'src [u8]> for RawSource<'src> {
    fn from(data: &'src [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl<'src> RawSource<'src> {
    fn remaining(&self) -> &'src [u8] {
        unsafe { &self.data.get_unchecked(self.pos..) }
    }
}

impl<'src> TextSource<'src> for RawSource<'src> {
    type Position = usize;
    fn position(&mut self) -> Result<Self::Position> {
        Ok(self.pos)
    }
    fn goto(&mut self, position: Self::Position) -> Result<()> {
        // Ensure that self.pos is a valid UTF-8 index into self.data
        let prev_pos = self.pos;
        if position > self.data.len() {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid UTF-8 index"));
        }
        // Verify that the bytes between prev_pos and position are valid UTF-8
        if position > prev_pos {
            std::str::from_utf8(&self.data[prev_pos..position]).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        }
        self.pos = position;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<char>> {
        match _next_char(self.remaining())? {
            Some((len, char)) => {
                self.pos += len;
                Ok(Some(char))
            }
            None => Ok(None),
        }
    }

    fn peek(&mut self) -> Result<Option<char>> {
        Ok(_next_char(self.remaining())?.map(|(_, char)| char))
    }

    fn peek_chars(&mut self, n: usize) -> Result<&str> {
        _next_chars(self.remaining(), n)
    }

    fn skip_bytes(&mut self, n: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        if self.pos + n > self.data.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "EOF while skipping bytes"));
        }
        // SAFETY: self.pos is a valid start index and self.pos + n is a valid end index.
        let skipped = unsafe { self.data.get_unchecked(self.pos..self.pos + n) };
        // Validate that the skipped bytes are valid UTF-8
        std::str::from_utf8(skipped).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        self.pos += n;
        Ok(())
    }

    unsafe fn skip_bytes_unchecked(&mut self, n: usize) {
        // SAFETY: The caller has verified that n is a valid 
        // offset to advance us by in self.data
        self.pos += n;
    }

    type Buffering<'s, 'tmp> = RawSourceBuf<'s, 'src> where Self: 's + 'tmp;
    fn buffering<'s, 'tmp>(&'s mut self, _scratch: &'tmp ScratchBuffer<str>) -> Self::Buffering<'s, 'tmp>
            where Self: 's + 'tmp {
        // SAFETY: self.pos is a valid index into self.data
        let buffer_start = self.pos;
        RawSourceBuf { parent: self, buffer_start }
    }
}

impl<'tmp, 'src, 'p> Into<Text<'src, 'tmp>> for RawSourceBuf<'p, 'src> {
    fn into(self) -> Text<'src, 'tmp> {
        // SAFETY: We know that buffer_start is a valid UTF-8 index 
        // into self.parent.data and that self.parent.pos 
        // is also a valid UTF-8 index.
        let buffered = unsafe { std::str::from_utf8_unchecked(
            self.parent.data.get_unchecked(self.buffer_start..self.parent.pos))
        };
        Text::Source(buffered)
    }
}

impl<'p, 'src> TextSource<'src> for RawSourceBuf<'p, 'src> {
    type Position = usize;
    fn position(&mut self) -> Result<Self::Position> { self.parent.position() }
    fn goto(&mut self, position: Self::Position) -> Result<()> { self.parent.goto(position) }
    fn next(&mut self) -> Result<Option<char>> { self.parent.next() }
    fn peek(&mut self) -> Result<Option<char>> { self.parent.peek() }
    fn peek_chars(&mut self, n: usize) -> Result<&str> { self.parent.peek_chars(n) }
    fn skip_bytes(&mut self, n: usize) -> Result<()> { self.parent.skip_bytes(n) }

    unsafe fn skip_bytes_unchecked(&mut self, n: usize) {
        // SAFETY: The caller has verified that n is a valid 
        // offset to advance us by in self.parent.data.
        unsafe { self.parent.skip_bytes_unchecked(n); }
    }

    type Buffering<'s, 'tmp> = RawSourceBuf<'s, 'src> where Self: 's + 'tmp;
    fn buffering<'s, 'tmp>(&'s mut self, _scratch: &'tmp ScratchBuffer<str>) -> Self::Buffering<'s, 'tmp>
            where Self: 's + 'tmp {
        let buffer_start = self.parent.pos;
        RawSourceBuf { parent: self.parent, buffer_start }
    }
}

impl<'p, 'src, 'buf> BufferedSource<'src, 'buf> for RawSourceBuf<'p, 'src> {}

pub struct RawStrSource<'src> {
    // The original data that we are consuming from.
    data: &'src str,
    pos: usize,
}

pub struct RawStrSourceBuf<'p, 'src> {
    parent: &'p mut RawStrSource<'src>,
    buffer_start: usize,
}

impl<'src> From<&'src str> for RawStrSource<'src> {
    fn from(data: &'src str) -> Self {
        Self { data, pos: 0 }
    }
}

impl<'src> RawStrSource<'src> {
    fn remaining(&self) -> &'src str {
        // SAFETY: self.pos is a valid UTF-8 index into self.data
        unsafe { self.data.get_unchecked(self.pos..) }
    }
}

impl<'src> TextSource<'src> for RawStrSource<'src> {
    type Position = usize;
    fn position(&mut self) -> Result<Self::Position> {
        Ok(self.pos)
    }
    fn goto(&mut self, position: Self::Position) -> Result<()> {
        // Verify that the position is a valid UTF-8 index into self.data
        self.data.get(0..position).ok_or_else(
            || Error::new(ErrorKind::InvalidData, "Invalid UTF-8 index")
        )?;
        self.pos = position;
        Ok(())
    }
    fn next(&mut self) -> Result<Option<char>> {
        let mut chars = self.remaining().char_indices();
        match chars.next() {
            Some((_, ch)) => {
                let next_idx = chars.offset();
                // SAFETY: pos remains a valid UTF-8 index into self.data
                self.pos += next_idx;
                Ok(Some(ch))
            }
            None => Ok(None),
        }
    }

    fn peek(&mut self) -> Result<Option<char>> {
        Ok(self.remaining().chars().next())
    }

    fn peek_chars(&mut self, n: usize) -> Result<&str> {
        // Count up to n chars into the remaining string.
        let remaining = self.remaining();
        let mut chars = remaining.char_indices();
        // Consume up to n chars
        (&mut chars).take(n).last();
        let offset = chars.offset();
        // SAFETY: we have verified that offset is a valid UTF-8 index into remaining
        let slice = unsafe { remaining.get_unchecked(..offset) };
        Ok(slice)
    }

    fn skip_bytes(&mut self, n: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        if self.pos + n > self.data.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "EOF while skipping bytes"));
        }
        // Verify that we are moving to a valid UTF-8 index.
        self.data.get(self.pos + n..).ok_or_else(
            || Error::new(ErrorKind::InvalidData, "Invalid UTF-8")
        )?;
        // SAFETY: self.pos + n is a valid UTF-8 index into self.data
        self.pos += n;
        Ok(())
    }

    unsafe fn skip_bytes_unchecked(&mut self, n: usize) {
        // SAFETY: self.pos + n is a valid UTF-8 index into self.data
        self.pos += n;
    }

    type Buffering<'s, 'tmp> = RawStrSourceBuf<'s, 'src> where Self: 's + 'tmp;
    fn buffering<'s, 'tmp>(&'s mut self, _scratch: &'tmp ScratchBuffer<str>)
            -> Self::Buffering<'s, 'tmp> where Self: 's + 'tmp {
        let buffer_start = self.pos;
        RawStrSourceBuf { parent: self, buffer_start }
    }
}

impl<'tmp, 'src, 'p> Into<Text<'src, 'tmp>> for RawStrSourceBuf<'p, 'src> {
    fn into(self) -> Text<'src, 'tmp> {
        // SAFETY: We know that buffer_start and self.parent.pos are valid UTF-8 indices
        // into self.parent.data (they were calculated from string slice positions)
        let buffered = unsafe {
            self.parent.data.get_unchecked(self.buffer_start..self.parent.pos)
        };
        Text::Source(buffered)
    }
}

impl<'src, 'p> TextSource<'src> for RawStrSourceBuf<'p, 'src> {
    type Position = usize;
    fn position(&mut self) -> Result<Self::Position> {
        Ok(self.parent.pos)
    }
    fn goto(&mut self, position: Self::Position) -> Result<()> {
        self.parent.goto(position)
    }
    fn next(&mut self) -> Result<Option<char>> { self.parent.next() }
    fn peek(&mut self) -> Result<Option<char>> { self.parent.peek() }
    fn peek_chars(&mut self, n: usize) -> Result<&str> { self.parent.peek_chars(n) }
    fn skip_bytes(&mut self, n: usize) -> Result<()> { self.parent.skip_bytes(n) }
    unsafe fn skip_bytes_unchecked(&mut self, n: usize) {
        // SAFETY: The caller has verified that n is a valid byte offset
        unsafe { self.parent.skip_bytes_unchecked(n); }
    }

    type Buffering<'s, 'tmp> = RawStrSourceBuf<'s, 'src> where Self: 's + 'tmp;
    fn buffering<'s, 'tmp>(&'s mut self, _scratch: &'tmp ScratchBuffer<str>)
            -> Self::Buffering<'s, 'tmp> where Self: 's + 'tmp {
        // SAFETY: self.parent.pos is a valid UTF-8 index into self.parent.data
        let buffer_start = self.parent.pos;
        RawStrSourceBuf { parent: self.parent, buffer_start }
    }
}
impl<'p, 'src, 'buf> BufferedSource<'src, 'buf> for RawStrSourceBuf<'p, 'src> {}