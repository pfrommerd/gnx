use std::io::{Result, Error, ErrorKind};

use super::peek::PeekRead;

pub enum Text<'src, 'tmp> {
    Source(&'src str),
    Temporary(&'tmp str)
}

impl<'src, 'tmp> Text<'src, 'tmp> {
    pub fn as_str<'a>(&self) -> &'a str  where 'src: 'a, 'tmp: 'a {
        match self {
            Text::Source(s) => s,
            Text::Temporary(s) => s,
        }
    }
}

// A utility for consuming characters from a UTF-8 source of text.
pub trait TextSource<'src> {
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

    // Peek a character and then call the provided function. If the function returns true
    // and the character is Some, we will also consume the character.
    fn peek_and<R, F: FnOnce(Option<char>) -> (bool, R)>(&mut self, f: F) -> Result<R> {
        let peek = self.peek()?;
        // SAFETY: Since we have captured self mutably, 
        // f cannot call self.next() or self.skip_bytes().
        let (consume, result) = f(peek);
        if consume && let Some(c) = peek {
            // SAFETY: We know that the next character is valid and so can consume its length in UTF-8 bytes.
            unsafe { self.skip_bytes_unchecked(c.len_utf8()); }
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

    // Start buffering raw source bytes.
    type Buffering<'s: 'tmp, 'tmp> : TextSource<'tmp> + Into<Text<'src, 'tmp>> where Self: 's;
    // The borrow on self must be as long as the borrow on the scratch buffer
    // so that we can use an internal buffer instead of the scratch buffer, if needed.
    fn buffering<'s: 'tmp, 'tmp>(&'s mut self, scratch: &'tmp mut Vec<u8>) -> Self::Buffering<'s, 'tmp>;
}

// Utility functions for consuming characters from a &[u8]
// TODO: These potentially valid the entire buffer, which is inefficient.
fn _next_char(data: &[u8]) -> Result<Option<(usize, char)>> {
    if data.is_empty() { return Ok(None); }
    let width = match data[0] {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => return Err(Error::new(ErrorKind::InvalidData, "Invalid UTF-8")),
    };
    if data.len() < width {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid UTF-8"));
    }
    // We have checked the width, but not that the rest of the data is valid UTF-8.
    let string = std::str::from_utf8(&data[width..]).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    let char = string.chars().next().unwrap();
    Ok(Some((width, char)))
}
// Returns up to n characters from the data.
fn _next_chars(data: &[u8], n: usize) -> Result<&str> {
    if data.len() == 0 { return Ok(&""); }
    // Figure out how many bytes to slice.
    let mut idx = 0;
    let mut count = 0;
    while count < n && idx < data.len() {
        let width = match data[idx] {
            0x00..=0x7F => 1,
            0xC2..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF4 => 4,
            _ => return Err(Error::new(ErrorKind::InvalidData, "Invalid UTF-8")),
        };
        idx += width;
        count += 1;
    }
    // Get the first count characters from the data as 
    let string = std::str::from_utf8(&data[idx..]).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    Ok(string)
}


// Any peekable source of text.
pub struct IoSource<R: PeekRead> {
    reader: R,
}

impl<R: PeekRead> IoSource<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<'src, R: PeekRead> TextSource<'src> for IoSource<R> {
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
            // SAFETY: Since 0 < amt <= buf.len(), there will always be a utf8-chunk
            //         and we can slice + unwrap() safely.
            let chunk = unsafe { buf.get_unchecked(..amt).utf8_chunks().next().unwrap_unchecked() };
            // If the next bytes are not valid UTF-8, return an error.
            if chunk.valid().is_empty() && !chunk.invalid().is_empty() {
                let valid_amt = chunk.valid().len();
                self.reader.consume(valid_amt);
                return Err(Error::new(ErrorKind::InvalidData, "Invalid UTF-8"));
            }
            // Reduces amt to the length of the valid bytes.
            let amt = chunk.valid().len();
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

    type Buffering<'s: 'tmp, 'tmp> = IoSourceBuf<'tmp, &'s mut R> where Self: 's;
    fn buffering<'s: 'tmp, 'tmp>(&'s mut self, scratch: &'tmp mut Vec<u8>) -> Self::Buffering<'s, 'tmp> {
        let buffer_start = scratch.len();
        IoSourceBuf { reader: &mut self.reader, buffer: scratch, buffer_start }
    }
}

pub struct IoSourceBuf<'buf, R: PeekRead> {
    reader: R,
    buffer: &'buf mut Vec<u8>,
    // Offset into the vector where the buffered data starts.
    // useful for nested buffering.
    buffer_start: usize,
}

impl<'tmp, 'src, R: PeekRead> Into<Text<'src, 'tmp>> for IoSourceBuf<'tmp, R> {
    fn into(self) -> Text<'src, 'tmp> {
        Text::Temporary(std::str::from_utf8(&self.buffer[self.buffer_start..]).unwrap())
    }
}

impl<'src, 'buf, R: PeekRead> TextSource<'src> for IoSourceBuf<'buf, R> {
    // Buffering versions of next, skip for IoSourceBuf
    fn next(&mut self) -> Result<Option<char>> {
        let ahead = self.reader.peek(4)?;
        match _next_char(ahead)? {
            Some((len, char)) => {
                // copy len bytes to the buffer
                self.buffer.extend_from_slice(&ahead[..len]);
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
            self.buffer.extend_from_slice(&buf[..amt]);
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
    type Buffering<'s: 'tmp, 'tmp> = IoSourceBuf<'tmp, &'s mut R> where Self: 's;
    fn buffering<'s: 'tmp, 'tmp>(&'s mut self, _scratch: &'tmp mut Vec<u8>) -> Self::Buffering<'s, 'tmp> {
        let buffer_start = self.buffer.len();
        IoSourceBuf { reader: &mut self.reader, buffer: self.buffer, buffer_start }
    }
}

pub struct RawSource<'src> {
    data: &'src [u8],
    pos: usize,
}

pub struct RawSourceBuf<'p, 'src> {
    parent: &'p mut RawSource<'src>,
    buffer_start: usize,
}

pub struct RawStrSource<'src> {
    // The original data that we are consuming from.
    data: &'src str,
    left: &'src str,
}

pub struct RawStrSourceBuf<'p, 'src> {
    parent: &'p mut RawStrSource<'src>,
    buffer_start: usize,
}

impl<'src> From<&'src [u8]> for RawSource<'src> {
    fn from(data: &'src [u8]) -> Self {
        Self { data, pos: 0 }
    }
}