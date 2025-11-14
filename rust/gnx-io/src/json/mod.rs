use crate::util::{ScratchBuffer, TextSource};

mod source;
pub use source::*;

#[derive(Debug)]
pub enum JsonError {
    UnterminatedString,
    InvalidEscape,
    UnexpectedEOF,
    Unexpected(char),
    ReadError(std::io::Error),
    Other(String),
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for JsonError {}
impl gnx::util::Error for JsonError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Self::Other(msg.to_string())
    }
}

impl From<std::io::Error> for JsonError {
    fn from(e: std::io::Error) -> Self {
        Self::ReadError(e)
    }
}

pub struct JsonParser<S> {
    source: S,
    scratch: ScratchBuffer<str>,
    remaining_depth: usize,
}

impl<'src, S: TextSource<'src>> JsonParser<S> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            scratch: ScratchBuffer::new(),
            remaining_depth: 0,
        }
    }
}