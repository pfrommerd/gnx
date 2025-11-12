use ordered_float::NotNan;

use std::io::Result;
use crate::Read;
use super::JsonParser;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Token<'s> {
    Comment,
    BlockComment,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Null,
    Bool(bool),
    Number(NotNan<f64>),
    String(&'s str),
}
// A parsed token is either borrowed from a source of lifetime 's
// or is temporary of 'p, which is the lifetime of the parser.
pub enum ParsedToken<'s, 'p> {
    Borrowed(Token<'s>),
    Temporary(Token<'p>),
}

pub struct SourceJsonParser<'s> {
    buf: &'s str
}

impl<'s> JsonParser<'s> for SourceJsonParser<'s> {
    fn next<'p>(&'p mut self) -> Result<Option<ParsedToken<'s, 'p>>> {
        todo!()
    }
}

pub struct StreamJsonParser<R: Read> {
    stream: R,
    buffer: String
}

impl<R: Read> StreamJsonParser<R> {
    pub fn new(stream: R) -> Self {
        Self {
            stream,
            buffer: String::new(),
        }
    }
}
