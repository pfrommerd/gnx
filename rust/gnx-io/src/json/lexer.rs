use ordered_float::NotNan;

use super::JsonParser;
use std::io::{Error, ErrorKind, Result};
use crate::{Read, Write};

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
    buffer: String,
}

impl<R: Read> StreamJsonParser<R> {
    pub fn new(stream: R) -> Self {
        Self { stream, buffer: String::new() }
    }
}

impl<'s, R: Read> JsonParser<'s> for StreamJsonParser<R> {
    fn next<'p>(&'p mut self) -> Result<Option<ParsedToken<'s, 'p>>> {
        let buf = self.stream.fill_buf()?;
        let mut consumed = 0;
        if let Some(chunk) = buf.utf8_chunks().next() {
            if chunk.valid().is_empty() && !chunk.invalid().is_empty() {
                return Err(Error::new(ErrorKind::InvalidData, "Invalid UTF-8"));
            }
            let chunk = chunk.valid();
            let trimmed = chunk.trim_start();
            consumed += chunk.len() - trimmed.len();
            if trimmed.is_empty() {
                self.stream.consume(consumed);
                return Ok(None);
            }
            match chunk.chars().next().unwrap() {
                '{' => {
                    self.stream.consume(consumed + 1);
                    Ok(Some(ParsedToken::Borrowed(Token::LBrace)))
                }
                // A string token
                '"' => {

                },
                _ => {
                    Err(Error::new(ErrorKind::InvalidData, "Invalid JSON"))
                }
            }
        } else {
            Ok(None)
        }
    }
}