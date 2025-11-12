use std::io::Result;

mod lexer;

use lexer::*;

use crate::{GraphDeserializer, DesVisitor};
#[derive(Debug)]
pub enum JsonError {
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

pub trait JsonParser<'s> {
    fn next<'p>(&'p mut self) -> Result<Option<ParsedToken<'s, 'p>>>;
}

// impl<'s, P: JsonParser<'s>> GraphDeserializer<'s> for P {

// }