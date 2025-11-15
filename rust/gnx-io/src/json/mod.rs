mod source;
mod value;
mod parser;

pub use source::*;
pub use value::*;
pub use parser::*;

#[derive(Debug)]
pub enum JsonError {
    UnterminatedString,
    InvalidEscape,
    InvalidNumber,
    InvalidBase64(base64::DecodeError),
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

impl From<base64::DecodeError> for JsonError {
    fn from(e: base64::DecodeError) -> Self {
        Self::InvalidBase64(e)
    }
}

impl From<std::num::ParseIntError> for JsonError {
    fn from(_: std::num::ParseIntError) -> Self {
        Self::InvalidNumber
    }
}

impl From<std::num::ParseFloatError> for JsonError {
    fn from(_: std::num::ParseFloatError) -> Self {
        Self::InvalidNumber
    }   
}

type Result<T> = std::result::Result<T, JsonError>;