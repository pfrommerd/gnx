mod parser;
mod source;
mod value;
mod writer;

pub use parser::*;
pub use source::*;
pub use value::*;
pub use writer::*;

#[derive(Debug)]
pub enum TomlError {
    UnterminatedString,
    InvalidEscape,
    InvalidNumber,
    InvalidDateTime,
    InvalidKey,
    DuplicateKey(String),
    RootMustBeTable,
    Unsupported(&'static str),
    UnexpectedEOF,
    Unexpected(char),
    Expected(&'static str),
    ReadError(std::io::Error),
    Other(String),
}

impl std::fmt::Display for TomlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnterminatedString => write!(f, "unterminated TOML string"),
            Self::InvalidEscape => write!(f, "invalid TOML escape sequence"),
            Self::InvalidNumber => write!(f, "invalid TOML number"),
            Self::InvalidDateTime => write!(f, "invalid TOML date/time"),
            Self::InvalidKey => write!(f, "invalid TOML key"),
            Self::DuplicateKey(key) => write!(f, "duplicate TOML key `{key}`"),
            Self::RootMustBeTable => write!(f, "a TOML document root must be a table"),
            Self::Unsupported(msg) => write!(f, "unsupported TOML value: {msg}"),
            Self::UnexpectedEOF => write!(f, "unexpected end of TOML input"),
            Self::Unexpected(ch) => write!(f, "unexpected TOML character `{ch}`"),
            Self::Expected(msg) => write!(f, "expected {msg}"),
            Self::ReadError(err) => write!(f, "{err}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for TomlError {}

impl gnx_graph::Error for TomlError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Self::Other(msg.to_string())
    }
}

impl From<std::io::Error> for TomlError {
    fn from(e: std::io::Error) -> Self {
        Self::ReadError(e)
    }
}

impl From<std::num::ParseIntError> for TomlError {
    fn from(_: std::num::ParseIntError) -> Self {
        Self::InvalidNumber
    }
}

impl From<std::num::ParseFloatError> for TomlError {
    fn from(_: std::num::ParseFloatError) -> Self {
        Self::InvalidNumber
    }
}

type Result<T> = std::result::Result<T, TomlError>;
