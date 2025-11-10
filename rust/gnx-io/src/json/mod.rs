use logos::{Logos, Span};
use ordered_float::NotNan;

use std::io::Read;

use crate::{GraphDeserializer, DesVisitor};

#[derive(Logos, Debug, Clone, Eq, PartialEq)]
#[logos(skip r"[ \t\n\r]+")]
pub enum Token<'s> {
    #[regex(r"//.*\n")]
    Comment,
    #[regex(r"/\*.*\*/")]
    BlockComment,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("null")]
    Null,
    #[token(r"(true)|(false)", |x| x.slice() == "true")]
    Bool(bool),
    #[regex(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", |lex| lex.slice().parse::<NotNan<f64>>().unwrap())]
    Number(NotNan<f64>),
    #[regex(r#""([^"\\\x00-\x1F]|\\(["\\bnfrt/]|u[a-fA-F0-9]{4}))*""#, |lex| lex.slice())]
    String(&'s str),
}

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

pub struct TokenStream<'de, I: Iterator<Item = (Token<'de>, Span)>> {
    inner: I,
    peeked: Option<(Token<'de>, Span)>,
}

impl<'de, I: Iterator<Item = (Token<'de>, Span)>> TokenStream<'de, I> {
    pub fn new(mut inner: I) -> Self {
        let peeked = inner.next();
        Self { inner, peeked }
    }
    pub fn next(&mut self) -> Option<(Token<'de>, Span)> {
        let mut next = self.inner.next();
        std::mem::swap(&mut self.peeked, &mut next);
        next
    }
    pub fn peek(&mut self) -> Option<(Token<'de>, Span)> {
        self.peeked.clone()
    }
    pub fn is_empty(&self) -> bool {
        self.peeked.is_none()
    }

    pub fn parse_string(&mut self) -> Result<&'de str, JsonError> {
        todo!()
    }
}