use crate::array::{Data, Item};
use crate::expr::Expr;
use super::value::ValueInfo;

use std::borrow::Cow;
use std::collections::BTreeMap;

// The static attributes of an operation.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Attr {
    Scalar(Item),
    // A literal array value.
    Literal(Data<'static>),
    String(Cow<'static, str>),
    Info(ValueInfo),
    Expr(Expr),
    List(Box<AttrList>),
    Map(Box<AttrMap>),
}


pub type AttrMap = BTreeMap<String, Attr>;
pub type AttrList = Vec<Attr>;
