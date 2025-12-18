use crate::array::{Item, Data};
use crate::value::ValueInfo;
use crate::expr::Expr;

use std::collections::BTreeMap;
use std::borrow::Cow;

// The static attributes of an operation.
#[derive(Clone, Hash, PartialEq, Eq)]
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


// inline the cases for 1-3 attrs

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct AttrMap {
}
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct AttrList {
}