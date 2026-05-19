use crate::*;
use crate::util::impl_lifetime_free;

use gnx_expr::trace::{TraceCellRef, TraceRef};

impl_lifetime_free!(TraceRef);
impl_lifetime_free!(TraceCellRef);

impl_leaf!(TraceRef);
impl_leaf!(TraceCellRef);

// graph implementations for Array/Device/ArrayRef


// serialization and deserialization for Array and ArrayRef