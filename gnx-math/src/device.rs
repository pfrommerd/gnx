use std::future::Future;
use std::io;

use gnx_expr::{Expr, Value};

use crate::buffer::TransferPolicy;

#[derive(Clone, Copy, Debug, Default)]
pub struct ExecOpts {
    pub transfers: TransferPolicy,
}

/// Executes a lowered expression. Arguments are captures followed by explicit
/// inputs; values are returned in `Expr::outputs()` order, including forwarded
/// parameters. This interface does not apply cell updates or reconstruct cell handles.
pub trait Device: Send + Sync {
    type Outputs: IntoIterator<Item = Value>;

    fn execute(
        &self,
        expr: Expr,
        args: impl IntoIterator<Item = Value>,
        opts: ExecOpts,
    ) -> impl Future<Output = io::Result<Self::Outputs>> + Send;
}
