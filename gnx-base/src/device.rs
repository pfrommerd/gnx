//! Static interface for device execution and explicit data transfers.
use std::{future::Future, io, sync::Arc};

use gnx_expr::{RootExpr, Value};

use crate::buffer::{DataHandle, HostBuffer, TransferPolicy};

#[derive(Clone, Copy, Debug, Default)]
pub struct ExecOpts {
    pub transfers: TransferPolicy,
}

pub trait Device: Send + Sync {
    type Outputs: IntoIterator<Item = Value>;

    /// Make storage available to this device, respecting the caller's copy policy.
    fn upload(
        &self,
        data: &DataHandle,
        policy: TransferPolicy,
    ) -> impl Future<Output = io::Result<DataHandle>> + Send;

    /// Return packed host storage whose bytes are ready to read.
    fn download(&self, data: &DataHandle) -> impl Future<Output = io::Result<HostBuffer>> + Send;

    /// Execute a lowered expression. Arguments are captures followed by explicit
    /// inputs; results follow `Expr::outputs()` order, including forwarded parameters.
    fn execute(
        &self,
        expr: Arc<RootExpr>,
        args: impl IntoIterator<Item = Value>,
        opts: ExecOpts,
    ) -> impl Future<Output = io::Result<Self::Outputs>> + Send;
}

/// Run a blocking native device operation off the polling thread.
pub async fn run_blocking<T: Send + 'static>(
    name: &'static str,
    work: impl FnOnce() -> io::Result<T> + Send + 'static,
) -> io::Result<T> {
    let (tx, rx) = futures_channel::oneshot::channel();
    std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let _ = tx.send(work());
        })?;
    rx.await
        .map_err(|_| io::Error::other(format!("{name} worker stopped without a result")))?
}
