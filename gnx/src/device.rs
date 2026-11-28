//! Device selection and execution.
pub use gnx_math::device::*;

use gnx_base::{DataHandle, HostBuffer, TransferPolicy, cache::Cache};
use gnx_expr::{RootExpr, Value};
use std::{future::Future, io, str::FromStr, sync::Arc};

/// A cheaply cloneable handle to a device provided by an enabled backend.
#[derive(Clone)]
pub enum DeviceHandle {
    #[cfg(feature = "xla")]
    Xla(Arc<gnx_xla::XlaDevice>),
    #[cfg(feature = "torch")]
    Torch(Arc<gnx_torch::TorchDevice>),
}

impl DeviceHandle {
    /// Open `xla/cpu`, `torch/cpu`, or `<backend>/cuda:<ordinal>` with the given cache.
    /// `default` tries XLA CUDA device 0, then XLA CPU. Explicit requests never fall back.
    pub fn open(spec: &str, cache: Cache) -> io::Result<Self> {
        let spec = if spec == "default" {
            #[cfg(feature = "xla")]
            if let Ok(device) = gnx_xla::XlaDevice::cuda(0, cache.clone()) {
                return Ok(Self::Xla(Arc::new(device)));
            }
            "xla/cpu"
        } else {
            spec
        };
        let invalid = || {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "invalid device {spec:?}; expected default, xla/cpu, torch/cpu, xla/cuda:N, or torch/cuda:N"
                ),
            )
        };
        let (backend, target) = spec.split_once('/').ok_or_else(invalid)?;
        let ordinal = match target {
            "cpu" => None,
            _ => {
                let n = target
                    .strip_prefix("cuda:")
                    .filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
                    .ok_or_else(invalid)?;
                Some(n.parse::<usize>().map_err(|_| invalid())?)
            }
        };
        match (backend, ordinal, cache) {
            #[cfg(feature = "xla")]
            ("xla", gpu, cache) => Ok(Self::Xla(Arc::new(match gpu {
                Some(n) => gnx_xla::XlaDevice::cuda(n, cache)?,
                None => gnx_xla::XlaDevice::cpu(cache)?,
            }))),
            #[cfg(feature = "torch")]
            ("torch", gpu, cache) => Ok(Self::Torch(Arc::new(match gpu {
                Some(n) => gnx_torch::TorchDevice::cuda(n, cache)?,
                None => gnx_torch::TorchDevice::cpu(cache)?,
            }))),
            #[cfg(not(feature = "xla"))]
            ("xla", _, _) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "XLA backend not built; enable the xla feature",
            )),
            #[cfg(not(feature = "torch"))]
            ("torch", _, _) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Torch backend not built; enable the torch feature",
            )),
            _ => Err(invalid()),
        }
    }
}

/// Parsing uses an in-memory compilation cache.
impl FromStr for DeviceHandle {
    type Err = io::Error;
    fn from_str(spec: &str) -> io::Result<Self> {
        Self::open(spec, Cache::in_memory())
    }
}

macro_rules! dispatch {
    ($handle:expr, $method:ident($($arg:expr),*)) => {
        match *$handle {
            #[cfg(feature = "xla")]
            DeviceHandle::Xla(ref d) => d.$method($($arg),*).await,
            #[cfg(feature = "torch")]
            DeviceHandle::Torch(ref d) => d.$method($($arg),*).await,
        }
    };
}

#[cfg_attr(not(any(feature = "xla", feature = "torch")), allow(unused_variables))]
impl Device for DeviceHandle {
    type Outputs = Vec<Value>;

    async fn upload(&self, data: &DataHandle, policy: TransferPolicy) -> io::Result<DataHandle> {
        dispatch!(self, upload(data, policy))
    }

    async fn download(&self, data: &DataHandle) -> io::Result<HostBuffer> {
        dispatch!(self, download(data))
    }

    fn execute(
        &self,
        expr: Arc<RootExpr>,
        args: impl IntoIterator<Item = Value>,
        opts: ExecOpts,
    ) -> impl Future<Output = io::Result<Self::Outputs>> + Send {
        let args: Vec<_> = args.into_iter().collect();
        async move { dispatch!(self, execute(expr, args, opts)) }
    }
}
