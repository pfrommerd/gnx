use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use gnx_math::{ArrayInfo, DataHandle, HostBuffer};
use serde::{Deserialize, Serialize};
use tarpc::{
    context,
    serde_transport::tcp,
    server::{BaseChannel, Channel},
    tokio_serde::formats::Bincode,
};
use tokio::task::{JoinHandle, JoinSet};
use uuid::Uuid;

use super::program::Program;
use super::{XlaDevice, invalid};

const VERSION: u32 = 1;
const MAX_FRAME: usize = 64 * 1024 * 1024;
pub(super) const MAX_TRANSFER: usize = MAX_FRAME - 1024 * 1024;

type Reply<T> = Result<T, String>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Identity {
    pub version: u32,
    pub owner: Uuid,
    pub host: Uuid,
    pub ordinal: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum Argument {
    Packed(Vec<u8>),
    Reference(Uuid),
}

#[tarpc::service]
pub(super) trait WorkerRpc {
    async fn identify() -> Identity;
    async fn compile(program: Program) -> Reply<Uuid>;
    async fn execute(kernel: Uuid, args: Vec<Argument>) -> Reply<Vec<Uuid>>;
    async fn download(allocation: Uuid) -> Reply<Vec<u8>>;
    async fn release(allocation: Uuid);
    async fn release_kernel(kernel: Uuid);
}

pub(super) fn request_context() -> context::Context {
    let mut context = context::current();
    context.deadline = Instant::now() + Duration::from_secs(120);
    context
}

pub(super) fn response<T>(response: Result<Reply<T>, tarpc::client::RpcError>) -> io::Result<T> {
    response
        .map_err(io::Error::other)?
        .map_err(io::Error::other)
}

pub(super) async fn connect(address: SocketAddr) -> io::Result<(WorkerRpcClient, Identity)> {
    let mut transport = tcp::connect(address, Bincode::default);
    transport.config_mut().max_frame_length(MAX_FRAME);
    let client = WorkerRpcClient::new(tarpc::client::Config::default(), transport.await?).spawn();
    let id = client
        .identify(request_context())
        .await
        .map_err(io::Error::other)?;
    if id.version != VERSION {
        return Err(invalid("unsupported GNX worker protocol version"));
    }
    Ok((client, id))
}

#[derive(Clone)]
struct Session(Arc<State>);

struct State {
    identity: Identity,
    device: XlaDevice,
    kernels: Mutex<HashMap<Uuid, Arc<Kernel>>>,
    allocations: Mutex<HashMap<Uuid, DataHandle>>,
}

struct Kernel {
    executable: xla::PjRtLoadedExecutable,
    inputs: Vec<ArrayInfo>,
    outputs: Vec<ArrayInfo>,
}

impl Session {
    fn new(device: XlaDevice) -> Self {
        let identity = Identity {
            version: VERSION,
            host: device.host().0,
            ordinal: device.ordinal(),
            owner: Uuid::new_v4(),
        };
        Self(Arc::new(State {
            identity,
            device,
            kernels: Mutex::default(),
            allocations: Mutex::default(),
        }))
    }
}

impl WorkerRpc for Session {
    async fn identify(self, _: context::Context) -> Identity {
        self.0.identity.clone()
    }

    async fn compile(self, _: context::Context, program: Program) -> Reply<Uuid> {
        let device = self.0.device.clone();
        let kernel = blocking(move || {
            let placement = program.placement()?;
            for placed in placement.inputs.iter().chain(&placement.outputs) {
                if placed.local().byte_len()? > MAX_TRANSFER {
                    return Err(invalid("mesh shard exceeds the RPC transfer limit"));
                }
            }
            let (bytes, placement) = program.partition()?;
            let executable = device.compile_partition(&bytes)?;
            let infos = |values: Vec<super::program::Placed>| {
                values
                    .into_iter()
                    .map(|placed| placed.local().info())
                    .collect()
            };
            Ok(Kernel {
                executable,
                inputs: infos(placement.inputs),
                outputs: infos(placement.outputs),
            })
        })
        .await?;
        let id = Uuid::new_v4();
        self.0.kernels.lock().unwrap().insert(id, Arc::new(kernel));
        Ok(id)
    }

    async fn execute(
        self,
        _: context::Context,
        kernel: Uuid,
        args: Vec<Argument>,
    ) -> Reply<Vec<Uuid>> {
        let kernel = self
            .0
            .kernels
            .lock()
            .unwrap()
            .get(&kernel)
            .cloned()
            .ok_or("unknown mesh kernel")?;
        if args.len() != kernel.inputs.len() {
            return Err("wrong mesh kernel input count".into());
        }
        let inputs = {
            let allocations = self.0.allocations.lock().unwrap();
            args.into_iter()
                .zip(&kernel.inputs)
                .map(|(argument, info)| match argument {
                    Argument::Packed(bytes) => Ok(DataHandle::from_host_on(
                        HostBuffer::new(info.clone(), bytes).map_err(|err| err.to_string())?,
                        self.0.device.host(),
                    )),
                    Argument::Reference(id) => {
                        let allocation = allocations.get(&id).ok_or("unknown mesh allocation")?;
                        if info != allocation.array_info() {
                            return Err("mesh allocation metadata mismatch".into());
                        }
                        Ok(allocation.clone())
                    }
                })
                .collect::<Reply<Vec<_>>>()?
        };
        let device = self.0.device.clone();
        let outputs =
            blocking(move || device.run_partition(&kernel.executable, &inputs, &kernel.outputs))
                .await?;
        let mut allocations = self.0.allocations.lock().unwrap();
        Ok(outputs
            .into_iter()
            .map(|value| {
                let id = Uuid::new_v4();
                allocations.insert(id, value);
                id
            })
            .collect())
    }

    async fn download(self, _: context::Context, allocation: Uuid) -> Reply<Vec<u8>> {
        let value = self
            .0
            .allocations
            .lock()
            .unwrap()
            .get(&allocation)
            .cloned()
            .ok_or("unknown mesh allocation")?;
        let device = self.0.device.clone();
        blocking(move || device.download(&value).map(|host| host.into_bytes())).await
    }

    async fn release(self, _: context::Context, allocation: Uuid) {
        let value = self.0.allocations.lock().unwrap().remove(&allocation);
        drop(value);
    }

    async fn release_kernel(self, _: context::Context, kernel: Uuid) {
        let value = self.0.kernels.lock().unwrap().remove(&kernel);
        drop(value);
    }
}

async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> io::Result<T> + Send + 'static,
) -> Reply<T> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|err| err.to_string())?
        .map_err(|err| err.to_string())
}

/// A worker listener. Dropping it closes its connections and releases sessions.
/// Use only on a trusted network; this TCP transport does not authenticate peers.
pub struct Worker {
    address: SocketAddr,
    task: JoinHandle<()>,
}

impl Worker {
    pub async fn bind(address: SocketAddr, device: XlaDevice) -> io::Result<Self> {
        let mut incoming = tcp::listen(address, Bincode::default).await?;
        incoming.config_mut().max_frame_length(MAX_FRAME);
        let address = incoming.local_addr();
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            loop {
                tokio::select! {
                    connection = incoming.next() => {
                        let Some(connection) = connection else { break; };
                        let Ok(connection) = connection else { continue; };
                        let session = Session::new(device.clone());
                        connections.spawn(async move {
                            BaseChannel::with_defaults(connection).execute(session.serve())
                                .for_each_concurrent(16, |request| request).await;
                        });
                    }
                    completed = connections.join_next(), if !connections.is_empty() => { drop(completed); }
                }
            }
        });
        Ok(Self { address, task })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.address
    }

    /// Drive the server until its listener closes.
    pub async fn wait(&mut self) -> io::Result<()> {
        (&mut self.task).await.map_err(io::Error::other)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.task.abort();
    }
}
