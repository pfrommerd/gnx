//! Single-controller distributed CPU XLA over tarpc.
//!
//! Each worker owns one addressable device and retains the allocations it
//! produces. Shardy/SPMD generates the local program. This first runtime
//! rejects programs that require cross-worker collectives.

mod program;
mod rpc;

use std::collections::{BTreeMap, HashSet};
use std::future::Future;
use std::io::{self, ErrorKind};
use std::net::SocketAddr;
use std::sync::Arc;

use futures::future::join_all;
use gnx_expr::pass::{Dce, FnPass, Pipeline};
use gnx_expr::{Expr, Value};
use gnx_math::{
    AllocationId, AllocationRef, DataHandle, Device, ExecOpts, ExecutionBoundary, HostBuffer,
    HostId, Lease, MemoryLocation, OwnerId, Shard, ShardStorage, TransferPolicy,
};
use gnx_transforms::ExpandTransforms;
use tokio::runtime::Handle;
use uuid::Uuid;

use crate::XlaDevice;
use program::{Layout, Placed, Placement, Program, Tensor};
pub use rpc::Worker;
use rpc::{Argument, MAX_TRANSFER, WorkerRpcClient, request_context, response};

/// Device and session identity reported by a connected worker.
#[derive(Clone, Debug)]
pub struct WorkerInfo {
    pub address: SocketAddr,
    pub host: HostId,
    pub allocation_owner: OwnerId,
    /// Logical CPU ordinal in the worker's runtime, not a hardware UUID.
    pub device_ordinal: u64,
}

#[derive(Clone)]
struct Peer {
    client: WorkerRpcClient,
    info: WorkerInfo,
}

/// One logical GNX device backed by an ordered set of remote workers.
#[derive(Clone)]
pub struct Mesh(Arc<Controller>);

struct Controller {
    axis: String,
    workers: Vec<Peer>,
    runtime: Handle,
}

impl Mesh {
    /// Connect from a Tokio runtime. Address order fixes the mesh coordinate
    /// of each worker. This protocol is for trusted peers.
    pub async fn connect(
        axis: impl Into<String>,
        addresses: impl IntoIterator<Item = SocketAddr>,
    ) -> io::Result<Self> {
        let axis = axis.into();
        let addresses: Vec<_> = addresses.into_iter().collect();
        if !gnx_math::shard::valid_axis(&axis)
            || axis.len() > 64
            || !(1..=256).contains(&addresses.len())
            || addresses.iter().collect::<HashSet<_>>().len() != addresses.len()
        {
            return Err(invalid(
                "mesh needs a valid axis and 1 to 256 distinct worker addresses",
            ));
        }
        let runtime = Handle::try_current().map_err(io::Error::other)?;
        let workers = join_all(addresses.into_iter().map(|address| async move {
            let (client, identity) = rpc::connect(address).await?;
            let info = WorkerInfo {
                address,
                host: HostId(identity.host),
                allocation_owner: OwnerId(identity.owner),
                device_ordinal: identity.ordinal,
            };
            Ok::<_, io::Error>(Peer { client, info })
        }))
        .await
        .into_iter()
        .collect::<io::Result<_>>()?;
        Ok(Self(Arc::new(Controller {
            axis,
            workers,
            runtime,
        })))
    }

    pub fn axis(&self) -> &str {
        &self.0.axis
    }
    pub fn workers(&self) -> impl ExactSizeIterator<Item = &WorkerInfo> {
        self.0.workers.iter().map(|worker| &worker.info)
    }

    /// Compile the global expression with Shardy/SPMD at every worker. The
    /// returned handle retains the local executables for repeat runs.
    pub async fn compile(&self, expr: &Expr) -> io::Result<MeshExecutable> {
        let (projected, boundary) = ExecutionBoundary::project(expr)?;
        let compiled = if let Some(projected) = projected {
            let (axis, workers) = (self.axis().to_owned(), self.0.workers.len());
            // Workers accept only the closed wire program, so the final stage
            // produces a `Program` rather than HLO.
            let pipeline = Pipeline::new()
                .with(ExpandTransforms)
                .with(Dce)
                .finalize(FnPass::new("mesh.lower", move |expr: Expr| {
                    Program::lower(&expr, &axis, workers)
                }));
            let program = pipeline.run(projected).map_err(crate::native::pass_error)?;
            let placement = program.placement()?;
            let kernels = join_all(self.0.workers.iter().map(|worker| async {
                let id = response(
                    worker
                        .client
                        .compile(request_context(), program.clone())
                        .await,
                )?;
                Ok::<_, io::Error>(RemoteKernel {
                    id,
                    client: worker.client.clone(),
                    runtime: self.0.runtime.clone(),
                })
            }))
            .await
            .into_iter()
            .collect::<io::Result<Vec<_>>>()?;
            Some((placement, kernels))
        } else {
            None
        };
        Ok(MeshExecutable {
            mesh: self.clone(),
            signature: expr.clone(),
            boundary,
            compiled,
        })
    }

    /// Explicitly assemble a packed host copy. This may fetch remote shards;
    /// inspecting or forwarding their DataHandle never does.
    pub async fn download(&self, data: &DataHandle) -> io::Result<HostBuffer> {
        if let Ok(host) = data.to_host() {
            return Ok(host);
        }
        if let Some(reference) = data.whole_allocation() {
            if self.peer(reference.id().owner).is_some() || data.shards().is_empty() {
                return HostBuffer::new(data.info(), self.fetch(reference).await?);
            }
        }
        let global = Tensor::of(data.array_info())?;
        let leading = global.dims.first().copied().unwrap_or(1);
        let mut groups = BTreeMap::<(usize, usize), Vec<usize>>::new();
        for (index, shard) in data.shards().iter().enumerate() {
            if !shard
                .bounds
                .iter()
                .zip(&global.dims)
                .skip(1)
                .all(|(bounds, dim)| bounds.start == 0 && bounds.end == *dim)
            {
                return Err(unsupported("mesh download requires leading-axis shards"));
            }
            let range = shard.bounds.first().cloned().unwrap_or(0..1);
            groups
                .entry((range.start, range.end))
                .or_default()
                .push(index);
        }
        let mut cursor = 0;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(global.byte_len()?)
            .map_err(io::Error::other)?;
        for ((start, end), replicas) in groups {
            if start != cursor {
                return Err(invalid("mesh shards do not cover the requested array"));
            }
            let mut last_error = None;
            let mut packed = None;
            for index in replicas {
                match self.download_shard(data, index).await {
                    Ok(shard) => {
                        packed = Some(shard);
                        break;
                    }
                    Err(err) => last_error = Some(err),
                }
            }
            let packed = packed.ok_or_else(|| {
                last_error.unwrap_or_else(|| unsupported("mesh has no readable replica"))
            })?;
            bytes.extend_from_slice(packed.bytes());
            cursor = end;
        }
        if cursor != leading {
            return Err(invalid("mesh shards do not cover the requested array"));
        }
        HostBuffer::new(data.info(), bytes)
    }

    async fn download_shard(&self, data: &DataHandle, index: usize) -> io::Result<HostBuffer> {
        let value = data.shard_data(index)?;
        if let Ok(host) = value.to_host() {
            return Ok(host);
        }
        let reference = value
            .single_allocation()
            .ok_or_else(|| unsupported("mesh shard has no remote reference"))?;
        HostBuffer::new(value.info(), self.fetch(reference).await?)
    }

    fn peer(&self, owner: OwnerId) -> Option<&Peer> {
        self.0
            .workers
            .iter()
            .find(|worker| worker.info.allocation_owner == owner)
    }

    async fn fetch(&self, reference: &AllocationRef) -> io::Result<Vec<u8>> {
        let id = reference.id();
        let worker = self
            .peer(id.owner)
            .ok_or_else(|| unsupported("mesh is not connected to this allocation owner"))?;
        response(worker.client.download(request_context(), id.id).await)
    }

    async fn inputs<'a>(
        &self,
        args: impl Iterator<Item = io::Result<&'a Value>>,
        placement: &Placement,
        policy: TransferPolicy,
    ) -> io::Result<Vec<Vec<Argument>>> {
        let mut inputs = vec![Vec::new(); self.0.workers.len()];
        let mut transfer = vec![0_usize; self.0.workers.len()];
        for (data, placed) in args.zip(&placement.inputs) {
            let data = data?
                .downcast::<DataHandle>()
                .map_err(|_| invalid("mesh requires concrete arrays"))?;
            if data.array_info() != &placed.tensor.info() {
                return Err(invalid("mesh argument metadata mismatch"));
            }
            let refs: Vec<_> =
                self.0
                    .workers
                    .iter()
                    .enumerate()
                    .map(|(index, peer)| {
                        let bounds = placed.bounds(index);
                        if is_full(data, &bounds) {
                            if let Some(reference) = data.whole_allocation().filter(|reference| {
                                reference.id().owner == peer.info.allocation_owner
                            }) {
                                return Some(reference.id().id);
                            }
                        }
                        data.shards()
                            .iter()
                            .filter(|shard| shard.bounds == bounds)
                            .find_map(|shard| {
                                let reference = match &shard.storage {
                                    ShardStorage::Reference(reference)
                                    | ShardStorage::Local {
                                        allocation: Some(reference),
                                        ..
                                    } => reference,
                                    _ => return None,
                                };
                                let id = reference.id();
                                (id.owner == peer.info.allocation_owner).then_some(id.id)
                            })
                    })
                    .collect();
            let host = if refs.iter().any(Option::is_none) {
                if policy == TransferPolicy::RequireZeroCopy {
                    return Err(unsupported("mesh input requires an upload or reshard"));
                }
                Some(self.download(data).await?)
            } else {
                None
            };
            for (index, reference) in refs.into_iter().enumerate() {
                if let Some(id) = reference {
                    inputs[index].push(Argument::Reference(id));
                } else {
                    let range = placed.byte_range(index)?;
                    if range.len() > MAX_TRANSFER - transfer[index] {
                        return Err(invalid("mesh input payload exceeds the RPC transfer limit"));
                    }
                    transfer[index] += range.len();
                    inputs[index].push(Argument::Packed(
                        host.as_ref().unwrap().bytes()[range].to_vec(),
                    ));
                }
            }
        }
        Ok(inputs)
    }

    fn retain(&self, peer: &Peer, id: Uuid) -> AllocationRef {
        let client = peer.client.clone();
        let runtime = self.0.runtime.clone();
        AllocationRef::new(
            AllocationId {
                owner: peer.info.allocation_owner,
                id,
            },
            Lease::on_drop(move || {
                runtime.spawn(async move {
                    let _ = client.release(request_context(), id).await;
                });
            }),
        )
    }

    fn output(
        &self,
        placed: &Placed,
        references: impl Iterator<Item = AllocationRef>,
    ) -> io::Result<Value> {
        let shards = self
            .0
            .workers
            .iter()
            .zip(references)
            .enumerate()
            .map(|(index, (worker, reference))| Shard {
                bounds: placed.bounds(index),
                replica: if placed.layout == Layout::Replicated {
                    index as u32
                } else {
                    0
                },
                location: MemoryLocation::host(worker.info.host),
                storage: ShardStorage::Reference(reference),
            })
            .collect();
        Ok(Value::new(DataHandle::from_shards(
            placed.tensor.info(),
            shards,
            None,
        )?))
    }
}

fn is_full(data: &DataHandle, bounds: &[std::ops::Range<usize>]) -> bool {
    bounds
        .iter()
        .zip(data.array_info().shape().dims())
        .all(|(range, dim)| range.start == 0 && *dim == gnx_math::Dim::Fixed(range.end))
}

struct RemoteKernel {
    id: Uuid,
    client: WorkerRpcClient,
    runtime: Handle,
}

impl Drop for RemoteKernel {
    fn drop(&mut self) {
        let client = self.client.clone();
        let id = self.id;
        self.runtime.spawn(async move {
            let _ = client.release_kernel(request_context(), id).await;
        });
    }
}

pub struct MeshExecutable {
    mesh: Mesh,
    signature: Expr,
    boundary: ExecutionBoundary,
    compiled: Option<(Placement, Vec<RemoteKernel>)>,
}

impl MeshExecutable {
    pub async fn run(&self, args: &[Value], opts: ExecOpts) -> io::Result<Vec<Value>> {
        ExecutionBoundary::validate_arguments(&self.signature, args)?;
        let Some((placement, kernels)) = &self.compiled else {
            return self.boundary.finish(args, &[]);
        };
        let inputs = self
            .mesh
            .inputs(self.boundary.arguments(args), placement, opts.transfers)
            .await?;
        let outputs = join_all(self.mesh.0.workers.iter().zip(kernels).zip(inputs).map(
            |((worker, kernel), args)| async move {
                let ids = response(
                    worker
                        .client
                        .execute(request_context(), kernel.id, args)
                        .await,
                )?;
                let outputs: Vec<_> = ids
                    .into_iter()
                    .map(|id| self.mesh.retain(worker, id))
                    .collect();
                if outputs.len() != placement.outputs.len() {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "mesh worker returned the wrong output count",
                    ));
                }
                Ok(outputs.into_iter())
            },
        ))
        .await;
        let mut outputs = outputs.into_iter().collect::<io::Result<Vec<_>>>()?;
        let computed = placement
            .outputs
            .iter()
            .map(|placed| {
                self.mesh.output(
                    placed,
                    outputs
                        .iter_mut()
                        .map(|worker| worker.next().expect("validated mesh output count")),
                )
            })
            .collect::<io::Result<Vec<_>>>()?;
        self.boundary.finish(args, &computed)
    }
}

impl Device for Mesh {
    type Outputs = Vec<Value>;
    fn execute(
        &self,
        expr: Expr,
        args: impl IntoIterator<Item = Value>,
        opts: ExecOpts,
    ) -> impl Future<Output = io::Result<Self::Outputs>> + Send {
        let args: Vec<_> = args.into_iter().collect();
        let mesh = self.clone();
        async move { mesh.compile(&expr).await?.run(&args, opts).await }
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}
fn unsupported(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::Unsupported, message.into())
}
