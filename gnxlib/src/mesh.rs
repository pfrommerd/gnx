use std::future::Future;
use std::net::SocketAddr;
use std::sync::OnceLock;

use gnx::buffer::TransferPolicy;
use gnx::shard::{self, Sharding};
use gnx::{Eval, ExecOpts};
use gnx_xla::{Mesh, WorkerInfo};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tokio::runtime::{Builder, Runtime};
use uuid::Uuid;

use crate::tracer::PyArray;

#[pyclass(name = "Mesh", module = "gnxlib.xla", frozen)]
pub struct PyMesh(Mesh);

#[pyclass(name = "MeshWorker", module = "gnxlib.xla", frozen)]
pub struct PyMeshWorker(WorkerInfo);

#[pymethods]
impl PyMeshWorker {
    #[getter]
    fn address(&self) -> String {
        self.0.address.to_string()
    }

    #[getter]
    fn host_uuid(&self) -> Uuid {
        self.0.host.0
    }

    #[getter]
    fn allocation_owner_uuid(&self) -> Uuid {
        self.0.allocation_owner.0
    }

    #[getter]
    fn device_ordinal(&self) -> u64 {
        self.0.device_ordinal
    }
}

#[pymethods]
impl PyMesh {
    #[staticmethod]
    async fn connect(axis: String, workers: Vec<String>) -> PyResult<Self> {
        let addresses = workers
            .iter()
            .map(|address| {
                address.parse::<SocketAddr>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "invalid worker address: {address:?}; expected IP:port"
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        tokio_task(async move { Ok(Self(Mesh::connect(axis, addresses).await?)) }).await
    }

    #[getter]
    fn axis(&self) -> &str {
        self.0.axis()
    }

    #[getter]
    fn workers(&self) -> Vec<PyMeshWorker> {
        self.0.workers().cloned().map(PyMeshWorker).collect()
    }

    /// Evaluate, placing inputs where the compiled program needs them.
    /// Shards already in the right place are reused without transferring data.
    async fn eval(&self, roots: Py<PyAny>) -> PyResult<()> {
        let roots = Python::attach(|py| crate::parse_roots(roots.bind(py)))?;
        let mesh = self.0.clone();
        tokio_task(async move {
            mesh.eval(
                roots.as_ref(),
                ExecOpts {
                    transfers: TransferPolicy::AllowCopy,
                },
            )
            .await
            .map_err(crate::eval_error)
        })
        .await
    }

    /// Explicitly place a concrete array on the mesh. With no axes, replicate it.
    #[pyo3(signature = (array, *, axes = None))]
    async fn upload(
        &self,
        array: Py<PyArray>,
        axes: Option<Vec<Vec<String>>>,
    ) -> PyResult<PyArray> {
        let array = Python::attach(|py| array.bind(py).borrow().0.clone());
        if array.data().is_none() {
            return Err(PyValueError::new_err(
                "mesh.upload requires a concrete array",
            ));
        }
        let placed = match axes {
            Some(axes) => constrain(&PyArray(array), axes)?.0,
            None => shard::replicate(&array),
        };
        let mesh = self.0.clone();
        tokio_task(async move {
            mesh.eval(
                &placed,
                ExecOpts {
                    transfers: TransferPolicy::AllowCopy,
                },
            )
            .await
            .map_err(crate::eval_error)?;
            Ok(PyArray(placed))
        })
        .await
    }

    /// Explicitly gather a concrete array as packed little-endian host bytes.
    async fn to_host_bytes(&self, array: Py<PyArray>) -> PyResult<Py<PyBytes>> {
        let data = Python::attach(|py| array.bind(py).borrow().0.data().cloned())
            .ok_or_else(|| PyValueError::new_err("array has not been evaluated"))?;
        let mesh = self.0.clone();
        let host = tokio_task(async move { Ok(mesh.download(&data).await?) }).await?;
        Ok(Python::attach(|py| PyBytes::new(py, host.bytes()).unbind()))
    }
}

/// Constrain the placement of an array in a Mesh computation.
#[pyfunction]
pub fn constrain(array: &PyArray, axes: Vec<Vec<String>>) -> PyResult<PyArray> {
    let spec = Sharding::new(axes).map_err(shard_error)?;
    Ok(PyArray(
        shard::constrain(&array.0, spec).map_err(shard_error)?,
    ))
}

/// Constrain an array to be replicated on the mesh.
#[pyfunction]
pub fn replicate(array: &PyArray) -> PyArray {
    PyArray(shard::replicate(&array.0))
}

fn shard_error(error: gnx::expr::LoweringError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

async fn tokio_task<T: Send + 'static>(
    task: impl Future<Output = PyResult<T>> + Send + 'static,
) -> PyResult<T> {
    // The Python event loop polls the PyO3 future, but tarpc needs Tokio's I/O
    // and timer drivers. Keep them alive across separate asyncio.run() calls.
    static RUNTIME: OnceLock<Result<Runtime, std::io::Error>> = OnceLock::new();
    let runtime = RUNTIME
        .get_or_init(|| {
            Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .thread_name("gnx-mesh")
                .build()
        })
        .as_ref()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    runtime
        .spawn(task)
        .await
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
}
