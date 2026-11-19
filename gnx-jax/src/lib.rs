//! JAX-specific device discovery, without a shared backend registry.
//!
//! Rust embedders initialize Python before discovery. Await `Device::execute`
//! outside an attached Python scope (or detach before blocking on it).
//! The optional `jax-tests` feature exercises a real CPU installation of JAX.
//! Run that integration test from outside the repository with
//! `cargo test --manifest-path /path/to/gnx/Cargo.toml -p gnx-jax --features jax-tests`
//! so the workspace's extension-only PyO3 configuration does not disable
//! linking to libpython. Select the interpreter with `PYO3_PYTHON` and make
//! its JAX installation visible to embedded Python via `PYTHONPATH` if needed.

mod buffer;
pub mod dlpack;
mod execute;
pub mod jaxpr;

pub use buffer::JaxBuffer;
pub use jaxpr::{JaxLowering, JaxLoweringError};

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use pyo3::prelude::*;
use pyo3::types::PyList;
use uuid::Uuid;

/// A JAX device and its native Python handle.
#[derive(Clone, Debug)]
pub struct JaxDevice {
    uuid: Uuid,
    platform: String,
    hardware_id: String,
    hardware_kind: String,
    ordinal: u64,
    native: Arc<Py<PyAny>>,
}

impl JaxDevice {
    pub fn from_native(device: Bound<'_, PyAny>) -> PyResult<Self> {
        let platform: String = device.getattr("platform")?.extract()?;
        let process: usize = device.getattr("process_index")?.extract()?;
        let id: usize = device.getattr("id")?.extract()?;
        let local_id: Option<usize> = device.getattr("local_hardware_id")?.extract()?;
        let hardware_kind = device.getattr("device_kind")?.extract()?;

        // Runtime ordinals are not physical identifiers. Namespace them to this
        // GNX process, and avoid relying on the Python wrapper's address.
        static NAMESPACE: OnceLock<Uuid> = OnceLock::new();
        let key = format!("{process}:{id}:{platform}");
        let uuid = Uuid::new_v5(NAMESPACE.get_or_init(Uuid::new_v4), key.as_bytes());
        Ok(Self {
            uuid,
            platform,
            hardware_id: local_id.unwrap_or(id).to_string(),
            hardware_kind,
            ordinal: id as u64,
            native: Arc::new(device.unbind()),
        })
    }

    /// A process-scoped JAX device ID, not a physical hardware UUID.
    pub fn uuid(&self) -> &Uuid {
        &self.uuid
    }

    pub fn platform(&self) -> &str {
        &self.platform
    }

    /// JAX's local hardware ordinal, or its device ID if no local ordinal exists.
    pub fn hardware_id(&self) -> &str {
        &self.hardware_id
    }

    pub fn hardware_kind(&self) -> &str {
        &self.hardware_kind
    }

    pub(crate) fn ordinal(&self) -> u64 {
        self.ordinal
    }

    pub fn native<'a, 'py>(&'a self, py: Python<'py>) -> &'a Bound<'py, PyAny> {
        self.native.bind(py)
    }
}

impl fmt::Display for JaxDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "JaxDevice(platform={}, kind={}, id={})",
            self.platform, self.hardware_kind, self.hardware_id
        )
    }
}

impl PartialEq for JaxDevice {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
    }
}

impl Eq for JaxDevice {}

impl Hash for JaxDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uuid.hash(state);
    }
}

/// List JAX devices for a platform such as `cpu` or `gpu`, or for JAX's default.
/// JAX is imported only when this function is called.
pub fn devices(py: Python<'_>, backend: Option<&str>) -> PyResult<Vec<JaxDevice>> {
    let jax = py.import("jax")?;
    let devices = match backend {
        Some(backend) => jax.call_method1("devices", (backend,))?,
        None => jax.call_method0("devices")?,
    };
    devices
        .cast_into::<PyList>()?
        .iter()
        .map(JaxDevice::from_native)
        .collect()
}
