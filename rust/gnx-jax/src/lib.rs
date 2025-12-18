use std::{
    collections::HashMap, sync::{Arc, Mutex, OnceLock, Weak as WeakArc}
};
use std::hash::{Hash, Hasher, DefaultHasher};

use std::fmt::{Debug, Display};
use uuid::Uuid;
use pyo3::prelude::*;

use gnx::expr::{Expr, Value};
use gnx::backend::{
    Backend, BackendImpl, DeviceImpl, Device, BackendHandle
};

#[derive(Debug)]
pub struct JaxDevice {
    uuid: Uuid,
    platform: String,
    hardware_kind: String,
    hardware_id: String,
    backend: BackendHandle,
    jax_device: Py<PyAny>,
}

impl Display for JaxDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JaxDevice(kind={}, id={}, jax_device={})",
            self.hardware_kind,
            self.hardware_id,
            self.jax_device
        )
    }
}

impl JaxDevice {
    fn new(backend: Arc<JaxBackend>, jax_device: Bound<'_, PyAny>) -> Result<Self, PyErr> {
        let platform = jax_device
            .getattr(pyo3::intern!(jax_device.py(), "platform"))?
            .extract::<String>()?;
        let hardware_id = jax_device
            .getattr(pyo3::intern!(jax_device.py(), "local_hardware_id"))?
            .extract::<usize>()?;
        let hardware_kind = jax_device
            .getattr(pyo3::intern!(jax_device.py(), "device_kind"))?
            .extract::<String>()?;

        // Use a hash of the platform + hardware_id and the
        // pointer to the JAX device to generate a unique UUID.
        // that is consistent even if the JAX device is re-created.
        let mut hasher = DefaultHasher::new();
        (&platform, hardware_id).hash(&mut hasher);
        let lower = hasher.finish() as u64;
        let upper = jax_device.as_ptr() as u64;
        let uuid = Uuid::from_u64_pair(upper, lower);

        Ok(JaxDevice {
            backend,
            uuid,
            platform,
            hardware_id: hardware_id.to_string(),
            hardware_kind,
            jax_device: jax_device.unbind(),
        })
    }
}

impl DeviceImpl for JaxDevice {
    fn uuid(&self) -> &Uuid { &self.uuid }
    fn platform(&self) -> &str { &self.platform }
    fn hardware_id(&self) -> &str { &self.hardware_id }
    fn hardware_kind(&self) -> &str { &self.hardware_kind }

    fn backend(&self) -> &Backend {
        let backend: &BackendHandle = &self.backend;
        backend.as_ref() // turn a &BackendHandle into a &Backend
    }
}

#[derive(Debug)]
pub struct JaxBackend {
    jax: Py<PyModule>,
    // A handle to the Arc of this backend.
    // So that devices can hold strong references to their backend.
    self_handle: OnceLock<WeakArc<JaxBackend>>,
    // The devices, owned weakly by this backend (as the devices own the backend).
    // Upon calling device(), we will re-create any devices that have been dropped.
    devices: Mutex<Vec<WeakArc<JaxDevice>>>,
}

impl JaxBackend {
    pub fn new() -> Result<Backend, PyErr> {
        let py_handle = Python::attach(|py| py.import("jax").map(|x| x.unbind()))?;
        let backend = Arc::new(JaxBackend {
            jax: py_handle,
            self_handle: Default::default(),
            devices: Default::default(),
        });
        backend.self_handle.set(Arc::downgrade(&backend)).unwrap();
        Ok(Backend::new(backend))
    }
}

impl Display for JaxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JaxBackend(jax={})", self.jax)
    }
}

#[async_trait::async_trait]
impl BackendImpl for JaxBackend {
    fn name(&self) -> &str {
        "jax"
    }

    #[rustfmt::skip]
    fn devices(&self) -> Result<Vec<Device>, std::io::Error> {
        // All "live" devices, indexed by their Py<PyAny> pointer.
        let mut old_devices = self.devices.lock().unwrap();
        let mut device_lookup: HashMap<usize, Arc<JaxDevice>> = {
            old_devices.iter().filter_map(|x| {
                let x = x.upgrade()?;
                Some((x.jax_device.as_ptr() as usize, x))
            }).collect()
        };
        // Go through all JAX devices, and use d to find existing ones.
        let self_handle = self.self_handle.get().unwrap().upgrade().unwrap();
        let jax_devices = Python::attach(|py| -> Result<Vec<Arc<JaxDevice>>, PyErr> {
            let jax = self.jax.bind(py);
            let py_devices = jax.call_method0(pyo3::intern!(py, "devices"))?;
            let py_devices: Vec<Bound<'_, PyAny>> = py_devices.extract()?;
            py_devices.into_iter().map(|dev| -> Result<Arc<JaxDevice>, PyErr> {
                // If we fid an existing device for this Py<PyAny>, use it.
                let m = device_lookup.remove(&(dev.as_ptr() as usize));
                if let Some(d) = m { return Ok(d) }
                Ok(Arc::new(JaxDevice::new(self_handle.clone(), dev)?))
            }).collect()
        })?;
        let (mut jax_devices, devices): (Vec<WeakArc<JaxDevice>>, Vec<Device>) = jax_devices.into_iter().map(|x| {
            (Arc::downgrade(&x), Device::new(x))
        }).unzip();
        // Update the devices list.
        std::mem::swap(&mut *old_devices, &mut jax_devices);
        Ok(devices)
    }

    async fn execute(&self, _expr: Expr, _args: Vec<Value>)
            -> Result<Vec<Value>, std::io::Error> {
        todo!()
    }
}

static JAX_BACKEND: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();

pub fn register() -> Backend {
    JAX_BACKEND
        .get_or_init(|| {
            let backend = JaxBackend::new().expect("Failed to initialize JAX backend");
            gnx::backend::register(backend.clone());
            backend
        })
        .clone()
}
