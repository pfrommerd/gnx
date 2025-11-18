use crate::array::Array;
use crate::{Expr, Value};

use std::sync::{Arc, Mutex};

pub trait DeviceImpl: Send + Sync {
    fn platform(&self) -> &str;
    fn platform_id(&self) -> usize;
    fn hardware_kind(&self) -> &str;

    fn backend(&self) -> Backend;
    fn put(&self, array: &Array) -> Result<Array, std::io::Error>;
}

pub trait BackendImpl: Send + Sync {
    fn name(&self) -> &str;
    // Note: All Values must belong to devices of this backend
    fn execute(&self, expr: Expr, args: Vec<Value>) -> Result<Vec<Value>, std::io::Error>;
    fn devices(&self) -> Result<Vec<Device>, std::io::Error>;
}

#[derive(Clone)]
pub struct Device(Arc<dyn DeviceImpl + Send + Sync>);

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Device {}

#[rustfmt::skip]
impl Device {
    pub fn new(impl_: Arc<dyn DeviceImpl + Send + Sync>) -> Self {
        Device(impl_)
    }

    pub fn platform(&self) -> &str { self.0.platform() }
    pub fn platform_id(&self) -> usize { self.0.platform_id() }
    pub fn hardware_kind(&self) -> &str { self.0.hardware_kind() }

    pub fn backend(&self) -> Backend { self.0.backend() }
    pub fn put(&self, array: &Array) -> Result<Array, std::io::Error> { self.0.put(array) }
}

#[derive(Clone)]
pub struct Backend(Arc<dyn BackendImpl + Send + Sync>);

#[rustfmt::skip]
impl Backend {
    pub fn from(impl_: Arc<dyn BackendImpl + Send + Sync>) -> Self {
        Backend(impl_)
    }

    pub fn name(&self) -> &str { self.0.name()  }
    pub fn execute(&self, expr: Expr, args: Vec<Value>)
        -> Result<Vec<Value>, std::io::Error> { self.0.execute(expr, args) }
    pub fn devices(&self) -> Result<Vec<Device>, std::io::Error> { self.0.devices() }
}

impl PartialEq for Backend {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Backend {}

static BACKENDS: Mutex<Vec<Backend>> = Mutex::new(Vec::new());

pub fn backends() -> Vec<Backend> {
    let backends = BACKENDS.lock().unwrap();
    backends.clone()
}

pub fn register(backend: Backend) {
    let mut backends = BACKENDS.lock().unwrap();
    backends.push(backend);
}

pub fn devices() -> Result<Vec<Device>, std::io::Error> {
    let mut devices = Vec::new();
    let backends = BACKENDS.lock().unwrap();
    for backend in backends.iter() {
        devices.extend(backend.devices()?);
    }
    Ok(devices)
}
