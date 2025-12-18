use crate::expr::Expr;
use crate::value::Value;

use std::sync::{Arc, Mutex};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use uuid::Uuid;

pub trait DeviceImpl: Debug + Display + Send + Sync {
    fn uuid(&self) -> &Uuid;
    fn platform(&self) -> &str;
    fn hardware_kind(&self) -> &str;
    fn hardware_id(&self) -> &str;

    fn backend(&self) -> &Backend;
}

pub type DeviceHandle = Arc<dyn DeviceImpl + Send + Sync>;


pub struct ExecOpts {

}

// Re-export the Device type from the device module.
pub use crate::device::Device;

#[async_trait::async_trait]
pub trait BackendImpl: Debug + Display + Send + Sync {
    fn name(&self) -> &str;
    fn devices(&self) -> Result<Vec<Device>, std::io::Error>;

    async fn execute(&self, expr: Expr, args: Vec<Value>, opts: ExecOpts)
        -> Result<Vec<Value>, std::io::Error>;
}

pub type BackendHandle = Arc<dyn BackendImpl>;

#[derive(Clone)]
#[repr(transparent)]
pub struct Backend(BackendHandle);

impl AsRef<Backend> for BackendHandle {
    fn as_ref(&self) -> &Backend {
        unsafe { std::mem::transmute(self) }
    }
}

impl Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl PartialEq for Backend {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Backend {}

impl Hash for Backend {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

#[rustfmt::skip]
impl Backend {
    pub fn new(v: Arc<dyn BackendImpl + Send + Sync>) -> Self {
        Backend(v)
    }

    pub fn name(&self) -> &str { self.0.name()  }
    pub fn devices(&self) -> Result<Vec<Device>, std::io::Error> {
        self.0.devices()
    }

    pub async fn execute(&self, expr: Expr, args: Vec<Value>, opts: ExecOpts)
            -> Result<Vec<Value>, std::io::Error> {
        self.0.execute(expr, args, opts).await
    }
}


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