use crate::trace::{Tracer, Traceable, ValueInfo};
use crate::value::{ConcreteValue, Value};
use crate::expr::Expr;
use crate::backend::{DeviceHandle, ExecOpts};

use std::hash::{Hash, Hasher};
use std::fmt::Display;
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct Device(Tracer<Device>);

// We don't have any device information :(
// without the physical device handle.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DeviceInfo;

impl Traceable for Device {
    type Concrete = DeviceHandle;
    type Info = DeviceInfo;
}

impl Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AbstractDevice")
    }
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.try_concrete() {
            Some(device) => write!(f, "{}", device),
            None => write!(f, "AbstractDevice"),
        }
    }
}

impl From<Device> for Tracer<Device> {
    fn from(value: Device) -> Self { value.0 }
}
impl From<Tracer<Device>> for Device {
    fn from(value: Tracer<Device>) -> Self { Device(value) }
}

impl ConcreteValue for DeviceHandle {
    fn to_info(&self) -> ValueInfo {
        ValueInfo::Device(DeviceInfo)
    }
}

#[rustfmt::skip]
impl Device {
    pub fn new(handle: DeviceHandle) -> Self {
        Device(Tracer::concrete(handle, DeviceInfo))
    }

    fn concrete(&self) -> &DeviceHandle {
        self.0.try_concrete().expect("Device is abstract! Cannot access concrete()")
    }
    // Helper methods to access the info fields directly.
    pub fn uuid(&self) -> &Uuid { self.concrete().uuid() }
    pub fn platform(&self) -> &str { self.concrete().platform() }
    pub fn hardware_kind(&self) -> &str { self.concrete().hardware_kind() }
    pub fn hardware_id(&self) -> &str { self.concrete().hardware_id() }

    pub async fn execute(&self, expr: Expr, args: Vec<Value>, opts: ExecOpts) -> Result<Vec<Value>, std::io::Error> {
        let handle: &DeviceHandle = self.0.try_concrete().ok_or_else(
            || std::io::Error::new(std::io::ErrorKind::Other, "Device is abstract! Cannot execute() using a device captured by a trace.")
        )?;
        // Wrap the provided expression
        // with a device::prefer op
        handle.backend().execute(expr, args, opts).await
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.uuid() == other.uuid()
    }
}
impl Eq for Device {}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uuid().hash(state);
    }
}