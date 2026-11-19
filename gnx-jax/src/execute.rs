use std::future::Future;
use std::io::{self, ErrorKind};
use std::sync::Arc;

use futures_channel::oneshot;
use gnx_expr::{Expr, Value};
use gnx_math::{DataHandle, Device, ExecOpts, ExecutionBoundary};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::{JaxBuffer, JaxDevice, jaxpr};

impl Device for JaxDevice {
    type Outputs = Vec<Value>;

    fn execute(
        &self,
        expr: Expr,
        args: impl IntoIterator<Item = Value>,
        opts: ExecOpts,
    ) -> impl Future<Output = io::Result<Self::Outputs>> + Send {
        let args = args.into_iter().collect::<Vec<_>>();
        let device = self.clone();
        async move {
            let (sender, receiver) = oneshot::channel();
            std::thread::Builder::new()
                .name("gnx-jax".into())
                .spawn(move || {
                    let result = Python::attach(|py| device.execute_blocking(py, expr, args, opts));
                    let _ = sender.send(result);
                })?;
            receiver.await.map_err(|_| {
                io::Error::other("JAX execution worker stopped before returning a result")
            })?
        }
    }
}

impl JaxDevice {
    fn execute_blocking(
        &self,
        py: Python<'_>,
        expr: Expr,
        args: Vec<Value>,
        opts: ExecOpts,
    ) -> io::Result<Vec<Value>> {
        ExecutionBoundary::validate_arguments(&expr, &args)?;
        let (projected, boundary) = ExecutionBoundary::project(&expr)?;
        let Some(expr) = projected else {
            return boundary.finish(&args, &[]);
        };
        let jax = py.import("jax")?;
        let expr = Arc::new(expr);
        let callable = jaxpr::callable(&jax, expr.clone()).map_err(io::Error::other)?;
        let inputs = boundary
            .arguments(&args)
            .map(|value| {
                let data = value?.downcast::<DataHandle>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidInput, "argument is not a concrete array")
                })?;
                self.import_buffer(py, data.clone(), opts.transfers)
            })
            .collect::<io::Result<Vec<_>>>()?;
        let sharding = self.single_device_sharding(py)?;
        let options = PyDict::new(py);
        options.set_item("in_shardings", &sharding)?;
        options.set_item("out_shardings", &sharding)?;
        let compiled = jax.call_method("jit", (callable,), Some(&options))?;
        let native_args = PyTuple::new(py, inputs.iter().map(|data| data.native(py)))?;
        let outputs = compiled.call1(native_args)?;
        let outputs = outputs.cast_into::<PyTuple>().map_err(PyErr::from)?;
        if outputs.len() != expr.outputs().len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "JAX returned an unexpected number of arrays",
            ));
        }
        let computed = outputs
            .iter()
            .zip(expr.outputs())
            .map(|(array, var)| {
                let result = JaxBuffer::from_native(array).map_err(io::Error::other)?;
                let expected =
                    jaxpr::expected(expr.value_info(var).expect("validated output metadata"))
                        .map_err(io::Error::other)?;
                if result.info() != expected || !self.native(py).eq(result.device().native(py))? {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "JAX returned an array with unexpected metadata or placement",
                    ));
                }
                // SAFETY: these are ready results of JAX execution on immutable inputs.
                Ok(Value::new(unsafe { result.into_data(py) }?))
            })
            .collect::<io::Result<Vec<_>>>()?;
        boundary.finish(&args, &computed)
    }
}
