use std::future::Future;
use std::io::{self, ErrorKind};
use std::sync::Arc;

use futures_channel::oneshot;
use gnx_expr::{Expr, Value};
use gnx_math::{DataHandle, DataImpl, Device, ExecOpts};
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
        let jax = py.import("jax")?;
        let expr = Arc::new(expr);
        let callable = jaxpr::callable(&jax, expr.clone()).map_err(io::Error::other)?;
        let expected = expr.closure().iter().chain(expr.inputs());
        let count = expr.closure().len() + expr.inputs().len();
        if count != args.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                format!("expression expects {count} arguments, got {}", args.len()),
            ));
        }
        let mut inputs = Vec::with_capacity(count);
        for (index, ((_, expected), value)) in expected.zip(args).enumerate() {
            let expected = jaxpr::expected(expected).map_err(io::Error::other)?;
            let data = value.downcast::<DataHandle>().cloned().map_err(|_| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("argument {index} is not a concrete array"),
                )
            })?;
            let actual = data.info();
            if actual != *expected {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("argument {index}: expected {expected}, got {actual}"),
                ));
            }
            inputs.push(data);
        }
        let inputs = inputs
            .into_iter()
            .map(|data| self.import_buffer(py, data, opts.transfers))
            .collect::<io::Result<Vec<_>>>()?;
        let sharding = self.single_device_sharding(py)?;
        let options = PyDict::new(py);
        options.set_item("in_shardings", &sharding)?;
        options.set_item("out_shardings", &sharding)?;
        let compiled = jax.call_method("jit", (callable,), Some(&options))?;
        let args = PyTuple::new(py, inputs.iter().map(|data| data.native(py)))?;
        let outputs = compiled.call1(args)?;
        jax.call_method1("block_until_ready", (&outputs,))?;
        let outputs = outputs.cast_into::<PyTuple>().map_err(PyErr::from)?;
        if outputs.len() != expr.outputs().len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "JAX returned an unexpected number of arrays",
            ));
        }
        outputs
            .iter()
            .zip(expr.outputs())
            .map(|(array, var)| {
                let result = JaxBuffer::from_native(array).map_err(io::Error::other)?;
                let expected =
                    jaxpr::expected(expr.value_info(var).expect("validated output metadata"))
                        .map_err(io::Error::other)?;
                if result.info() != *expected || !self.native(py).eq(result.device().native(py))? {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "JAX returned an array with unexpected metadata or placement",
                    ));
                }
                Ok(Value::new(DataHandle::new(result)))
            })
            .collect()
    }
}
