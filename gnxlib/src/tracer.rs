use gnx::array::{Array, ArrayInfo, ArrayRef, DType, Dim, Shape};
use gnx::expr::{CellTracer, ObjectTracer, TraceContext, Tracer, ValueInfo};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

/// A symbolic array. It describes a traced value, not a materialized array.
#[pyclass(name = "Array", module = "gnxlib.array", frozen)]
pub struct PyArray(pub(crate) Array);

#[pymethods]
impl PyArray {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        shape_tuple(py, self.0.shape())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.0.dtype().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "Array(shape={}, dtype='{}', symbolic=True)",
            self.0.shape(),
            self.0.dtype()
        )
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "a symbolic GNX array has no truth value",
        ))
    }
}

/// A mutable cell holding a symbolic array. Use [:] or [...] to access it whole.
#[pyclass(name = "ArrayRef", module = "gnxlib.array", frozen)]
pub struct PyArrayRef(pub(crate) ArrayRef);

#[pymethods]
impl PyArrayRef {
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        check_whole_array_key(key)?;
        Ok(PyArray(self.0.get()))
    }

    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: &PyArray) -> PyResult<()> {
        check_whole_array_key(key)?;
        let value = value.0.tracer();
        if self.0.tracer().cell_ref().type_info() != value.trace_ref().info() {
            return Err(PyValueError::new_err(
                "array cell assignment requires the same shape and dtype",
            ));
        }
        self.0.set(value.clone());
        Ok(())
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        shape_tuple(py, self.info().shape())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.info().dtype().into()
    }
}

/// Create a symbolic array input outside a JIT trace.
#[pyfunction(signature = (shape, dtype = "float32"))]
pub fn placeholder(shape: Vec<usize>, dtype: &str) -> PyResult<PyArray> {
    if TraceContext::current().is_some() {
        return Err(PyTypeError::new_err(
            "symbolic array inputs must be created outside a JIT trace",
        ));
    }
    let dtype = parse_dtype(dtype)?;
    let info = ValueInfo::new(ArrayInfo::new(Shape::fixed(shape), dtype));
    Ok(PyArray(Tracer::<Array>::placeholder(info).into()))
}

/// Create a mutable cell initialized with the symbolic array x.
#[pyfunction]
pub fn new_ref(x: &PyArray) -> PyArrayRef {
    PyArrayRef(ArrayRef::new(x.0.tracer().clone()))
}

impl PyArrayRef {
    fn info(&self) -> &ArrayInfo {
        self.0
            .tracer()
            .cell_ref()
            .type_info()
            .downcast()
            .expect("typed array cell")
    }
}

fn check_whole_array_key(key: &Bound<'_, PyAny>) -> PyResult<()> {
    if key.is(key.py().Ellipsis())
        || (key.is_instance_of::<PySlice>()
            && key.getattr("start")?.is_none()
            && key.getattr("stop")?.is_none()
            && key.getattr("step")?.is_none())
    {
        return Ok(());
    }
    Err(PyIndexError::new_err(
        "ArrayRef only supports whole-array access with [:] or [...]",
    ))
}

pub(crate) fn extract_tracer(obj: &Bound<'_, PyAny>) -> Option<ObjectTracer> {
    if let Ok(value) = obj.cast::<PyArray>() {
        Some(value.borrow().0.tracer().clone().into())
    } else if let Ok(cell) = obj.cast::<PyArrayRef>() {
        Some(cell.borrow().0.tracer().clone().into())
    } else {
        None
    }
}

pub(crate) fn wrap_tracer(py: Python<'_>, tracer: ObjectTracer) -> PyResult<Bound<'_, PyAny>> {
    let wrong_kind = || PyTypeError::new_err("unsupported GNX tracer type");
    match tracer {
        ObjectTracer::Ref(value) => {
            let typed = Tracer::from(value)
                .try_cast::<Array>()
                .map_err(|_| wrong_kind())?;
            Ok(Bound::new(py, PyArray(typed.into()))?.into_any())
        }
        ObjectTracer::Cell(cell) => {
            let typed = CellTracer::from(cell)
                .try_cast::<Array>()
                .map_err(|_| wrong_kind())?;
            Ok(Bound::new(py, PyArrayRef(typed.into()))?.into_any())
        }
    }
}

fn shape_tuple<'py>(py: Python<'py>, shape: &Shape) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        shape.dims().iter().map(|dim| match dim {
            Dim::Fixed(n) => Some(*n),
            _ => None,
        }),
    )
}

fn parse_dtype(dtype: &str) -> PyResult<DType> {
    const DTYPES: &[DType] = &[
        DType::Bool,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::C64,
        DType::C128,
    ];
    DTYPES
        .iter()
        .copied()
        .find(|kind| {
            let name: &'static str = (*kind).into();
            name == dtype
        })
        .ok_or_else(|| PyValueError::new_err(format!("unsupported dtype: {dtype}")))
}
