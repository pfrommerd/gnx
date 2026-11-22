use gnx::array::{Array, ArrayInfo, ArrayRef, DType, Dim, Shape};
use gnx::buffer::{DataHandle, HostBuffer};
use gnx::expr::{CellTracer, Invocation, ObjectTracer, TraceContext, Tracer, VarInfo};
use numpy::npyffi::{NpyTypes, NPY_ARRAY_CARRAY, NPY_TYPES, PY_ARRAY_API};
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PySlice, PyTuple};

/// An array that can be symbolic or backed by concrete storage.
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
            "Array(shape={}, dtype='{}', symbolic={})",
            self.0.shape(),
            self.0.dtype(),
            if self.0.data().is_some() {
                "False"
            } else {
                "True"
            },
        )
    }

    fn __add__(&self, rhs: &PyArray) -> PyResult<Self> {
        let lhs = self.0.tracer();
        let rhs = rhs.0.tracer();
        if lhs.trace_ref().info() != rhs.trace_ref().info() {
            return Err(PyValueError::new_err(
                "array addition requires the same shape and dtype",
            ));
        }
        let output = Invocation::invoke(
            gnx::ops::add_op(),
            vec![],
            vec![lhs.clone(), rhs.clone()],
            vec![lhs.trace_ref().info().clone()],
        )
        .pop()
        .expect("one array output");
        Ok(Self(
            output
                .try_cast::<Array>()
                .ok()
                .expect("typed array output")
                .into(),
        ))
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
    let info = VarInfo::new(ArrayInfo::new(Shape::fixed(shape), dtype));
    Ok(PyArray(Tracer::<Array>::placeholder(info).into()))
}

/// Copy packed little-endian C-order bytes into a concrete host array.
#[pyfunction]
pub fn from_host_bytes(
    shape: Vec<usize>,
    dtype: &str,
    bytes: &Bound<'_, PyBytes>,
) -> PyResult<PyArray> {
    let info = ArrayInfo::new(Shape::fixed(shape), parse_dtype(dtype)?);
    let host = HostBuffer::copy_from_slice(info, bytes.as_bytes())?;
    Ok(PyArray(Array::from_data(DataHandle::new(host))))
}

/// Snapshot a NumPy ndarray directly into aligned GNX storage, including
/// strided views and arrays with non-native byte order.
#[pyfunction]
pub fn from_numpy_array(source: &Bound<'_, PyUntypedArray>) -> PyResult<PyArray> {
    let py = source.py();
    let source_dtype = source.dtype();
    let number = source_dtype.num();
    // The temporary destination must not be passed to a custom dtype's cast
    // implementation. Accept only built-in numeric descriptors.
    if !(0..NPY_TYPES::NPY_OBJECT as i32).contains(&number) && number != NPY_TYPES::NPY_HALF as i32
    {
        return Err(PyTypeError::new_err(
            "expected a built-in NumPy numeric or boolean dtype",
        ));
    }
    let dtype = parse_dtype(&source_dtype.getattr("name")?.extract::<String>()?)?;
    let shape = source.shape();
    let info = ArrayInfo::new(Shape::fixed(shape.iter().copied()), dtype);
    let mut dims = shape
        .iter()
        .map(|&n| isize::try_from(n).map_err(|_| PyValueError::new_err("NumPy dimension overflow")))
        .collect::<PyResult<Vec<_>>>()?;
    let rank =
        i32::try_from(dims.len()).map_err(|_| PyValueError::new_err("NumPy rank overflow"))?;
    let dtype_name: &'static str = dtype.into();
    let descriptor = PyArrayDescr::new(py, dtype_name)?;
    // Select an exact built-in little-endian dtype, of the size used by GNX.
    let descriptor = unsafe {
        let ptr = PY_ARRAY_API.PyArray_DescrNewByteorder(py, descriptor.as_dtype_ptr(), b'<' as _);
        Bound::from_owned_ptr_or_err(py, ptr.cast())?.cast_into_unchecked::<PyArrayDescr>()
    };
    let host = HostBuffer::try_fill(info, |bytes| -> PyResult<()> {
        // SAFETY: a temporary base ndarray views the entire aligned buffer.
        // It doesn't own the bytes or escape. Only NumPy's C copy routine
        // receives it, and the view is dropped before storage is published.
        unsafe {
            let ptr = PY_ARRAY_API.PyArray_NewFromDescr(
                py,
                PY_ARRAY_API.get_type_object(py, NpyTypes::PyArray_Type),
                descriptor.into_dtype_ptr(),
                rank,
                dims.as_mut_ptr(),
                std::ptr::null_mut(),
                bytes.as_mut_ptr().cast(),
                NPY_ARRAY_CARRAY,
                std::ptr::null_mut(),
            );
            let destination = Bound::from_owned_ptr_or_err(py, ptr)?;
            if PY_ARRAY_API.PyArray_CopyInto(py, destination.as_ptr().cast(), source.as_array_ptr())
                < 0
            {
                return Err(PyErr::fetch(py));
            }
        }
        Ok(())
    })?;
    Ok(PyArray(Array::from_data(DataHandle::new(host))))
}

/// Create a mutable cell initialized with the array x.
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
        DType::BF16,
        DType::F8E4M3FN,
        DType::F8E5M2,
        DType::F8E8M0FNU,
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
