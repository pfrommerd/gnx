//! Trace lowered GNX expressions with JAX and inspect the resulting JAXPR.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use gnx_expr::{Eqn as GnxEqn, Expr, Kind, Op, Operation, ValueInfo, Var, builtins::CallOperation};
use gnx_math::{ArrayInfo, DType, Dim, Shape, ops::AddOperation};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyModule, PyTuple};

#[doc(hidden)]
pub use inventory as __inventory;

#[derive(Debug, thiserror::Error)]
pub enum JaxLoweringError {
    #[error("invalid GNX expression: {0}")]
    InvalidExpr(#[from] gnx_expr::LoweringError),
    #[error("no JAX lowering for {dialect}.{name}")]
    UnsupportedOperation {
        dialect: &'static str,
        name: &'static str,
    },
    #[error("multiple JAX lowerings registered for {dialect}.{name}")]
    DuplicateRegistration {
        dialect: &'static str,
        name: &'static str,
    },
    #[error("equation {index} ({op}): {source}")]
    Equation {
        index: usize,
        op: String,
        source: Box<Self>,
    },
    #[error("{0}")]
    InvalidValue(String),
    #[error(transparent)]
    Python(#[from] PyErr),
}

impl JaxLoweringError {
    fn at(self, index: usize, eqn: &GnxEqn) -> Self {
        Self::Equation {
            index,
            op: eqn.op().to_string(),
            source: Box::new(self),
        }
    }

    fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match self {
            Self::Python(error) => error,
            Self::Equation { index, op, source } => {
                let error = PyRuntimeError::new_err(format!("equation {index} ({op}): {source}"));
                error.set_cause(py, Some(source.into_pyerr(py)));
                error
            }
            error => PyValueError::new_err(error.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, JaxLoweringError>;

macro_rules! python_wrapper {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(Clone, Debug)]
        pub struct $name<'py>(Bound<'py, PyAny>);

        impl<'py> $name<'py> {
            pub fn from_native(value: Bound<'py, PyAny>) -> Self {
                Self(value)
            }

            pub fn as_native(&self) -> &Bound<'py, PyAny> {
                &self.0
            }

            pub fn into_native(self) -> Bound<'py, PyAny> {
                self.0
            }
        }
    };
}

python_wrapper!(/// A live JAX value while tracing, not an atom in a finished JAXPR.
    TraceValue);
python_wrapper!(ClosedJaxpr);
python_wrapper!(Jaxpr);
python_wrapper!(Eqn);
python_wrapper!(Atom);

impl TraceValue<'_> {
    pub fn info(&self) -> Result<ArrayInfo> {
        array_info(self.as_native())
    }
}

impl<'py> ClosedJaxpr<'py> {
    pub fn jaxpr(&self) -> PyResult<Jaxpr<'py>> {
        Ok(Jaxpr(self.0.getattr("jaxpr")?))
    }

    pub fn consts(&self) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.0.getattr("consts")?.extract()
    }
}

impl<'py> Jaxpr<'py> {
    pub fn inputs(&self) -> PyResult<Vec<Atom<'py>>> {
        atoms(&self.0, "invars")
    }

    pub fn outputs(&self) -> PyResult<Vec<Atom<'py>>> {
        atoms(&self.0, "outvars")
    }

    pub fn constvars(&self) -> PyResult<Vec<Atom<'py>>> {
        atoms(&self.0, "constvars")
    }

    pub fn equations(&self) -> PyResult<Vec<Eqn<'py>>> {
        self.0
            .getattr("eqns")?
            .try_iter()?
            .map(|item| item.map(Eqn))
            .collect()
    }
}

impl<'py> Eqn<'py> {
    pub fn primitive(&self) -> PyResult<String> {
        self.0.getattr("primitive")?.getattr("name")?.extract()
    }

    pub fn inputs(&self) -> PyResult<Vec<Atom<'py>>> {
        atoms(&self.0, "invars")
    }

    pub fn outputs(&self) -> PyResult<Vec<Atom<'py>>> {
        atoms(&self.0, "outvars")
    }

    pub fn params(&self) -> PyResult<Bound<'py, PyDict>> {
        Ok(self.0.getattr("params")?.cast_into()?)
    }
}

impl<'py> Atom<'py> {
    pub fn aval(&self) -> PyResult<Bound<'py, PyAny>> {
        self.0.getattr("aval")
    }

    pub fn literal(&self) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.0.getattr_opt("val")
    }
}

fn atoms<'py>(obj: &Bound<'py, PyAny>, attr: &str) -> PyResult<Vec<Atom<'py>>> {
    obj.getattr(attr)?
        .try_iter()?
        .map(|item| item.map(Atom))
        .collect()
}

pub trait JaxLowering {
    /// Lower the equation's explicit operands. Child captures are already
    /// bound in `ctx`; return one live value per GNX result in equation order.
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>>;
}

type Dispatch = for<'py> fn(
    &dyn Operation,
    &mut TraceContext<'py>,
    &GnxEqn,
    &[TraceValue<'py>],
) -> Result<Vec<TraceValue<'py>>>;

pub struct JaxLoweringRegistration {
    type_id: fn() -> TypeId,
    dispatch: Dispatch,
}

impl JaxLoweringRegistration {
    pub const fn of<O: Operation + JaxLowering>() -> Self {
        Self {
            type_id: TypeId::of::<O>,
            dispatch: dispatch::<O>,
        }
    }
}

fn dispatch<'py, O: Operation + JaxLowering>(
    op: &dyn Operation,
    ctx: &mut TraceContext<'py>,
    eqn: &GnxEqn,
    operands: &[TraceValue<'py>],
) -> Result<Vec<TraceValue<'py>>> {
    let typed = (op as &dyn Any)
        .downcast_ref::<O>()
        .expect("JAX rule indexed by operation type");
    typed.lower_jax(ctx, eqn, operands)
}

inventory::collect!(JaxLoweringRegistration);

/// Link a typed JAX lowering for an operation into inventory.
#[macro_export]
macro_rules! register_jax_lowering {
    ($op:ty) => {
        $crate::jaxpr::__inventory::submit! {
            $crate::jaxpr::JaxLoweringRegistration::of::<$op>()
        }
    };
}

static RULES: LazyLock<HashMap<TypeId, Option<Dispatch>>> = LazyLock::new(|| {
    let mut rules = HashMap::new();
    for entry in inventory::iter::<JaxLoweringRegistration> {
        rules
            .entry((entry.type_id)())
            .and_modify(|rule| *rule = None)
            .or_insert(Some(entry.dispatch));
    }
    rules
});

fn rule(op: &dyn Operation) -> Result<Dispatch> {
    let dialect = op.dialect().name();
    let name = op.name();
    match RULES.get(&(op as &dyn Any).type_id()) {
        Some(Some(dispatch)) => Ok(*dispatch),
        Some(None) => Err(JaxLoweringError::DuplicateRegistration { dialect, name }),
        None => Err(JaxLoweringError::UnsupportedOperation { dialect, name }),
    }
}

impl JaxLowering for dyn Operation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        rule(self)?(self, ctx, eqn, operands)
    }
}

impl JaxLowering for Op {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        self.operation().lower_jax(ctx, eqn, operands)
    }
}

struct Child<'py> {
    expr: Arc<Expr>,
    captures: Vec<TraceValue<'py>>,
}

pub struct TraceContext<'py> {
    py: Python<'py>,
    children: Vec<Child<'py>>,
}

impl<'py> TraceContext<'py> {
    pub fn python(&self) -> Python<'py> {
        self.py
    }

    pub fn lax(&self) -> PyResult<Lax<'py>> {
        Ok(Lax(self.py.import("jax.lax")?))
    }

    /// Trace a child with its already-bound parent captures.
    pub fn trace_subexpr(
        &self,
        index: usize,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let child = self
            .children
            .get(index)
            .ok_or_else(|| invalid(format!("no subexpression at index {index}")))?;
        trace_body(
            self.py,
            &child.expr,
            child.captures.iter().chain(operands).cloned(),
        )
    }
}

pub struct Lax<'py>(Bound<'py, PyModule>);

impl<'py> Lax<'py> {
    pub fn add(&self, lhs: &TraceValue<'py>, rhs: &TraceValue<'py>) -> PyResult<TraceValue<'py>> {
        Ok(TraceValue(
            self.0
                .call_method1("add", (lhs.as_native(), rhs.as_native()))?,
        ))
    }
}

impl JaxLowering for AddOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let [lhs, rhs] = operands else {
            return Err(invalid("array.add requires two operands"));
        };
        if !eqn.subexprs().is_empty() || !eqn.op().attrs().is_empty() {
            return Err(invalid(
                "array.add does not take attributes or subexpressions",
            ));
        }
        Ok(vec![ctx.lax()?.add(lhs, rhs)?])
    }
}
register_jax_lowering!(AddOperation);

impl JaxLowering for CallOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        _eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        ctx.trace_subexpr(0, operands)
    }
}
register_jax_lowering!(CallOperation);

fn invalid(msg: impl Into<String>) -> JaxLoweringError {
    JaxLoweringError::InvalidValue(msg.into())
}

pub(crate) fn dimensions(info: &ArrayInfo) -> Result<Vec<usize>> {
    info.shape()
        .dims()
        .iter()
        .map(|dim| match dim {
            Dim::Fixed(size) => Ok(*size),
            _ => Err(invalid("JAX lowering currently requires fixed dimensions")),
        })
        .collect()
}

pub(crate) fn array_info(value: &Bound<'_, PyAny>) -> Result<ArrayInfo> {
    let shape: Vec<usize> = value.getattr("shape")?.extract()?;
    let dtype = value
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?;
    const TYPES: [DType; 14] = [
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
    let dtype = TYPES
        .into_iter()
        .find(|ty| Into::<&'static str>::into(*ty) == dtype)
        .ok_or_else(|| invalid(format!("unsupported JAX dtype {dtype}")))?;
    Ok(ArrayInfo::new(Shape::fixed(shape), dtype))
}

pub(crate) fn expected(info: &ValueInfo) -> Result<&ArrayInfo> {
    info.downcast()
        .map_err(|_| invalid(format!("JAX lowering only supports arrays, found {info}")))
}

fn check_value(value: &TraceValue<'_>, info: &ValueInfo) -> Result<()> {
    let actual = array_info(value.as_native())?;
    let expected = expected(info)?;
    if actual != *expected {
        return Err(invalid(format!("expected {expected}, got {actual}")));
    }
    Ok(())
}

fn trace_body<'py>(
    py: Python<'py>,
    expr: &Expr,
    args: impl IntoIterator<Item = TraceValue<'py>>,
) -> Result<Vec<TraceValue<'py>>> {
    let mut args = args.into_iter();
    let mut values = HashMap::<Var, TraceValue<'py>>::new();
    for (var, _, info) in expr.closure().iter().chain(expr.inputs()) {
        let value = args
            .next()
            .ok_or_else(|| invalid("too few arguments for the expression"))?;
        check_value(&value, info)?;
        values.insert(*var, value);
    }
    if args.next().is_some() {
        return Err(invalid("too many arguments for the expression"));
    }
    for (index, eqn) in expr.eqns().iter().enumerate() {
        (|| {
            let operands: Vec<_> = eqn
                .inputs()
                .iter()
                .map(|(var, _)| values[var].clone())
                .collect();
            let children = eqn
                .subexprs()
                .iter()
                .map(|child| Child {
                    expr: child.expr.clone(),
                    captures: child
                        .captures
                        .iter()
                        .map(|(var, _)| values[var].clone())
                        .collect(),
                })
                .collect();
            let mut ctx = TraceContext { py, children };
            let outputs = eqn.op().lower_jax(&mut ctx, eqn, &operands)?;
            if outputs.len() != eqn.outputs().len() {
                return Err(invalid(format!(
                    "lowering returned {} values, expected {}",
                    outputs.len(),
                    eqn.outputs().len()
                )));
            }
            for ((var, kind), value) in eqn.outputs().iter().zip(outputs) {
                let Kind::Value(info) = kind else {
                    unreachable!("validated Expr");
                };
                check_value(&value, info)?;
                values.insert(*var, value);
            }
            Ok(())
        })()
        .map_err(|error: JaxLoweringError| error.at(index, eqn))?;
    }
    Ok(expr
        .outputs()
        .iter()
        .map(|var| values.remove(var).expect("validated Expr output"))
        .collect())
}

fn validate(jax: &Bound<'_, PyModule>, expr: &Expr) -> Result<()> {
    let check = |info| -> Result<()> {
        let info = expected(info)?;
        dimensions(info)?;
        let dtype: &'static str = info.dtype().into();
        let canonical = jax
            .getattr("dtypes")?
            .call_method1("canonicalize_dtype", (dtype,))?;
        if canonical.getattr("name")?.extract::<String>()? != dtype {
            return Err(invalid(format!(
                "JAX would change dtype {dtype}; check its dtype configuration"
            )));
        }
        Ok(())
    };
    for (_, _, info) in expr.closure().iter().chain(expr.inputs()) {
        check(info)?;
    }
    for (index, eqn) in expr.eqns().iter().enumerate() {
        (|| {
            rule(eqn.op().operation())?;
            for (_, kind) in eqn.outputs() {
                let Kind::Value(info) = kind else {
                    unreachable!("validated Expr");
                };
                check(info)?;
            }
            for child in eqn.subexprs() {
                validate(jax, &child.expr)?;
            }
            Ok(())
        })()
        .map_err(|error: JaxLoweringError| error.at(index, eqn))?;
    }
    Ok(())
}

pub(crate) fn callable<'py>(
    jax: &Bound<'py, PyModule>,
    expr: Arc<Expr>,
) -> Result<Bound<'py, PyCFunction>> {
    expr.check()?;
    validate(jax, &expr)?;
    Ok(PyCFunction::new_closure(
        jax.py(),
        Some(c"gnx_expr"),
        None,
        move |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
            let py = args.py();
            let outputs = trace_body(py, &expr, args.iter().map(TraceValue))
                .map_err(|err| err.into_pyerr(py))?;
            Ok::<_, PyErr>(
                PyTuple::new(py, outputs.into_iter().map(TraceValue::into_native))?.unbind(),
            )
        },
    )?)
}

/// Use shape/dtype-only inputs to trace a validated lowered expression.
pub fn trace_expr<'py>(py: Python<'py>, expr: &Expr) -> Result<ClosedJaxpr<'py>> {
    let jax = py.import("jax")?;
    let callable = callable(&jax, Arc::new(expr.clone()))?;
    let signature = expr
        .closure()
        .iter()
        .chain(expr.inputs())
        .map(|(_, _, info)| {
            let info = expected(info)?;
            let dtype: &'static str = info.dtype().into();
            Ok(jax.call_method1(
                "ShapeDtypeStruct",
                (PyTuple::new(py, dimensions(info)?)?, dtype),
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    let trace = jax.call_method1("make_jaxpr", (callable,))?;
    Ok(ClosedJaxpr(trace.call1(PyTuple::new(py, signature)?)?))
}
