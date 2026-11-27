pub use crate::scalar::{AsDType, IntoScalar, Scalar};
use ordered_float::OrderedFloat;
use std::fmt::{Debug, Display};

use gnx_expr::{CellTracer, ConcreteValue, LoweringError, Trace, Traceable, Tracer, Value};

pub use crate::buffer::DataHandle;

#[derive(
    Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug, rkyv::Archive, rkyv::Serialize,
)]
pub enum Dim {
    Fixed(usize),
    /// A runtime length in 0..=max. Symbols are scoped to an expression.
    Bounded {
        symbol: gnx_expr::Symbol,
        max: u32,
    },
    // unknown size, can be converted
    // to a fixed size by truncating if too big or padding if too small.
    Dynamic,
    // Dependent on another dimension
    // to resolve the dynamic size.
    // Contains the index of the dependent dimension.
    // The other dimension can potentially also be jagged.
    Jagged(usize),
}

impl Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Fixed(n) => write!(f, "{}", n),
            Dim::Bounded { symbol, max } => write!(f, "{symbol}<={max}"),
            Dim::Dynamic => write!(f, "?"),
            Dim::Jagged(i) => write!(f, "jagged({})", i),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, rkyv::Archive, rkyv::Serialize)]
pub struct Shape(Vec<Dim>);

/// A user-supplied array shape. Tuple and integer container axes are fixed;
/// an existing `Shape` retains symbolic dimensions.
pub trait IntoShape {
    fn into_shape(self) -> Shape;
}
impl IntoShape for Shape {
    fn into_shape(self) -> Shape {
        self
    }
}
impl IntoShape for &Shape {
    fn into_shape(self) -> Shape {
        self.clone()
    }
}
impl IntoShape for () {
    fn into_shape(self) -> Shape {
        Shape::fixed([])
    }
}
impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> Shape {
        Shape::fixed(self)
    }
}
impl<const N: usize> IntoShape for &[usize; N] {
    fn into_shape(self) -> Shape {
        Shape::fixed(self.iter().copied())
    }
}
impl IntoShape for &[usize] {
    fn into_shape(self) -> Shape {
        Shape::fixed(self.iter().copied())
    }
}
impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Shape {
        Shape::fixed(self)
    }
}
impl IntoShape for &Vec<usize> {
    fn into_shape(self) -> Shape {
        self.as_slice().into_shape()
    }
}
macro_rules! tuple_shape {
    (@ty $x:ident) => {usize};
    ($first:ident $(,$rest:ident)*) => {
        impl IntoShape for (tuple_shape!(@ty $first), $(tuple_shape!(@ty $rest),)*) {
            fn into_shape(self) -> Shape {
                let ($first, $($rest,)*) = self;
                Shape::fixed([$first, $($rest),*])
            }
        }
        tuple_shape!($($rest),*);
    };
    () => {};
}
tuple_shape!(a, b, c, d, e, f, g, h);

impl Shape {
    pub fn fixed(dims: impl IntoIterator<Item = usize>) -> Self {
        Shape(dims.into_iter().map(Dim::Fixed).collect())
    }

    pub fn dims(&self) -> &[Dim] {
        &self.0
    }
}

impl FromIterator<Dim> for Shape {
    fn from_iter<T: IntoIterator<Item = Dim>>(iter: T) -> Self {
        Shape(iter.into_iter().collect())
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, ")")
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[rustfmt::skip]
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Portable)]
#[rkyv(as = Self)]
#[repr(u8)]
pub enum DType {
    Bool,
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F16, BF16, F32, F64,
    F8E4M3FN, F8E5M2, F8E8M0FNU,
    C64, C128,
    KeyThreeFry2x32, KeyPhilox4x32,
}

impl DType {
    pub fn is_key(self) -> bool {
        matches!(self, Self::KeyThreeFry2x32 | Self::KeyPhilox4x32)
    }
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Item {
    Bool(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F16(OrderedFloat<f32>),
    F32(OrderedFloat<f32>),
    F64(OrderedFloat<f64>),
    C64(OrderedFloat<f32>, OrderedFloat<f32>),
    C128(OrderedFloat<f64>, OrderedFloat<f64>),
}

impl std::fmt::Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Bool(b) => write!(f, "{}", b),
            Item::U8(u) => write!(f, "{}", u),
            Item::U16(u) => write!(f, "{}", u),
            Item::U32(u) => write!(f, "{}", u),
            Item::U64(u) => write!(f, "{}", u),
            Item::I8(i) => write!(f, "{}", i),
            Item::I16(i) => write!(f, "{}", i),
            Item::I32(i) => write!(f, "{}", i),
            Item::I64(i) => write!(f, "{}", i),
            Item::F16(v) => write!(f, "{}", v),
            Item::F32(v) => write!(f, "{}", v),
            Item::F64(v) => write!(f, "{}", v),
            Item::C64(r, i) => write!(f, "{} + {}i", r, i),
            Item::C128(r, i) => write!(f, "{} + {}i", r, i),
        }
    }
}

use std::borrow::Cow;

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Data<'r> {
    Bool(Cow<'r, [bool]>),
    U8(Cow<'r, [u8]>),
    U16(Cow<'r, [u16]>),
    U32(Cow<'r, [u32]>),
    U64(Cow<'r, [u64]>),
    I8(Cow<'r, [i8]>),
    I16(Cow<'r, [i16]>),
    I32(Cow<'r, [i32]>),
    I64(Cow<'r, [i64]>),
    F16(Cow<'r, [OrderedFloat<f32>]>),
    F32(Cow<'r, [OrderedFloat<f32>]>),
    F64(Cow<'r, [OrderedFloat<f64>]>),
    C64(Cow<'r, [OrderedFloat<f32>]>, Cow<'r, [OrderedFloat<f32>]>),
    C128(Cow<'r, [OrderedFloat<f64>]>, Cow<'r, [OrderedFloat<f64>]>),
}

fn fmt_display_slice<T: Display>(f: &mut std::fmt::Formatter<'_>, slice: &[T]) -> std::fmt::Result {
    write!(f, "[")?;
    let mut first = true;
    for x in slice {
        if !first {
            write!(f, ", ")?;
        }
        first = false;
        write!(f, "{}", x)?;
    }
    write!(f, "]")
}

impl<'r> std::fmt::Display for Data<'r> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Data::Bool(v) => fmt_display_slice(f, v.as_ref()),
            Data::U8(v) => fmt_display_slice(f, v.as_ref()),
            Data::U16(v) => fmt_display_slice(f, v.as_ref()),
            Data::U32(v) => fmt_display_slice(f, v.as_ref()),
            Data::U64(v) => fmt_display_slice(f, v.as_ref()),
            Data::I8(v) => fmt_display_slice(f, v.as_ref()),
            Data::I16(v) => fmt_display_slice(f, v.as_ref()),
            Data::I32(v) => fmt_display_slice(f, v.as_ref()),
            Data::I64(v) => fmt_display_slice(f, v.as_ref()),
            Data::F16(v) => fmt_display_slice(f, v.as_ref()),
            Data::F32(v) => fmt_display_slice(f, v.as_ref()),
            Data::F64(v) => fmt_display_slice(f, v.as_ref()),
            Data::C64(re, im) => {
                write!(f, "[")?;
                let mut first = true;
                for (r, i) in re.iter().zip(im.iter()) {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    write!(f, "{} + {}i", r, i)?;
                }
                write!(f, "]")
            }
            Data::C128(re, im) => {
                write!(f, "[")?;
                let mut first = true;
                for (r, i) in re.iter().zip(im.iter()) {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    write!(f, "{} + {}i", r, i)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl From<&'static str> for DType {
    fn from(s: &'static str) -> Self {
        s.parse()
            .unwrap_or_else(|_| panic!("Unsupported data type: {s}"))
    }
}

impl std::str::FromStr for DType {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "bool" => DType::Bool,
            "uint8" => DType::U8,
            "uint16" => DType::U16,
            "uint32" => DType::U32,
            "uint64" => DType::U64,
            "int8" => DType::I8,
            "int16" => DType::I16,
            "int32" => DType::I32,
            "int64" => DType::I64,
            "bfloat16" => DType::BF16,
            "float8_e4m3fn" => DType::F8E4M3FN,
            "float8_e5m2" => DType::F8E5M2,
            "float8_e8m0fnu" => DType::F8E8M0FNU,
            "float16" => DType::F16,
            "float32" => DType::F32,
            "float64" => DType::F64,
            "complex64" => DType::C64,
            "complex128" => DType::C128,
            "key<fry>" => DType::KeyThreeFry2x32,
            "key<philox>" => DType::KeyPhilox4x32,
            _ => return Err("unsupported data type"),
        })
    }
}

impl Into<&'static str> for DType {
    fn into(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::BF16 => "bfloat16",
            DType::F8E4M3FN => "float8_e4m3fn",
            DType::F8E5M2 => "float8_e5m2",
            DType::F8E8M0FNU => "float8_e8m0fnu",
            DType::F16 => "float16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::C64 => "complex64",
            DType::C128 => "complex128",
            DType::KeyThreeFry2x32 => "key<fry>",
            DType::KeyPhilox4x32 => "key<philox>",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dtype_str: &'static str = (*self).into();
        write!(f, "{}", dtype_str)
    }
}

/// The metadata of an array: its shape and element type. This is the form
/// it is constructed and inspected in; stored in an expression it is an
/// [`ArrayInfoData`].
#[derive(Clone, Debug, Hash, PartialEq, Eq, rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = ArrayInfoData)]
pub struct ArrayInfo {
    shape: Shape,
    dtype: DType,
    symbols: Vec<gnx_expr::Symbol>,
}

impl ArrayInfoData {
    pub fn shape(&self) -> Shape {
        (self.shape.0.iter())
            .map(|dim| match dim {
                ArchivedDim::Fixed(n) => Dim::Fixed(n.to_native() as usize),
                ArchivedDim::Bounded { symbol, max } => Dim::Bounded {
                    symbol: *symbol,
                    max: max.to_native(),
                },
                ArchivedDim::Dynamic => Dim::Dynamic,
                ArchivedDim::Jagged(i) => Dim::Jagged(i.to_native() as usize),
            })
            .collect()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The metadata in the form it is constructed and inspected in.
    pub fn to_native(&self) -> ArrayInfo {
        ArrayInfo::new(self.shape(), self.dtype())
    }
}

impl gnx_expr::Info for ArrayInfoData {
    fn symbols(&self) -> &[gnx_expr::Symbol] {
        &self.symbols
    }

    fn remap_symbols(
        &self,
        f: &mut dyn FnMut(gnx_expr::Symbol) -> gnx_expr::Symbol,
    ) -> Option<gnx_expr::VarInfo> {
        if self.symbols.is_empty() {
            return None;
        }
        let shape: Shape = self
            .shape()
            .0
            .into_iter()
            .map(|d| match d {
                Dim::Bounded { symbol, max } => Dim::Bounded {
                    symbol: f(symbol),
                    max,
                },
                d => d,
            })
            .collect();
        Some(gnx_expr::VarInfo::new(ArrayInfo::new(shape, self.dtype)))
    }
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.to_native(), f)
    }
}

gnx_expr::dialect! {
    dyn gnx_expr::Info, "array" => { "info" => ArrayInfoData }
}

impl Display for ArrayInfoData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.to_native(), f)
    }
}

impl ArrayInfo {
    /// The array metadata `info` holds, if that is what it is.
    pub fn from_info(info: &gnx_expr::VarInfo) -> Option<Self> {
        info.downcast::<ArrayInfoData>()
            .map(ArrayInfoData::to_native)
    }
}

impl ArrayInfo {
    pub fn new(shape: impl IntoShape, dtype: DType) -> Self {
        let shape = shape.into_shape();
        let symbols = shape
            .dims()
            .iter()
            .filter_map(|d| match d {
                Dim::Bounded { symbol, .. } => Some(*symbol),
                _ => None,
            })
            .collect();
        ArrayInfo {
            shape,
            dtype,
            symbols,
        }
    }

    pub fn has_bounded_axes(&self) -> bool {
        !self.symbols.is_empty()
    }

    /// Abstract a selected fixed axis when constructing a JIT input
    /// signature. Reuse the symbol for inputs whose lengths must agree.
    pub fn with_bounded_axis(
        &self,
        axis: usize,
        symbol: gnx_expr::Symbol,
        max: usize,
    ) -> Result<Self, LoweringError> {
        let mut shape = self.shape.clone();
        let Some(dim) = shape.0.get_mut(axis) else {
            return Err(LoweringError::RegionSignature("axis out of range"));
        };
        if !matches!(dim,Dim::Fixed(n) if *n<=max) {
            return Err(LoweringError::RegionSignature(
                "bounded promotion needs a fixed extent within its bound",
            ));
        }
        *dim = Dim::Bounded {
            symbol,
            max: u32::try_from(max)
                .map_err(|_| LoweringError::RegionSignature("axis bound exceeds u32"))?,
        };
        Ok(Self::new(shape, self.dtype))
    }

    /// Statically sized storage used by bounded lowerings.
    pub fn capacity(&self) -> Self {
        Self::new(
            self.shape
                .dims()
                .iter()
                .map(|d| match *d {
                    Dim::Bounded { max, .. } => Dim::Fixed(max as usize),
                    other => other,
                })
                .collect::<Shape>(),
            self.dtype,
        )
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

impl Display for ArrayInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dtype = match self.dtype {
            DType::Bool => "bool",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::BF16 => "bf16",
            DType::F8E4M3FN => "f8e4m3fn",
            DType::F8E5M2 => "f8e5m2",
            DType::F8E8M0FNU => "f8e8m0fnu",
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::C64 => "c64",
            DType::C128 => "c128",
            DType::KeyThreeFry2x32 => "key<fry>",
            DType::KeyPhilox4x32 => "key<philox>",
        };
        write!(f, "<{dtype}>{}", self.shape)
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct Array(Tracer<Array>);

// SAFETY: The Array and Tracer<Array> have the same layout.
impl From<&Tracer<Array>> for &Array {
    fn from(value: &Tracer<Array>) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<Array> for Tracer<Array> {
    fn from(value: Array) -> Self {
        value.0
    }
}
impl From<Tracer<Array>> for Array {
    fn from(value: Tracer<Array>) -> Self {
        Array(value)
    }
}

impl Traceable for Array {
    type Concrete = DataHandle;
    type Info = ArrayInfoData;
}

impl Array {
    /// A shape/dtype argument for JIT lowering; no storage is allocated.
    pub fn placeholder(info: ArrayInfo) -> Self {
        Tracer::placeholder(gnx_expr::VarInfo::new(info)).into()
    }

    /// Wrap a concrete storage reference without transferring its data.
    pub fn from_data(data: DataHandle) -> Self {
        let info = data.to_info();
        let trace = Trace::concrete(Value::new(data), info);
        // SAFETY: both the value type and its metadata came from this array buffer.
        Self(unsafe { Tracer::unchecked_new(trace) })
    }

    pub fn data(&self) -> Option<&DataHandle> {
        self.0.try_concrete()
    }

    pub fn dtype(&self) -> DType {
        self.0.info().dtype()
    }
    pub fn shape(&self) -> Shape {
        self.0.info().shape()
    }

    /// Return the concrete extents, or an error for non-fixed dimensions.
    pub fn fixed_dims(&self) -> Result<Vec<usize>, LoweringError> {
        fixed_dims(&ArrayInfo::new(self.shape(), self.dtype()))
    }

    /// Reshape to fixed dimensions while preserving element order and dtype.
    pub fn reshape(&self, dims: impl IntoShape) -> Self {
        let info = ArrayInfo::new(dims, self.dtype());
        assert!(
            info.shape()
                .dims()
                .iter()
                .all(|d| matches!(d, Dim::Fixed(_))),
            "reshape requires fixed output dimensions"
        );
        gnx_expr::Invocation::invoke(
            crate::ops::reshape_op(),
            vec![],
            vec![self.tracer().clone()],
            vec![gnx_expr::VarInfo::new(info)],
        )
        .remove(0)
        .try_cast::<Array>()
        .expect("reshape produces an array")
        .into()
    }

    pub fn tracer(&self) -> &Tracer<Array> {
        &self.0
    }
    pub fn into_tracer(self) -> Tracer<Array> {
        self.0
    }
}

/// Mutable array traced via [`CellTracer`] (JAX-style ref).
#[derive(Clone)]
pub struct ArrayRef(CellTracer<Array>);

impl ArrayRef {
    /// Allocate a separate slot from this cell's home initializer and context.
    pub fn independent(&self) -> Self {
        Self(self.0.independent())
    }
    pub fn new(base: impl Into<Tracer<Array>>) -> Self {
        ArrayRef(CellTracer::new(base.into()))
    }

    pub fn cell(&self) -> &CellTracer<Array> {
        &self.0
    }

    pub fn get(&self) -> Array {
        Array(self.0.get())
    }

    pub fn set(&self, value: impl Into<Tracer<Array>>) {
        self.0.set(value.into());
    }

    pub fn dtype(&self) -> DType {
        self.0.get().info().dtype()
    }

    pub fn shape(&self) -> Shape {
        self.0.get().info().shape()
    }

    pub fn tracer(&self) -> &CellTracer<Array> {
        &self.0
    }
    pub fn into_tracer(self) -> CellTracer<Array> {
        self.0
    }
}

// SAFETY: ArrayRef and CellTracer<Array> have the same layout.
impl From<&CellTracer<Array>> for &ArrayRef {
    fn from(value: &CellTracer<Array>) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<CellTracer<Array>> for ArrayRef {
    fn from(value: CellTracer<Array>) -> Self {
        ArrayRef(value)
    }
}

impl From<ArrayRef> for CellTracer<Array> {
    fn from(value: ArrayRef) -> Self {
        value.0
    }
}

/// Fixed dimensions of an array, as required by every structural operation.
pub fn fixed_dims(info: &ArrayInfo) -> Result<Vec<usize>, LoweringError> {
    info.shape()
        .dims()
        .iter()
        .map(|dim| match dim {
            Dim::Fixed(size) => Ok(*size),
            _ => Err(LoweringError::RegionSignature(
                "array operation requires fixed dimensions",
            )),
        })
        .collect()
}
