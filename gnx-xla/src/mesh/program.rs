use std::any::Any;
use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::fmt::Write;
use std::io;
use std::ops::Range;

use gnx_expr::{Expr, Var};
use gnx_math::{ArrayInfo, DType, Dim, Shape, ops::AddOperation, shard};
use gnx_transforms::cf::CallOperation;
use serde::{Deserialize, Serialize};

use super::{invalid, unsupported};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) enum Kind {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    C64,
    C128,
}

impl Kind {
    fn dtype(self) -> DType {
        use Kind::*;
        match self {
            Bool => DType::Bool,
            U8 => DType::U8,
            U16 => DType::U16,
            U32 => DType::U32,
            U64 => DType::U64,
            I8 => DType::I8,
            I16 => DType::I16,
            I32 => DType::I32,
            I64 => DType::I64,
            F16 => DType::F16,
            F32 => DType::F32,
            F64 => DType::F64,
            C64 => DType::C64,
            C128 => DType::C128,
        }
    }
    fn of(dtype: DType) -> Self {
        use DType::*;
        match dtype {
            Bool => Self::Bool,
            U8 => Self::U8,
            U16 => Self::U16,
            U32 => Self::U32,
            U64 => Self::U64,
            I8 => Self::I8,
            I16 => Self::I16,
            I32 => Self::I32,
            I64 => Self::I64,
            F16 => Self::F16,
            F32 => Self::F32,
            F64 => Self::F64,
            C64 => Self::C64,
            C128 => Self::C128,
        }
    }
    fn mlir(self) -> &'static str {
        use Kind::*;
        match self {
            Bool => "i1",
            U8 => "ui8",
            U16 => "ui16",
            U32 => "ui32",
            U64 => "ui64",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            F16 => "f16",
            F32 => "f32",
            F64 => "f64",
            C64 => "complex<f32>",
            C128 => "complex<f64>",
        }
    }
    fn width(self) -> usize {
        use Kind::*;
        match self {
            Bool | I8 | U8 => 1,
            I16 | U16 | F16 => 2,
            I32 | U32 | F32 => 4,
            I64 | U64 | F64 | C64 => 8,
            C128 => 16,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) struct Tensor {
    pub dims: Vec<usize>,
    kind: Kind,
}

impl Tensor {
    pub fn of(info: &ArrayInfo) -> io::Result<Self> {
        let dims = info
            .shape()
            .dims()
            .iter()
            .map(|d| match d {
                Dim::Fixed(n) => Ok(*n),
                _ => Err(unsupported("mesh requires fixed array shapes")),
            })
            .collect::<io::Result<_>>()?;
        Ok(Self {
            dims,
            kind: Kind::of(info.dtype()),
        })
    }
    pub fn info(&self) -> ArrayInfo {
        ArrayInfo::new(Shape::fixed(self.dims.iter().copied()), self.kind.dtype())
    }
    fn mlir(&self) -> String {
        let prefix = self
            .dims
            .iter()
            .map(|n| format!("{n}x"))
            .collect::<String>();
        format!("tensor<{prefix}{}>", self.kind.mlir())
    }
    pub fn byte_len(&self) -> io::Result<usize> {
        if self.dims.contains(&0) {
            return Ok(0);
        }
        self.dims
            .iter()
            .try_fold(self.kind.width(), |bytes, &dim| bytes.checked_mul(dim))
            .ok_or_else(|| invalid("mesh array size overflow"))
    }
    fn check(&self) -> io::Result<()> {
        if self.dims.len() > 32 || self.dims.iter().any(|&n| n > i64::MAX as usize) {
            return Err(invalid("mesh tensor dimensions out of range"));
        }
        self.byte_len()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) enum Layout {
    Replicated,
    Leading,
}

#[derive(Clone, Debug)]
pub(super) struct Placed {
    pub tensor: Tensor,
    pub layout: Layout,
    pub workers: usize,
}

impl Placed {
    pub fn bounds(&self, worker: usize) -> Vec<Range<usize>> {
        let mut bounds: Vec<_> = self.tensor.dims.iter().map(|&size| 0..size).collect();
        if self.layout == Layout::Leading {
            let len = self.tensor.dims[0] / self.workers;
            bounds[0] = worker * len..(worker + 1) * len;
        }
        bounds
    }
    pub fn local(&self) -> Tensor {
        let mut tensor = self.tensor.clone();
        if self.layout == Layout::Leading {
            tensor.dims[0] /= self.workers;
        }
        tensor
    }
    pub fn byte_range(&self, worker: usize) -> io::Result<Range<usize>> {
        let width = self.local().byte_len()?;
        let start = if self.layout == Layout::Leading {
            worker * width
        } else {
            0
        };
        Ok(start..start + width)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Op {
    Add(usize, usize),
    Constraint { input: usize, layout: Layout },
}

/// A closed wire IR. Workers do not accept arbitrary HLO, MLIR, or custom calls.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Program {
    axis: String,
    workers: usize,
    inputs: Vec<Tensor>,
    ops: Vec<Op>,
    outputs: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(super) struct Placement {
    pub inputs: Vec<Placed>,
    pub outputs: Vec<Placed>,
}

impl Program {
    pub fn lower(expr: &Expr, axis: &str, workers: usize) -> io::Result<Self> {
        let infos = expr
            .closure()
            .iter()
            .chain(expr.inputs())
            .map(|(_, info)| {
                Tensor::of(
                    info.downcast::<ArrayInfo>()
                        .map_err(|_| unsupported("mesh only executes arrays"))?,
                )
            })
            .collect::<io::Result<Vec<_>>>()?;
        let mut program = Self {
            axis: axis.into(),
            workers,
            inputs: infos,
            ops: vec![],
            outputs: vec![],
        };
        program.outputs = program.emit(expr, &(0..program.inputs.len()).collect::<Vec<_>>())?;
        Ok(program)
    }

    fn emit(&mut self, expr: &Expr, args: &[usize]) -> io::Result<Vec<usize>> {
        let vars: Vec<_> = expr
            .closure()
            .iter()
            .chain(expr.inputs())
            .map(|(var, _)| *var)
            .collect();
        if vars.len() != args.len() {
            return Err(invalid("mesh call arity"));
        }
        let mut values: HashMap<Var, usize> = vars.into_iter().zip(args.iter().copied()).collect();
        for eqn in expr.eqns() {
            let op = eqn.op().operation() as &dyn Any;
            let ids: Vec<_> = eqn.inputs().iter().map(|var| values[var]).collect();
            let out = if op.is::<CallOperation>() {
                let [child] = eqn.subexprs() else {
                    return Err(invalid("mesh call needs one region"));
                };
                let args: Vec<_> = child
                    .closure()
                    .iter()
                    .map(|var| values[var])
                    .chain(ids)
                    .collect();
                self.emit(child.expr(), &args)?
            } else {
                if !eqn.subexprs().is_empty() {
                    return Err(unsupported("mesh only supports call regions"));
                }
                let [(_, info)] = eqn.outputs() else {
                    return Err(unsupported("mesh operation must have one result"));
                };
                let info = info
                    .downcast::<ArrayInfo>()
                    .map_err(|_| unsupported("mesh only executes arrays"))?;
                let tensor = Tensor::of(info)?;
                let operation = if op.is::<AddOperation>() {
                    let [a, b] = ids.as_slice() else {
                        return Err(invalid("mesh add arity"));
                    };
                    if !eqn.op().attrs().is_empty() {
                        return Err(invalid("mesh add does not take attributes"));
                    }
                    Op::Add(*a, *b)
                } else if op.is::<shard::Constraint>() || op.is::<shard::Replicate>() {
                    let [input] = ids.as_slice() else {
                        return Err(invalid("mesh annotation arity"));
                    };
                    let spec = shard::Sharding::from_op(eqn.op(), tensor.dims.len())
                        .map_err(io::Error::other)?;
                    let layout = if spec.is_replicated() {
                        Layout::Replicated
                    } else if matches!(spec.axes(), [first, rest @ ..] if first.as_slice() == [self.axis.as_str()] && rest.iter().all(Vec::is_empty))
                    {
                        Layout::Leading
                    } else {
                        return Err(unsupported(
                            "mesh currently partitions only the leading dimension on its one named axis",
                        ));
                    };
                    Op::Constraint {
                        input: *input,
                        layout,
                    }
                } else {
                    return Err(unsupported(format!(
                        "mesh does not lower {}.{}",
                        eqn.op().operation().dialect().name(),
                        eqn.op().operation().name()
                    )));
                };
                let idx = self.inputs.len() + self.ops.len();
                self.ops.push(operation);
                if self.ops.len() > 16_384 {
                    return Err(unsupported("mesh program is too large"));
                }
                let actual = match self.ops.last().unwrap() {
                    Op::Add(a, _) | Op::Constraint { input: a, .. } => self.source_tensor(*a),
                };
                if actual != &tensor {
                    return Err(invalid(
                        "mesh operation declared a mismatched output shape or dtype",
                    ));
                }
                vec![idx]
            };
            if out.len() != eqn.outputs().len() {
                return Err(invalid("mesh equation result arity"));
            }
            for ((var, info), id) in eqn.outputs().iter().zip(out) {
                if self.source_tensor(id).info()
                    != *info
                        .downcast::<ArrayInfo>()
                        .map_err(|_| invalid("mesh call result is not an array"))?
                {
                    return Err(invalid("mesh call result metadata mismatch"));
                }
                values.insert(*var, id);
            }
        }
        Ok(expr.outputs().iter().map(|v| values[v]).collect())
    }

    fn source_tensor(&self, mut i: usize) -> &Tensor {
        while i >= self.inputs.len() {
            i = match self.ops[i - self.inputs.len()] {
                Op::Add(a, _) | Op::Constraint { input: a, .. } => a,
            };
        }
        &self.inputs[i]
    }

    fn analyze(&self) -> io::Result<(Vec<Tensor>, Vec<Layout>, Placement)> {
        if !shard::valid_axis(&self.axis)
            || self.axis.len() > 64
            || !(1..=256).contains(&self.workers)
            || self.ops.len() > 16_384
            || self.inputs.len() > 1024
            || self.outputs.is_empty()
            || self.outputs.len() > 1024
        {
            return Err(invalid("invalid mesh kernel size or topology"));
        }
        fn root(parent: &mut [usize], mut k: usize) -> usize {
            while parent[k] != k {
                parent[k] = parent[parent[k]];
                k = parent[k];
            }
            k
        }
        let mut tensors = self.inputs.clone();
        for tensor in &tensors {
            tensor.check()?;
        }
        let mut parent: Vec<_> = (0..self.inputs.len() + self.ops.len()).collect();
        for op in &self.ops {
            let tensor = |id: usize| {
                tensors
                    .get(id)
                    .ok_or_else(|| invalid("invalid mesh SSA reference"))
            };
            let i = tensors.len();
            let source = match *op {
                Op::Add(a, b) => {
                    if tensor(a)? != tensor(b)? {
                        return Err(unsupported(
                            "distributed addition requires equal shapes and dtypes",
                        ));
                    }
                    let (a, b) = (root(&mut parent, a), root(&mut parent, b));
                    parent[b] = a;
                    a
                }
                Op::Constraint { input, .. } => {
                    tensor(input)?;
                    input
                }
            };
            parent[i] = root(&mut parent, source);
            tensors.push(tensor(source)?.clone());
        }
        let mut constraints = vec![None; tensors.len()];
        for (i, op) in self.ops.iter().enumerate() {
            let Op::Constraint { layout, .. } = op else {
                continue;
            };
            let root = root(&mut parent, i + self.inputs.len());
            if constraints[root]
                .replace(*layout)
                .is_some_and(|old| old != *layout)
            {
                return Err(unsupported(
                    "conflicting sharding constraints require a collective or reshard",
                ));
            }
        }
        let layouts = (0..tensors.len())
            .map(|i| {
                let layout = constraints[root(&mut parent, i)].unwrap_or(Layout::Replicated);
                let dims = &tensors[i].dims;
                if layout == Layout::Leading
                    && (dims.is_empty()
                        || dims.contains(&0)
                        || !dims[0].is_multiple_of(self.workers))
                {
                    return Err(unsupported(
                        "mesh leading dimension must be nonempty and evenly divisible by workers",
                    ));
                }
                Ok(layout)
            })
            .collect::<io::Result<Vec<_>>>()?;
        let place = |i: usize| -> io::Result<Placed> {
            Ok(Placed {
                tensor: tensors
                    .get(i)
                    .ok_or_else(|| invalid("invalid mesh output reference"))?
                    .clone(),
                layout: layouts[i],
                workers: self.workers,
            })
        };
        let placement = Placement {
            inputs: (0..self.inputs.len())
                .map(place)
                .collect::<io::Result<_>>()?,
            outputs: self
                .outputs
                .iter()
                .copied()
                .map(place)
                .collect::<io::Result<_>>()?,
        };
        Ok((tensors, layouts, placement))
    }

    pub fn placement(&self) -> io::Result<Placement> {
        Ok(self.analyze()?.2)
    }

    fn mlir(&self) -> io::Result<(String, Placement)> {
        let (tensors, layouts, placement) = self.analyze()?;
        let sharding = |i: usize| {
            let dims = (0..tensors[i].dims.len())
                .map(|d| {
                    if d == 0 && layouts[i] == Layout::Leading {
                        format!("{{\"{}\"}}", self.axis)
                    } else {
                        "{}".into()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("<@mesh, [{dims}]>")
        };
        let attrs = |i| format!(" {{sdy.sharding = #sdy.sharding{}}}", sharding(i));
        let args = (0..self.inputs.len())
            .map(|i| format!("%v{i}: {}{}", tensors[i].mlir(), attrs(i)))
            .collect::<Vec<_>>()
            .join(", ");
        let outputs = self
            .outputs
            .iter()
            .map(|&i| format!("{}{}", tensors[i].mlir(), attrs(i)))
            .collect::<Vec<_>>()
            .join(", ");
        let mut text = format!(
            "module {{\n  sdy.mesh @mesh = <[\"{}\"={}]>\n  func.func public @main({args}) -> ({outputs}) {{\n",
            self.axis, self.workers
        );
        for (offset, op) in self.ops.iter().enumerate() {
            let i = offset + self.inputs.len();
            let ty = tensors[i].mlir();
            match *op {
                Op::Add(a, b) => {
                    let name = if tensors[i].kind == Kind::Bool {
                        "or"
                    } else {
                        "add"
                    };
                    writeln!(text, "    %v{i} = stablehlo.{name} %v{a}, %v{b} : {ty}").unwrap();
                }
                Op::Constraint { input, .. } => writeln!(
                    text,
                    "    %v{i} = sdy.sharding_constraint %v{input} {} : {ty}",
                    sharding(i)
                )
                .unwrap(),
            }
        }
        let names = self
            .outputs
            .iter()
            .map(|i| format!("%v{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let types = self
            .outputs
            .iter()
            .map(|&i| tensors[i].mlir())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(text, "    return {names} : {types}\n  }}\n}}").unwrap();
        Ok((text, placement))
    }

    pub fn partition(&self) -> io::Result<(Vec<u8>, Placement)> {
        let (text, placement) = self.mlir()?;
        unsafe extern "C" fn store(context: *mut c_void, data: *const c_char, len: usize) {
            // SAFETY: the bridge supplies borrowed bytes and our synchronous output context.
            unsafe {
                (*context.cast::<Vec<u8>>())
                    .extend_from_slice(std::slice::from_raw_parts(data.cast(), len));
            }
        }
        let mut bytes = Vec::<u8>::new();
        // SAFETY: the bridge borrows the input and calls store before returning.
        let success = unsafe {
            gnx_partition_shardy(
                text.as_ptr().cast(),
                text.len(),
                self.workers as i32,
                (&mut bytes as *mut Vec<u8>).cast(),
                store,
            )
        };
        if !success {
            return Err(unsupported(format!(
                "Shardy could not partition mesh program: {}",
                String::from_utf8_lossy(&bytes)
            )));
        }
        Ok((bytes, placement))
    }
}

unsafe extern "C" {
    fn gnx_partition_shardy(
        input: *const c_char,
        len: usize,
        partitions: i32,
        context: *mut c_void,
        output: unsafe extern "C" fn(*mut c_void, *const c_char, usize),
    ) -> bool;
}
