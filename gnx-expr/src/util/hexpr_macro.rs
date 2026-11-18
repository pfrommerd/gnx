use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{AttrMap, HEqn, HExpr, HInstr, HSubExpr, Kind, ValueInfo, Var, VarScope};

#[doc(hidden)]
pub struct HExprBuilder {
    expr: HExpr,
    scope: VarScope,
    vars: BTreeMap<&'static str, Var>,
    info: Option<ValueInfo>,
}

impl HExprBuilder {
    pub fn new(info: Option<ValueInfo>) -> Self {
        Self {
            expr: HExpr {
                closure: vec![],
                inputs: vec![],
                instructions: vec![],
                outputs: vec![],
            },
            scope: VarScope::new(),
            vars: BTreeMap::new(),
            info,
        }
    }

    fn define(&mut self, name: &'static str) -> Var {
        assert!(!self.vars.contains_key(name), "duplicate variable `{name}`");
        let var = self.scope.create_var_with_hint(name);
        self.vars.insert(name, var);
        var
    }

    fn lookup(&self, name: &str, cell: bool) -> (Var, &ValueInfo) {
        let var = *self
            .vars
            .get(name)
            .unwrap_or_else(|| panic!("undefined variable `{name}`"));
        let info = match (self.expr.kind(&var), cell) {
            (Some(Kind::Cell(info)), true) | (Some(Kind::Value(info)), false) => info,
            (_, true) => panic!("expected a cell for `{name}`"),
            (_, false) => panic!("expected a value for `{name}`"),
        };
        (var, info)
    }

    pub fn parameter(
        &mut self,
        closure: bool,
        name: &'static str,
        cell: bool,
        info: Option<ValueInfo>,
    ) {
        let info = info
            .or_else(|| self.info.clone())
            .expect("missing ValueInfo annotation");
        let kind = if cell {
            Kind::Cell(info)
        } else {
            Kind::Value(info)
        };
        let var = self.define(name);
        let params = if closure {
            &mut self.expr.closure
        } else {
            &mut self.expr.inputs
        };
        params.push((var, kind));
    }

    pub fn create_cell(&mut self, name: &'static str, initial: &str) {
        let (initial, info) = self.lookup(initial, false);
        let kind = Kind::Cell(info.clone());
        let cell = self.define(name);
        self.expr.instructions.push(HInstr::CreateCell {
            cell,
            initial,
            kind,
        });
    }

    pub fn read_cell(&mut self, name: &'static str, cell: &str) {
        let (cell, info) = self.lookup(cell, true);
        let kind = Kind::Value(info.clone());
        let output = self.define(name);
        self.expr
            .instructions
            .push(HInstr::ReadCell { cell, output, kind });
    }

    pub fn write_cell(&mut self, cell: &str, value: &str) {
        let (cell, cell_info) = self.lookup(cell, true);
        let (value, value_info) = self.lookup(value, false);
        assert_eq!(cell_info, value_info, "cell assignment type mismatch");
        self.expr
            .instructions
            .push(HInstr::WriteCell { cell, value });
    }

    pub fn operand(&self, name: &str) -> Var {
        self.lookup(name, false).0
    }

    pub fn cell_operand(&self, name: &str) -> Var {
        self.lookup(name, true).0
    }

    pub fn subexpr(&self, body: impl Into<Arc<HExpr>>, captures: Vec<Var>) -> HSubExpr {
        HSubExpr::new(body.into(), captures)
    }

    pub fn output(&mut self, name: &'static str) -> (Var, Kind) {
        self.typed_output(
            name,
            self.info.clone().expect("missing ValueInfo annotation"),
        )
    }

    pub fn typed_output(&mut self, name: &'static str, info: ValueInfo) -> (Var, Kind) {
        // Even an unused equation result needs a fresh SSA variable.
        let var = if name == "_" {
            self.scope.create_var()
        } else {
            self.define(name)
        };
        (var, Kind::Value(info))
    }

    pub fn eqn(
        &mut self,
        outputs: Vec<(Var, Kind)>,
        dialect: &'static str,
        op: &'static str,
        attrs: AttrMap,
        inputs: Vec<Var>,
        subexprs: Vec<HSubExpr>,
    ) {
        let op = super::expr_macro::op_by_name(dialect, op, attrs);
        self.expr
            .instructions
            .push(HInstr::Apply(HEqn::new(op, subexprs, inputs, outputs)));
    }

    pub fn result(&self, name: &str, cell: bool) -> Var {
        self.lookup(name, cell).0
    }

    pub fn finish(mut self, outputs: Vec<Var>) -> HExpr {
        self.expr.outputs = outputs;
        self.expr
    }
}

/// Build a mutation-aware `HExpr` without tracing a function.
///
/// Annotate parameters and equation results with Rust expressions yielding
/// `ValueInfo`: a value moves in, and `&info` clones it. The older
/// `hexpr!(info; { ... })` form supplies default metadata. Cell reads and
/// creation inherit metadata; ownership is assigned during lowering.
/// Nested bodies are `HExpr` or `Arc<HExpr>`, never immutable `Expr`.
/// End the lambda header with `=>` and return with `in (x)`, `in (x, y)`, or `in ()`.
///
/// ```
/// use gnx_expr::{HInstr, ValueInfo, hexpr};
/// let info = ValueInfo::new(0usize);
/// let body = hexpr!({
///     lambda captured[:]: &info; value: info =>
///     local[:] = value;
///     captured[:] <- value;
///     result <- captured[:];
///     in (result, local[:])
/// });
/// assert!(matches!(body.instructions()[0], HInstr::CreateCell { .. }));
/// assert!(matches!(body.instructions()[1], HInstr::WriteCell { .. }));
/// assert!(matches!(body.instructions()[2], HInstr::ReadCell { .. }));
/// assert_eq!(body.to_string(), concat!(
///     "{ lambda captured[:]: 0; value: 0 =>\n",
///     "  local[:] = value;\n",
///     "  captured[:] <- value;\n",
///     "  result <- captured[:];\n",
///     "  in (result, local[:]) }",
/// ));
/// ```
/// ```compile_fail
/// let info = gnx_expr::ValueInfo::new(0usize);
/// let _ = gnx_expr::hexpr!({ lambda cell[:]: info => in cell[:] }); // missing parentheses
/// ```
/// ```compile_fail
/// use gnx_expr::{ValueInfo, expr, hexpr};
/// let info = ValueInfo::new(0usize);
/// let low = expr!({ lambda x: &info => in (x) });
/// let _ = hexpr!({ lambda x: &info => y: &info = builtins::call[low => ()] x; in (y) });
/// ```
#[macro_export]
macro_rules! hexpr {
    ({ lambda $($body:tt)* }) => {{
        let mut __builder = $crate::util::hexpr_macro::HExprBuilder::new(None);
        $crate::__typed_lambda!(__builder, __hexpr_typed_params, __hexpr_body; first [] $($body)*)
    }};
    ($info:expr; { lambda
        $($input:ident $([$icell:tt])? $(($iinfo:expr))?),* $(,)? =>
        $($body:tt)*
    }) => {
        $crate::hexpr!($info; { lambda ; $($input $([$icell])? $(($iinfo))?),* : $($body)* })
    };
    ($info:expr; { lambda
        $($capture:ident $([$ccell:tt])? $(($cinfo:expr))?),* $(,)? ;
        $($input:ident $([$icell:tt])? $(($iinfo:expr))?),* $(,)? =>
        $($body:tt)*
    }) => {
        $crate::hexpr!($info; { lambda $($capture $([$ccell])? $(($cinfo))?),* ;
            $($input $([$icell])? $(($iinfo))?),* : $($body)* })
    };
    ($info:expr; { lambda
        $($input:ident $([$icell:tt])? $(($iinfo:expr))?),* $(,)? :
        $($body:tt)*
    }) => {
        $crate::hexpr!($info; { lambda ; $($input $([$icell])? $(($iinfo))?),* : $($body)* })
    };
    ($info:expr; { lambda
        $($input:ident $([$icell:tt])? $(($iinfo:expr))?),* $(,)? .
        $($body:tt)*
    }) => {
        $crate::hexpr!($info; { lambda $($input $([$icell])? $(($iinfo))?),* : $($body)* })
    };
    ($info:expr; { lambda
        $($capture:ident $([$ccell:tt])? $(($cinfo:expr))?),* $(,)? ;
        $($input:ident $([$icell:tt])? $(($iinfo:expr))?),* $(,)? .
        $($body:tt)*
    }) => {
        $crate::hexpr!($info; { lambda $($capture $([$ccell])? $(($cinfo))?),* ;
            $($input $([$icell])? $(($iinfo))?),* : $($body)* })
    };
    ($info:expr; { lambda
        $($capture:ident $([$ccell:tt])? $(($cinfo:expr))?),* $(,)? ;
        $($input:ident $([$icell:tt])? $(($iinfo:expr))?),* $(,)? :
        $($body:tt)*
    }) => {{
        let mut __builder = $crate::util::hexpr_macro::HExprBuilder::new(Some($crate::ValueInfo::from($info)));
        $(__builder.parameter(true, stringify!($capture), $crate::__hexpr_cell!($($ccell)?),
            $crate::__hexpr_info!($($cinfo)?));)*
        $(__builder.parameter(false, stringify!($input), $crate::__hexpr_cell!($($icell)?),
            $crate::__hexpr_info!($($iinfo)?));)*
        $crate::__hexpr_body!(__builder; $($body)*)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __hexpr_cell {
    () => {
        false
    };
    (:) => {
        true
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __hexpr_info {
    () => {
        None
    };
    ($info:expr) => {
        Some($crate::ValueInfo::from($info))
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __hexpr_typed_params {
    ($builder:ident, $closure:expr;) => {};
    ($builder:ident, $closure:expr; ,) => {};
    ($builder:ident, $closure:expr; $name:ident $([$cell:tt])? : $info:expr, $($rest:tt)*) => {{
        $builder.parameter($closure, stringify!($name), $crate::__hexpr_cell!($($cell)?),
            Some($crate::ValueInfo::from($info)));
        $crate::__hexpr_typed_params!($builder, $closure; $($rest)*);
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __hexpr_body {
    ($builder:ident; $out:tt : $($tail:tt)*) => {
        $crate::__typed_eqn!($builder, __hexpr_body; [$out :] $($tail)*)
    };
    ($builder:ident; in ($($result:ident $([$cell:tt])?),* $(,)?) $(;)?) => {{
        let __outputs = vec![$($builder.result(stringify!($result), $crate::__hexpr_cell!($($cell)?))),*];
        $builder.finish(__outputs)
    }};
    ($builder:ident; $cell:ident [:] = $value:ident; $($rest:tt)*) => {{
        $builder.create_cell(stringify!($cell), stringify!($value));
        $crate::__hexpr_body!($builder; $($rest)*)
    }};
    ($builder:ident; $cell:ident [:] <- $value:ident; $($rest:tt)*) => {{
        $builder.write_cell(stringify!($cell), stringify!($value));
        $crate::__hexpr_body!($builder; $($rest)*)
    }};
    ($builder:ident; $value:ident <- $cell:ident [:]; $($rest:tt)*) => {{
        $builder.read_cell(stringify!($value), stringify!($cell));
        $crate::__hexpr_body!($builder; $($rest)*)
    }};
    ($builder:ident; $dialect:ident :: $op:ident $($tail:tt)*) => {
        $crate::__expr_operation!($builder, __hexpr_body; [] $dialect :: $op $($tail)*)
    };
    ($builder:ident; $($outs:tt),+ = $dialect:ident :: $op:ident $($tail:tt)*) => {
        $crate::__expr_operation!($builder, __hexpr_body; [$($outs),+] $dialect :: $op $($tail)*)
    };
}
