use std::collections::BTreeMap;

use crate::{
    HEqnSource, HExprBuilder as HExpr, HExprRef, HInstrSource, HSubExprBuilder, Kind, Op, Var,
    VarInfo,
};

// What `hexpr!` expands to calls of: builds a high-level expression whose
// variables are referred to by name.
#[doc(hidden)]
pub struct HExprBuilder<'a> {
    expr: HExpr,
    vars: BTreeMap<&'static str, Var>,
    info: Option<VarInfo>,
    outer: Option<&'a mut dyn FnMut(&'static str) -> Option<(Kind, VarInfo)>>,
}

impl HExprBuilder<'_> {
    pub fn new(info: Option<VarInfo>) -> Self {
        Self {
            expr: HExpr::new(),
            vars: BTreeMap::new(),
            info,
            outer: None,
        }
    }

    fn define(&mut self, name: &'static str) -> Var {
        assert!(!self.vars.contains_key(name), "duplicate variable `{name}`");
        let var = self.expr.fresh_var();
        self.vars.insert(name, var);
        var
    }

    fn resolve(&mut self, name: &'static str) -> Option<(Var, Kind, VarInfo)> {
        if let Some(&var) = self.vars.get(name) {
            return self
                .expr
                .definition(var)
                .map(|(kind, info)| (var, kind, info.clone()));
        }
        let (kind, info) = (self.outer.as_mut()?)(name)?;
        let var = self.define(name);
        self.expr.closure_inputs.push((var, kind, info.clone()));
        Some((var, kind, info))
    }

    fn lookup(&mut self, name: &'static str, cell: bool) -> (Var, VarInfo) {
        let (var, kind, info) = self
            .resolve(name)
            .unwrap_or_else(|| panic!("undefined variable `{name}`"));
        let info = match (kind, cell) {
            (Kind::Cell, true) | (Kind::Value, false) => info,
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
        info: Option<VarInfo>,
    ) {
        let info = info
            .or_else(|| self.info.clone())
            .expect("missing VarInfo annotation");
        let kind = if cell { Kind::Cell } else { Kind::Value };
        let var = self.define(name);
        let params = if closure {
            &mut self.expr.closure_inputs
        } else {
            &mut self.expr.inputs
        };
        params.push((var, kind, info));
    }

    pub fn create_cell(&mut self, name: &'static str, initial: &'static str) {
        let (initial, info) = self.lookup(initial, false);
        let cell = self.define(name);
        self.expr.instrs.push(HInstrSource::CreateCell {
            cell,
            initial,
            info,
        });
    }

    pub fn read_cell(&mut self, name: &'static str, cell: &'static str) {
        let (cell, info) = self.lookup(cell, true);
        let output = self.define(name);
        self.expr
            .instrs
            .push(HInstrSource::ReadCell { cell, output, info });
    }

    pub fn write_cell(&mut self, cell: &'static str, value: &'static str) {
        let (cell, cell_info) = self.lookup(cell, true);
        let (value, value_info) = self.lookup(value, false);
        assert_eq!(cell_info, value_info, "cell assignment type mismatch");
        self.expr
            .instrs
            .push(HInstrSource::WriteCell { cell, value });
    }

    pub fn operand(&mut self, name: &'static str) -> Var {
        self.lookup(name, false).0
    }

    pub fn cell_operand(&mut self, name: &'static str) -> Var {
        self.lookup(name, true).0
    }

    pub fn subexpr(&self, body: impl Into<HExprRef>, captures: Vec<Var>) -> HSubExprBuilder {
        HSubExprBuilder::new(body).with_vars(captures)
    }

    pub fn inline_subexpr(
        &mut self,
        build: impl FnOnce(HExprBuilder<'_>) -> HExpr,
    ) -> HSubExprBuilder {
        let mut child = HExprBuilder::new(self.info.clone());
        let mut captures = Vec::new();
        let mut outer = |name| {
            let (var, kind, info) = self.resolve(name)?;
            captures.push(var);
            Some((kind, info))
        };
        child.outer = Some(&mut outer);
        HSubExprBuilder::new(build(child)).with_vars(captures)
    }

    pub fn output(&mut self, name: &'static str) -> (Var, Kind, VarInfo) {
        self.typed_output(name, self.info.clone().expect("missing VarInfo annotation"))
    }

    pub fn typed_output(&mut self, name: &'static str, info: VarInfo) -> (Var, Kind, VarInfo) {
        // A hole discards the result.
        let var = if name == "_" {
            Var::hole()
        } else {
            self.define(name)
        };
        (var, Kind::Value, info)
    }

    pub fn eqn(
        &mut self,
        outputs: Vec<(Var, Kind, VarInfo)>,
        op: Op,
        inputs: Vec<Var>,
        subexprs: Vec<HSubExprBuilder>,
    ) {
        self.expr.instrs.push(HInstrSource::Apply(HEqnSource::new(
            op, subexprs, inputs, outputs,
        )));
    }

    pub fn result(&mut self, name: &'static str, cell: bool) -> Var {
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
/// `VarInfo`: a value moves in, and `&info` clones it. The older
/// `hexpr!(info; { ... })` form supplies default metadata. Cell reads and
/// creation inherit metadata.
/// Operations resolve through the caller's `ops` module, just as in
/// [`expr!`](crate::expr). Write constructor arguments in angle brackets and
/// equation operands without parentheses.
/// Inline a region with `dialect::op [|| => (outer[:])]` to refer to
/// an enclosing cell directly. Shared definitions go before `in |...| => { ... }`
/// and are referenced with `dialect::op [@name{outer[:]}]`. To attach a prebuilt
/// expression (an `HExprBuilder`, an `HExprRef` or an `HExprHandle`), use
/// `dialect::op[body => (captures)]`. The result is an `HExprBuilder`: call
/// `build()` on it for a root.
/// Write parameters as `|captured[:]: &info; value: &info| => { ... }` and
/// return with `in (x)`, `in (x, y)`, or `in ()`. A body with no instructions
/// can be just the result tuple.
///
/// ```
/// use gnx_expr::{HInstr, VarInfo, hexpr};
/// let info = VarInfo::new(0usize);
/// let root = hexpr!(|captured[:]: &info; value: info| => {
///     local[:] = value;
///     captured[:] <- value;
///     result <- captured[:];
///     in (result, local[:])
/// })
/// .build()
/// .unwrap();
/// let body = root.view();
/// let instrs: Vec<_> = body.instrs().collect();
/// assert!(matches!(instrs[0], HInstr::CreateCell { .. }));
/// assert!(matches!(instrs[1], HInstr::WriteCell { .. }));
/// assert!(matches!(instrs[2], HInstr::ReadCell { .. }));
/// assert_eq!(body.to_string(), concat!(
///     "|a[:]: 0; b: 0| => {\n",
///     "  c[:] = b;\n",
///     "  a[:] <- b;\n",
///     "  d <- a[:];\n",
///     "  in (d, c[:])\n}",
/// ));
/// ```
/// ```compile_fail
/// let info = gnx_expr::VarInfo::new(0usize);
/// let _ = gnx_expr::hexpr!({ lambda cell[:]: info => in cell[:] }); // missing parentheses
/// ```
/// ```compile_fail
/// use gnx_expr::{VarInfo, expr, hexpr};
/// mod ops {
///     pub mod test {
///         pub fn region() -> gnx_expr::Op { unreachable!() }
///     }
/// }
/// let info = VarInfo::new(0usize);
/// let low = expr!({ lambda x: &info => in (x) });
/// let _ = hexpr!({ lambda x: &info => y: &info = test::region[low => ()] x; in (y) });
/// ```
/// ```compile_fail
/// mod ops {
///     pub mod test {
///         pub fn op() -> gnx_expr::Op { unreachable!() }
///     }
/// }
/// let info = gnx_expr::VarInfo::new(0usize);
/// let _ = gnx_expr::hexpr!(&info; |x| => { y = test::op (x); in (y) });
/// ```
#[macro_export]
macro_rules! hexpr {
    (|| $($body:tt)*) => { $crate::hexpr!(| | $($body)*) };
    (| $($body:tt)*) => {{
        let mut __builder = $crate::util::hexpr_macro::HExprBuilder::new(None);
        $crate::__typed_lambda!(__builder, __hexpr_typed_params, __hexpr_body; [|] first [] $($body)*)
    }};
    ({ || $($body:tt)* }) => { $crate::hexpr!(|| $($body)*) };
    ({ | $($body:tt)* }) => { $crate::hexpr!(| $($body)*) };
    (lambda $($body:tt)*) => {
        $crate::hexpr!({ lambda $($body)* })
    };
    ({ lambda $($body:tt)* }) => {{
        let mut __builder = $crate::util::hexpr_macro::HExprBuilder::new(None);
        $crate::__typed_lambda!(__builder, __hexpr_typed_params, __hexpr_body; [] first [] $($body)*)
    }};
    ($name:ident = $($tail:tt)*) => {
        $crate::__expr_named!(hexpr, []; $name = $($tail)*)
    };
    ($info:expr; $name:ident = $($tail:tt)*) => {{
        let __default = $crate::VarInfo::from($info);
        $crate::__expr_named!(hexpr, [&__default;]; $name = $($tail)*)
    }};
    ($info:expr; || $($body:tt)*) => { $crate::hexpr!($info; | | $($body)*) };
    ($info:expr; { || $($body:tt)* }) => { $crate::hexpr!($info; || $($body)*) };
    ($info:expr; { | $($body:tt)* }) => { $crate::hexpr!($info; | $($body)*) };
    ($info:expr; | $($body:tt)*) => {{
        let mut __builder = $crate::util::hexpr_macro::HExprBuilder::new(Some($crate::VarInfo::from($info)));
        $crate::__typed_lambda!(__builder, __hexpr_typed_params, __hexpr_body; [|] first [] $($body)*)
    }};
    ($info:expr; lambda $($body:tt)*) => {
        $crate::hexpr!($info; { lambda $($body)* })
    };
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
        let mut __builder = $crate::util::hexpr_macro::HExprBuilder::new(Some($crate::VarInfo::from($info)));
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
        Some($crate::VarInfo::from($info))
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __hexpr_typed_params {
    ($builder:ident, $closure:expr;) => {};
    ($builder:ident, $closure:expr; ,) => {};
    ($builder:ident, $closure:expr; $name:ident $([$cell:tt])? : $info:expr, $($rest:tt)*) => {{
        $builder.parameter($closure, stringify!($name), $crate::__hexpr_cell!($($cell)?),
            Some($crate::VarInfo::from($info)));
        $crate::__hexpr_typed_params!($builder, $closure; $($rest)*);
    }};
    ($builder:ident, $closure:expr; $name:ident $([$cell:tt])?, $($rest:tt)*) => {{
        $builder.parameter($closure, stringify!($name), $crate::__hexpr_cell!($($cell)?), None);
        $crate::__hexpr_typed_params!($builder, $closure; $($rest)*);
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __hexpr_body {
    ($builder:ident; { $($body:tt)* }) => { $crate::__hexpr_body!($builder; $($body)*) };
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
