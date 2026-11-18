use crate::dialects::dialects;
use crate::{
    Attr, AttrList, AttrMap, Eqn, Expr, Kind, Op, Ownership, SubExpr, ValueInfo, Var, VarScope,
};
use std::collections::BTreeMap;
use std::sync::Arc;

pub fn op_by_name(dialect_name: &'static str, op_name: &'static str, attrs: AttrMap) -> Op {
    let dialect = dialects()
        .get(dialect_name)
        .unwrap_or_else(|| panic!("unknown dialect `{dialect_name}`"));
    let operation = dialect
        .create(op_name)
        .unwrap_or_else(|| panic!("unknown operation `{dialect_name}::{op_name}`"));
    Op::new_with_attrs(operation, attrs)
}

#[doc(hidden)]
pub struct ExprBuilder {
    var_scope: VarScope,
    vars: BTreeMap<&'static str, Var>,
    closure: Vec<(Var, Ownership, ValueInfo)>,
    inputs: Vec<(Var, Ownership, ValueInfo)>,
    eqns: Vec<Eqn>,
    info: Option<ValueInfo>,
}

impl ExprBuilder {
    pub fn new(info: Option<ValueInfo>) -> Self {
        Self {
            var_scope: VarScope::new(),
            vars: BTreeMap::new(),
            closure: Vec::new(),
            inputs: Vec::new(),
            eqns: Vec::new(),
            info,
        }
    }

    pub fn closure(&mut self, name: &'static str) {
        self.parameter(true, name, Ownership::Move, self.default_info());
    }

    pub fn input(&mut self, name: &'static str) {
        self.parameter(false, name, Ownership::Move, self.default_info());
    }

    pub fn operand(&mut self, name: &'static str) -> (Var, Ownership) {
        (self.var(name), Ownership::Move)
    }

    pub fn borrowed_operand(&mut self, name: &'static str) -> (Var, Ownership) {
        (self.var(name), Ownership::Borrow)
    }

    pub fn borrowed_closure(&mut self, name: &'static str) {
        self.parameter(true, name, Ownership::Borrow, self.default_info());
    }

    pub fn borrowed_input(&mut self, name: &'static str) {
        self.parameter(false, name, Ownership::Borrow, self.default_info());
    }

    fn default_info(&self) -> ValueInfo {
        self.info.clone().expect("missing ValueInfo annotation")
    }

    pub fn parameter(
        &mut self,
        closure: bool,
        name: &'static str,
        ownership: Ownership,
        info: ValueInfo,
    ) {
        let var = self.var(name);
        let params = if closure {
            &mut self.closure
        } else {
            &mut self.inputs
        };
        params.push((var, ownership, info));
    }

    pub fn output(&mut self, name: &'static str) -> (Var, Kind) {
        self.typed_output(name, self.default_info())
    }

    pub fn typed_output(&mut self, name: &'static str, info: ValueInfo) -> (Var, Kind) {
        (self.result(name), Kind::Value(info))
    }

    pub fn result(&mut self, name: &'static str) -> Var {
        if name == "_" {
            Var::hole()
        } else {
            self.var(name)
        }
    }

    pub fn subexpr(&self, body: impl Into<Arc<Expr>>, captures: Vec<(Var, Ownership)>) -> SubExpr {
        SubExpr::new(body.into(), captures)
    }

    pub fn eqn(
        &mut self,
        outputs: Vec<(Var, Kind)>,
        dialect: &'static str,
        op: &'static str,
        attrs: AttrMap,
        inputs: Vec<(Var, Ownership)>,
        subexprs: Vec<SubExpr>,
    ) {
        let op = op_by_name(dialect, op, attrs);
        self.eqns
            .push(Eqn::new_with_subexprs(op, subexprs, inputs, outputs));
    }

    pub fn finish(self, outputs: Vec<Var>) -> Expr {
        Expr::new(self.closure, self.inputs, self.eqns, outputs)
    }

    fn var(&mut self, name: &'static str) -> Var {
        if let Some(var) = self.vars.get(name) {
            var.clone()
        } else {
            let var = self.var_scope.create_var_with_hint(name);
            self.vars.insert(name, var.clone());
            var
        }
    }
}

#[doc(hidden)]
pub fn subexpr(body: impl Into<Arc<Expr>>, captures: Vec<(Var, Ownership)>) -> SubExpr {
    SubExpr::new(body.into(), captures)
}

pub fn string_attr(value: &'static str) -> Attr {
    Attr::String(value.into())
}

pub fn info_attr<T>(value: T) -> Attr
where
    T: crate::value::AnyInfo,
{
    Attr::Info(ValueInfo::new(value))
}

pub fn list_attr(values: AttrList) -> Attr {
    Attr::List(Box::new(values))
}

pub fn map_attr(values: AttrMap) -> Attr {
    Attr::Map(Box::new(values))
}

#[macro_export]
macro_rules! attr {
    (info($value:expr)) => {
        $crate::expr_macro::info_attr($value)
    };
    ([$($value:tt),* $(,)?]) => {
        $crate::expr_macro::list_attr(vec![$($crate::attr!($value)),*])
    };
    ({$($key:ident : $value:tt),* $(,)?}) => {{
        let mut attrs = $crate::AttrMap::default();
        $(attrs.insert(stringify!($key).to_string(), $crate::attr!($value));)*
        $crate::expr_macro::map_attr(attrs)
    }};
    ($value:literal) => {
        $crate::expr_macro::string_attr($value)
    };
}

/// Build an immutable computation expression. Cell actions belong to HExpr.
/// Annotate definitions with Rust expressions yielding `ValueInfo`. An owned
/// `ValueInfo` is moved in; `&info` clones it. The older `expr!(info; { ... })`
/// form supplies default metadata for unannotated definitions.
/// In parameter, operand, and nested capture lists, `x` means owned or moved
/// and `&x` means borrowed. Returns always transfer ownership.
/// Use `;` between captures and inputs, and `=>` before the body. For example,
/// `lambda &captured: &info; x: make_info() =>`. Return values with `in (x)`,
/// `in (x, y)`, or `in ()`. Write operation operands without
/// parentheses: `z: &info = dialect::op &captured x;`.
/// If an annotation can be confused with a DSL delimiter, wrap it in braces.
/// Attach a nested expression with `dialect::op[body => (captures)] x y`;
/// `body` can be an `Expr` or `Arc<Expr>`.
///
/// ```compile_fail
/// let _ = gnx_expr::expr!(gnx_expr::ValueInfo::new(0usize); { lambda cell[:]: in (cell) });
/// ```
/// ```compile_fail
/// let info = gnx_expr::ValueInfo::new(0usize);
/// let _ = gnx_expr::expr!({ lambda x: info => in x }); // missing parentheses
/// ```
#[macro_export]
macro_rules! expr {
    ({ lambda $($body:tt)* }) => {{
        let mut __builder = $crate::expr_macro::ExprBuilder::new(None);
        $crate::__typed_lambda!(__builder, __expr_typed_params, __expr_eqns; first [] $($body)*)
    }};
    ($info:expr; { lambda $($body:tt)* }) => {{
        let mut __builder = $crate::expr_macro::ExprBuilder::new(Some($crate::ValueInfo::from($info)));
        $crate::__expr_params!(__builder; closure [] $($body)*)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __typed_lambda {
    ($builder:ident, $params:ident, $body:ident; first [$($captures:tt)*] ; $($rest:tt)*) => {{
        $crate::$params!($builder, true; $($captures)* ,);
        $crate::__typed_lambda!($builder, $params, $body; inputs [] $($rest)*)
    }};
    ($builder:ident, $params:ident, $body:ident; $phase:ident [$($args:tt)*] => $($rest:tt)*) => {{
        $crate::$params!($builder, false; $($args)* ,);
        $crate::$body!($builder; $($rest)*)
    }};
    ($builder:ident, $params:ident, $body:ident; $phase:ident [$($args:tt)*] $head:tt $($rest:tt)*) => {
        $crate::__typed_lambda!($builder, $params, $body; $phase [$($args)* $head] $($rest)*)
    };
    ($builder:ident, $params:ident, $body:ident; $phase:ident [$($args:tt)*]) => {
        compile_error!("expected `=>` after the lambda parameters");
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_typed_params {
    ($builder:ident, $closure:expr;) => {};
    ($builder:ident, $closure:expr; ,) => {};
    ($builder:ident, $closure:expr; & $name:ident : $info:expr, $($rest:tt)*) => {{
        $builder.parameter($closure, stringify!($name), $crate::Ownership::Borrow, $crate::ValueInfo::from($info));
        $crate::__expr_typed_params!($builder, $closure; $($rest)*);
    }};
    ($builder:ident, $closure:expr; $name:ident : $info:expr, $($rest:tt)*) => {{
        $builder.parameter($closure, stringify!($name), $crate::Ownership::Move, $crate::ValueInfo::from($info));
        $crate::__expr_typed_params!($builder, $closure; $($rest)*);
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __typed_eqn {
    ($builder:ident, $next:ident; [$($defs:tt)*] = $dialect:ident :: $op:ident $($tail:tt)*) => {
        $crate::__expr_operation!($builder, $next; [@typed $($defs)* ,] $dialect :: $op $($tail)*)
    };
    ($builder:ident, $next:ident; [$($defs:tt)*] $head:tt $($tail:tt)*) => {
        $crate::__typed_eqn!($builder, $next; [$($defs)* $head] $($tail)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_typed_outputs {
    ($builder:ident; [$($outputs:expr,)*] $(,)?) => { vec![$($outputs,)*] };
    ($builder:ident; [$($outputs:expr,)*] $name:tt : $info:expr, $($rest:tt)*) => {
        $crate::__expr_typed_outputs!($builder;
            [$($outputs,)* $builder.typed_output(stringify!($name), $crate::ValueInfo::from($info)),]
            $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_params {
    ($builder:ident; $phase:ident [$($args:tt)*] => $($rest:tt)*) => {
        $crate::__expr_params!($builder; $phase [$($args)*] : $($rest)*)
    };
    ($builder:ident; closure [$($inputs:tt)*] : $($rest:tt)*) => {{
        $crate::__expr_bind_params!($builder; input; $($inputs)*);
        $crate::__expr_eqns!($builder; $($rest)*)
    }};
    ($builder:ident; closure [$($inputs:tt)*] . $($rest:tt)*) => {{
        $crate::__expr_params!($builder; closure [$($inputs)*] : $($rest)*)
    }};
    ($builder:ident; closure [$($closure:tt)*] ; $($rest:tt)*) => {
        {
        $crate::__expr_bind_params!($builder; closure; $($closure)*);
        $crate::__expr_params!($builder; inputs [] $($rest)*)
        }
    };
    ($builder:ident; closure [$($closure:tt)*] $next:tt $($rest:tt)*) => {
        $crate::__expr_params!($builder; closure [$($closure)* $next] $($rest)*)
    };
    ($builder:ident; inputs [$($inputs:tt)*] : $($rest:tt)*) => {{
        $crate::__expr_bind_params!($builder; input; $($inputs)*);
        $crate::__expr_eqns!($builder; $($rest)*)
    }};
    ($builder:ident; inputs [$($inputs:tt)*] . $($rest:tt)*) => {
        {
        $crate::__expr_bind_params!($builder; input; $($inputs)*);
        $crate::__expr_eqns!($builder; $($rest)*)
        }
    };
    ($builder:ident; inputs [$($inputs:tt)*] $next:tt $($rest:tt)*) => {
        $crate::__expr_params!($builder; inputs [$($inputs)* $next] $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_bind_params {
    ($builder:ident; $kind:ident;) => {};
    ($builder:ident; $kind:ident; , $($rest:tt)*) => {
        $crate::__expr_bind_params!($builder; $kind; $($rest)*)
    };
    ($builder:ident; $kind:ident; $name:ident [:] $($rest:tt)*) => {
        compile_error!("cell mutation is structural HExpr syntax, not a computation parameter annotation");
    };
    ($builder:ident; closure; & $name:ident $($rest:tt)*) => {
        $builder.borrowed_closure(stringify!($name));
        $crate::__expr_bind_params!($builder; closure; $($rest)*)
    };
    ($builder:ident; input; & $name:ident $($rest:tt)*) => {
        $builder.borrowed_input(stringify!($name));
        $crate::__expr_bind_params!($builder; input; $($rest)*)
    };
    ($builder:ident; closure; $name:ident $($rest:tt)*) => {
        $builder.closure(stringify!($name));
        $crate::__expr_bind_params!($builder; closure; $($rest)*)
    };
    ($builder:ident; input; $name:ident $($rest:tt)*) => {
        $builder.input(stringify!($name));
        $crate::__expr_bind_params!($builder; input; $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_eqns {
    ($builder:ident; $out:tt : $($tail:tt)*) => { $crate::__typed_eqn!($builder, __expr_eqns; [$out :] $($tail)*) };
    ($builder:ident; in ($($outputs:tt)*) $(;)?) => {
        $crate::__expr_finish!($builder; [] $($outputs)*)
    };
    ($builder:ident; $dialect:ident :: $op:ident $($tail:tt)*) => {
        $crate::__expr_operation!($builder, __expr_eqns; [] $dialect :: $op $($tail)*)
    };
    ($builder:ident; $($outs:tt),+ = $dialect:ident :: $op:ident $($tail:tt)*) => {
        $crate::__expr_operation!($builder, __expr_eqns; [$($outs),+] $dialect :: $op $($tail)*)
    };
}

// Shared operation grammar for expr! and hexpr!.
#[doc(hidden)]
#[macro_export]
macro_rules! __expr_operation {
    ($builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident < $($tail:tt)*) => {
        $crate::__expr_operation!(@attrs $builder, $next; [$($outs)*] $dialect :: $op [] $($tail)*)
    };
    (@attrs $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*] > $($tail:tt)*) => {
        $crate::__expr_operation!(@args $builder, $next; [$($outs)*] $dialect :: $op [$($attrs)*] $($tail)*)
    };
    (@attrs $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*] $head:tt $($tail:tt)*) => {
        $crate::__expr_operation!(@attrs $builder, $next; [$($outs)*] $dialect :: $op [$($attrs)* $head] $($tail)*)
    };
    ($builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident $($tail:tt)*) => {
        $crate::__expr_operation!(@args $builder, $next; [$($outs)*] $dialect :: $op [] $($tail)*)
    };
    (@args $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] [] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        [$($subexprs:tt)*] [$body:expr => ($($captures:tt)*)] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] [$($subexprs)* [$body => ($($captures)*)]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        [$($subexprs:tt)*] $($tail:tt)*) => {
        $crate::__expr_operation!(@bare $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] [$($subexprs)*] [] $($tail)*)
    };
    (@bare $builder:ident, __hexpr_body; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        [$($subexprs:tt)*] [] ($($inputs:tt)*) ; $($rest:tt)*) => {
        $crate::__expr_operation!(@emit $builder, __hexpr_body;
            [$($outs)*] $dialect :: $op [$($attrs)*] $($subexprs)* ($($inputs)*) ; $($rest)*)
    };
    (@bare $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        [$($subexprs:tt)*] [$($inputs:tt)*] ; $($rest:tt)*) => {
        $crate::__expr_operation!(@emit $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] $($subexprs)* ($($inputs)*) ; $($rest)*)
    };
    (@bare $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        [$($subexprs:tt)*] [$($inputs:tt)*] $head:tt $($tail:tt)*) => {
        $crate::__expr_operation!(@bare $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] [$($subexprs)*] [$($inputs)* $head] $($tail)*)
    };
    (@emit $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        $([$body:expr => ($($captures:tt)*)])* ($($inputs:tt)*) ; $($rest:tt)*) => {{
        let __inputs = $crate::__expr_inputs!($builder; [] $($inputs)*);
        let __subexprs = vec![$({
            let __captures = $crate::__expr_inputs!($builder; [] $($captures)*);
            $builder.subexpr($body, __captures)
        }),*];
        let __outputs = $crate::__expr_outputs!($builder; [] $($outs)*);
        $builder.eqn(__outputs, stringify!($dialect), stringify!($op),
            $crate::__expr_attrs!($($attrs)*), __inputs, __subexprs);
        $crate::$next!($builder; $($rest)*)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_attrs {
    () => {
        $crate::AttrMap::default()
    };
    ($($key:ident : $value:tt),+ $(,)?) => {{
        let mut attrs = $crate::AttrMap::default();
        $(attrs.insert(stringify!($key).to_string(), $crate::attr!($value));)+
        attrs
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_inputs {
    ($builder:ident; [$($inputs:expr,)*]) => {
        vec![$($inputs,)*]
    };
    ($builder:ident; [$($inputs:expr,)*] , $($rest:tt)*) => {
        $crate::__expr_inputs!($builder; [$($inputs,)*] $($rest)*)
    };
    ($builder:ident; [$($inputs:expr,)*] $name:ident [:] $($rest:tt)*) => {
        $crate::__expr_inputs!($builder; [$($inputs,)* $builder.cell_operand(stringify!($name)),] $($rest)*)
    };
    ($builder:ident; [$($inputs:expr,)*] move $name:ident $($rest:tt)*) => {
        compile_error!("write `x` for a moved operand or `&x` for a borrowed operand");
    };
    ($builder:ident; [$($inputs:expr,)*] & $name:ident $($rest:tt)*) => {
        $crate::__expr_inputs!($builder; [$($inputs,)* $builder.borrowed_operand(stringify!($name)),] $($rest)*)
    };
    ($builder:ident; [$($inputs:expr,)*] $name:ident $($rest:tt)*) => {
        $crate::__expr_inputs!($builder; [$($inputs,)* $builder.operand(stringify!($name)),] $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_outputs {
    ($builder:ident; [] @typed $($defs:tt)*) => { $crate::__expr_typed_outputs!($builder; [] $($defs)*) };
    ($builder:ident; [$($outputs:expr,)*]) => {
        vec![$($outputs,)*]
    };
    ($builder:ident; [$($outputs:expr,)*] , $($rest:tt)*) => {
        $crate::__expr_outputs!($builder; [$($outputs,)*] $($rest)*)
    };
    ($builder:ident; [$($outputs:expr,)*] _ $($rest:tt)*) => {
        $crate::__expr_outputs!($builder; [$($outputs,)* $builder.output("_"),] $($rest)*)
    };
    ($builder:ident; [$($outputs:expr,)*] $name:ident $($rest:tt)*) => {
        $crate::__expr_outputs!($builder; [$($outputs,)* $builder.output(stringify!($name)),] $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_finish {
    ($builder:ident; [$($outputs:expr,)*]) => {
        {
            let __outputs = vec![$($outputs,)*];
            $builder.finish(__outputs)
        }
    };
    ($builder:ident; [$($outputs:expr,)*] , $($rest:tt)*) => {
        $crate::__expr_finish!($builder; [$($outputs,)*] $($rest)*)
    };
    ($builder:ident; [$($outputs:expr,)*] $name:ident $($rest:tt)*) => {
        $crate::__expr_finish!($builder; [$($outputs,)* $builder.result(stringify!($name)),] $($rest)*)
    };
}
