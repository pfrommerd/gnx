use crate::dialects::dialects;
use crate::{Attr, AttrList, AttrMap, Eqn, Expr, Op, SubExpr, ValueInfo, Var, VarScope};
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
pub struct ExprBuilder<'a> {
    var_scope: VarScope,
    vars: BTreeMap<&'static str, Var>,
    closure: Vec<(Var, ValueInfo)>,
    inputs: Vec<(Var, ValueInfo)>,
    eqns: Vec<Eqn>,
    info: Option<ValueInfo>,
    outer: Option<&'a mut dyn FnMut(&'static str) -> Option<ValueInfo>>,
}

impl ExprBuilder<'_> {
    pub fn new(info: Option<ValueInfo>) -> Self {
        Self {
            var_scope: VarScope::new(),
            vars: BTreeMap::new(),
            closure: Vec::new(),
            inputs: Vec::new(),
            eqns: Vec::new(),
            info,
            outer: None,
        }
    }

    pub fn closure(&mut self, name: &'static str) {
        self.parameter(true, name, self.default_info());
    }

    pub fn input(&mut self, name: &'static str) {
        self.parameter(false, name, self.default_info());
    }

    pub fn operand(&mut self, name: &'static str) -> Var {
        if let Some(&var) = self.vars.get(name) {
            return var;
        }
        self.resolve(name)
            .map_or_else(|| self.var(name), |(var, _)| var)
    }

    fn default_info(&self) -> ValueInfo {
        self.info.clone().expect("missing ValueInfo annotation")
    }

    pub fn parameter(&mut self, closure: bool, name: &'static str, info: ValueInfo) {
        let var = self.var(name);
        let params = if closure {
            &mut self.closure
        } else {
            &mut self.inputs
        };
        params.push((var, info));
    }

    pub fn output(&mut self, name: &'static str) -> (Var, ValueInfo) {
        self.typed_output(name, self.default_info())
    }

    pub fn typed_output(&mut self, name: &'static str, info: ValueInfo) -> (Var, ValueInfo) {
        let var = if name == "_" {
            Var::hole()
        } else {
            self.var(name)
        };
        (var, info)
    }

    pub fn result(&mut self, name: &'static str) -> Var {
        if name == "_" {
            Var::hole()
        } else {
            self.operand(name)
        }
    }

    pub fn subexpr(&self, body: impl Into<Arc<Expr>>, captures: Vec<Var>) -> SubExpr {
        SubExpr::new(body.into(), captures)
    }

    pub fn inline_subexpr(&mut self, build: impl FnOnce(ExprBuilder<'_>) -> Expr) -> SubExpr {
        let mut child = ExprBuilder::new(self.info.clone());
        let mut captures = Vec::new();
        // Resolving through this builder also captures names used only by deeper regions.
        let mut outer = |name| {
            let (var, info) = self.resolve(name)?;
            captures.push(var);
            Some(info)
        };
        child.outer = Some(&mut outer);
        SubExpr::new(Arc::new(build(child)), captures)
    }

    pub fn eqn(
        &mut self,
        outputs: Vec<(Var, ValueInfo)>,
        dialect: &'static str,
        op: &'static str,
        attrs: AttrMap,
        inputs: Vec<Var>,
        subexprs: Vec<SubExpr>,
    ) {
        let op = op_by_name(dialect, op, attrs);
        self.eqns
            .push(Eqn::new_with_subexprs(op, subexprs, inputs, outputs));
    }

    pub fn finish(self, outputs: Vec<Var>) -> Expr {
        Expr::new(self.closure, self.inputs, self.eqns, outputs)
    }

    fn resolve(&mut self, name: &'static str) -> Option<(Var, ValueInfo)> {
        if let Some(&var) = self.vars.get(name) {
            let info = self
                .closure
                .iter()
                .chain(&self.inputs)
                .chain(self.eqns.iter().flat_map(Eqn::outputs))
                .find_map(|(defined, info)| (*defined == var).then(|| info.clone()))?;
            return Some((var, info));
        }
        let info = (self.outer.as_mut()?)(name)?;
        let var = self.var(name);
        self.closure.push((var, info.clone()));
        Some((var, info))
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
pub fn subexpr(body: impl Into<Arc<Expr>>, captures: Vec<Var>) -> SubExpr {
    SubExpr::new(body.into(), captures)
}

pub fn string_attr(value: &'static str) -> Attr {
    Attr::from(value)
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
        $crate::Attr::from($value)
    };
}

/// Build an immutable computation expression. Cell actions belong to HExpr.
/// Annotate definitions with Rust expressions yielding `ValueInfo`. An owned
/// `ValueInfo` is moved in; `&info` clones it. The older `expr!(info; { ... })`
/// form supplies default metadata for unannotated definitions.
/// Use `;` between captures and inputs: `|captured: &info; x: make_info()| => { ... }`.
/// Return values with `in (x)`,
/// `in (x, y)`, or `in ()`. Write operation operands without
/// parentheses: `z: &info = dialect::op captured x;`.
/// If an annotation can be confused with a DSL delimiter, wrap it in braces.
/// Inline a region with `dialect::op [|x: &info| => { in (outer, x) }] x`.
/// Only explicit inputs appear in the region's header; references to enclosing
/// variables (like `outer`) are captured implicitly. Use `name = |...| =>
/// { ... }` before `in |...| => { ... }` to define a shared region, then refer to it
/// with `dialect::op [@name{outer}] x`. Captures are positional. To attach a prebuilt
/// `Expr` or `Arc<Expr>`, use `dialect::op[body => (captures)] x y`.
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
    (|| $($body:tt)*) => { $crate::expr!(| | $($body)*) };
    (| $($body:tt)*) => {{
        let mut __builder = $crate::expr_macro::ExprBuilder::new(None);
        $crate::__typed_lambda!(__builder, __expr_typed_params, __expr_eqns; [|] first [] $($body)*)
    }};
    ({ || $($body:tt)* }) => { $crate::expr!(|| $($body)*) };
    ({ | $($body:tt)* }) => { $crate::expr!(| $($body)*) };
    (lambda $($body:tt)*) => {
        $crate::expr!({ lambda $($body)* })
    };
    ({ lambda $($body:tt)* }) => {{
        let mut __builder = $crate::expr_macro::ExprBuilder::new(None);
        $crate::__typed_lambda!(__builder, __expr_typed_params, __expr_eqns; [] first [] $($body)*)
    }};
    ($name:ident = $($tail:tt)*) => {
        $crate::__expr_named!(expr, []; $name = $($tail)*)
    };
    ($info:expr; $name:ident = $($tail:tt)*) => {{
        let __default = $crate::ValueInfo::from($info);
        $crate::__expr_named!(expr, [&__default;]; $name = $($tail)*)
    }};
    ($info:expr; || $($body:tt)*) => { $crate::expr!($info; | | $($body)*) };
    ($info:expr; { || $($body:tt)* }) => { $crate::expr!($info; || $($body)*) };
    ($info:expr; { | $($body:tt)* }) => { $crate::expr!($info; | $($body)*) };
    ($info:expr; | $($body:tt)*) => {{
        let mut __builder = $crate::expr_macro::ExprBuilder::new(Some($crate::ValueInfo::from($info)));
        $crate::__typed_lambda!(__builder, __expr_typed_params, __expr_eqns; [|] first [] $($body)*)
    }};
    ($info:expr; lambda $($body:tt)*) => {
        $crate::expr!($info; { lambda $($body)* })
    };
    ($info:expr; { lambda $($body:tt)* }) => {{
        let mut __builder = $crate::expr_macro::ExprBuilder::new(Some($crate::ValueInfo::from($info)));
        $crate::__expr_params!(__builder; closure [] $($body)*)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_named {
    ($macro:ident, [$($default:tt)*]; in $($body:tt)*) => {
        $crate::$macro!($($default)* $($body)*)
    };
    ($macro:ident, $default:tt; $name:ident = || $($tail:tt)*) => {
        $crate::__expr_named!($macro, $default; $name = | | $($tail)*)
    };
    ($macro:ident, $default:tt; $name:ident = | $($tail:tt)*) => {
        $crate::__expr_named!(@header $macro, $default, $name; [|] [] $($tail)*)
    };
    ($macro:ident, $default:tt; $name:ident = lambda $($tail:tt)*) => {
        $crate::__expr_named!(@header $macro, $default, $name; [] [] $($tail)*)
    };
    (@header $macro:ident, $default:tt, $name:ident;
        [|] [$($header:tt)*] | => { $($body:tt)* } $($tail:tt)*) => {
        $crate::__expr_named!(@define $macro, $default, $name; [| $($header)* | => { $($body)* }] $($tail)*)
    };
    (@header $macro:ident, $default:tt, $name:ident;
        [] [$($header:tt)*] => { $($body:tt)* } $($tail:tt)*) => {
        $crate::__expr_named!(@define $macro, $default, $name; [lambda $($header)* => { $($body)* }] $($tail)*)
    };
    (@header $macro:ident, $default:tt, $name:ident;
        $end:tt [$($header:tt)*] $head:tt $($tail:tt)*) => {
        $crate::__expr_named!(@header $macro, $default, $name; $end [$($header)* $head] $($tail)*)
    };
    (@define $macro:ident, [$($default:tt)*], $name:ident;
        [$($definition:tt)*] $($tail:tt)*) => {{
        let $name = ::std::sync::Arc::new(
            $crate::$macro!($($default)* $($definition)*)
                .with_name_hint(stringify!($name))
        );
        $crate::__expr_named!($macro, [$($default)*]; $($tail)*)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __typed_lambda {
    ($builder:ident, $params:ident, $body:ident; $end:tt first [$($captures:tt)*] ; $($rest:tt)*) => {{
        $crate::$params!($builder, true; $($captures)* ,);
        $crate::__typed_lambda!($builder, $params, $body; $end inputs [] $($rest)*)
    }};
    ($builder:ident, $params:ident, $body:ident; [|] $phase:ident [$($args:tt)*] | => { $($rest:tt)* }) => {{
        $crate::$params!($builder, false; $($args)* ,);
        $crate::$body!($builder; $($rest)*)
    }};
    ($builder:ident, $params:ident, $body:ident; [] $phase:ident [$($args:tt)*] => $($rest:tt)*) => {{
        $crate::$params!($builder, false; $($args)* ,);
        $crate::$body!($builder; $($rest)*)
    }};
    ($builder:ident, $params:ident, $body:ident; $end:tt $phase:ident [$($args:tt)*] $head:tt $($rest:tt)*) => {
        $crate::__typed_lambda!($builder, $params, $body; $end $phase [$($args)* $head] $($rest)*)
    };
    ($builder:ident, $params:ident, $body:ident; [|] $phase:ident [$($args:tt)*]) => {
        compile_error!("expected `| => { ... }` after the parameters");
    };
    ($builder:ident, $params:ident, $body:ident; [] $phase:ident [$($args:tt)*]) => {
        compile_error!("expected `=>` after the lambda parameters");
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_typed_params {
    ($builder:ident, $closure:expr;) => {};
    ($builder:ident, $closure:expr; ,) => {};
    ($builder:ident, $closure:expr; $name:ident : $info:expr, $($rest:tt)*) => {{
        $builder.parameter($closure, stringify!($name), $crate::ValueInfo::from($info));
        $crate::__expr_typed_params!($builder, $closure; $($rest)*);
    }};
    ($builder:ident, $closure:expr; $name:ident, $($rest:tt)*) => {{
        if $closure {
            $builder.closure(stringify!($name));
        } else {
            $builder.input(stringify!($name));
        }
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
    ($builder:ident; { $($body:tt)* }) => { $crate::__expr_eqns!($builder; $($body)*) };
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
        [$($subexprs:tt)*] { lambda $($body:tt)* } $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] [$($subexprs)* [lambda $($body)*]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        [$($subexprs:tt)*] [{ lambda $($body:tt)* }] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] [$($subexprs)* [lambda $($body)*]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        [$($subexprs:tt)*] [lambda $($body:tt)*] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] [$($subexprs)* [lambda $($body)*]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        [$($subexprs:tt)*] [|| $($body:tt)*] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] [$($subexprs)* [| | $($body)*]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        [$($subexprs:tt)*] [| $($body:tt)*] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] [$($subexprs)* [| $($body)*]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident [$($attrs:tt)*]
        [$($subexprs:tt)*] [@$name:ident $({$($captures:tt)*})?] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] $dialect :: $op [$($attrs)*] [$($subexprs)* [@$name $({$($captures)*})?]] $($tail)*)
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
        $([$($region:tt)*])* ($($inputs:tt)*) ; $($rest:tt)*) => {{
        let __inputs = $crate::__expr_inputs!($builder; [] $($inputs)*);
        let __subexprs = vec![$($crate::__expr_subexpr!($builder, $next; $($region)*)),*];
        let __outputs = $crate::__expr_outputs!($builder; [] $($outs)*);
        $builder.eqn(__outputs, stringify!($dialect), stringify!($op),
            $crate::__expr_attrs!($($attrs)*), __inputs, __subexprs);
        $crate::$next!($builder; $($rest)*)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_subexpr {
    ($builder:ident, $next:ident; | $($body:tt)*) => {
        $crate::__expr_subexpr!(@inline $builder, $next; [|] $($body)*)
    };
    ($builder:ident, $next:ident; lambda $($body:tt)*) => {
        $crate::__expr_subexpr!(@inline $builder, $next; [] $($body)*)
    };
    (@inline $builder:ident, __expr_eqns; $end:tt $($body:tt)*) => {
        $builder.inline_subexpr(|mut __region| {
            $crate::__typed_lambda!(__region, __expr_typed_params, __expr_eqns; $end first [] $($body)*)
        })
    };
    (@inline $builder:ident, __hexpr_body; $end:tt $($body:tt)*) => {
        $builder.inline_subexpr(|mut __region| {
            $crate::__typed_lambda!(__region, __hexpr_typed_params, __hexpr_body; $end first [] $($body)*)
        })
    };
    ($builder:ident, $next:ident; @$name:ident $({$($captures:tt)*})?) => {{
        let __captures = $crate::__expr_inputs!($builder; [] $($($captures)*)?);
        $builder.subexpr(::std::sync::Arc::clone(&$name), __captures)
    }};
    ($builder:ident, $next:ident; $body:expr => ($($captures:tt)*)) => {{
        let __captures = $crate::__expr_inputs!($builder; [] $($captures)*);
        $builder.subexpr($body, __captures)
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
