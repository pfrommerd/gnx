use crate::{EqnSource, ExprBuilder as Expr, ExprRef, Op, SubExprBuilder, Var, VarInfo, VarScope};
use std::collections::BTreeMap;

// What `expr!` expands to calls of: builds an expression whose variables
// are referred to by name.
#[doc(hidden)]
pub struct ExprBuilder<'a> {
    var_scope: VarScope,
    vars: BTreeMap<&'static str, Var>,
    closure: Vec<(Var, VarInfo)>,
    inputs: Vec<(Var, VarInfo)>,
    eqns: Vec<EqnSource>,
    info: Option<VarInfo>,
    outer: Option<&'a mut dyn FnMut(&'static str) -> Option<VarInfo>>,
}

impl ExprBuilder<'_> {
    pub fn new(info: Option<VarInfo>) -> Self {
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

    fn default_info(&self) -> VarInfo {
        self.info.clone().expect("missing VarInfo annotation")
    }

    pub fn parameter(&mut self, closure: bool, name: &'static str, info: VarInfo) {
        let var = self.var(name);
        let params = if closure {
            &mut self.closure
        } else {
            &mut self.inputs
        };
        params.push((var, info));
    }

    pub fn output(&mut self, name: &'static str) -> (Var, VarInfo) {
        self.typed_output(name, self.default_info())
    }

    pub fn typed_output(&mut self, name: &'static str, info: VarInfo) -> (Var, VarInfo) {
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

    pub fn subexpr(&self, body: impl Into<ExprRef>, captures: Vec<Var>) -> SubExprBuilder {
        SubExprBuilder::new(body).with_vars(captures)
    }

    pub fn inline_subexpr(
        &mut self,
        build: impl FnOnce(ExprBuilder<'_>) -> Expr,
    ) -> SubExprBuilder {
        let mut child = ExprBuilder::new(self.info.clone());
        let mut captures = Vec::new();
        // Resolving through this builder also captures names used only by deeper regions.
        let mut outer = |name| {
            let (var, info) = self.resolve(name)?;
            captures.push(var);
            Some(info)
        };
        child.outer = Some(&mut outer);
        SubExprBuilder::new(build(child)).with_vars(captures)
    }

    pub fn eqn(
        &mut self,
        outputs: Vec<(Var, VarInfo)>,
        op: Op,
        inputs: Vec<Var>,
        subexprs: Vec<SubExprBuilder>,
    ) {
        self.eqns
            .push(EqnSource::new(op, subexprs, inputs, outputs));
    }

    pub fn finish(self, outputs: Vec<Var>) -> Expr {
        Expr::from_parts(vec![], self.closure, self.inputs, self.eqns, outputs)
    }

    fn resolve(&mut self, name: &'static str) -> Option<(Var, VarInfo)> {
        if let Some(&var) = self.vars.get(name) {
            let info = self
                .closure
                .iter()
                .chain(&self.inputs)
                .chain(self.eqns.iter().flat_map(EqnSource::outputs))
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
            *var
        } else {
            let var = self.var_scope.create_var();
            self.vars.insert(name, var);
            var
        }
    }
}

/// Build an immutable computation expression. Cell actions belong to HExpr.
/// Annotate definitions with Rust expressions yielding `VarInfo`. An owned
/// `VarInfo` is moved in; `&info` clones it. The older `expr!(info; { ... })`
/// form supplies default metadata for unannotated definitions.
/// Use `;` between captures and inputs: `|captured: &info; x: make_info()| => { ... }`.
/// Return values with `in (x)`, `in (x, y)`, or `in ()`. A body with no
/// equations can be just the result tuple: `|x: &info| => (x, x)`.
/// Write operation operands without
/// parentheses: `z: &info = dialect::op captured x;`. Import an `ops` module
/// at the call site. The macro calls `ops::dialect::op()`, or forwards Rust
/// arguments from `dialect::op<arg, ...>`. Group arguments containing `>`
/// with parentheses or braces.
/// If an annotation can be confused with a DSL delimiter, wrap it in braces.
/// Inline a region with `dialect::op [|x: &info| => { in (outer, x) }] x`.
/// Only explicit inputs appear in the region's header; references to enclosing
/// variables (like `outer`) are captured implicitly. Use `name = |...| =>
/// { ... }` before `in |...| => { ... }` to define a shared region, then refer to it
/// with `dialect::op [@name{outer}] x`. Captures are positional. To attach a prebuilt
/// expression (an `ExprBuilder`, an `ExprRef` or an `ExprHandle`), use
/// `dialect::op[body => (captures)] x y`.
///
/// The result is an `ExprBuilder`: call `build()` on it for a root. A result
/// written `_` is a hole, which discards the value.
///
/// ```compile_fail
/// let _ = gnx_expr::expr!(gnx_expr::VarInfo::new(0usize); { lambda cell[:]: in (cell) });
/// ```
/// ```compile_fail
/// let info = gnx_expr::VarInfo::new(0usize);
/// let _ = gnx_expr::expr!({ lambda x: info => in x }); // missing parentheses
/// ```
/// ```compile_fail
/// mod ops {
///     pub mod test {
///         pub fn op(_: usize) -> gnx_expr::Op { unreachable!() }
///     }
/// }
/// let info = gnx_expr::VarInfo::new(0usize);
/// let _ = gnx_expr::expr!(&info; |x| => { y = test::op<axis: 1> x; in (y) });
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
        let __default = $crate::VarInfo::from($info);
        $crate::__expr_named!(expr, [&__default;]; $name = $($tail)*)
    }};
    ($info:expr; || $($body:tt)*) => { $crate::expr!($info; | | $($body)*) };
    ($info:expr; { || $($body:tt)* }) => { $crate::expr!($info; || $($body)*) };
    ($info:expr; { | $($body:tt)* }) => { $crate::expr!($info; | $($body)*) };
    ($info:expr; | $($body:tt)*) => {{
        let mut __builder = $crate::expr_macro::ExprBuilder::new(Some($crate::VarInfo::from($info)));
        $crate::__typed_lambda!(__builder, __expr_typed_params, __expr_eqns; [|] first [] $($body)*)
    }};
    ($info:expr; lambda $($body:tt)*) => {
        $crate::expr!($info; { lambda $($body)* })
    };
    ($info:expr; { lambda $($body:tt)* }) => {{
        let mut __builder = $crate::expr_macro::ExprBuilder::new(Some($crate::VarInfo::from($info)));
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
        [|] [$($header:tt)*] | => $body:tt $($tail:tt)*) => {
        $crate::__expr_named!(@define $macro, $default, $name; [| $($header)* | => $body] $($tail)*)
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
        let $name = $crate::Ref::from(
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
    ($builder:ident, $params:ident, $body:ident; [|] $phase:ident [$($args:tt)*] | => ($($outputs:tt)*)) => {{
        $crate::$params!($builder, false; $($args)* ,);
        $crate::$body!($builder; in ($($outputs)*))
    }};
    ($builder:ident, $params:ident, $body:ident; [] $phase:ident [$($args:tt)*] => $($rest:tt)*) => {{
        $crate::$params!($builder, false; $($args)* ,);
        $crate::$body!($builder; $($rest)*)
    }};
    ($builder:ident, $params:ident, $body:ident; $end:tt $phase:ident [$($args:tt)*] $head:tt $($rest:tt)*) => {
        $crate::__typed_lambda!($builder, $params, $body; $end $phase [$($args)* $head] $($rest)*)
    };
    ($builder:ident, $params:ident, $body:ident; [|] $phase:ident [$($args:tt)*]) => {
        compile_error!("expected `| =>` followed by a block or a result tuple");
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
        $builder.parameter($closure, stringify!($name), $crate::VarInfo::from($info));
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
            [$($outputs,)* $builder.typed_output(stringify!($name), $crate::VarInfo::from($info)),]
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
        $crate::__expr_operation!(@constructor $builder, $next;
            [$($outs)*] [ops::$dialect::$op] [] $($tail)*)
    };
    ($builder:ident, $next:ident; [$($outs:tt)*] $dialect:ident :: $op:ident $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] [ops::$dialect::$op()] [] $($tail)*)
    };
    (@constructor $builder:ident, $next:ident; [$($outs:tt)*] [$constructor:path]
        [$($args:tt)*] > $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] [$constructor($($args)*)] [] $($tail)*)
    };
    (@constructor $builder:ident, $next:ident; [$($outs:tt)*] [$constructor:path]
        [$($args:tt)*] $head:tt $($tail:tt)*) => {
        $crate::__expr_operation!(@constructor $builder, $next;
            [$($outs)*] [$constructor] [$($args)* $head] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        [$($subexprs:tt)*] { lambda $($body:tt)* } $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] [$operation] [$($subexprs)* [lambda $($body)*]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        [$($subexprs:tt)*] [{ lambda $($body:tt)* }] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] [$operation] [$($subexprs)* [lambda $($body)*]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        [$($subexprs:tt)*] [lambda $($body:tt)*] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] [$operation] [$($subexprs)* [lambda $($body)*]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        [$($subexprs:tt)*] [|| $($body:tt)*] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] [$operation] [$($subexprs)* [| | $($body)*]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        [$($subexprs:tt)*] [| $($body:tt)*] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] [$operation] [$($subexprs)* [| $($body)*]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        [$($subexprs:tt)*] [@$name:ident $({$($captures:tt)*})?] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] [$operation] [$($subexprs)* [@$name $({$($captures)*})?]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        [$($subexprs:tt)*] [$body:expr => ($($captures:tt)*)] $($tail:tt)*) => {
        $crate::__expr_operation!(@nested $builder, $next;
            [$($outs)*] [$operation] [$($subexprs)* [$body => ($($captures)*)]] $($tail)*)
    };
    (@nested $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        [$($subexprs:tt)*] $($tail:tt)*) => {
        $crate::__expr_operation!(@bare $builder, $next;
            [$($outs)*] [$operation] [$($subexprs)*] [] $($tail)*)
    };
    (@bare $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        [$($subexprs:tt)*] [$($inputs:tt)*] ; $($rest:tt)*) => {
        $crate::__expr_operation!(@emit $builder, $next;
            [$($outs)*] [$operation] $($subexprs)* ($($inputs)*) ; $($rest)*)
    };
    (@bare $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        [$($subexprs:tt)*] [$($inputs:tt)*] $head:tt $($tail:tt)*) => {
        $crate::__expr_operation!(@bare $builder, $next;
            [$($outs)*] [$operation] [$($subexprs)*] [$($inputs)* $head] $($tail)*)
    };
    (@emit $builder:ident, $next:ident; [$($outs:tt)*] [$operation:expr]
        $([$($region:tt)*])* ($($inputs:tt)*) ; $($rest:tt)*) => {{
        let __inputs = $crate::__expr_inputs!($builder; [] $($inputs)*);
        let __subexprs = vec![$($crate::__expr_subexpr!($builder, $next; $($region)*)),*];
        let __outputs = $crate::__expr_outputs!($builder; [] $($outs)*);
        $builder.eqn(__outputs, $operation, __inputs, __subexprs);
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
        $builder.subexpr(::core::clone::Clone::clone(&$name), __captures)
    }};
    ($builder:ident, $next:ident; $body:expr => ($($captures:tt)*)) => {{
        let __captures = $crate::__expr_inputs!($builder; [] $($captures)*);
        $builder.subexpr($body, __captures)
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
