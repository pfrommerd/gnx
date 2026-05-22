use crate::dialects::dialects;
use crate::{
    Attr, AttrList, AttrMap, Effect, Eqn, Expr, Input, Op, SubExpr, ValueInfo, Var, VarScope,
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
    Op::new(operation, attrs)
}

#[doc(hidden)]
pub struct ExprBuilder {
    var_scope: VarScope,
    vars: BTreeMap<&'static str, Var>,
    closure: Vec<Input>,
    inputs: Vec<Input>,
    eqns: Vec<Eqn>,
}

impl ExprBuilder {
    pub fn new() -> Self {
        Self {
            var_scope: VarScope::new(),
            vars: BTreeMap::new(),
            closure: Vec::new(),
            inputs: Vec::new(),
            eqns: Vec::new(),
        }
    }

    pub fn closure(&mut self, name: &'static str, effect: Effect) {
        let var = self.var(name);
        self.closure.push(Input::new(var, effect));
    }

    pub fn input(&mut self, name: &'static str, effect: Effect) {
        let var = self.var(name);
        self.inputs.push(Input::new(var, effect));
    }

    pub fn read(&mut self, name: &'static str) -> Input {
        Input::new(self.var(name), Effect::Read)
    }

    pub fn write(&mut self, name: &'static str) -> Input {
        Input::new(self.var(name), Effect::Write)
    }

    pub fn output(&mut self, name: &'static str) -> Var {
        if name == "_" {
            Var::hole()
        } else {
            self.var(name)
        }
    }

    pub fn eqn(
        &mut self,
        outputs: Vec<Var>,
        dialect: &'static str,
        op: &'static str,
        attrs: AttrMap,
        inputs: Vec<Input>,
        subexprs: Option<Vec<Expr>>,
    ) {
        let op = op_by_name(dialect, op, attrs);
        let subexprs = subexprs
            .unwrap_or_default()
            .into_iter()
            .map(|expr| SubExpr::new(Arc::new(expr), Vec::new()))
            .collect();
        self.eqns
            .push(Eqn::new_with_subexprs(op, subexprs, inputs, outputs));
    }

    pub fn finish(self, outputs: Vec<Var>) -> Expr {
        Expr::new(
            self.closure,
            self.inputs,
            self.eqns,
            outputs,
            self.var_scope,
        )
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

#[macro_export]
macro_rules! expr {
    ({ lambda $($body:tt)* }) => {{
        let mut __builder = $crate::expr_macro::ExprBuilder::new();
        $crate::__expr_params!(__builder; closure [] $($body)*)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_params {
    ($builder:ident; closure [$($closure:tt)*] ; $($rest:tt)*) => {
        {
        $crate::__expr_bind_params!($builder; closure; $($closure)*);
        $crate::__expr_params!($builder; inputs [] $($rest)*)
        }
    };
    ($builder:ident; closure [$($closure:tt)*] $next:tt $($rest:tt)*) => {
        $crate::__expr_params!($builder; closure [$($closure)* $next] $($rest)*)
    };
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
    ($builder:ident; closure; $name:ident [:] $($rest:tt)*) => {
        $builder.closure(stringify!($name), $crate::Effect::Write);
        $crate::__expr_bind_params!($builder; closure; $($rest)*)
    };
    ($builder:ident; input; $name:ident [:] $($rest:tt)*) => {
        $builder.input(stringify!($name), $crate::Effect::Write);
        $crate::__expr_bind_params!($builder; input; $($rest)*)
    };
    ($builder:ident; closure; $name:ident $($rest:tt)*) => {
        $builder.closure(stringify!($name), $crate::Effect::Read);
        $crate::__expr_bind_params!($builder; closure; $($rest)*)
    };
    ($builder:ident; input; $name:ident $($rest:tt)*) => {
        $builder.input(stringify!($name), $crate::Effect::Read);
        $crate::__expr_bind_params!($builder; input; $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_eqns {
    ($builder:ident; in ($($outputs:tt)*) $(;)?) => {
        $crate::__expr_finish!($builder; [] $($outputs)*)
    };
    ($builder:ident; $($outs:tt),+ = $dialect:ident :: $op:ident < $($attrs:tt)* > [$($closure:tt)*] ($($inputs:tt)*) ; $($rest:tt)*) => {
        {
        let __outputs = $crate::__expr_outputs!($builder; [] $($outs),+);
        let __attrs = $crate::__expr_attrs!($($attrs)*);
        let __closure = $crate::__expr_inputs!($builder; [] $($closure)*);
        let __inputs = $crate::__expr_inputs!($builder; [] $($inputs)*);
        $builder.eqn(
            __outputs,
            stringify!($dialect),
            stringify!($op),
            __attrs,
            __closure,
            __inputs,
            None,
        );
        $crate::__expr_eqns!($builder; $($rest)*)
        }
    };
    ($builder:ident; $($outs:tt),+ = $dialect:ident :: $op:ident < $($attrs:tt)* > ($($inputs:tt)*) ; $($rest:tt)*) => {
        {
        let __outputs = $crate::__expr_outputs!($builder; [] $($outs),+);
        let __attrs = $crate::__expr_attrs!($($attrs)*);
        let __inputs = $crate::__expr_inputs!($builder; [] $($inputs)*);
        $builder.eqn(
            __outputs,
            stringify!($dialect),
            stringify!($op),
            __attrs,
            Vec::new(),
            __inputs,
            None,
        );
        $crate::__expr_eqns!($builder; $($rest)*)
        }
    };
    ($builder:ident; $($outs:tt),+ = $dialect:ident :: $op:ident [$($closure:tt)*] ($($inputs:tt)*) ; $($rest:tt)*) => {
        {
        let __outputs = $crate::__expr_outputs!($builder; [] $($outs),+);
        let __closure = $crate::__expr_inputs!($builder; [] $($closure)*);
        let __inputs = $crate::__expr_inputs!($builder; [] $($inputs)*);
        $builder.eqn(
            __outputs,
            stringify!($dialect),
            stringify!($op),
            $crate::AttrMap::default(),
            __closure,
            __inputs,
            None,
        );
        $crate::__expr_eqns!($builder; $($rest)*)
        }
    };
    ($builder:ident; $($outs:tt),+ = $dialect:ident :: $op:ident ($($inputs:tt)*) ; $($rest:tt)*) => {
        {
        let __outputs = $crate::__expr_outputs!($builder; [] $($outs),+);
        let __inputs = $crate::__expr_inputs!($builder; [] $($inputs)*);
        $builder.eqn(
            __outputs,
            stringify!($dialect),
            stringify!($op),
            $crate::AttrMap::default(),
            __inputs,
            None,
        );
        $crate::__expr_eqns!($builder; $($rest)*)
        }
    };
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
        $crate::__expr_inputs!($builder; [$($inputs,)* $builder.write(stringify!($name)),] $($rest)*)
    };
    ($builder:ident; [$($inputs:expr,)*] $name:ident $($rest:tt)*) => {
        $crate::__expr_inputs!($builder; [$($inputs,)* $builder.read(stringify!($name)),] $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __expr_outputs {
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
        $crate::__expr_finish!($builder; [$($outputs,)* $builder.output(stringify!($name)),] $($rest)*)
    };
}
