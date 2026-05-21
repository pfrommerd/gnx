use crate::{
    Attr, AttrList, AttrMap, Effect, Eqn, Expr, Input, Op, ValueInfo, Var, VarScope, dialects,
};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fmt;

pub fn fmt_attr(attr: &Attr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match attr {
        Attr::String(string) => write!(f, "{string:?}"),
        Attr::Info(info) => write!(f, "info({info})"),
        Attr::List(list) => {
            write!(f, "[")?;
            for (i, attr) in list.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_attr(attr, f)?;
            }
            write!(f, "]")
        }
        Attr::Map(map) => {
            write!(f, "{{")?;
            for (i, (key, value)) in map.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{key}: ")?;
                fmt_attr(value, f)?;
            }
            write!(f, "}}")
        }
    }
}

#[derive(Debug)]
pub struct PrettyVarScope {
    names: RefCell<BTreeMap<Var, String>>,
    next_name: Cell<usize>,
}

impl Clone for PrettyVarScope {
    fn clone(&self) -> Self {
        Self {
            names: RefCell::new(self.names.borrow().clone()),
            next_name: Cell::new(self.next_name.get()),
        }
    }
}

impl PrettyVarScope {
    pub fn new() -> Self {
        Self {
            names: RefCell::new(BTreeMap::new()),
            next_name: Cell::new(0),
        }
    }

    pub fn bind(&self, var: Var) -> String {
        self.name_for(var)
    }

    pub fn alias(&self, var: Var, name: impl Into<String>) {
        if var.id().is_some() {
            let name = name.into();
            if let Some(index) = alphabetic_index(&name) {
                self.next_name.set(self.next_name.get().max(index + 1));
            }
            self.names.borrow_mut().insert(var, name);
        }
    }

    pub fn fmt_var(&self, var: &Var, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_for(*var))
    }

    pub fn fmt_input(&self, input: &Input, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_for(*input.var()))?;
        match input.effect() {
            Effect::Unused | Effect::Read => Ok(()),
            Effect::Write | Effect::ReadWrite => write!(f, "[:]"),
        }
    }

    pub fn fmt_eqn(&self, eqn: &Eqn, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, output) in eqn.outputs().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.name_for(*output))?;
        }
        if !eqn.outputs().is_empty() {
            write!(f, " = ")?;
        }
        self.fmt_eqn_op(eqn, f)?;
        if !eqn.closure().is_empty() {
            write!(f, " [")?;
            for (i, input) in eqn.closure().iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                self.fmt_input(input, f)?;
            }
            write!(f, "]")?;
        }
        if !eqn.inputs().is_empty() {
            write!(f, " ")?;
            for (i, input) in eqn.inputs().iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                self.fmt_input(input, f)?;
            }
        }
        Ok(())
    }

    pub fn fmt_expr(&self, expr: &Expr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ lambda ")?;
        for (i, input) in expr.closure().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            self.bind(*input.var());
            self.fmt_input(input, f)?;
        }
        write!(f, " ; ")?;
        for (i, input) in expr.inputs().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            self.bind(*input.var());
            self.fmt_input(input, f)?;
        }
        writeln!(f, " .")?;
        for eqn in expr.eqns() {
            write!(f, "  ")?;
            self.fmt_eqn(eqn, f)?;
            writeln!(f)?;
        }
        write!(f, "  in (")?;
        for (i, output) in expr.outputs().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.name_for(*output))?;
        }
        write!(f, ") }}")
    }

    fn fmt_eqn_op(&self, eqn: &Eqn, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = eqn.op();
        write!(f, "{}::{}", op.impl_.dialect().name(), op.impl_.name())?;
        if !op.attrs().is_empty() {
            write!(f, "<")?;
            for (i, (key, value)) in op.attrs().iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{key}: ")?;
                fmt_attr(value, f)?;
            }
            write!(f, ">")?;
        }
        if let Some(subexprs) = op.subexprs() {
            for subexpr in subexprs {
                write!(f, " ")?;
                let child = PrettyVarScope::new();
                for (sub_input, outer_input) in subexpr.closure().iter().zip(eqn.closure()) {
                    child.alias(*sub_input.var(), self.name_for(*outer_input.var()));
                }
                child.fmt_expr(subexpr, f)?;
            }
        }
        Ok(())
    }

    fn name_for(&self, var: Var) -> String {
        if var.id().is_none() {
            return "_".to_string();
        }
        if let Some(name) = self.names.borrow().get(&var).cloned() {
            return name;
        }
        let name = alphabetic_name(self.next_name.get());
        self.next_name.set(self.next_name.get() + 1);
        self.names.borrow_mut().insert(var, name.clone());
        name
    }
}

impl Default for PrettyVarScope {
    fn default() -> Self {
        Self::new()
    }
}

fn alphabetic_name(mut index: usize) -> String {
    let mut chars = Vec::new();
    loop {
        chars.push((b'a' + (index % 26) as u8) as char);
        index /= 26;
        if index == 0 {
            break;
        }
        index -= 1;
    }
    chars.iter().rev().collect()
}

fn alphabetic_index(name: &str) -> Option<usize> {
    let mut index = 0usize;
    for byte in name.bytes() {
        if !byte.is_ascii_lowercase() {
            return None;
        }
        index = index.checked_mul(26)?;
        index = index.checked_add((byte - b'a') as usize + 1)?;
    }
    index.checked_sub(1)
}

pub fn op_by_name(
    dialect_name: &'static str,
    op_name: &'static str,
    attrs: AttrMap,
    subexprs: Option<Vec<Expr>>,
) -> Op {
    let dialect = dialects()
        .get(dialect_name)
        .unwrap_or_else(|| panic!("unknown dialect `{dialect_name}`"));
    let operation = dialect
        .create(op_name)
        .unwrap_or_else(|| panic!("unknown operation `{dialect_name}.{op_name}`"));
    Op::new(operation, attrs, subexprs)
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
        closure: Vec<Input>,
        inputs: Vec<Input>,
        subexprs: Option<Vec<Expr>>,
    ) {
        self.eqns.push(Eqn::new(
            op_by_name(dialect, op, attrs, subexprs),
            closure,
            inputs,
            outputs,
        ));
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
            *var
        } else {
            let var = self.var_scope.create_var();
            self.vars.insert(name, var);
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
        $crate::pretty::info_attr($value)
    };
    ([$($value:tt),* $(,)?]) => {
        $crate::pretty::list_attr(vec![$($crate::attr!($value)),*])
    };
    ({$($key:ident : $value:tt),* $(,)?}) => {{
        let mut attrs = $crate::AttrMap::default();
        $(attrs.insert(stringify!($key).to_string(), $crate::attr!($value));)*
        $crate::pretty::map_attr(attrs)
    }};
    ($value:literal) => {
        $crate::pretty::string_attr($value)
    };
}

#[macro_export]
macro_rules! expr {
    ({ lambda $($body:tt)* }) => {{
        let mut __builder = $crate::pretty::ExprBuilder::new();
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
            Vec::new(),
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
