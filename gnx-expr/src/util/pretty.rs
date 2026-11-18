use crate::{Eqn, Expr, HExpr, HInstr, Kind, Op, Ownership, SubExpr, ValueInfo, Var};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub struct PrettyVarScope {
    names: RefCell<BTreeMap<Var, String>>,
    used_names: RefCell<BTreeSet<String>>,
    next_name: Cell<usize>,
}

impl Clone for PrettyVarScope {
    fn clone(&self) -> Self {
        Self {
            names: RefCell::new(self.names.borrow().clone()),
            used_names: RefCell::new(self.used_names.borrow().clone()),
            next_name: Cell::new(self.next_name.get()),
        }
    }
}

impl PrettyVarScope {
    pub fn new() -> Self {
        Self {
            names: RefCell::new(BTreeMap::new()),
            used_names: RefCell::new(BTreeSet::new()),
            next_name: Cell::new(0),
        }
    }

    pub fn bind(&self, var: &Var) -> String {
        self.name_for(var)
    }

    pub fn alias(&self, var: &Var, name: impl Into<String>) {
        if var.id().is_some() {
            let name = name.into();
            if let Some(index) = alphabetic_index(&name) {
                self.next_name.set(self.next_name.get().max(index + 1));
            }
            self.used_names.borrow_mut().insert(name.clone());
            self.names.borrow_mut().insert(var.clone(), name);
        }
    }

    pub fn fmt_var(&self, var: &Var, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_for(var))
    }

    pub fn fmt_input(&self, input: &(Var, Ownership), f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if input.1 == Ownership::Borrow {
            write!(f, "&")?;
        }
        write!(f, "{}", self.name_for(&input.0))
    }

    fn fmt_param(
        &self,
        (var, ownership, info): &(Var, Ownership, ValueInfo),
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        self.fmt_input(&(*var, *ownership), f)?;
        write!(f, ": {info}")
    }

    fn fmt_definition(&self, (var, kind): &(Var, Kind), f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_var(var, f)?;
        match kind {
            Kind::Value(info) => write!(f, ": {info}"),
            Kind::Cell(info) => write!(f, "[:]: {info}"),
        }
    }

    pub fn fmt_eqn(&self, eqn: &Eqn, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut printer = PrettyPrinter::new(PrettyConfig::new());
        self.fmt_eqn_with_printer(eqn, f, &mut printer, 0)
    }

    fn fmt_eqn_with_printer(
        &self,
        eqn: &Eqn,
        f: &mut fmt::Formatter<'_>,
        printer: &mut PrettyPrinter,
        indent_level: usize,
    ) -> fmt::Result {
        fmt_list(f, eqn.outputs(), |output, f| self.fmt_definition(output, f))?;
        if !eqn.outputs().is_empty() {
            write!(f, " = ")?;
        }
        self.fmt_eqn_op(eqn, f, printer, indent_level)?;
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
        let mut printer = PrettyPrinter::new(PrettyConfig::new());
        self.fmt_expr_with_printer(expr, f, &mut printer, 0)
    }

    /// Format the HExpr body and display metadata at its explicit definitions.
    pub fn fmt_hexpr(&self, expr: &HExpr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cell_var = |var: &Var, f: &mut fmt::Formatter<'_>| {
            self.fmt_var(var, f)?;
            if matches!(expr.kind(var), Some(Kind::Cell(_))) {
                write!(f, "[:]")?;
            }
            Ok(())
        };
        fmt_lambda(f, expr.closure(), expr.inputs(), |param, f| {
            self.fmt_definition(param, f)
        })?;
        for instruction in expr.instructions() {
            write!(f, "  ")?;
            match instruction {
                HInstr::Apply(eqn) => self.fmt_macro_eqn(eqn, f, 1, false)?,
                HInstr::CreateCell { cell, initial, .. } => {
                    cell_var(cell, f)?;
                    write!(f, " = ")?;
                    self.fmt_var(initial, f)?;
                }
                HInstr::ReadCell { cell, output, .. } => {
                    self.fmt_var(output, f)?;
                    write!(f, " <- ")?;
                    cell_var(cell, f)?;
                }
                HInstr::WriteCell { cell, value } => {
                    cell_var(cell, f)?;
                    write!(f, " <- ")?;
                    self.fmt_var(value, f)?;
                }
            }
            writeln!(f, ";")?;
        }
        write!(f, "  ")?;
        fmt_return(f, expr.outputs(), cell_var)
    }

    // Print nested bodies and their bindings so they can sit inside `expr!(...)`.
    fn fmt_macro_expr(
        &self,
        expr: &Expr,
        f: &mut fmt::Formatter<'_>,
        indent: usize,
    ) -> fmt::Result {
        fmt_lambda(f, expr.closure(), expr.inputs(), |input, f| {
            self.fmt_param(input, f)
        })?;
        let padding = (indent + 1) * 2;
        for eqn in expr.eqns() {
            write!(f, "{:padding$}", "")?;
            self.fmt_macro_eqn(eqn, f, indent + 1, true)?;
            writeln!(f, ";")?;
        }
        write!(f, "{:padding$}", "")?;
        fmt_return(f, expr.outputs(), |var, f| self.fmt_var(var, f))
    }

    fn fmt_macro_eqn(
        &self,
        eqn: &Eqn,
        f: &mut fmt::Formatter<'_>,
        indent: usize,
        ownership: bool,
    ) -> fmt::Result {
        if !eqn.outputs().is_empty() {
            fmt_list(f, eqn.outputs(), |output, f| self.fmt_definition(output, f))?;
            write!(f, " = ")?;
        }
        fmt_op(eqn.op(), f)?;
        let operand = |input: &(Var, Ownership), f: &mut fmt::Formatter<'_>| {
            if ownership {
                self.fmt_input(input, f)
            } else {
                self.fmt_var(&input.0, f)
            }
        };
        for subexpr in eqn.subexprs() {
            write!(f, "[expr!(")?;
            PrettyVarScope::new().fmt_macro_expr(subexpr.expr(), f, indent)?;
            write!(f, ") => (")?;
            fmt_list(f, subexpr.closure(), operand)?;
            write!(f, ")]")?;
        }
        for input in eqn.inputs() {
            write!(f, " ")?;
            operand(input, f)?;
        }
        Ok(())
    }

    fn fmt_expr_with_printer(
        &self,
        expr: &Expr,
        f: &mut fmt::Formatter<'_>,
        printer: &mut PrettyPrinter,
        indent_level: usize,
    ) -> fmt::Result {
        fmt_lambda(f, expr.closure(), expr.inputs(), |input, f| {
            self.fmt_param(input, f)
        })?;
        for eqn in expr.eqns() {
            printer.write_indent(f, indent_level + 1)?;
            self.fmt_eqn_with_printer(eqn, f, printer, indent_level)?;
            writeln!(f, ";")?;
        }
        printer.write_indent(f, indent_level + 1)?;
        fmt_return(f, expr.outputs(), |var, f| self.fmt_var(var, f))
    }

    fn fmt_eqn_op(
        &self,
        eqn: &Eqn,
        f: &mut fmt::Formatter<'_>,
        printer: &mut PrettyPrinter,
        indent_level: usize,
    ) -> fmt::Result {
        fmt_op(eqn.op(), f)?;
        for subexpr in eqn.subexprs() {
            let expr = subexpr.expr().as_ref();
            write!(f, " ")?;
            if printer.should_defer(expr) {
                write!(f, "@{}", printer.defer(subexpr, self))?;
            } else {
                let child = PrettyVarScope::new();
                for (sub_input, outer_input) in expr.closure().iter().zip(subexpr.closure()) {
                    child.alias(&sub_input.0, self.name_for(&outer_input.0));
                }
                child.fmt_expr_with_printer(
                    expr,
                    f,
                    printer,
                    indent_level + printer.config.inline_subexpr_indent,
                )?;
            }
            if !subexpr.closure().is_empty() {
                write!(f, " [")?;
                for (i, var) in subexpr.closure().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_input(var, f)?;
                }
                write!(f, "]")?;
            }
        }
        Ok(())
    }

    fn name_for(&self, var: &Var) -> String {
        if var.id().is_none() {
            return "_".to_string();
        }
        if let Some(name) = self.names.borrow().get(&var).cloned() {
            return name;
        }
        let name = if let Some(hint) = var.name_hint() {
            self.unique_hint_name(hint)
        } else {
            self.next_generated_name()
        };
        self.names.borrow_mut().insert(var.clone(), name.clone());
        name
    }

    fn unique_hint_name(&self, hint: &str) -> String {
        let base = if hint.is_empty() { "v" } else { hint };
        let mut suffix = 1;
        loop {
            let candidate = if suffix == 1 {
                base.to_string()
            } else {
                format!("{base}_{suffix}")
            };
            if self.used_names.borrow_mut().insert(candidate.clone()) {
                return candidate;
            }
            suffix += 1;
        }
    }

    fn next_generated_name(&self) -> String {
        loop {
            let name = alphabetic_name(self.next_name.get());
            self.next_name.set(self.next_name.get() + 1);
            if self.used_names.borrow_mut().insert(name.clone()) {
                return name;
            }
        }
    }
}

impl Default for PrettyVarScope {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct PrettyConfig {
    max_inline_eqns: Option<usize>,
    indent: String,
    inline_subexpr_indent: usize,
}

impl PrettyConfig {
    pub fn new() -> Self {
        Self {
            max_inline_eqns: None,
            indent: "  ".to_string(),
            inline_subexpr_indent: 1,
        }
    }

    pub fn max_inline_eqns(mut self, max_inline_eqns: usize) -> Self {
        self.max_inline_eqns = Some(max_inline_eqns);
        self
    }

    pub fn indent(mut self, indent: impl Into<String>) -> Self {
        self.indent = indent.into();
        self
    }

    pub fn indent_width(self, spaces: usize) -> Self {
        self.indent(" ".repeat(spaces))
    }

    pub fn inline_subexpr_indent(mut self, indent_level: usize) -> Self {
        self.inline_subexpr_indent = indent_level;
        self
    }
}

impl Default for PrettyConfig {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PrettyExpr<'a> {
    expr: &'a Expr,
    config: PrettyConfig,
}

impl<'a> PrettyExpr<'a> {
    pub fn new(expr: &'a Expr) -> Self {
        Self {
            expr,
            config: PrettyConfig::new(),
        }
    }

    pub fn with_config(mut self, config: PrettyConfig) -> Self {
        self.config = config;
        self
    }

    pub fn max_inline_eqns(mut self, max_inline_eqns: usize) -> Self {
        self.config = self.config.max_inline_eqns(max_inline_eqns);
        self
    }

    pub fn indent(mut self, indent: impl Into<String>) -> Self {
        self.config = self.config.indent(indent);
        self
    }

    pub fn indent_width(mut self, spaces: usize) -> Self {
        self.config = self.config.indent_width(spaces);
        self
    }

    pub fn inline_subexpr_indent(mut self, indent_level: usize) -> Self {
        self.config = self.config.inline_subexpr_indent(indent_level);
        self
    }

    pub fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut printer = PrettyPrinter::new(self.config.clone());
        let scope = PrettyVarScope::new();
        scope.fmt_expr_with_printer(self.expr, f, &mut printer, 0)?;
        printer.fmt_deferred(f)
    }
}

type EqnKey = usize;

struct PrettyPrinter {
    config: PrettyConfig,
    deferred: Vec<DeferredExpr>,
    deferred_names: BTreeMap<EqnKey, String>,
    used_expr_names: BTreeSet<String>,
}

struct DeferredExpr {
    name: String,
    expr: Arc<Expr>,
    aliases: Vec<(Var, String)>,
}

impl PrettyPrinter {
    fn new(config: PrettyConfig) -> Self {
        Self {
            config,
            deferred: Vec::new(),
            deferred_names: BTreeMap::new(),
            used_expr_names: BTreeSet::new(),
        }
    }

    fn write_indent(&self, f: &mut fmt::Formatter<'_>, level: usize) -> fmt::Result {
        for _ in 0..level {
            write!(f, "{}", self.config.indent)?;
        }
        Ok(())
    }

    fn should_defer(&self, expr: &Expr) -> bool {
        self.config
            .max_inline_eqns
            .is_some_and(|max_inline_eqns| expr.eqns().len() > max_inline_eqns)
    }

    fn defer(&mut self, subexpr: &SubExpr, outer_scope: &PrettyVarScope) -> String {
        let expr = subexpr.expr();
        let key = Arc::as_ptr(expr) as EqnKey;
        if let Some(name) = self.deferred_names.get(&key).cloned() {
            return name;
        }
        let name = self.unique_expr_name(expr.name_hint().unwrap_or("expr"));
        let aliases = expr
            .closure()
            .iter()
            .zip(subexpr.closure())
            .map(|(sub_input, outer_input)| (sub_input.0, outer_scope.name_for(&outer_input.0)))
            .collect();
        self.deferred.push(DeferredExpr {
            name: name.clone(),
            expr: Arc::clone(expr),
            aliases,
        });
        self.deferred_names.insert(key, name.clone());
        name
    }

    fn unique_expr_name(&mut self, hint: &str) -> String {
        let base = if hint.is_empty() { "expr" } else { hint };
        let mut suffix = 1;
        loop {
            let candidate = if suffix == 1 {
                base.to_string()
            } else {
                format!("{base}-{suffix}")
            };
            if self.used_expr_names.insert(candidate.clone()) {
                return candidate;
            }
            suffix += 1;
        }
    }

    fn fmt_deferred(&mut self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut index = 0;
        while index < self.deferred.len() {
            let deferred = DeferredExpr {
                name: self.deferred[index].name.clone(),
                expr: self.deferred[index].expr.clone(),
                aliases: self.deferred[index].aliases.clone(),
            };
            if index == 0 {
                writeln!(f)?;
                writeln!(f, "where")?;
            }
            write!(f, "@{} = ", deferred.name)?;
            let scope = PrettyVarScope::new();
            for (var, name) in &deferred.aliases {
                scope.alias(var, name.clone());
            }
            scope.fmt_expr_with_printer(&deferred.expr, f, self, 0)?;
            index += 1;
            if index < self.deferred.len() {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

fn fmt_lambda<T>(
    f: &mut fmt::Formatter<'_>,
    closure: &[T],
    inputs: &[T],
    format: impl Fn(&T, &mut fmt::Formatter<'_>) -> fmt::Result,
) -> fmt::Result {
    write!(f, "{{ lambda")?;
    if !closure.is_empty() {
        write!(f, " ")?;
        fmt_list(f, closure, &format)?;
        write!(f, ";")?;
    }
    if !inputs.is_empty() {
        write!(f, " ")?;
        fmt_list(f, inputs, format)?;
    }
    writeln!(f, " =>")
}

fn fmt_return(
    f: &mut fmt::Formatter<'_>,
    outputs: &[Var],
    format: impl Fn(&Var, &mut fmt::Formatter<'_>) -> fmt::Result,
) -> fmt::Result {
    write!(f, "in (")?;
    fmt_list(f, outputs, format)?;
    write!(f, ") }}")
}

fn fmt_list<T>(
    f: &mut fmt::Formatter<'_>,
    items: &[T],
    format: impl Fn(&T, &mut fmt::Formatter<'_>) -> fmt::Result,
) -> fmt::Result {
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            write!(f, ", ")?;
        }
        format(item, f)?;
    }
    Ok(())
}

fn fmt_op(op: &Op, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}::{}", op.impl_.dialect().name(), op.impl_.name())?;
    if !op.attrs().is_empty() {
        write!(f, "<")?;
        for (index, (key, value)) in op.attrs().iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{key}: {value}")?;
        }
        write!(f, ">")?;
    }
    Ok(())
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
