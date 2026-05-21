use crate::{Effect, Eqn, Expr, Input, SubExpr, Var};
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

    pub fn fmt_input(&self, input: &Input, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_for(input.var()))?;
        match input.effect() {
            Effect::Unused | Effect::Read => Ok(()),
            Effect::Write | Effect::ReadWrite => write!(f, "[:]"),
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
        for (i, output) in eqn.outputs().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.name_for(output))?;
        }
        if !eqn.outputs().is_empty() {
            write!(f, " = ")?;
        }
        self.fmt_eqn_op(eqn, f, printer, indent_level)?;
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
        let mut printer = PrettyPrinter::new(PrettyConfig::new());
        self.fmt_expr_with_printer(expr, f, &mut printer, 0)
    }

    fn fmt_expr_with_printer(
        &self,
        expr: &Expr,
        f: &mut fmt::Formatter<'_>,
        printer: &mut PrettyPrinter,
        indent_level: usize,
    ) -> fmt::Result {
        write!(f, "{{ lambda ")?;
        for (i, input) in expr.closure().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            self.bind(input.var());
            self.fmt_input(input, f)?;
        }
        write!(f, " ; ")?;
        for (i, input) in expr.inputs().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            self.bind(input.var());
            self.fmt_input(input, f)?;
        }
        writeln!(f, " .")?;
        for eqn in expr.eqns() {
            printer.write_indent(f, indent_level + 1)?;
            self.fmt_eqn_with_printer(eqn, f, printer, indent_level)?;
            writeln!(f)?;
        }
        printer.write_indent(f, indent_level + 1)?;
        write!(f, "in (")?;
        for (i, output) in expr.outputs().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.name_for(output))?;
        }
        write!(f, ") }}")
    }

    fn fmt_eqn_op(
        &self,
        eqn: &Eqn,
        f: &mut fmt::Formatter<'_>,
        printer: &mut PrettyPrinter,
        indent_level: usize,
    ) -> fmt::Result {
        let op = eqn.op();
        write!(f, "{}::{}", op.impl_.dialect().name(), op.impl_.name())?;
        if !op.attrs().is_empty() {
            write!(f, "<")?;
            for (i, (key, value)) in op.attrs().iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{key}: {value}")?;
            }
            write!(f, ">")?;
        }
        for subexpr in eqn.subexprs() {
            let expr = subexpr.expr().as_ref();
            write!(f, " ")?;
            if printer.should_defer(expr) {
                write!(f, "@{}", printer.defer(subexpr, self))?;
            } else {
                let child = PrettyVarScope::new();
                for (sub_input, outer_var) in expr.closure().iter().zip(subexpr.closure()) {
                    child.alias(sub_input.var(), self.name_for(outer_var));
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
                    write!(f, "{}", self.name_for(var))?;
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
            .map(|(sub_input, outer_var)| {
                (sub_input.var().clone(), outer_scope.name_for(outer_var))
            })
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
