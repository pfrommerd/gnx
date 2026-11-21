use crate::{Eqn, Expr, HExpr, HInstr, Kind, ValueInfo, Var};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub struct PrettyVarScope {
    names: RefCell<BTreeMap<Var, String>>,
    used_names: RefCell<BTreeSet<String>>,
    next_name: Cell<usize>,
    nested: bool,
}

impl Clone for PrettyVarScope {
    fn clone(&self) -> Self {
        Self {
            names: RefCell::new(self.names.borrow().clone()),
            used_names: RefCell::new(self.used_names.borrow().clone()),
            next_name: Cell::new(self.next_name.get()),
            nested: self.nested,
        }
    }
}

impl PrettyVarScope {
    pub fn new() -> Self {
        Self {
            names: RefCell::new(BTreeMap::new()),
            used_names: RefCell::new(BTreeSet::new()),
            next_name: Cell::new(0),
            nested: false,
        }
    }

    pub fn bind(&self, var: &Var) -> String {
        self.name_for(var)
    }

    pub fn alias(&self, var: &Var, name: impl Into<String>) {
        if var.id().is_some() {
            let name = name.into();
            self.used_names.borrow_mut().insert(name.clone());
            self.names.borrow_mut().insert(var.clone(), name);
        }
    }

    fn child(aliases: &[(Var, String)]) -> Self {
        let scope = Self {
            nested: true,
            ..Self::new()
        };
        for (var, name) in aliases {
            scope.alias(var, name.clone());
        }
        scope
    }

    fn aliases(
        &self,
        parameters: impl Iterator<Item = Var>,
        captures: &[Var],
    ) -> Vec<(Var, String)> {
        parameters
            .zip(captures)
            .map(|(parameter, outer)| (parameter, self.name_for(outer)))
            .collect()
    }

    fn visible_closure<'a, T>(&self, closure: &'a [T]) -> &'a [T] {
        if self.nested { &[] } else { closure }
    }

    pub fn fmt_var(&self, var: &Var, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_for(var))
    }

    fn fmt_param(&self, (var, info): &(Var, ValueInfo), f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_var(var, f)?;
        write!(f, ": {info}")
    }

    fn fmt_definition(
        &self,
        (var, kind, info): &(Var, Kind, ValueInfo),
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        self.fmt_var(var, f)?;
        match kind {
            Kind::Value => write!(f, ": {info}"),
            Kind::Cell => write!(f, "[:]: {info}"),
        }
    }

    pub fn fmt_eqn(&self, eqn: &Eqn, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let printer = PrettyPrinter::new(
            PrettyConfig::new(),
            eqn.subexprs().iter().map(|sub| sub.expr()),
        );
        printer.fmt_definitions(f)?;
        self.fmt_eqn_with_printer(eqn, f, &printer, 0)
    }

    fn fmt_eqn_with_printer(
        &self,
        eqn: &Eqn,
        f: &mut fmt::Formatter<'_>,
        printer: &PrettyPrinter<'_, Expr>,
        indent_level: usize,
    ) -> fmt::Result {
        fmt_list(f, eqn.outputs(), |output, f| self.fmt_param(output, f))?;
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
                self.fmt_var(input, f)?;
            }
        }
        Ok(())
    }

    pub fn fmt_expr(&self, expr: &Expr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PrettyExpr::new(expr).fmt_with_scope(self, f)
    }

    /// Format the HExpr body and display metadata at its explicit definitions.
    pub fn fmt_hexpr(&self, expr: &HExpr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PrettyExpr::new(expr).fmt_with_scope(self, f)
    }

    fn fmt_hexpr_at(
        &self,
        expr: &HExpr,
        f: &mut fmt::Formatter<'_>,
        printer: &PrettyPrinter<'_, HExpr>,
        indent: usize,
    ) -> fmt::Result {
        let cell_var = |var: &Var, f: &mut fmt::Formatter<'_>| {
            self.fmt_var(var, f)?;
            if matches!(expr.kind(var), Some(Kind::Cell)) {
                write!(f, "[:]")?;
            }
            Ok(())
        };
        fmt_symbol_params(f, expr.symbol_closure(), expr.symbol_inputs())?;
        fmt_lambda(
            f,
            self.visible_closure(expr.closure()),
            expr.inputs(),
            |param, f| self.fmt_definition(param, f),
        )?;
        if expr.instructions().is_empty() {
            return fmt_tuple(f, expr.outputs(), cell_var);
        }
        writeln!(f, "{{")?;
        for instruction in expr.instructions() {
            printer.write_indent(f, indent + 1)?;
            match instruction {
                HInstr::Apply(eqn) => {
                    if !eqn.outputs().is_empty() {
                        fmt_list(f, eqn.outputs(), |output, f| self.fmt_definition(output, f))?;
                        write!(f, " = ")?;
                    }
                    fmt::Display::fmt(eqn.op(), f)?;
                    for sub in eqn.subexprs() {
                        write!(f, " [")?;
                        fmt_symbol_args(f, &sub.symbol_captures)?;
                        printer.fmt_region(f, self, sub.expr(), sub.closure(), indent, cell_var)?;
                        write!(f, "]")?;
                    }
                    for var in eqn.inputs() {
                        write!(f, " ")?;
                        cell_var(var, f)?;
                    }
                }
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
        printer.fmt_return(f, indent, expr.outputs(), cell_var)
    }

    fn fmt_expr_with_printer(
        &self,
        expr: &Expr,
        f: &mut fmt::Formatter<'_>,
        printer: &PrettyPrinter<'_, Expr>,
        indent_level: usize,
    ) -> fmt::Result {
        fmt_symbol_params(f, expr.symbol_closure(), expr.symbol_inputs())?;
        fmt_lambda(
            f,
            self.visible_closure(expr.closure()),
            expr.inputs(),
            |input, f| self.fmt_param(input, f),
        )?;
        let value_var = |var: &Var, f: &mut fmt::Formatter<'_>| self.fmt_var(var, f);
        if expr.eqns().is_empty() {
            return fmt_tuple(f, expr.outputs(), value_var);
        }
        writeln!(f, "{{")?;
        for eqn in expr.eqns() {
            printer.write_indent(f, indent_level + 1)?;
            self.fmt_eqn_with_printer(eqn, f, printer, indent_level)?;
            writeln!(f, ";")?;
        }
        printer.fmt_return(f, indent_level, expr.outputs(), value_var)
    }

    fn fmt_eqn_op(
        &self,
        eqn: &Eqn,
        f: &mut fmt::Formatter<'_>,
        printer: &PrettyPrinter<'_, Expr>,
        indent_level: usize,
    ) -> fmt::Result {
        fmt::Display::fmt(eqn.op(), f)?;
        for subexpr in eqn.subexprs() {
            write!(f, " [")?;
            fmt_symbol_args(f, &subexpr.symbol_captures)?;
            printer.fmt_region(
                f,
                self,
                subexpr.expr(),
                subexpr.closure(),
                indent_level,
                |var, f| self.fmt_var(var, f),
            )?;
            write!(f, "]")?;
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
    max_inline_eqns: usize,
    indent: String,
    inline_subexpr_indent: usize,
}

impl PrettyConfig {
    pub fn new() -> Self {
        Self {
            max_inline_eqns: 8,
            indent: "  ".to_string(),
            inline_subexpr_indent: 1,
        }
    }

    /// Maximum number of instructions in an inline region (default: 8).
    pub fn max_inline_eqns(mut self, max_inline_eqns: usize) -> Self {
        self.max_inline_eqns = max_inline_eqns;
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

pub struct PrettyExpr<'a, E = Expr> {
    expr: &'a E,
    config: PrettyConfig,
}

impl<'a, E> PrettyExpr<'a, E> {
    pub fn new(expr: &'a E) -> Self {
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
}

pub type PrettyHExpr<'a> = PrettyExpr<'a, HExpr>;

impl PrettyExpr<'_, Expr> {
    pub fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_scope(&PrettyVarScope::new(), f)
    }
}

impl fmt::Display for PrettyHExpr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_scope(&PrettyVarScope::new(), f)
    }
}

impl<E> PrettyExpr<'_, E> {
    fn fmt_with_scope(&self, scope: &PrettyVarScope, f: &mut fmt::Formatter<'_>) -> fmt::Result
    where
        E: PrintableExpr,
    {
        let printer = PrettyPrinter::new(self.config.clone(), self.expr.regions());
        printer.fmt_definitions(f)?;
        self.expr.fmt_body(f, scope, &printer, 0)
    }
}

trait PrintableExpr: Sized {
    fn regions(&self) -> impl Iterator<Item = &Arc<Self>>;
    fn body_len(&self) -> usize;
    fn name_hint(&self) -> Option<&str>;
    fn closure_vars(&self) -> impl Iterator<Item = Var>;
    fn fmt_body(
        &self,
        f: &mut fmt::Formatter<'_>,
        scope: &PrettyVarScope,
        printer: &PrettyPrinter<'_, Self>,
        indent: usize,
    ) -> fmt::Result;
}

impl PrintableExpr for Expr {
    fn regions(&self) -> impl Iterator<Item = &Arc<Self>> {
        self.eqns()
            .iter()
            .flat_map(|eqn| eqn.subexprs())
            .map(|sub| sub.expr())
    }
    fn body_len(&self) -> usize {
        self.eqns().len()
    }
    fn name_hint(&self) -> Option<&str> {
        self.name_hint()
    }
    fn closure_vars(&self) -> impl Iterator<Item = Var> {
        self.closure().iter().map(|v| v.0)
    }
    fn fmt_body(
        &self,
        f: &mut fmt::Formatter<'_>,
        scope: &PrettyVarScope,
        printer: &PrettyPrinter<'_, Self>,
        indent: usize,
    ) -> fmt::Result {
        scope.fmt_expr_with_printer(self, f, printer, indent)
    }
}

impl PrintableExpr for HExpr {
    fn regions(&self) -> impl Iterator<Item = &Arc<Self>> {
        self.instructions()
            .iter()
            .filter_map(HInstr::as_apply)
            .flat_map(|eqn| eqn.subexprs())
            .map(|sub| sub.expr())
    }
    fn body_len(&self) -> usize {
        self.instructions().len()
    }
    fn name_hint(&self) -> Option<&str> {
        self.name_hint()
    }
    fn closure_vars(&self) -> impl Iterator<Item = Var> {
        self.closure().iter().map(|v| v.0)
    }
    fn fmt_body(
        &self,
        f: &mut fmt::Formatter<'_>,
        scope: &PrettyVarScope,
        printer: &PrettyPrinter<'_, Self>,
        indent: usize,
    ) -> fmt::Result {
        scope.fmt_hexpr_at(self, f, printer, indent)
    }
}

struct PrettyPrinter<'a, E> {
    config: PrettyConfig,
    definitions: Vec<&'a E>,
    names: BTreeMap<*const E, String>,
}

impl<'a, E: PrintableExpr + 'a> PrettyPrinter<'a, E> {
    fn new(config: PrettyConfig, regions: impl IntoIterator<Item = &'a Arc<E>>) -> Self {
        let mut seen = BTreeSet::new();
        let mut definitions = Vec::new();
        if config.max_inline_eqns != usize::MAX {
            for region in regions {
                Self::collect(region, config.max_inline_eqns, &mut seen, &mut definitions);
            }
        }
        // Reserve actual hints so anonymous bodies don't take their names.
        let hint = |expr: &'a E| expr.name_hint().filter(|hint| !hint.is_empty());
        let reserved: BTreeSet<_> = definitions.iter().filter_map(|&expr| hint(expr)).collect();
        let mut used = BTreeSet::new();
        let mut generated = (0..).map(alphabetic_name);
        let names = definitions
            .iter()
            .map(|&expr| {
                let available =
                    |candidate: &str| !reserved.contains(candidate) && !used.contains(candidate);
                let name = match hint(expr) {
                    Some(hint) if !used.contains(hint) => hint.to_owned(),
                    Some(hint) => (2..)
                        .map(|n| format!("{hint}_{n}"))
                        .find(|name| available(name))
                        .unwrap(),
                    None => generated.by_ref().find(|name| available(name)).unwrap(),
                };
                used.insert(name.clone());
                (std::ptr::from_ref(expr), name)
            })
            .collect();
        Self {
            config,
            definitions,
            names,
        }
    }

    fn collect(
        expr: &'a E,
        limit: usize,
        seen: &mut BTreeSet<*const E>,
        definitions: &mut Vec<&'a E>,
    ) {
        if !seen.insert(std::ptr::from_ref(expr)) {
            return;
        }
        for region in expr.regions() {
            Self::collect(region, limit, seen, definitions);
        }
        // Dependencies come before their users. A shared body is listed once,
        // independent of which positional capture bindings its uses supply.
        if expr.body_len() > limit {
            definitions.push(expr);
        }
    }

    fn name(&self, expr: &E) -> Option<&str> {
        self.names
            .get(&std::ptr::from_ref(expr))
            .map(String::as_str)
    }

    fn fmt_definitions(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &expr in &self.definitions {
            write!(f, "{} = ", self.name(expr).unwrap())?;
            expr.fmt_body(f, &PrettyVarScope::new(), self, 0)?;
            writeln!(f)?;
        }
        if !self.definitions.is_empty() {
            writeln!(f, "in")?;
        }
        Ok(())
    }

    fn fmt_region(
        &self,
        f: &mut fmt::Formatter<'_>,
        scope: &PrettyVarScope,
        expr: &E,
        captures: &[Var],
        indent: usize,
        fmt_capture: impl Fn(&Var, &mut fmt::Formatter<'_>) -> fmt::Result,
    ) -> fmt::Result {
        if let Some(name) = self.name(expr) {
            write!(f, "@{name}")?;
            if !captures.is_empty() {
                write!(f, "{{")?;
                fmt_list(f, captures, fmt_capture)?;
                write!(f, "}}")?;
            }
            Ok(())
        } else {
            let aliases = scope.aliases(expr.closure_vars(), captures);
            expr.fmt_body(
                f,
                &PrettyVarScope::child(&aliases),
                self,
                indent + self.config.inline_subexpr_indent,
            )
        }
    }

    fn write_indent(&self, f: &mut fmt::Formatter<'_>, level: usize) -> fmt::Result {
        for _ in 0..level {
            write!(f, "{}", self.config.indent)?;
        }
        Ok(())
    }

    fn fmt_return(
        &self,
        f: &mut fmt::Formatter<'_>,
        indent: usize,
        outputs: &[Var],
        format: impl Fn(&Var, &mut fmt::Formatter<'_>) -> fmt::Result,
    ) -> fmt::Result {
        self.write_indent(f, indent + 1)?;
        write!(f, "in ")?;
        fmt_tuple(f, outputs, format)?;
        writeln!(f)?;
        self.write_indent(f, indent)?;
        write!(f, "}}")
    }
}

fn fmt_symbol_args(f: &mut fmt::Formatter<'_>, args: &[crate::Symbol]) -> fmt::Result {
    if !args.is_empty() {
        write!(f, "bind[")?;
        fmt_list(f, args, |s, f| write!(f, "{s}"))?;
        write!(f, "] ")?;
    }
    Ok(())
}
fn fmt_symbol_params(
    f: &mut fmt::Formatter<'_>,
    closure: &[(crate::Symbol, crate::SymbolInfo)],
    inputs: &[(crate::Symbol, crate::SymbolInfo)],
) -> fmt::Result {
    if !closure.is_empty() || !inputs.is_empty() {
        write!(f, "[")?;
        let decl = |(s, t): &(crate::Symbol, crate::SymbolInfo), f: &mut fmt::Formatter<'_>| {
            write!(f, "{s}: {t}")
        };
        if !closure.is_empty() {
            fmt_list(f, closure, decl)?;
            write!(f, "; ")?;
        }
        fmt_list(f, inputs, decl)?;
        write!(f, "] ")?;
    }
    Ok(())
}
fn fmt_lambda<T>(
    f: &mut fmt::Formatter<'_>,
    closure: &[T],
    inputs: &[T],
    format: impl Fn(&T, &mut fmt::Formatter<'_>) -> fmt::Result,
) -> fmt::Result {
    write!(f, "|")?;
    if !closure.is_empty() {
        fmt_list(f, closure, &format)?;
        write!(f, ";")?;
    }
    if !inputs.is_empty() {
        if !closure.is_empty() {
            write!(f, " ")?;
        }
        fmt_list(f, inputs, format)?;
    }
    write!(f, "| => ")
}

fn fmt_tuple<T>(
    f: &mut fmt::Formatter<'_>,
    items: &[T],
    format: impl Fn(&T, &mut fmt::Formatter<'_>) -> fmt::Result,
) -> fmt::Result {
    write!(f, "(")?;
    fmt_list(f, items, format)?;
    write!(f, ")")
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
