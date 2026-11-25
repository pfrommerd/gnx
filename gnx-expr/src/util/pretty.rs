//! Textual form of expressions.
//!
//! Variables have no names of their own: they are shown by generated names
//! (`a`, `b`, ...). Large or shared subexpressions can be hoisted into
//! definitions ahead of the expression, named by their name hints. Metadata is displayed as a
//! value, not serialized back to Rust source.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{Eqn, Expr, HEqn, HExpr, HInstr, HParam, Kind, Param, Symbol, Var};

// The names of the variables of one expression
struct PrettyVarScope {
    names: RefCell<BTreeMap<Var, String>>,
    used_names: RefCell<BTreeSet<String>>,
    next_name: Cell<usize>,
    nested: bool,
}

impl PrettyVarScope {
    fn new() -> Self {
        Self {
            names: RefCell::new(BTreeMap::new()),
            used_names: RefCell::new(BTreeSet::new()),
            next_name: Cell::new(0),
            nested: false,
        }
    }

    fn alias(&self, var: Var, name: String) {
        if !var.is_hole() {
            self.used_names.borrow_mut().insert(name.clone());
            self.names.borrow_mut().insert(var, name);
        }
    }

    // The scope of a subexpression shown inline: its closure inputs go
    // by the names of what they are provided.
    fn child(aliases: Vec<(Var, String)>) -> Self {
        let scope = Self {
            nested: true,
            ..Self::new()
        };
        for (var, name) in aliases {
            scope.alias(var, name);
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
            .map(|(parameter, outer)| (parameter, self.name_for(*outer)))
            .collect()
    }

    fn fmt_var(&self, var: Var, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_for(var))
    }

    fn name_for(&self, var: Var) -> String {
        if var.is_hole() {
            return "_".to_string();
        }
        if let Some(name) = self.names.borrow().get(&var).cloned() {
            return name;
        }
        let name = self.next_generated_name();
        self.names.borrow_mut().insert(var, name.clone());
        name
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

/// How an expression is laid out when shown.
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
    /// Larger subexpressions are shown as named definitions.
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

/// An expression with the layout to show it in. Made by `Expr::pretty`
/// and `HExpr::pretty`.
pub struct PrettyExpr<E> {
    expr: E,
    config: PrettyConfig,
}

impl<E> PrettyExpr<E> {
    fn new(expr: E) -> Self {
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

impl<'a> Expr<'a> {
    pub fn pretty(self) -> PrettyExpr<Expr<'a>> {
        PrettyExpr::new(self)
    }
}

impl<'a> HExpr<'a> {
    pub fn pretty(self) -> PrettyExpr<HExpr<'a>> {
        PrettyExpr::new(self)
    }
}

impl<'a, E: PrintableExpr<'a>> fmt::Display for PrettyExpr<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let printer = PrettyPrinter::new(self.config.clone(), self.expr.regions());
        printer.fmt_definitions(f)?;
        (self.expr).fmt_body(f, &PrettyVarScope::new(), &printer, 0)
    }
}

// What Expr and HExpr have in common, as far as showing them goes
trait PrintableExpr<'a>: Copy + 'a {
    // Identifies the stored expression
    fn key(self) -> *const ();
    // The expressions used as subexpressions, in order
    fn regions(self) -> Vec<Self>;
    fn body_len(self) -> usize;
    fn name_hint(self) -> Option<&'a str>;
    fn closure_vars(self) -> Vec<Var>;
    fn fmt_body(
        self,
        f: &mut fmt::Formatter<'_>,
        scope: &PrettyVarScope,
        printer: &PrettyPrinter<'a, Self>,
        indent: usize,
    ) -> fmt::Result;
}

impl<'a> PrintableExpr<'a> for Expr<'a> {
    fn key(self) -> *const () {
        self.data_ptr()
    }
    fn regions(self) -> Vec<Self> {
        (self.eqns())
            .flat_map(|eqn| eqn.subexprs())
            .map(|sub| sub.expr())
            .collect()
    }
    fn body_len(self) -> usize {
        self.eqns().len()
    }
    fn name_hint(self) -> Option<&'a str> {
        Expr::name_hint(self)
    }
    fn closure_vars(self) -> Vec<Var> {
        self.closure_inputs().map(|param| param.var()).collect()
    }
    fn fmt_body(
        self,
        f: &mut fmt::Formatter<'_>,
        scope: &PrettyVarScope,
        printer: &PrettyPrinter<'a, Self>,
        indent: usize,
    ) -> fmt::Result {
        fmt_expr(self, f, scope, printer, indent)
    }
}

impl<'a> PrintableExpr<'a> for HExpr<'a> {
    fn key(self) -> *const () {
        self.data_ptr()
    }
    fn regions(self) -> Vec<Self> {
        (self.instrs())
            .filter_map(HInstr::as_apply)
            .flat_map(|eqn| eqn.subexprs())
            .map(|sub| sub.expr())
            .collect()
    }
    fn body_len(self) -> usize {
        self.instrs().len()
    }
    fn name_hint(self) -> Option<&'a str> {
        HExpr::name_hint(self)
    }
    fn closure_vars(self) -> Vec<Var> {
        self.closure_inputs().map(|param| param.var()).collect()
    }
    fn fmt_body(
        self,
        f: &mut fmt::Formatter<'_>,
        scope: &PrettyVarScope,
        printer: &PrettyPrinter<'a, Self>,
        indent: usize,
    ) -> fmt::Result {
        fmt_hexpr(self, f, scope, printer, indent)
    }
}

fn fmt_param(scope: &PrettyVarScope, param: Param<'_>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    scope.fmt_var(param.var(), f)?;
    write!(f, ": {}", param.info_ref())
}

fn fmt_definition(
    scope: &PrettyVarScope,
    param: HParam<'_>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    scope.fmt_var(param.var(), f)?;
    match param.kind() {
        Kind::Value => write!(f, ": {}", param.info_ref()),
        Kind::Cell => write!(f, "[:]: {}", param.info_ref()),
    }
}

// An equation, without the `;` that ends it in a body
fn fmt_eqn<'a>(
    eqn: Eqn<'a>,
    f: &mut fmt::Formatter<'_>,
    scope: &PrettyVarScope,
    printer: &PrettyPrinter<'a, Expr<'a>>,
    indent: usize,
) -> fmt::Result {
    let outputs: Vec<_> = eqn.outputs().collect();
    fmt_list(f, &outputs, |output, f| fmt_param(scope, *output, f))?;
    if !outputs.is_empty() {
        write!(f, " = ")?;
    }
    fmt::Display::fmt(&eqn.op(), f)?;
    for sub in eqn.subexprs() {
        write!(f, " [")?;
        fmt_symbol_args(f, sub.provided_symbols())?;
        let captures = sub.provided_vars();
        printer.fmt_region(f, scope, sub.expr(), captures, indent, |var, f| {
            scope.fmt_var(*var, f)
        })?;
        write!(f, "]")?;
    }
    for input in eqn.inputs() {
        write!(f, " ")?;
        scope.fmt_var(*input, f)?;
    }
    Ok(())
}

fn fmt_expr<'a>(
    expr: Expr<'a>,
    f: &mut fmt::Formatter<'_>,
    scope: &PrettyVarScope,
    printer: &PrettyPrinter<'a, Expr<'a>>,
    indent: usize,
) -> fmt::Result {
    fmt_symbol_params(f, expr.closure_symbols(), &expr.input_symbols())?;
    // The closure of a subexpression shown inline goes by outer names.
    let closure: Vec<_> = (expr.closure_inputs()).filter(|_| !scope.nested).collect();
    let inputs: Vec<_> = expr.inputs().collect();
    fmt_lambda(f, &closure, &inputs, |param, f| fmt_param(scope, *param, f))?;
    let value_var = |var: &Var, f: &mut fmt::Formatter<'_>| scope.fmt_var(*var, f);
    if expr.eqns().len() == 0 {
        return fmt_tuple(f, expr.outputs(), value_var);
    }
    writeln!(f, "{{")?;
    for eqn in expr.eqns() {
        printer.write_indent(f, indent + 1)?;
        fmt_eqn(eqn, f, scope, printer, indent)?;
        writeln!(f, ";")?;
    }
    printer.fmt_return(f, indent, expr.outputs(), value_var)
}

/// Format the body, displaying metadata at its explicit definitions.
fn fmt_hexpr<'a>(
    expr: HExpr<'a>,
    f: &mut fmt::Formatter<'_>,
    scope: &PrettyVarScope,
    printer: &PrettyPrinter<'a, HExpr<'a>>,
    indent: usize,
) -> fmt::Result {
    let cell_var = |var: &Var, f: &mut fmt::Formatter<'_>| {
        scope.fmt_var(*var, f)?;
        if expr.kind(*var) == Some(Kind::Cell) {
            write!(f, "[:]")?;
        }
        Ok(())
    };
    fmt_symbol_params(f, expr.closure_symbols(), &expr.input_symbols())?;
    let closure: Vec<_> = (expr.closure_inputs()).filter(|_| !scope.nested).collect();
    let inputs: Vec<_> = expr.inputs().collect();
    fmt_lambda(f, &closure, &inputs, |param, f| {
        fmt_definition(scope, *param, f)
    })?;
    if expr.instrs().len() == 0 {
        return fmt_tuple(f, expr.outputs(), cell_var);
    }
    writeln!(f, "{{")?;
    for instruction in expr.instrs() {
        printer.write_indent(f, indent + 1)?;
        fmt_instr(instruction, f, scope, printer, indent, &cell_var)?;
        writeln!(f, ";")?;
    }
    printer.fmt_return(f, indent, expr.outputs(), cell_var)
}

// An instruction, without the `;` that ends it in a body
fn fmt_instr<'a>(
    instruction: HInstr<'a>,
    f: &mut fmt::Formatter<'_>,
    scope: &PrettyVarScope,
    printer: &PrettyPrinter<'a, HExpr<'a>>,
    indent: usize,
    cell_var: &dyn Fn(&Var, &mut fmt::Formatter<'_>) -> fmt::Result,
) -> fmt::Result {
    match instruction {
        HInstr::Apply(eqn) => {
            let outputs: Vec<_> = eqn.outputs().collect();
            if !outputs.is_empty() {
                fmt_list(f, &outputs, |output, f| fmt_definition(scope, *output, f))?;
                write!(f, " = ")?;
            }
            fmt::Display::fmt(&eqn.op(), f)?;
            for sub in eqn.subexprs() {
                write!(f, " [")?;
                fmt_symbol_args(f, sub.provided_symbols())?;
                let captures = sub.provided_vars();
                printer.fmt_region(f, scope, sub.expr(), captures, indent, cell_var)?;
                write!(f, "]")?;
            }
            for var in eqn.inputs() {
                write!(f, " ")?;
                cell_var(var, f)?;
            }
            Ok(())
        }
        HInstr::CreateCell { cell, initial, .. } => {
            // The cell is being defined, so its kind is known.
            scope.fmt_var(cell, f)?;
            write!(f, "[:] = ")?;
            scope.fmt_var(initial, f)
        }
        HInstr::ReadCell { cell, output, .. } => {
            scope.fmt_var(output, f)?;
            write!(f, " <- ")?;
            cell_var(&cell, f)
        }
        HInstr::WriteCell { cell, value } => {
            cell_var(&cell, f)?;
            write!(f, " <- ")?;
            scope.fmt_var(value, f)
        }
    }
}

// Decides which subexpressions are hoisted into named definitions.
struct PrettyPrinter<'a, E> {
    config: PrettyConfig,
    definitions: Vec<E>,
    names: BTreeMap<*const (), String>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, E: PrintableExpr<'a>> PrettyPrinter<'a, E> {
    fn new(config: PrettyConfig, regions: impl IntoIterator<Item = E>) -> Self {
        let mut seen = BTreeSet::new();
        let mut definitions = Vec::new();
        if config.max_inline_eqns != usize::MAX {
            for region in regions {
                Self::collect(region, config.max_inline_eqns, &mut seen, &mut definitions);
            }
        }
        // Reserve actual hints so anonymous bodies don't take their names.
        let hint = |expr: E| expr.name_hint().filter(|hint| !hint.is_empty());
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
                (expr.key(), name)
            })
            .collect();
        Self {
            config,
            definitions,
            names,
            _marker: std::marker::PhantomData,
        }
    }

    fn collect(expr: E, limit: usize, seen: &mut BTreeSet<*const ()>, definitions: &mut Vec<E>) {
        if !seen.insert(expr.key()) {
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

    fn name(&self, expr: E) -> Option<&str> {
        self.names.get(&expr.key()).map(String::as_str)
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
        expr: E,
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
            let aliases = scope.aliases(expr.closure_vars().into_iter(), captures);
            expr.fmt_body(
                f,
                &PrettyVarScope::child(aliases),
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

fn fmt_symbol_args(f: &mut fmt::Formatter<'_>, args: &[Symbol]) -> fmt::Result {
    if !args.is_empty() {
        write!(f, "bind[")?;
        fmt_list(f, args, |s, f| write!(f, "{s}"))?;
        write!(f, "] ")?;
    }
    Ok(())
}
fn fmt_symbol_params(
    f: &mut fmt::Formatter<'_>,
    closure: &[Symbol],
    inputs: &[Symbol],
) -> fmt::Result {
    if !closure.is_empty() || !inputs.is_empty() {
        write!(f, "[")?;
        let decl = |s: &Symbol, f: &mut fmt::Formatter<'_>| write!(f, "{s}");
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

impl fmt::Display for Expr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.pretty().fmt(f)
    }
}

impl fmt::Display for HExpr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.pretty().fmt(f)
    }
}

// An equation on its own is shown in a scope of its own, after the
// definitions of any subexpressions that are hoisted.
impl<'a> fmt::Display for Eqn<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let regions = self.subexprs().map(|sub| sub.expr());
        let printer = PrettyPrinter::new(PrettyConfig::new(), regions);
        printer.fmt_definitions(f)?;
        fmt_eqn(*self, f, &PrettyVarScope::new(), &printer, 0)
    }
}

impl<'a> fmt::Display for HEqn<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let regions = self.subexprs().map(|sub| sub.expr());
        let printer = PrettyPrinter::new(PrettyConfig::new(), regions);
        printer.fmt_definitions(f)?;
        let scope = PrettyVarScope::new();
        // Without its expression, the kinds of its operands are unknown.
        let var = |var: &Var, f: &mut fmt::Formatter<'_>| scope.fmt_var(*var, f);
        fmt_instr(HInstr::Apply(*self), f, &scope, &printer, 0, &var)
    }
}
