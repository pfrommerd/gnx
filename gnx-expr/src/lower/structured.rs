//! Fixed region signatures for control flow. Ordinary calls can forward a
//! result statically; a control-flow join or backedge needs a logical slot.

use super::*;

/// A canonical cell identity in the parent region. The handle never reaches Expr.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegionCell(pub(super) CellBinding);

/// An explicit region parameter is either a distinct runtime argument or a
/// statically bound cell. Captures are bound separately by the subexpression.
#[derive(Clone, Copy)]
pub enum RegionArgument {
    Value,
    Cell(Var),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Slot {
    Argument(usize),
    Capture(Var),
    Cell(RegionCell),
}

pub struct StructuredRegion {
    symbol_captures: Vec<Symbol>,
    program: LoweredExpr,
    slots: BTreeMap<InputId, Slot>,
    num_arguments: usize,
}

pub struct FixedRegion {
    symbol_captures: Vec<Symbol>,
    expr: ExprRef,
    trace_body: HExprRef,
    captures: Vec<Var>,
}

impl FixedRegion {
    pub fn subexpr(&self) -> RegionUse {
        RegionUse {
            symbol_captures: self.symbol_captures.clone(),
            expr: self.expr.clone(),
            trace_body: self.trace_body.clone(),
            captures: self.captures.clone(),
        }
    }
}

impl LoweringContext<'_> {
    pub(super) fn ssa_info(&self, var: Var) -> Result<VarInfo, LoweringError> {
        self.lowerer
            .bound_inputs
            .values()
            .find_map(|(_, v, info)| (*v == var).then(|| info.clone()))
            .or_else(|| {
                self.lowerer
                    .eqns
                    .iter()
                    .rev()
                    .flat_map(EqnSource::outputs)
                    .find_map(|(v, info)| (*v == var).then(|| info.clone()))
            })
            .ok_or(LoweringError::UndefinedVariable(var))
    }

    pub fn value_operand(&self, var: Var) -> Result<(Var, VarInfo), LoweringError> {
        let var = self.lowerer.value(var)?;
        Ok((var, self.ssa_info(var)?))
    }

    pub fn cell_state(&self, cell: RegionCell) -> Result<(Var, VarInfo), LoweringError> {
        let var = self.lowerer.states[&cell.0];
        Ok((var, self.ssa_info(var)?))
    }

    /// Compile the region with the parent's cell aliases, but don't deduplicate
    /// dynamic arguments even if their initial SSA values happen to be equal.
    pub fn lower_structured_region(
        &self,
        sub: HSubExpr<'_>,
        arguments: &[RegionArgument],
        argument_infos: &[VarInfo],
    ) -> Result<StructuredRegion, LoweringError> {
        let body = sub.expr();
        if sub.provided_vars().len() != body.closure_inputs().len()
            || arguments.len() != body.inputs().len()
        {
            return Err(LoweringError::RegionSignature("region binding arity"));
        }
        let mut values = argument_infos.iter();
        let actual = arguments
            .iter()
            .map(|arg| match arg {
                RegionArgument::Value => values.next().cloned().ok_or(LoweringError::CallArity),
                RegionArgument::Cell(v) => self.object_info(*v, Kind::Cell),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let symbols = self.child_bindings(sub, &actual)?;
        // Lower the body in the symbol namespace of the caller.
        let substituted;
        let body = if symbols.is_empty() {
            body
        } else {
            substituted = body.substitute_symbols(&symbols).build()?;
            substituted.view()
        };
        let mut identities = BTreeMap::new();
        let mut slots = BTreeMap::new();
        let mut spec_binding = |slot, id, param: HParam<'_>| {
            let id = *identities.entry(slot).or_insert(id);
            slots.insert(id, slot);
            (id, param.kind(), param.info_ref().to_owned())
        };
        let mut num_arguments = 0;
        let mut inputs = Vec::new();
        for (index, (argument, param)) in arguments.iter().zip(body.inputs()).enumerate() {
            let info = param.info_ref();
            let slot = match (argument, param.kind()) {
                (RegionArgument::Value, Kind::Value)
                    if (argument_infos.get(num_arguments))
                        .is_some_and(|actual| info == *actual) =>
                {
                    let slot = Slot::Argument(num_arguments);
                    num_arguments += 1;
                    slot
                }
                (RegionArgument::Cell(var), Kind::Cell) => {
                    let cell = RegionCell(self.lowerer.cell(*var)?);
                    if info != self.cell_state(cell)?.1 {
                        return Err(LoweringError::RegionSignature("region cell metadata"));
                    }
                    Slot::Cell(cell)
                }
                _ => return Err(LoweringError::RegionSignature("region argument metadata")),
            };
            inputs.push(spec_binding(slot, InputId::Input(index), param));
        }
        if num_arguments != argument_infos.len() {
            return Err(LoweringError::RegionSignature("region argument arity"));
        }
        let mut closure = Vec::new();
        for (index, (&var, param)) in (sub.provided_vars().iter())
            .zip(body.closure_inputs())
            .enumerate()
        {
            let info = param.info_ref();
            let slot = match param.kind() {
                Kind::Value => {
                    let (var, actual) = self.value_operand(var)?;
                    if info != actual {
                        return Err(LoweringError::RegionSignature("region capture metadata"));
                    }
                    Slot::Capture(var)
                }
                Kind::Cell => {
                    let cell = RegionCell(self.lowerer.cell(var)?);
                    if info != self.cell_state(cell)?.1 {
                        return Err(LoweringError::RegionSignature(
                            "region capture cell metadata",
                        ));
                    }
                    Slot::Cell(cell)
                }
            };
            closure.push(spec_binding(slot, InputId::Closure(index), param));
        }
        let program = body.lower(&InputSpec { closure, inputs })?;
        Ok(StructuredRegion {
            symbol_captures: body.closure_symbols().to_vec(),
            program,
            slots,
            num_arguments,
        })
    }

    /// Bind a region operation's joined results and advance the chosen parent
    /// cell states. State slots are interleaved at `position` in physical results.
    pub fn bind_structured_results<'b>(
        &mut self,
        logical: impl ExactSizeIterator<Item = HParam<'b>>,
        cells: &[RegionCell],
        position: usize,
        physical: &[(Var, VarInfo)],
    ) -> Result<(), LoweringError> {
        if position > logical.len() || physical.len() != logical.len() + cells.len() {
            return Err(LoweringError::RegionSignature("structured result arity"));
        }
        let (before, rest) = physical.split_at(position);
        let (states, after) = rest.split_at(cells.len());
        for (param, (value, actual)) in logical.zip(before.iter().chain(after)) {
            if param.kind() != Kind::Value || param.info_ref() != *actual {
                return Err(LoweringError::RegionSignature("structured result metadata"));
            }
            self.lowerer.values.insert(param.var(), *value);
        }
        for (&cell, (value, info)) in cells.iter().zip(states) {
            if *info != self.cell_state(cell)?.1 {
                return Err(LoweringError::RegionSignature("structured state metadata"));
            }
            self.lowerer.states.insert(cell.0, *value);
            self.lowerer.written.insert(cell.0);
        }
        Ok(())
    }
}

impl StructuredRegion {
    pub fn writes(&self) -> impl Iterator<Item = RegionCell> + '_ {
        self.program
            .updates
            .iter()
            .map(|(id, _)| match self.slots[id] {
                Slot::Cell(cell) => cell,
                _ => unreachable!("updates are only defined for cells"),
            })
    }

    /// Fix argument and result arities across regions. Unwritten cells yield
    /// their incoming contents; read-only cells outside this layout stay
    /// captured. Borrowed results forward their parameters without an equation.
    pub fn fix(
        &self,
        ctx: &LoweringContext<'_>,
        states: &[(RegionCell, VarInfo)],
        input_position: usize,
        output_position: Option<usize>,
        expected_results: &[VarInfo],
    ) -> Result<FixedRegion, LoweringError> {
        let original = &self.program.source;
        if input_position > self.num_arguments
            || expected_results.len() != self.program.results().len()
            || output_position.is_some_and(|p| p > expected_results.len())
            || (output_position.is_none() && self.writes().next().is_some())
            || self
                .writes()
                .any(|cell| !states.iter().any(|(slot, _)| *slot == cell))
        {
            return Err(LoweringError::RegionSignature(
                "incompatible fixed region layout",
            ));
        }
        let id_params: BTreeMap<_, _> = (self.program.closure.iter())
            .chain(&self.program.inputs)
            .copied()
            .zip((original.closure_inputs.iter().chain(&original.inputs)).cloned())
            .collect();
        let mut slot_params: BTreeMap<_, _> = self
            .slots
            .iter()
            .map(|(id, slot)| (*slot, id_params[id].clone()))
            .collect();
        // New parameters come after every variable of the lowered region.
        let mut scope = original.scope.clone();
        let mut inputs: Vec<_> = (0..self.num_arguments)
            .map(|idx| {
                slot_params
                    .remove(&Slot::Argument(idx))
                    .expect("runtime argument parameter")
            })
            .collect();
        let mut state_params = BTreeMap::new();
        let mut state_inputs = Vec::new();
        for (cell, info) in states {
            let parameter = slot_params
                .remove(&Slot::Cell(*cell))
                .unwrap_or_else(|| (scope.create_var(), info.clone()));
            if parameter.1 != *info || state_params.insert(*cell, parameter.0).is_some() {
                return Err(LoweringError::RegionSignature("incompatible state slots"));
            }
            state_inputs.push(parameter);
        }
        inputs.splice(input_position..input_position, state_inputs);

        let mut closure = Vec::new();
        let mut captures = Vec::new();
        for id in self.program.closure.iter().chain(&self.program.inputs) {
            let slot = self.slots[id];
            let Some(parameter) = slot_params.remove(&slot) else {
                continue;
            };
            let outer = match slot {
                Slot::Argument(_) => unreachable!(),
                Slot::Capture(value) => value,
                Slot::Cell(cell) => ctx.cell_state(cell)?.0,
            };
            closure.push(parameter);
            captures.push(outer);
        }
        let source = |output: &Output| match *output {
            Output::Value { result_idx } => original
                .outputs
                .get(result_idx)
                .copied()
                .ok_or(LoweringError::InvalidResultIndex(result_idx)),
            _ => Err(LoweringError::RegionSignature(
                "control flow cannot return cell handles",
            )),
        };
        let mut results = Vec::new();
        for (output, expected) in self.program.results().iter().zip(expected_results) {
            let var = source(output)?;
            if original.value_info(var) != Some(expected) {
                return Err(LoweringError::RegionSignature(
                    "incompatible region result metadata",
                ));
            }
            results.push(var);
        }
        if let Some(position) = output_position {
            let updates: BTreeMap<_, _> = self
                .writes()
                .zip(self.program.updates.iter().map(|(_, idx)| *idx))
                .collect();
            let mut state_results = Vec::new();
            for (cell, info) in states {
                let var = if let Some(&idx) = updates.get(cell) {
                    let var = original
                        .outputs
                        .get(idx)
                        .copied()
                        .ok_or(LoweringError::InvalidResultIndex(idx))?;
                    if original.value_info(var) != Some(info) {
                        return Err(LoweringError::RegionSignature(
                            "incompatible region state result",
                        ));
                    }
                    var
                } else {
                    state_params[cell]
                };
                state_results.push(var);
            }
            results.splice(position..position, state_results);
        }

        let trace_params = |params: &[(Var, VarInfo)]| {
            params
                .iter()
                .map(|(var, info)| (*var, Kind::Value, info.clone()))
                .collect()
        };
        let name = original.name_hint().map(str::to_owned);
        let trace_body = HExprBuilder::from_parts(
            original.closure_symbols.clone(),
            trace_params(&closure),
            trace_params(&inputs),
            self.program.trace_source.instrs.clone(),
            results.clone(),
        )
        .with_name_hint_opt(name.clone())
        .build()?;
        let expr = ExprBuilder::from_parts(
            original.closure_symbols.clone(),
            closure,
            inputs,
            original.eqns.clone(),
            results,
        )
        .with_name_hint_opt(name)
        .build()?;
        expr.view().check()?;
        Ok(FixedRegion {
            symbol_captures: self.symbol_captures.clone(),
            expr: expr.handle().into(),
            trace_body: trace_body.handle().into(),
            captures,
        })
    }
}

impl LoweringContext<'_> {
    /// Lower an operation that has one region, takes values, and yields
    /// values, emitting it unchanged with the lowered region. The region
    /// is called with values, one per input; it may read the cells it is
    /// provided, but not write them.
    ///
    /// `check` validates the signature first. It is given the infos of the
    /// region's inputs and results, and of the equation's operands and
    /// results.
    pub fn lower_value_region_apply(
        &mut self,
        eqn: HEqn<'_>,
        check: impl FnOnce(&[VarInfo], &[VarInfo], &[VarInfo], &[VarInfo]) -> Result<(), LoweringError>,
    ) -> Result<(), LoweringError> {
        let mut subexprs = eqn.subexprs();
        let (Some(sub), None) = (subexprs.next(), subexprs.next()) else {
            return Err(LoweringError::RegionSignature(
                "expected exactly one region",
            ));
        };
        let value = |param: HParam<'_>| match param.kind() {
            Kind::Value => Ok(param.info_ref().to_owned()),
            Kind::Cell => Err(LoweringError::ExpectedValue(param.var())),
        };
        let body = sub.expr();
        let body_inputs = body.inputs().map(value).collect::<Result<Vec<_>, _>>()?;
        let body_outputs = (body.outputs().iter())
            .map(|&var| match body.definition(var) {
                Some(param) => value(param),
                None => Err(LoweringError::UndefinedVariable(var)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (values, operands): (Vec<_>, Vec<_>) = (eqn.inputs().iter())
            .map(|&var| self.value_operand(var))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let results = eqn.outputs().map(value).collect::<Result<Vec<_>, _>>()?;
        check(&body_inputs, &body_outputs, &operands, &results)?;

        let region = self.lower_structured_region(
            sub,
            &vec![RegionArgument::Value; body_inputs.len()],
            &body_inputs,
        )?;
        if region.writes().next().is_some() {
            return Err(LoweringError::RegionSignature(
                "the region cannot write the cells it is provided",
            ));
        }
        let fixed = region.fix(self, &[], body_inputs.len(), None, &body_outputs)?;
        let physical = self.allocate_results(&results);
        self.bind_structured_results(eqn.outputs(), &[], 0, &physical)?;
        self.emit(eqn.op().to_owned(), values, physical, vec![fixed.subexpr()]);
        Ok(())
    }
}
