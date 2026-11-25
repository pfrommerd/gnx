use std::{collections::BTreeMap, rc::Rc};

use rkyv::{
    Archive, Place, Portable, Serialize,
    bytecheck::{self, CheckBytes},
    rancor::Fallible,
    rend::i32_le,
};

use super::ExprHandle;
use crate::op::{OpBuilder, OpTypeTable};
use crate::symbol::Symbol;
use crate::util::Blake3Hash;
use crate::var::{InfoTypeTable, Var, VarInfo};

#[derive(Archive)]
#[rkyv(archived = RootData)]
pub struct RootBuilder {
    pub root: ExprRef,
    pub exprs: ExprsTable,
    // The type tables are written from what has been interned
    // so far, so they must stay after exprs.
    pub op_types: OpTypeTable,
    pub info_types: InfoTypeTable,
}

impl Default for RootBuilder {
    fn default() -> Self {
        Self {
            root: ExprRef::Internal(Rc::default()),
            exprs: ExprsTable,
            op_types: OpTypeTable::default(),
            info_types: InfoTypeTable::default(),
        }
    }
}

impl RootBuilder {
    pub fn new() -> Self {
        Default::default()
    }
}

#[derive(Archive, Serialize, Default)]
#[rkyv(archived = ExprData)]
pub struct ExprBuilder {
    pub closure_symbols: Vec<Symbol>,
    pub closure_inputs: Vec<(Var, VarInfo)>,
    pub inputs: Vec<(Var, VarInfo)>,
    pub eqns: Vec<EqnBuilder>,
    pub outputs: Vec<Var>,
}

#[derive(Archive)]
#[rkyv(archived = ExtExprData)]
pub struct ExtExprBuilder {
    pub root_hash: Blake3Hash,
    pub idx: ExprIdx,
}

#[derive(Archive, Serialize)]
#[rkyv(archived = EqnData)]
pub struct EqnBuilder {
    pub op: OpBuilder,
    pub subexpr: Vec<SubExprBuilder>,
}

#[derive(Archive, Serialize)]
#[rkyv(archived = SubExprData)]
pub struct SubExprBuilder {
    pub handle: ExprRef,
    pub provided_symbols: Vec<Symbol>,
    pub provided_vars: Vec<Var>,
}

#[derive(Clone, Copy, Archive, Portable, CheckBytes, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[rkyv(as = Self)]
#[bytecheck(crate = bytecheck)]
#[repr(transparent)]
pub struct ExprIdx(pub(super) i32_le);

impl ExprIdx {
    // External slots 0, 1, ... are encoded as -1, -2, ... .
    fn external(slot: usize) -> Self {
        let slot = i32::try_from(slot).expect("external expression index exceeds i32::MAX");
        Self((!slot).into())
    }

    pub fn is_internal(&self) -> bool {
        self.0 >= 0
    }
}
impl From<ExprIdx> for usize {
    fn from(idx: ExprIdx) -> usize {
        let raw = idx.0.to_native();
        (if raw < 0 { !raw } else { raw }) as usize
    }
}

pub enum ExprRef {
    Internal(Rc<ExprBuilder>),
    External(ExprHandle),
}

pub trait ExprRefContext {
    fn intern_expr(&mut self, expr: &ExprRef) -> ExprIdx;
    fn internal_exprs(&mut self) -> Vec<Rc<ExprBuilder>>;
    fn external_exprs(&mut self) -> Vec<ExprHandle>;
}

impl Archive for ExprRef {
    type Archived = ExprIdx;
    type Resolver = ExprIdx;

    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        rkyv::munge::munge!(let ExprIdx(index) = out);
        index.write(resolver.0);
    }
}

impl<S: Fallible + ExprRefContext + ?Sized> Serialize<S> for ExprRef {
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        Ok(serializer.intern_expr(self))
    }
}

// Local expressions are stored by data, external expressions by hash reference.
pub struct ExprsTable;

#[derive(Archive)]
#[rkyv(archived=ExprsTableData, resolver=ExprsTableResolver)]
pub struct ExprsTableRaw {
    pub internal: Vec<ExprBuilder>,
}

impl Archive for ExprsTable {
    type Archived = ExprsTableData;
    type Resolver = (ExprsTableRaw, ExprsTableResolver);

    fn resolve(&self, (table, resolver): Self::Resolver, out: Place<Self::Archived>) {
        table.resolve(resolver, out);
    }
}

pub struct ExprRefCtx {
    internal_ids: BTreeMap<*const ExprBuilder, ExprIdx>,
    internal: Vec<Rc<ExprBuilder>>,
    external_ids: BTreeMap<ExprHandle, ExprIdx>,
    external: Vec<ExprHandle>,
}

impl ExprRefContext for ExprRefCtx {
    fn intern_expr(&mut self, expr: &ExprRef) -> ExprIdx {
        match expr {
            ExprRef::External(handle) => {
                if let Some(&idx) = self.external_ids.get(handle) {
                    return idx;
                }
                let idx = ExprIdx::external(self.external.len());
                self.external.push(handle.clone());
                self.external_ids.insert(handle.clone(), idx);
                idx
            }
            ExprRef::Internal(handle) => {
                if let Some(&idx) = self.internal_ids.get(&Rc::as_ptr(handle)) {
                    return idx;
                }
                let idx = ExprIdx((self.internal.len() as i32).into());
                self.internal.push(handle.clone());
                self.internal_ids.insert(Rc::as_ptr(handle), idx);
                idx
            }
        }
    }
    fn internal_exprs(&mut self) -> Vec<Rc<ExprBuilder>> {
        std::mem::take(&mut self.internal)
    }
    fn external_exprs(&mut self) -> Vec<ExprHandle> {
        std::mem::take(&mut self.external)
    }
}
