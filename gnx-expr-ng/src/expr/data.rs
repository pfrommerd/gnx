use std::rc::Rc;

use rkyv::rancor::Fallible;
use rkyv::rend::i32_le;
use rkyv::vec::VecResolver;
use rkyv::{Archive, Archived, Place, Portable, Serialize};
use std::collections::BTreeMap;

use crate::ExprHandle;
use crate::op::{OpData, OpTypeContext, OpTypeTable};
use crate::symbol::Symbol;
use crate::var::{Var, VarInfo};

#[derive(Archive)]
pub struct RootData {
    // the root expr (actually stored in the exprs table)
    pub root: ExprRef,
    // the table of all expressions,
    // both internal and external
    pub exprs: ExprTable,
    pub op_types: OpTypeTable,
}

#[derive(Archive, Serialize)]
#[rkyv(serialize_bounds(__S: OpTypeContext))]
pub struct ExprData {
    closure_symbols: Vec<Symbol>,
    closure_inputs: Vec<(Var, VarInfo)>,
    inputs: Vec<(Var, VarInfo)>,
    // expression eqns
    eqns: Vec<EqnData>,
    outputs: Vec<Var>,
}

#[derive(Archive)]
pub struct ExtExprData {
    root_hash: String,
    idx: ExprIdx,
}

#[derive(Archive, Serialize)]
#[rkyv(serialize_bounds(__S: OpTypeContext))]
pub struct EqnData {
    op: OpData,
    subexpr: Vec<SubExprData>,
}

#[derive(Archive, Serialize)]
#[rkyv(serialize_bounds(__S: OpTypeContext))]
pub struct SubExprData {
    handle: ExprRef,
    // from the closure of exprref
    provided_symbols: Vec<Symbol>,
    provided_vars: Vec<Var>,
}

pub enum ExprRef {
    Internal(Rc<ExprData>),
    External(ExprHandle),
}

#[derive(Clone, Copy, Archive, Portable, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[rkyv(as = Self)]
#[repr(transparent)]
pub struct ExprIdx(i32_le);

#[rustfmt::skip]
impl ExprIdx {
    pub fn is_internal(&self) -> bool { return self.0 >= 0; }
}
impl From<ExprIdx> for usize {
    fn from(idx: ExprIdx) -> usize {
        (idx.0.to_native() as isize).abs() as usize
    }
}

pub trait ExprRefContext {
    fn intern_expr(&mut self, expr: &ExprRef) -> ExprIdx;
    fn internal_exprs(&mut self) -> Vec<Rc<ExprData>>;
    fn external_exprs(&mut self) -> Vec<ExprHandle>;
}

impl Archive for ExprRef {
    type Archived = Archived<ExprIdx>;
    type Resolver = ExprIdx;

    fn resolve(&self, resolver: Self::Resolver, out: rkyv::Place<Self::Archived>) {
        resolver.resolve(ExprIdxResolver(()), out)
    }
}

impl<S> Serialize<S> for ExprRef
where
    S: Fallible + ExprRefContext + ?Sized,
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, <S as Fallible>::Error> {
        Ok(serializer.intern_expr(&self))
    }
}

// An expression table contains both
// (a) local expressions (stored by data)
// (b) external expressins (stored by hash reference)
pub struct ExprTable;

type RawExprTable = (Vec<ExprData>, Vec<ExtExprData>);
impl Archive for ExprTable {
    type Archived = Archived<RawExprTable>;
    type Resolver = (RawExprTable, (VecResolver, VecResolver));

    fn resolve(&self, (table, resolver): Self::Resolver, out: Place<Self::Archived>) {
        table.resolve(resolver, out);
    }
}

pub struct ExprRefCtx {
    internal_ids: BTreeMap<*const ExprData, ExprIdx>,
    internal: Vec<Rc<ExprData>>,
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
                let idx = ExprIdx((-(self.external.len() as i32)).into());
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
    fn internal_exprs(&mut self) -> Vec<Rc<ExprData>> {
        std::mem::take(&mut self.internal)
    }
    fn external_exprs(&mut self) -> Vec<ExprHandle> {
        std::mem::take(&mut self.external)
    }
}
