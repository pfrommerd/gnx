use rkyv::{Archive, Serialize};

use crate::util::{ArchivedTypeTable, AsDyn, TypeImpls, TypeTable, TypeTableBuilder, dyn_archive};
use std::rc::Rc;

dyn_archive! {
    // Concrete operations are named in a dialect!
    pub trait Operation {}
}

// Serialization helpers: OpTypeTable and OpTypeImpls

// Used during serialization to build the OpTypeTable
pub type OpTypeCtx = TypeTableBuilder<dyn Operation>;

// The archived version of an optypetable
// contains the (dialect, opname) values.
// The unarchived version is a zero-sized type.
pub type OpTypeTable = TypeTable<dyn Operation>;
pub type OpTypeTableData = ArchivedTypeTable<dyn Operation>;

// The external table for operation ids, resolved
// from an OpTypeTableData at load time
pub type OpTypeImpls = TypeImpls<dyn Operation>;

#[derive(Archive, Serialize)]
#[rkyv(archived = OpData)]
pub struct OpBuilder {
    #[rkyv(with = AsDyn)]
    pub op: Rc<dyn Operation>,
}

impl OpBuilder {
    pub fn new(op: Rc<dyn Operation>) -> Self {
        Self { op }
    }
}
