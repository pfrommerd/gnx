use std::sync::Arc;

use rkyv::{
    Archive, Portable, Serialize,
    bytecheck::{self, CheckBytes},
    rend::u32_le,
};

use crate::util::{
    ArchivedDyn, ArchivedTypeTable, AsDyn, TypeImpls, TypeTable, TypeTableBuilder, dyn_archive,
};

#[derive(Archive, Serialize, Portable, CheckBytes, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[rkyv(as = Self)]
#[bytecheck(crate = bytecheck)]
#[repr(transparent)]
pub struct Var(u32_le);
impl Var {
    pub fn new(id: u32) -> Self {
        Self(id.into())
    }
    pub fn id(self) -> u32 {
        self.0.to_native()
    }
}

dyn_archive! {
    // The dialect-defined metadata of a variable.
    // Concrete infos are named in a dialect!
    pub trait Info {}
}

// Serialization helpers: InfoTypeTable and InfoTypeImpls

// Used during serialization to build the InfoTypeTable
pub type InfoTypeCtx = TypeTableBuilder<dyn Info>;

// The archived version contains the (dialect, name) values.
// The unarchived version is a zero-sized type.
pub type InfoTypeTable = TypeTable<dyn Info>;
pub type InfoTypeTableData = ArchivedTypeTable<dyn Info>;

// The external table for info ids, resolved
// from an InfoTypeTableData at load time
pub type InfoTypeImpls = TypeImpls<dyn Info>;

#[derive(Portable)]
#[repr(C)]
pub struct VarInfoData {
    info: ArchivedDyn<dyn Info>,
}

impl VarInfoData {
    // SAFETY: The caller certifies that impls was resolved from the
    // InfoTypeTable of the archive this data is in, and that the archive is valid.
    pub unsafe fn get(&self, impls: &InfoTypeImpls) -> Option<&dyn Info> {
        unsafe { self.info.get(impls) }
    }
}

// An owned, cheaply cloned Info of any concrete type.
#[derive(Archive, Serialize, Clone)]
#[rkyv(as = VarInfoData)]
pub struct VarInfo {
    #[rkyv(with = AsDyn)]
    info: Arc<dyn Info>,
}

impl VarInfo {
    pub fn new<T: Info + 'static>(info: T) -> Self {
        Self {
            info: Arc::new(info),
        }
    }
}

impl std::ops::Deref for VarInfo {
    type Target = dyn Info;

    fn deref(&self) -> &Self::Target {
        &*self.info
    }
}
