use rkyv::ptr_meta::DynMetadata;
use rkyv::rancor::{Fallible, Strategy};
use rkyv::rend::u32_le;
use rkyv::ser::Serializer;
use rkyv::vec::VecResolver;
use rkyv::{Archive, Archived, Place, Serialize};

use crate::util::Blob;
use std::collections::HashMap;

pub trait Operation {}

// Serialization helpers: OpType, OpTypeIdx, OpTypeTable, and OpTypeImpls

#[derive(Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct OpType(&'static str, &'static str);

#[derive(Archive, Serialize, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct OpTypeIdx(u32_le);

pub trait OpTypeContext {
    fn intern_op_type(&mut self, dialect: &'static str, name: &'static str) -> OpTypeIdx;
    fn interned_op_ids(&self) -> &[(&'static str, &'static str)];
}

impl Archive for OpType {
    type Archived = Archived<OpTypeIdx>;
    type Resolver = OpTypeIdx;

    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        resolver.resolve(OpTypeIdxResolver(()), out);
    }
}

impl<S> Serialize<S> for OpType
where
    S: Fallible + OpTypeContext + ?Sized,
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, <S as Fallible>::Error> {
        Ok(serializer.intern_op_type(self.0, self.1))
    }
}

// The archived version of an optypetable
// contains the (dialect, opname) values.
// The unarchived version is a zero-sized type.
pub struct OpTypeTable;

// The external table for operation ids
pub struct OpTypeImpls(Vec<DynMetadata<dyn Operation>>);

impl Archive for OpTypeTable {
    type Archived = Archived<Vec<(String, String)>>;
    type Resolver = (Vec<(String, String)>, VecResolver);

    fn resolve(&self, (ids, resolver): Self::Resolver, out: Place<Self::Archived>) {
        ids.resolve(resolver, out);
    }
}

impl OpTypeImpls {
    // SAFETY: The caller certifies that
    // the id belongs to the same table that the
    // resolver is associated with
    pub unsafe fn lookup<'op>(
        &self,
        _id: OpTypeIdx,
        _data: &'op Archived<Blob>,
    ) -> &'op dyn Operation {
        todo!()
    }
}

// Used during serialization to build the OpTypeTable
pub struct OpTypeCtx {
    indices: HashMap<(&'static str, &'static str), OpTypeIdx>,
    ids: Vec<(&'static str, &'static str)>,
}

impl<W, A, E> OpTypeContext for Strategy<Serializer<W, A, OpTypeCtx>, E> {
    fn intern_op_type(&mut self, dialect: &'static str, name: &'static str) -> OpTypeIdx {
        if let Some(&idx) = self.sharing.indices.get(&(dialect, name)) {
            return idx;
        }
        let idx = OpTypeIdx(u32_le::try_from(self.sharing.ids.len()).unwrap());
        self.sharing.ids.push((dialect, name));
        self.sharing.indices.insert((dialect, name), idx);
        idx
    }
    fn interned_op_ids(&self) -> &[(&'static str, &'static str)] {
        self.sharing.ids.as_slice()
    }
}

#[derive(Archive, Serialize)]
#[rkyv(serialize_bounds(__S: OpTypeContext))]
pub struct OpData {
    type_id: OpType,
    data: Blob,
}
