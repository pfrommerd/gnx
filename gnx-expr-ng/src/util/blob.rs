use rkyv::{
    Archive, Place, Serialize,
    rancor::{Fallible, Source},
    ser::{Writer, WriterExt},
    util::AlignedVec,
    vec::{ArchivedVec, VecResolver},
};

// Contains a complete, already serialized archive.
pub struct Blob(AlignedVec);

impl Archive for Blob {
    type Archived = ArchivedVec<u8>;
    type Resolver = VecResolver;

    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        ArchivedVec::resolve_from_len(self.0.len(), resolver, out);
    }
}

impl<S> Serialize<S> for Blob
where
    S: Fallible + Writer + ?Sized,
    S::Error: Source,
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let position = serializer.align(16)?;
        serializer.write(self.0.as_slice())?;
        Ok(VecResolver::from_pos(position))
    }
}
