use std::marker::PhantomData;
use std::ops::Deref;

use rkyv::{
    Archive, Portable,
    api::{
        access_pos_unchecked,
        high::{HighValidator, access_pos},
    },
    bytecheck::CheckBytes,
    rancor::Error,
    util::AlignedVec,
};

pub struct OwnedArchive<T: Portable> {
    bytes: AlignedVec,
    offset: usize,
    _type: PhantomData<T>,
}

#[allow(type_alias_bounds)]
pub type OwnedArchived<T: Archive> = OwnedArchive<T::Archived>;

impl<T: Portable> OwnedArchive<T> {
    pub fn new(bytes: AlignedVec, offset: usize) -> Result<Self, Error>
    where
        T: for<'a> CheckBytes<HighValidator<'a, Error>>,
    {
        access_pos::<T, Error>(&bytes[..], offset)?;

        Ok(Self {
            bytes,
            offset,
            _type: PhantomData,
        })
    }

    pub fn get(&self) -> &T {
        // SAFETY: construction validated this location, and the private
        // backing buffer is never exposed for mutation.
        unsafe { access_pos_unchecked::<T>(&self.bytes[..], self.offset) }
    }
}

impl<T: Portable> Deref for OwnedArchive<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}
