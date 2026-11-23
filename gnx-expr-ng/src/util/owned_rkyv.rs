use std::marker::PhantomData;
use std::ops::Deref;

use rkyv::{
    Archive,
    api::{
        access_pos_unchecked,
        high::{HighValidator, access_pos},
    },
    bytecheck::CheckBytes,
    rancor::Error,
    util::AlignedVec,
};

pub struct OwnedArchive<T> {
    bytes: AlignedVec,
    offset: usize,
    _type: PhantomData<T>,
}

impl<T: Archive> OwnedArchive<T> {
    pub fn new(bytes: AlignedVec, offset: usize) -> Result<Self, Error>
    where
        T::Archived: for<'a> CheckBytes<HighValidator<'a, Error>>,
    {
        access_pos::<<T as Archive>::Archived, Error>(&bytes[..], offset)?;

        Ok(Self {
            bytes,
            offset,
            _type: PhantomData,
        })
    }

    pub fn get(&self) -> &T::Archived {
        // SAFETY: construction validated this location, and the private
        // backing buffer is never exposed for mutation.
        unsafe { access_pos_unchecked::<T::Archived>(&self.bytes[..], self.offset) }
    }
}

impl<T: Archive> Deref for OwnedArchive<T> {
    type Target = T::Archived;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}
