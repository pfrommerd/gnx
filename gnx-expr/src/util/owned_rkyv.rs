use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;

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

/// Immutable bytes owned elsewhere, such as a memory-mapped file.
pub type SharedBytes = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// The alignment an archive's first byte must have for the values in it to
/// be aligned.
pub const ARCHIVE_ALIGNMENT: usize = AlignedVec::<16>::ALIGNMENT;

// Keeps the bytes of an archive alive. Their address is stable: an
// AlignedVec is never resized here, and shared bytes are immutable.
enum Backing {
    Owned(#[allow(dead_code)] AlignedVec),
    Shared(#[allow(dead_code)] SharedBytes),
}

/// An archived `T` together with the bytes that hold it.
pub struct OwnedArchive<T: Portable> {
    address: NonNull<u8>,
    len: usize,
    offset: usize,
    _backing: Backing,
    _type: PhantomData<T>,
}

// SAFETY: the bytes are immutable and kept alive by the backing, which is
// itself Send and Sync; the archive only gives out shared references to T.
unsafe impl<T: Portable + Send + Sync> Send for OwnedArchive<T> {}
unsafe impl<T: Portable + Send + Sync> Sync for OwnedArchive<T> {}

#[allow(type_alias_bounds)]
pub type OwnedArchived<T: Archive> = OwnedArchive<T::Archived>;

impl<T: Portable> OwnedArchive<T> {
    pub fn new(bytes: AlignedVec, offset: usize) -> Result<Self, Error>
    where
        T: for<'a> CheckBytes<HighValidator<'a, Error>>,
    {
        access_pos::<T, Error>(&bytes[..], offset)?;
        // SAFETY: just validated.
        Ok(unsafe { Self::new_unchecked(bytes, offset) })
    }

    // SAFETY: the caller certifies that bytes holds a valid archived T at
    // offset, e.g. because it was just written by serializing one.
    pub unsafe fn new_unchecked(bytes: AlignedVec, offset: usize) -> Self {
        Self {
            address: NonNull::from(&bytes[..]).cast(),
            len: bytes.len(),
            offset,
            _backing: Backing::Owned(bytes),
            _type: PhantomData,
        }
    }

    /// An archive over bytes owned elsewhere.
    ///
    /// # Safety
    /// `bytes` must hold a valid archived `T` at `offset`, and must return
    /// the same slice for as long as it lives.
    ///
    /// # Panics
    /// If the bytes do not start at a multiple of [`ARCHIVE_ALIGNMENT`].
    pub unsafe fn from_shared(bytes: SharedBytes, offset: usize) -> Self {
        let slice: &[u8] = (*bytes).as_ref();
        assert!(
            slice.as_ptr().addr().is_multiple_of(ARCHIVE_ALIGNMENT),
            "archive bytes are not aligned"
        );
        Self {
            address: NonNull::from(slice).cast(),
            len: slice.len(),
            offset,
            _backing: Backing::Shared(bytes),
            _type: PhantomData,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the backing keeps len bytes alive at address, unchanged.
        unsafe { std::slice::from_raw_parts(self.address.as_ptr(), self.len) }
    }

    pub fn get(&self) -> &T {
        // SAFETY: construction certified this location, and the backing
        // bytes are never exposed for mutation.
        unsafe { access_pos_unchecked::<T>(self.bytes(), self.offset) }
    }
}

impl<T: Portable> Deref for OwnedArchive<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}
