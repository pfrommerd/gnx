mod dyn_rkyv;
pub mod expr_macro;
pub mod hexpr_macro;
mod owned_rkyv;
pub mod pretty;

pub use dyn_rkyv::*;
pub use owned_rkyv::*;

use rkyv::bytecheck::{self, CheckBytes};
use rkyv::{Archive, Deserialize, Portable, Serialize};

#[derive(
    Archive,
    Serialize,
    Deserialize,
    Portable,
    CheckBytes,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
#[rkyv(as = Self)]
#[bytecheck(crate = bytecheck)]
#[repr(transparent)]
pub struct Blake3Hash([u8; 32]);

impl Blake3Hash {
    pub fn of(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Display for Blake3Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

impl std::fmt::Debug for ArchivedTypeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", self.dialect(), self.name())
    }
}

// Formats the items of an iterator as a list.
pub(crate) struct DebugList<I>(pub I);

impl<I: Iterator + Clone> std::fmt::Debug for DebugList<I>
where
    I::Item: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.0.clone()).finish()
    }
}
