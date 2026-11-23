mod blob;
mod owned_rkyv;

pub use blob::*;
pub use owned_rkyv::*;

use rkyv::{Archive, Deserialize, Portable, Serialize};

#[derive(
    Archive,
    Serialize,
    Deserialize,
    Portable,
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
#[repr(transparent)]
pub struct Blake3Hash([u8; 32]);
