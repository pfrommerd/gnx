mod dyn_rkyv;
mod owned_rkyv;

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
