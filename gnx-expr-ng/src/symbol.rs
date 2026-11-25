use rkyv::{
    Archive, Portable, Serialize,
    bytecheck::{self, CheckBytes},
    rend::u32_le,
};

#[derive(Archive, Serialize, Portable, CheckBytes, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[rkyv(as = Self)]
#[bytecheck(crate = bytecheck)]
#[repr(transparent)]
pub struct Symbol(u32_le);
impl Symbol {
    pub fn new(id: u32) -> Self {
        Self(id.into())
    }
    pub fn id(self) -> u32 {
        self.0.to_native()
    }
}
