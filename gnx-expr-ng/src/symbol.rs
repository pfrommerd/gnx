use rkyv::Archive;
use rkyv::rend::u32_le;

#[derive(Archive)]
pub struct Symbol(u32_le);
