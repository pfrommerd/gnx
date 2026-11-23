use rkyv::Archive;
use rkyv::rend::u32_le;

#[derive(Archive)]
pub struct Var(u32_le);

#[derive(Archive)]
pub struct VarInfo;
