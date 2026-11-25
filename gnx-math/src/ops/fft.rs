use super::{Op, arity, result_rank, signature};
use gnx_common::ops::{indices, stored_indices};

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Portable,
)]
#[rkyv(as = Self)]
#[repr(u8)]
pub enum FftKind {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = FftOperation)]
pub struct Fft {
    kind: FftKind,
    lengths: Vec<u32>,
}
impl FftOperation {
    pub fn kind(&self) -> FftKind {
        self.kind
    }
    pub fn lengths(&self) -> Vec<usize> {
        indices(&self.lengths)
    }
}
gnx_common::array_operation!(
    FftOperation,
    "fft",
    |op, eqn| {
        arity(eqn, 1)?;
        if op.lengths.len() > 3 || op.lengths.len() > result_rank(eqn)? {
            return Err(signature("invalid FFT rank"));
        }
        Ok(())
    },
    lengths
);
pub fn fft(kind: FftKind, lengths: &[usize]) -> Op {
    Op::new(Fft {
        kind,
        lengths: stored_indices(lengths),
    })
}
