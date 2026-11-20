use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt, result_rank, signature};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FftKind {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct FftOperation {
    kind: FftKind,
    lengths: Vec<usize>,
}
impl FftOperation {
    pub fn kind(&self) -> FftKind {
        self.kind
    }
    pub fn lengths(&self) -> &[usize] {
        &self.lengths
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
    Op::new(FftOperation {
        kind,
        lengths: lengths.to_vec(),
    })
}
