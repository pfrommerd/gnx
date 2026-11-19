use gnx_expr::{Eqn, InterfaceError, LoweringError};

/// An error tracing, recording, or expanding a function transformation.
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    #[error(transparent)]
    Interface(#[from] InterfaceError),
    #[error("{0}")]
    Signature(#[from] LoweringError),
    #[error("{0}")]
    Invalid(String),
    #[error("equation {index} ({op}): {source}")]
    Equation {
        index: usize,
        op: String,
        source: Box<TransformError>,
    },
    #[error("expanding {op}: {source}")]
    Expanding {
        op: String,
        source: Box<TransformError>,
    },
}

impl TransformError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    pub(crate) fn at(self, index: usize, eqn: &Eqn) -> Self {
        Self::Equation {
            index,
            op: eqn.op().to_string(),
            source: Box::new(self),
        }
    }
}

pub type TransformResult<T> = Result<T, TransformError>;

/// The error of a fallible transformed function: the transformation failed,
/// or the traced body itself returned an error.
#[derive(Debug, thiserror::Error)]
pub enum ApplyError<E> {
    #[error(transparent)]
    Transform(#[from] TransformError),
    #[error("transformed function failed: {0}")]
    Body(E),
}
