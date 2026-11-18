use std::convert::Infallible;

use crate::Graph;

/// Normalize a function's return type into a graph or an error.
/// `from_result` and `into_result` must round-trip without changing the result.
pub trait GraphResult {
    type Output: Graph;
    type Error;

    fn into_result(self) -> Result<Self::Output, Self::Error>;
    fn from_result(result: Result<Self::Output, Self::Error>) -> Self;
}

impl<G: Graph> GraphResult for G {
    type Output = G;
    type Error = Infallible;

    fn into_result(self) -> Result<G, Infallible> {
        Ok(self)
    }

    fn from_result(result: Result<G, Infallible>) -> Self {
        result.unwrap_or_else(|never| match never {})
    }
}

impl<G: Graph, E> GraphResult for Result<G, E> {
    type Output = G;
    type Error = E;

    fn into_result(self) -> Result<G, E> {
        self
    }

    fn from_result(result: Result<G, E>) -> Self {
        result
    }
}
