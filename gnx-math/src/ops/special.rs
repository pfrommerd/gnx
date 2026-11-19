use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt};

simple_operations! {
    ErfOperation, erf_op, erf, 1;
    ErfcOperation, erfc_op, erfc, 1;
    ErfInvOperation, erf_inv_op, erf_inv, 1;
    LgammaOperation, lgamma_op, lgamma, 1;
    DigammaOperation, digamma_op, digamma, 1;
    BesselI0eOperation, bessel_i0e_op, bessel_i0e, 1;
    BesselI1eOperation, bessel_i1e_op, bessel_i1e, 1;
    IgammaOperation, igamma_op, igamma, 2;
    IgammacOperation, igammac_op, igammac, 2;
    IgammaGradAOperation, igamma_grad_a_op, igamma_grad_a, 2;
    PolygammaOperation, polygamma_op, polygamma, 2;
    ZetaOperation, zeta_op, zeta, 2;
    RegularizedIncompleteBetaOperation, regularized_incomplete_beta_op, regularized_incomplete_beta, 3;
}
