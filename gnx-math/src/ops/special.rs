use super::{Op, arity};

simple_operations! {
    ErfOperation, erf_op, erf, "erf", 1;
    ErfcOperation, erfc_op, erfc, "erfc", 1;
    ErfInvOperation, erf_inv_op, erf_inv, "erf_inv", 1;
    LgammaOperation, lgamma_op, lgamma, "lgamma", 1;
    DigammaOperation, digamma_op, digamma, "digamma", 1;
    BesselI0eOperation, bessel_i0e_op, bessel_i0e, "bessel_i0e", 1;
    BesselI1eOperation, bessel_i1e_op, bessel_i1e, "bessel_i1e", 1;
    IgammaOperation, igamma_op, igamma, "igamma", 2;
    IgammacOperation, igammac_op, igammac, "igammac", 2;
    IgammaGradAOperation, igamma_grad_a_op, igamma_grad_a, "igamma_grad_a", 2;
    PolygammaOperation, polygamma_op, polygamma, "polygamma", 2;
    ZetaOperation, zeta_op, zeta, "zeta", 2;
    RegularizedIncompleteBetaOperation, regularized_incomplete_beta_op, regularized_incomplete_beta, "regularized_incomplete_beta", 3;
}
