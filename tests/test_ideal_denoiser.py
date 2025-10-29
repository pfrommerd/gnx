import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from gnx.methods.diffusion import IdealDiffuser, IdentityFlowParam
from gnx.methods.noise_schedule import NoiseSchedule

from gnx.util.distribution import Gaussian, Mixture


def compute_expected_v(dist, schedule, xt, t):
    @jax.vmap
    def est_log_prob(x0):
        eps = (xt - schedule.alpha(t) * x0) / schedule.sigma(t)
        log_pdf = jnp.sum(stats.norm.logpdf(eps))
        return log_pdf

    # use the means so we don't have to sample
    x0s = dist.components.mean
    est_log_probs = est_log_prob(x0s)
    weights = jax.nn.softmax(est_log_probs)

    x0_hat = jnp.sum(x0s * weights, axis=0)
    x1_hat = (xt - x0_hat * schedule.alpha(t)) / schedule.sigma(t)
    alpha_dot = jax.grad(schedule.alpha)(t)
    sigma_dot = jax.grad(schedule.sigma)(t)
    v_expected = alpha_dot * x0_hat + sigma_dot * x1_hat
    return v_expected


def compute_expected_score(dist, schedule, xt, t):
    @jax.vmap
    def est_log_prob(x0):
        eps = (xt - schedule.alpha(t) * x0) / schedule.sigma(t)
        log_pdf = jnp.sum(stats.norm.logpdf(eps))
        return log_pdf

    # use the means so we don't have to sample
    x0s = dist.components.mean
    est_log_probs = est_log_prob(x0s)
    weights = jax.nn.softmax(est_log_probs)
    x0_hat = jnp.sum(x0s * weights, axis=0)
    score = (x0_hat * schedule.alpha(t) - xt) / (schedule.sigma(t) ** 2)
    return score


def test_ideal():
    components = Gaussian(
        mean=jnp.array([1.0, -1.0]),
        std=0.0 * jnp.ones((2,)),
    )
    dist = Mixture(components)
    schedule = (
        NoiseSchedule.log_linear_noise(1e-3, 100)
        .anneal_signal_linear()
        .constant_variance()
    )
    denoiser = IdealDiffuser(dist, schedule, IdentityFlowParam())
    t = jnp.array(0.5)
    x = jnp.array(0.5)
    # x = jnp.array([0.1, 0.2])
    transformed = schedule.transform(dist, t)

    log_pdf = transformed.log_pdf(x)

    #
    comp_means = components.mean * schedule.alpha(t)
    comp_stds = jnp.sqrt(
        components.std**2 * schedule.alpha(t) ** 2 + schedule.sigma(t) ** 2
    )
    comp_log_pdfs = jnp.array(
        [
            stats.norm.logpdf(x, loc=comp_means[0], scale=comp_stds[0]),
            stats.norm.logpdf(x, loc=comp_means[1], scale=comp_stds[1]),
        ]
    )
    expected_log_pdf = jax.nn.logsumexp(comp_log_pdfs - jnp.log(2.0))
    assert jnp.isclose(
        log_pdf, expected_log_pdf
    ), f"Expected {expected_log_pdf}, got {log_pdf}"

    score_ideal = jnp.array(schedule.transform(dist, t).score(x))
    score_expected = compute_expected_score(dist, schedule, x, t)
    assert jnp.isclose(
        score_ideal, score_expected
    ), f"Expected {score_expected}, got {score_ideal}"

    v_ideal = denoiser(x, t=t)
    v_expected = compute_expected_v(dist, schedule, x, t)
    assert jnp.isclose(v_ideal, v_expected), f"Expected {v_expected}, got {v_ideal}"
