import jax
import jax.numpy as jnp

from gnx.util.distribution import Empirical, Gaussian, Mixture, Noise


def test_gaussian():
    gaus = Gaussian(
        mean=jnp.array([0.0, 1.0]),
        std=jnp.array([0.1, 0.01]),
    )

    log_pdf = gaus.log_pdf(jnp.array([0.25, 0.95]))
    gt_log_pdf = jax.scipy.stats.norm.logpdf(
        0.25, loc=0.0, scale=0.1
    ) + jax.scipy.stats.norm.logpdf(0.95, loc=1.0, scale=0.01)
    assert jnp.isclose(log_pdf, gt_log_pdf), f"Expected {gt_log_pdf}, got {log_pdf}"

    samples = gaus.sample(jax.random.key(42), shape=(16,))

    gt_score = jax.vmap(jax.grad(gaus.log_pdf))(samples)
    our_score = jax.vmap(lambda x: gaus.score(x))(samples)

    print("GT scores:", gt_score)
    print("Our scores:", our_score)
    assert jnp.allclose(gt_score, our_score, atol=1e-5), "Scores do not match"


def test_mixture():
    dist = Mixture(
        Gaussian(
            mean=jnp.array([[0.0, 1.0], [-1.0, 0.0]]),
            std=jnp.array([[0.1, 0.1], [0.1, 0.1]]),
        )
    )
    sample_points = dist.sample(jax.random.key(42), shape=(16,))

    gt_scores = jax.vmap(jax.grad(dist.log_pdf))(sample_points)
    gt_score_potentials = jax.vmap(jax.grad(dist.log_potential))(sample_points)
    our_scores = jax.vmap(dist.score)(sample_points)
    print("GT scores:", gt_scores)
    print("GT (potential) scores:", gt_scores)
    assert jnp.all(
        jnp.abs(gt_scores - gt_score_potentials) <= 1e-5
    ), "Scores do not match"
    print("Our scores:", our_scores)
    assert jnp.all(jnp.abs(gt_scores - our_scores) <= 1e-5), "Scores do not match"

    empirical = Empirical(jnp.array([[0.0, 1.0], [-1.0, 0.0]]), sigma=0.1)
    emp_scores = jax.vmap(empirical.score)(sample_points)
    print("Our (empirical) scores:", emp_scores)
    assert jnp.all(
        jnp.abs(emp_scores - our_scores) <= 1e-5
    ), "Empirical scores do not match"

    dist.transform(Noise(0.1))
