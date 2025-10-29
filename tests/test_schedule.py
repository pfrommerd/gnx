import jax
import jax.numpy as jnp

from gnx.methods.noise_schedule import NoiseSchedule


def test_parameterization():
    schedule = NoiseSchedule.log_linear_noise(1e-4, 20).constant_variance()
    x0 = jax.random.normal(jax.random.key(43), (2,))
    x1 = jax.random.normal(jax.random.key(42), (2,))
    t = jnp.array(0.0)
    xt = schedule.alpha(t) * x0 + schedule.sigma(t) * x1
    xt_dot = schedule.alpha_dot(t) * x0 + schedule.sigma_dot(t) * x1
    # epsilon parameterization
    eps = schedule.parameterize(0.0, 1.0)
    pred = eps.flow_to_output(xt_dot, xt, t=t)
    flow = eps.output_to_flow(pred, xt, t=t)
    assert jnp.allclose(pred, x1, atol=1e-5), "Epsilon parameterization failed"
    assert jnp.allclose(flow, xt_dot, atol=1e-5)

    # x0 parameterization
    denoise = schedule.parameterize(1.0, 0.0)
    pred = denoise.flow_to_output(xt_dot, xt, t=t)
    flow = denoise.output_to_flow(pred, xt, t=t)
    assert jnp.allclose(pred, x0, atol=1e-5), "Denoiser parameterization failed"
    assert jnp.allclose(flow, xt_dot, atol=1e-5)
