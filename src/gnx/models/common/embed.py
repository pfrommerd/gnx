import jax
import jax.numpy as jnp

from ...core import nn
from ...methods.noise_schedule import NoiseSchedule


def _sinusoidal_embedding(x, dim, min_period=0.01, max_period=10000.0):
    assert x.shape == (), "x must be a scalar"
    half_dim = dim // 2
    freqs = jnp.exp(-jnp.linspace(jnp.log(min_period), jnp.log(max_period), half_dim))
    emb = x * freqs
    emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
    if dim % 2 == 1:
        emb = jnp.concatenate((emb, jnp.zeros(())))
    return emb


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, min_period=0.01, max_period: float = 10000.0):
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period

    @jax.jit
    def __call__(self, x: jax.Array) -> jax.Array:
        return _sinusoidal_embedding(x, self.dim, self.min_period, self.max_period)


class DualSinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.max_period = max_period

    @jax.jit
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        a = _sinusoidal_embedding(x, self.dim // 2, self.max_period)
        b = _sinusoidal_embedding(y, self.dim // 2, self.max_period)
        return jnp.concatenate((a, b), axis=-1)


class LogSNRSinusoidEmbed(nn.Module):
    def __init__(self, schedule: NoiseSchedule):
        super().__init__()
        self.schedule = schedule

    def __call__(self, t):
        # rescale by the alpha of the schedule
        # to get the noise-to-signal ratio
        snr = self.schedule.sigma(t) / self.schedule.alpha(t)
        log_snr = 0.5 * jnp.log(snr)
        embed = jnp.stack((jnp.sin(log_snr), jnp.cos(log_snr)), axis=-1)
        return embed
