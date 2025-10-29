import jax
import jax.numpy as jnp


from gnx.util import datasource
from gnx.util.distribution import Empirical
from gnx.core import nn, graph


@jax.jit
def next[Data](
    data_iterator: datasource.DataIterator[Data],
    rngs: nn.Rngs,
) -> Data:
    data = data_iterator.cyclic_next(rngs.data)
    return data


def test_pytree():
    ds = datasource.pytree(jnp.zeros((10, 2)))
    next(ds.sampler(jax.random.key(42)), nn.Rngs(42))
    ds = datasource.zip(datasource.rng(), ds)
    next(ds.sampler(jax.random.key(42)), nn.Rngs(42))


def test_distribution():
    empirical = Empirical(jnp.zeros((10, 2)))
    next(empirical.sampler(jax.random.key(42)), nn.Rngs(42))
    combined = datasource.zip(datasource.rng(), empirical).batch((1024,))
    next(combined.sampler(jax.random.key(42)), nn.Rngs(42))
