import jax
import jax.numpy as jnp
import asyncio

from gnx.core import nn, graph
from gnx.util import optimizers, datasource
from gnx.util.trainer import Trainer
from gnx.util.trainer.objectives import TrackModel, Minimize


class Params(nn.Module):
    def __init__(self, value):
        self.param = nn.Param(value)


def test_update():
    def loss(model, data):
        return (model.param[...] - 16.0) ** 2, {}

    model = Params(0.0)
    objective = Minimize(loss, optimizers.sgd(0.1), model)
    objective.update(None)


def test_simple():
    def loss(model, data):
        return (model.param[...] - 16.0) ** 2, {}

    model = Params(0.0)
    trainer = Trainer(
        Minimize(
            loss, optimizers.sgd(optimizers.cosine_decay_schedule(0.1, 100)), model
        ),
        data=datasource.none(),
        iterations=100,
        shuffle_rng=nn.Rngs(42).data,
    )
    # update_orignal=True means that the original model will be updated
    trainer.run(iterations=100)
    assert jnp.abs(model.param[...] - 16.0) < 1e-3


class DummyState(graph.Object):
    def __init__(self, init: jax.Array):
        self.model = Params(init)

    def update(self, value: jax.Array):
        self.model.param[...] = value
        return {}


def test_ema():
    state = DummyState(jnp.ones((10,)))
    ema = TrackModel(state.model, state, optimizers.ema_tracker(0.9))
    decay = 0.9
    a, b = jnp.ones((10,)), 2 * jnp.ones((10,))
    # initial ema is all zeros
    assert jnp.allclose(ema.eval_model().param[...], jnp.zeros((10,)))
    # update the wrapped model
    ema.update(a)
    assert jnp.allclose(ema.eval_model().param[...], (1 - decay) * a)
    ema.update(b)
    assert jnp.allclose(
        ema.eval_model().param[...], (1 - decay) * decay * a + (1 - decay) * b
    )
