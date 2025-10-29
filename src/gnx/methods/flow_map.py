import math
import functools
import typing as tp

import jax
import jax.numpy as jnp

from ..core import nn, graph_util
from ..core.dataclasses import dataclass

from ..util import optimizers
from ..util.trainer import Trainer, Plugin, ModelTrainState
from ..util.trainer.objectives import Minimize, TrackModel
from ..util.datasource import DataSource
from ..util.distribution import Distribution

from . import GenerativeModel
from .diffusion import ForwardProcess, Time, TimeLike


class FlowMap[T, Cond, FlowAux = None](tp.Protocol):
    def __call__(
        self, x_s: T, /, s: TimeLike, t: TimeLike, *, cond: Cond, aux: FlowAux
    ) -> T: ...


@dataclass(frozen=True)
class CouplingCond[Cond]:
    s: Time
    t: Time
    cond: Cond = None  # type: ignore[assignment]


class FlowMapModel[T, Cond, FlowAux](GenerativeModel[T, Cond], nn.Module):
    def __init__(
        self,
        flow_map: FlowMap[T, Cond, FlowAux],
        forward_process: ForwardProcess[T, Cond],
        flow_aux_data: DataSource[FlowAux],
        nfe: int = 1,
    ):
        self.flow_map = flow_map
        self.forward_process = forward_process
        self.flow_aux_data = flow_aux_data
        self.nfe = nfe

    @functools.partial(jax.jit, static_argnames=("nfe", "sample_path"))
    def _sample(
        self,
        key: jax.Array,
        cond: Cond,
        initial: T | None = None,
        sample_path: bool = False,
        nfe: int | None = None,
    ) -> T:
        nfe = nfe or self.nfe

        i_key, a_key = jax.random.split(key, 2)
        initial = (
            self.forward_process.approximate_final.sample(i_key)
            if initial is None
            else initial
        )
        times = jnp.linspace(0.0, 1.0, nfe + 1)
        s, t = times[::-1][:-1], times[::-1][1:]

        aux = (
            self.flow_aux_data.sample(a_key, shape=(nfe,))
            if self.flow_aux_data is not None
            else None
        )

        def step(x, input):
            s, t, aux = input
            next = self.flow_map(x, s=s, t=t, cond=cond, aux=aux)
            return next, x

        sample, history = jax.lax.scan(step, initial, (s, t, aux))

        if sample_path:
            return jax.tree.map(
                lambda xs, final: jnp.concatenate((xs, final[None, ...]))[::-1],
                history,
                sample,
            )
        else:
            return sample

    @functools.partial(jax.jit, static_argnames=("shape", "nfe", "sample_path"))
    def sample(
        self,
        key: jax.Array,
        shape: tuple[int, ...] = (),
        *,
        cond: Cond,
        initial: T | None = None,
        sample_path: bool = False,
        nfe: int | None = None,
    ) -> T:
        if shape == ():
            return self._sample(
                key, cond, initial=initial, sample_path=sample_path, nfe=nfe
            )
        else:
            N = math.prod(shape)
            keys = jax.random.split(key, N)
            samples = jax.vmap(
                lambda initial, k: self._sample(k, cond, initial=initial, nfe=nfe),
            )(initial, keys)
            samples = jax.tree.map(
                lambda x: jnp.reshape(x, shape + x.shape[1:]), samples
            )
            return samples


class UniformTimePairs(Distribution[tuple[Time, Time]]):
    @jax.jit
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> tuple[Time, Time]:
        s, t = jax.random.uniform(key, shape=(2,) + shape, minval=0.0, maxval=1.0)
        return s, t


class EndpointTimePairs(Distribution[tuple[Time, Time]]):
    @jax.jit
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> tuple[Time, Time]:
        return jnp.ones(shape), jnp.zeros(shape)


# Will take a distribution of pairs (s, t) and sort them such that t < s.
class ReverseTimePairs(Distribution[tuple[Time, Time]]):
    def __init__(self, distribution: Distribution[tuple[Time, Time]]):
        self.distribution = distribution

    @jax.jit
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> tuple[Time, Time]:
        s, t = self.distribution.sample(key, shape=shape)
        return jnp.maximum(s, t), jnp.minimum(s, t)


# Similarly will take a distribution of pairs (s, t) and sort them such that s < t.
class ForwardTimePairs(Distribution[tuple[Time, Time]]):
    def __init__(self, distribution: Distribution[tuple[Time, Time]]):
        self.distribution = distribution

    @jax.jit
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> tuple[Time, Time]:
        s, t = self.distribution.sample(key, shape=shape)
        return jnp.minimum(s, t), jnp.maximum(s, t)


# A simple flow map model that takes supervised
# pairs of (x_s, x_t, s, t) and learns to map x_s to x_t.


@dataclass(kw_only=True)
class FlowMapDistillSample[T, Cond, FlowAux]:
    x_s: T
    x_t: T
    s: Time
    t: Time
    cond: Cond
    aux: FlowAux


class DistilledFlowMapModel[T, Cond = None, FlowAux = None](
    FlowMapModel[T, Cond, FlowAux]
):
    def __init__(
        self,
        flow_map: FlowMap[T, Cond, FlowAux],
        forward_process: ForwardProcess[T, Cond],
        flow_aux_data: DataSource[FlowAux],
        nfe: int = 1,
    ):
        super().__init__(
            flow_map=flow_map,
            forward_process=forward_process,
            flow_aux_data=flow_aux_data,
            nfe=nfe,
        )

    @jax.jit
    def loss(
        self,
        batch: FlowMapDistillSample[T, Cond, FlowAux],
    ):
        @jax.vmap
        def sample_loss(sample):
            x_t_pred = self.flow_map(
                sample.x_s, s=sample.s, t=sample.t, cond=sample.cond, aux=sample.aux
            )
            return graph_util.mse(x_t_pred, sample.x_t)

        loss = jnp.mean(sample_loss(batch))
        return loss, {"loss": loss}

    def trainer(
        self,
        data: DataSource[FlowMapDistillSample[T, Cond, FlowAux]],
        *,
        shuffle_rng: nn.RngStream,
        optimizer: optimizers.Optimizer,
        model_tracker: optimizers.ModelTracker | None = None,
        batch_size: int,
        iterations: int,
        plugins: tp.Sequence[Plugin] = (),
        logging_plugins: tp.Sequence[Plugin] = (),
    ) -> Trainer[ModelTrainState[tp.Self], FlowMapDistillSample[T, Cond, FlowAux]]:
        # Optionally batch the data if batch_size is provided
        data = data.batch((batch_size,))
        objective = Minimize(self, optimizer)
        objective = TrackModel(self, objective, tracker=model_tracker)
        return Trainer(
            objective,
            data=data,
            iterations=iterations,
            shuffle_rng=shuffle_rng,
            plugins=plugins,
            logging_plugins=logging_plugins,
        )
