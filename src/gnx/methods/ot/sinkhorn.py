import typing as tp
import functools
import math

import ott.geometry.geometry as ott_geometry
import ott.problems.linear.linear_problem as ott_linear
import ott.solvers.linear.sinkhorn as ott_sinkhorn

import jax
import jax.numpy as jnp

from ...core import graph, graph_util, nn
from ...util.datasource import DataSource, DataIterator
from ...util.distribution import Distribution
from ...util import datasource

from .. import TrainSample
from ..diffusion import ForwardProcess, Time
from ..flow_map import CouplingCond, FlowMap
from ..gan import Generator

from .flow_map import FlowOTCost
from .generator import GeneratorOTCost
from . import OTSample, OTSampler, OTCostFn, OTTrainSample


class SinkhornSampler[Input, A, B, Cond](OTSampler[Input, A, B, Cond]):
    def __init__(
        self,
        aux_transform: tp.Callable[[jax.Array, Input], OTTrainSample[A, B, Cond]],
        batch_shape: tuple[int, ...],
        ot_batch_size: int,
        ot_batches: int,
        cost_fn: OTCostFn[A, B, Cond],
        ott_solver: ott_sinkhorn.Sinkhorn | None = None,
        rel_epsilon: float = 0.05,
    ):
        self.batch_shape = batch_shape
        self.ot_batch_size = ot_batch_size
        self.ot_batches = ot_batches
        assert ot_batch_size * ot_batches >= math.prod(
            batch_shape
        ), """
            The total number of samples in the OT batches must be at least as large as the requested batch shape.
        """

        self.aux_transform = aux_transform
        self.cost_fn = cost_fn
        self.rel_epsilon = rel_epsilon
        self.ott_solver = ott_solver or ott_sinkhorn.Sinkhorn()

    def _solve_batch(self, key: jax.Array, aux_batch: Input) -> tuple[A, B, Cond]:
        batch_key, proj_key = jax.random.split(key)
        batch = self.aux_transform(batch_key, aux_batch)
        C = self.cost_fn.all_pairs(batch.a, batch.b, cond=batch.cond)
        geom = ott_geometry.Geometry(
            C, relative_epsilon="mean", epsilon=self.rel_epsilon
        )
        prob = ott_linear.LinearProblem(geom)
        solution = self.ott_solver(prob)

        @jax.vmap
        def sample_rows(key, C_row: jax.Array):
            log_prob = (solution.g - C_row) / geom.epsilon
            idx = jax.random.categorical(key, log_prob)
            return idx

        pair_idxs = sample_rows(jax.random.split(proj_key, C.shape[0]), C)
        targets = jax.tree.map(lambda x: jnp.take(x, pair_idxs, axis=0), batch.b)
        # expand the conditioning values to match the number of samples
        n = graph_util.axis_size(batch.a, 0)
        samples_cond = jax.tree.map(
            lambda c: jnp.repeat(c[jnp.newaxis, ...], n, 0), batch.cond
        )
        return batch.a, targets, samples_cond

    @jax.jit
    def sample(self, key: jax.Array, input: Input) -> OTSample[A, B, Cond]:
        aux_batches: Input = jax.tree.map(
            lambda x: jnp.reshape(
                x, (self.ot_batches, self.ot_batch_size) + x.shape[1:]
            ),
            input,
        )
        keys = jax.random.split(key, self.ot_batches)
        a, b, cond = jax.vmap(self._solve_batch)(keys, aux_batches)
        cond = jax.tree.map(
            lambda x: jnp.repeat(
                x[:, jnp.newaxis], repeats=self.ot_batch_size, axis=-1
            ),
            cond,
        )
        samples = OTSample(a, b, cond)
        samples = jax.tree.map(
            lambda x: jnp.reshape(
                x, (self.ot_batches * self.ot_batch_size,) + x.shape[2:]
            ),
            samples,
        )
        # select only the number of samples needed to fill the requested batch shape
        if math.prod(self.batch_shape) < self.ot_batches * self.ot_batch_size:
            idxs = jax.random.choice(
                key,
                self.ot_batches * self.ot_batch_size,
                (math.prod(self.batch_shape),),
            )
            a, b, cond = jax.tree.map(lambda x: x[idxs], (a, b, cond))
        samples = jax.tree.map(
            lambda x: jnp.reshape(x, self.batch_shape + x.shape[1:]), samples
        )
        return samples


class SinkhornOTSamplerFactory[T, Noise, FlowAux](graph.Object):
    def __init__(
        self,
        ot_batch_size: int,
        ott_solver=None,
        rel_epsilon: float = 0.05,
        ot_batches: int = 1,
    ):
        self.ot_batch_size = ot_batch_size
        self.ott_solver = ott_solver
        self.rel_epsilon = rel_epsilon
        self.ot_batches = ot_batches

    @jax.jit
    def transform_flow_batch(
        self,
        forward_process: ForwardProcess[T],
        time_dist: Distribution[tuple[Time, Time]],
        flow_aux_data: DataSource[FlowAux],
        #
        key: jax.Array,
        batch: TrainSample[T],
        cond: CouplingCond | None,
    ) -> OTTrainSample[tuple[T, FlowAux], T, CouplingCond]:
        time_key, aux_key, xs_key, xt_key = jax.random.split(key, 4)
        if cond is None:
            s, t = time_dist.sample(time_key)
            cond = CouplingCond(s=s, t=t, cond=None)
        s, t = cond.s, cond.t

        # noise batch.value to times s and t
        @functools.partial(jax.vmap, in_axes=(0, None, 0))
        def fp(key, x0: T, t):
            return forward_process.forward(key, x0, t=t, cond=cond.cond)

        N = graph_util.axis_size(batch.value, 0)
        xs = fp(jax.random.split(xs_key, N), batch.value, s)
        xt = fp(jax.random.split(xt_key, N), batch.value, t)
        aux: FlowAux = flow_aux_data.sample(aux_key, (N,))
        return OTTrainSample((xs, aux), xt, cond)

    @jax.jit
    def transform_generator_batch(
        self,
        noise_dist: Distribution[Noise],
        key: jax.Array,
        batch: TrainSample[T],
    ) -> OTTrainSample[Noise, T]:
        N = graph_util.axis_size(batch.value, 0)
        a = noise_dist.sample(key, (N,))
        b = batch.value
        return OTTrainSample(a, b)

    def create_flow_sampler(
        self,
        data: DataSource[TrainSample[T]],
        batch_shape: tuple[int, ...],
        flow_map: FlowMap[T, None, FlowAux],
        *,
        forward_process: ForwardProcess[T],
        time_distribution: Distribution[tuple[Time, Time]],
        flow_aux_data: DataSource[FlowAux],
    ) -> tp.Any:  # type: ignore

        transform_batch = graph_util.Partial(
            type(self).transform_flow_batch,
            self,
            forward_process,
            time_distribution,
            flow_aux_data,
        )
        cost_fn = FlowOTCost(flow_map)
        data = data.batch((self.ot_batch_size,))
        sampler = SinkhornSampler[TrainSample[T], tuple[T, FlowAux], T, CouplingCond](
            aux_transform=transform_batch,
            cost_fn=cost_fn,
            rel_epsilon=self.rel_epsilon,
            batch_shape=batch_shape,
            ott_solver=self.ott_solver,
            ot_batch_size=self.ot_batch_size,
            ot_batches=self.ot_batches,
        )
        return data, sampler

    def create_generator_sampler(
        self,
        data: DataSource[TrainSample[T]],
        batch_shape: tuple[int, ...],
        generator: Generator[T, None, Noise],
    ) -> tp.Any:  # type: ignore
        transform_batch = graph_util.Partial(
            type(self).transform_generator_batch,
            self,
            generator.noise_distribution,
        )
        cost_fn = GeneratorOTCost(generator)
        data = data.batch((self.ot_batch_size,))
        sampler = SinkhornSampler[TrainSample[T], Noise, T, None](
            aux_transform=transform_batch,
            cost_fn=cost_fn,
            rel_epsilon=self.rel_epsilon,
            batch_shape=batch_shape,
            ott_solver=self.ott_solver,
            ot_batches=self.ot_batches,
            ot_batch_size=self.ot_batch_size,
        )
        return data, sampler
