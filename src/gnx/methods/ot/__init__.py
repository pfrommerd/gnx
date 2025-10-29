import typing as tp

import jax
import jax.numpy as jnp
import typing as tp

from ...core import graph, graph_util, asserts
from ...util.datasource import DataSource
from ...util.distribution import Distribution

from ..flow_map import FlowMap, FlowMapDistillSample, ForwardProcess, Time
from ..gan import Generator, GeneratorDistillSample
from .. import TrainSample


class OTCostFn[A, B, Cond = None](graph.Object):
    def __call__(self, a: A, b: B, /, cond: Cond) -> jax.Array:
        raise NotImplementedError()

    def all_pairs(self, a: A, b: B, /, cond: Cond) -> jax.Array:
        """Compute the cost for all pairs of points in a and b."""
        return jax.vmap(
            jax.vmap(lambda a, b: self(a, b, cond=cond), in_axes=(None, 0)),
            in_axes=(0, None),
        )(a, b)


class SqrEuclideanCost[T](OTCostFn[T, T]):
    def __call__(self, a: T, b: T, /, cond=None) -> jax.Array:
        asserts.graphs_equal_shapes_and_dtypes(a, b)
        a_flat, _ = graph_util.ravel(a)
        b_flat, _ = graph_util.ravel(b)
        a_norm = jnp.sum(a_flat**2)
        b_norm = jnp.sum(b_flat**2)
        cross = -2.0 * jnp.vdot(a_flat, b_flat)
        return a_norm + b_norm + cross


class EuclideanCost[T](OTCostFn[T, T]):
    def __call__(self, a: T, b: T, /, cond=None) -> jax.Array:
        return graph_util.norm(graph_util.sub(a, b))


class MSECost[T](OTCostFn[T, T]):
    """Mean Squared Error cost function for OT problems."""

    def __call__(self, a: T, b: T, /, cond=None) -> jax.Array:
        asserts.graphs_equal_shapes_and_dtypes(a, b)
        a_flat, _ = graph_util.ravel(a)
        b_flat, _ = graph_util.ravel(b)
        a_norm = jnp.mean(a_flat**2)
        b_norm = jnp.mean(b_flat**2)
        cross = -2.0 * jnp.vdot(a_flat, b_flat) / a_flat.size
        return a_norm + b_norm + cross


class OTSamplerFactory[T, Cond = None, Noise = tp.Any, FlowAux = tp.Any](tp.Protocol):
    def create_flow_sampler(
        self,
        data: DataSource[TrainSample[T, Cond]],
        batch_shape: tuple[int, ...],
        flow_map: FlowMap[T, Cond, FlowAux],
        *,
        forward_process: ForwardProcess[T, Cond],
        time_distribution: Distribution[tuple[Time, Time]],
        flow_aux_data: DataSource[FlowAux],
    ) -> DataSource[FlowMapDistillSample[T, Cond, FlowAux]]: ...

    def create_generator_sampler(
        self,
        data: DataSource[TrainSample[T, Cond]],
        batch_shape: tuple[int, ...],
        generator: Generator[T, Cond, Noise],
    ) -> DataSource[GeneratorDistillSample[T, Cond, Noise]]: ...
