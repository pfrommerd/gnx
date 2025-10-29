import jax
import functools

from ..core import graph, graph_util
from ..util.datasource import DataSource

from ..methods import GenerativeModel

import ott.solvers.linear.sinkhorn as ott_sinkhorn
import ott.geometry.costs as ott_costs
import ott.geometry.pointcloud as ott_pointcloud
import ott.problems.linear.linear_problem as ott_linear


class OTEvaluator[T](graph.Object):
    def __init__(
        self,
        splits: dict[str, DataSource[T]],
        ott_solver=None,
        batch_size: int = 10_000,
    ):
        self.ott_solver = ott_solver or ott_sinkhorn.Sinkhorn()
        self.splits = splits
        self.batch_size = batch_size

    @functools.partial(jax.jit, static_argnames=("batch_size"))
    def _evaluate(
        self,
        key: jax.Array,
        model: GenerativeModel[T, None],
        *,
        batch_size: int,
    ):
        gen_key, splits_key = jax.random.split(key, 2)
        gen_batch = model.sample(gen_key, (batch_size,), cond=None)
        split_keys = jax.random.split(splits_key, len(self.splits))
        costs = {}
        for (split_name, data), split_key in zip(self.splits.items(), split_keys):
            data_key, solve_key = jax.random.split(split_key)
            data_batch = data.sample(data_key, (batch_size,))
            # flatten the data and generated batches
            data_flat, _ = graph_util.ravel(data_batch, axis=0)
            gen_flat, _ = graph_util.ravel(gen_batch, axis=0)
            geom = ott_pointcloud.PointCloud(
                data_flat,
                gen_flat,
                ott_costs.Euclidean(),
                relative_epsilon="std",
                epsilon=0.01,
            )
            solution = self.ott_solver(ott_linear.LinearProblem(geom))
            costs[split_name] = solution.primal_cost
        return costs

    def __call__(
        self, key: jax.Array, model: GenerativeModel[T, None], *, full: bool = True
    ):
        return self._evaluate(key, model, batch_size=self.batch_size)
