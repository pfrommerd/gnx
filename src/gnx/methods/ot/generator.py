import math
import jax
import jax.numpy as jnp
import typing as tp

from ...core import nn, graph, graph_util
from ...util import optimizers, datasource
from ...util.trainer import Plugin, Trainer, Metrics, ModelTrainState
from ...util.trainer.objectives import Minimize, TrackModel
from ...util.datasource import DataSource

from .. import GenerativeModel, TrainSample
from ..gan import Generator

from . import OTCostFn, OTSample, OTSamplerFactory, OTTrainSample, TrainableOTSampler


class GeneratorOTCost[T, Cond, Noise](OTCostFn[Noise, T, Cond]):
    def __init__(self, generator: Generator[T, Cond, Noise]):
        self.generator = generator

    @jax.jit
    def __call__(self, source: Noise, target: T, cond: Cond | None = None):
        c: Cond = cond  # type: ignore
        pred = self.generator(source, cond=c)
        return graph_util.mse(pred, target)


class OTGeneratorModel[T, Cond, Noise = jax.Array](GenerativeModel[T, Cond]):
    def __init__(self, generator: Generator[T, Cond, Noise]):
        self.generator = generator

    def _sample(self, key: jax.Array, cond: Cond = None):
        pass

    def sample(
        self,
        key: jax.Array,
        shape: tuple[int, ...] = (),
        cond: Cond = None,
        noise: Noise | None = None,
    ) -> T:
        if shape == ():
            noise = (
                self.generator.noise_distribution.sample(key)
                if noise is None
                else noise
            )
            return self.generator(noise, cond=cond)
        else:
            N = math.prod(shape)
            noise = (
                self.generator.noise_distribution.sample(key, (N,))
                if noise is None
                else noise
            )
            samples = jax.vmap(
                lambda noise: self.generator(noise, cond),
            )(noise)
            samples = jax.tree.map(
                lambda x: jnp.reshape(x, shape + x.shape[1:]), samples
            )
            return samples

    @jax.jit
    def loss(self, data: OTSample[Noise, T, Cond]) -> tuple[jax.Array, Metrics]:
        @jax.vmap
        def sample_loss(src, tgt, cond):
            pred = self.generator(src, cond=cond)
            return graph_util.mse(pred, tgt)

        loss = sample_loss(data.a, data.b, data.cond)
        loss = jnp.mean(loss)
        return loss, {"mse": loss}

    def trainer(
        self,
        data: DataSource[TrainSample[T, Cond]],
        shuffle_rng: nn.RngStream | None,
        *,
        solver: (
            TrainableOTSampler[Noise, T, Cond]
            | OTSamplerFactory[T, Cond, Noise, tp.Any]
        ),
        model_tracker: optimizers.ModelTracker | None = None,
        iterations: int,
        **kwargs,
    ) -> Trainer:
        assert not kwargs
        if not hasattr(solver, "trainer"):
            factory: OTSamplerFactory[T, Cond, Noise, tp.Any] = solver  # type: ignore
            optimizer: optimizers.Optimizer | None = kwargs.pop("optimizer", None)
            batch_size: int | None = kwargs.pop("batch_size", None)
            assert optimizer is not None, "Optimizer must be provided for OTSolver"
            assert batch_size is not None, "Batch size must be provided for OTSolver"
            data, ot_sampler = factory.create_generator_sampler(
                data, batch_shape=(batch_size,), generator=self.generator
            )
            ot_data = data.map(lambda _, sample: ot_sampler.sample(sample))
            objective = Minimize(self, optimizer)
            objective = TrackModel(self, objective, tracker=model_tracker)
            return Trainer(
                objective,
                iterations=iterations,
                data=ot_data,
                shuffle_rng=shuffle_rng,
                **kwargs,
            )
        else:
            sampler: TrainableOTSampler[Noise, T, Cond] = solver  # type: ignore

            def process(
                key: jax.Array,
                data: TrainSample[T, Cond],
            ) -> OTTrainSample[Noise, T, Cond]:
                noise = self.generator.noise_distribution.sample(key)
                return OTTrainSample(a=noise, b=data.value, cond=data.cond)

            proc_data = data.map(process)
            cost_fn = GeneratorOTCost(self.generator)
            trainer = sampler.trainer(
                iterations=iterations,
                data=proc_data,
                shuffle_rng=shuffle_rng,
                cost=cost_fn,
                optimize_cost_params=True,
                **kwargs,
            )
            # wrap the state of the OT trainer to include ema as well as the current model
            trainer: Trainer[ModelTrainState[tp.Self]] = trainer.replace_state(
                TrackModel(self, trainer.state, tracker=model_tracker)
            )
            return trainer
