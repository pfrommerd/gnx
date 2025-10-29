import math
import typing as typ

import jax
import jax.numpy as jnp
import functools


from ..core import graph, graph_util, nn, filters
from ..core.dataclasses import dataclass
from ..util import datasource, optimizers

from ..util.datasource import DataSource
from ..util.distribution import Distribution
from ..util.trainer import Trainer, Plugin
from ..util.trainer.objectives import TrackModel

from . import GenerativeModel, TrainSample


class Generator[T, Cond, Noise = jax.Array](typ.Protocol):
    @property
    def noise_distribution(self) -> Distribution[Noise]: ...

    def __call__(self, noise: Noise, /, cond: Cond) -> T: ...


class Discriminator[T, Cond](typ.Protocol):
    def __call__(self, sample: T, /, cond: Cond) -> jax.Array: ...


class AdversarialModel[T, Cond, Noise = jax.Array](GenerativeModel[T, Cond], nn.Module):
    def __init__(
        self,
        generator: Generator[T, Cond, Noise],
        discriminator: Discriminator[T, Cond],
    ):
        self.generator = generator
        self.discriminator = discriminator

    @functools.partial(jax.jit, static_argnames=("shape",))
    def sample(
        self,
        key: jax.Array,
        shape: tuple[int, ...] = (),
        *,
        cond: Cond,
        noise: Noise | None = None,
    ) -> T:
        N = math.prod(shape)
        noise = (
            self.generator.noise_distribution.sample(key, (N,))
            if noise is None
            else noise
        )
        samples = jax.vmap(
            lambda noise: self.generator(noise, cond),
        )(noise)
        samples = jax.tree.map(lambda x: jnp.reshape(x, shape + x.shape[1:]), samples)
        return samples

    def trainer(
        self,
        data: DataSource[TrainSample[T, Cond]],
        *,
        shuffle_rng: nn.RngStream | None,
        update_ratio: int | None,
        batch_size: int,
        gen_optimizer: optimizers.Optimizer,
        disc_optimizer: optimizers.Optimizer,
        model_tracker: optimizers.ModelTracker | None = None,
        #
        plugins: typ.Sequence[Plugin] = (),
        logging_plugins: typ.Sequence[Plugin] = (),
        iterations: int,
        **kwargs,
    ) -> Trainer:
        assert not kwargs
        combined = datasource.zip(datasource.rng(), data.batch((batch_size,)))
        objective = AdversarialTrainState(
            self,
            gen_optimizer,
            disc_optimizer,
            update_ratio=update_ratio,
        )
        objective = TrackModel(
            self,
            objective,
            tracker=model_tracker,
            track_wrt=filters.All(nn.Param, filters.PathPrefix("generator")),
        )
        return Trainer(
            AdversarialTrainState(
                self,
                gen_optimizer,
                disc_optimizer,
                update_ratio,
            ),
            data=combined,
            shuffle_rng=shuffle_rng,
            plugins=plugins,
            logging_plugins=logging_plugins,
            iterations=iterations,
        )


class AdversarialTrainState[T, Noise, Cond](nn.Module):
    def __init__(
        self,
        model: AdversarialModel[T, Noise, Cond],
        gen_opt: optimizers.Optimizer,
        disc_opt: optimizers.Optimizer,
        update_ratio: int | None = None,
    ):
        self.iteration = nn.Variable(jnp.zeros((), dtype=jnp.int32))
        self.model = model
        self.gen_opt = gen_opt.init(self.model.generator, wrt=nn.Param)
        self.disc_opt = disc_opt.init(self.model.discriminator, wrt=nn.Param)
        self.update_ratio = update_ratio

    def eval_model(self) -> AdversarialModel[T, Noise, Cond]:
        model = graph_util.duplicate(self.model)
        model = graph.thaw(model)
        model.eval_mode()
        return model

    @jax.jit
    def update(self, data: tuple[jax.Array, TrainSample[T, Cond]]):
        def sample_losses(
            generator: Generator[T, Cond, Noise],
            discriminator: Discriminator[T, Cond],
            key: jax.Array,
            sample: TrainSample[T, Cond],
        ):
            noise = generator.noise_distribution.sample(key)
            gen_sample = generator(noise, cond=sample.cond)
            disc_gen = discriminator(gen_sample, cond=sample.cond)
            disc_real = discriminator(sample.value, cond=sample.cond)
            gen_loss = -disc_gen
            disc_loss = jax.nn.relu(1 + disc_gen) + jax.nn.relu(1 - disc_real)
            return {"gen_loss": gen_loss, "disc_loss": disc_loss}

        def batch_loss(
            gen_graphdef: graph.GraphDef[Generator[T, Cond, Noise]],
            gen_params: graph.GraphLeaves,
            gen_leaves: graph.GraphLeaves,
            disc_graphdef: graph.GraphDef[Discriminator[T, Cond]],
            disc_params: graph.GraphLeaves,
            disc_leaves: graph.GraphLeaves,
            key: jax.Array, batch: TrainSample[T, Cond], loss_gen: bool
        ) -> tuple[jax.Array, dict[str, jax.Array]]: # fmt: skip
            generator = graph.merge(gen_graphdef, gen_params, gen_leaves)
            discriminator = graph.merge(disc_graphdef, disc_params, disc_leaves)
            N = graph_util.axis_size(batch, 0)
            losses = jax.vmap(
                sample_losses,
                in_axes=(None, None, 0, 0),
            )(generator, discriminator, jax.random.split(key, N), batch) # fmt: skip
            losses = jax.tree.map(jnp.mean, losses)
            return (losses["gen_loss"] if loss_gen else losses["disc_loss"]), losses

        def step(self: AdversarialTrainState[T, Noise, Cond],
            key: jax.Array, batch: TrainSample[T, Cond],
            gen_step: bool = True, disc_step: bool = True,
        ): # fmt: skip
            gen = graph.split(self.model.generator, self.gen_opt.wrt, ...)
            disc = graph.split(self.model.discriminator, self.disc_opt.wrt, ...)
            gen_grads, gen_metrics = jax.grad(
                functools.partial(batch_loss, loss_gen=True),
                argnums=1, has_aux=True,
            )(*gen, *disc, key, batch) # fmt: skip
            disc_grads, disc_metrics = jax.grad(
                functools.partial(batch_loss, loss_gen=False),
                argnums=4, has_aux=True,
            )(*gen, *disc, key, batch) # fmt: skip
            if gen_step:
                self.gen_opt.update(self.model.generator, gen_grads)
            if disc_step:
                self.disc_opt.update(self.model.discriminator, disc_grads)
            return disc_metrics if disc_step else gen_metrics

        if self.update_ratio:
            metrics = jax.lax.cond(
                self.iteration[...] % (1 + self.update_ratio) == 0,
                functools.partial(step, gen_step=True, disc_step=False),
                functools.partial(step, gen_step=False, disc_step=True),
                self,
                *data,
            )
        else:
            metrics = step(self, *data)

        # Update the iteration count
        self.iteration[...] += 1

        return metrics


@dataclass
class GeneratorDistillSample[T, Cond, Noise]:
    value: T
    cond: Cond
    noise: Noise
