import contextlib
import typing as tp
import functools

import jax

from ..core import nn
from ..core.dataclasses import dataclass
from ..datasets import TrainSample, Visualizable
from ..util.datasource import DataSource
from ..util.trainer import CallbackFn, Plugin, Step, Trainer, ModelTrainState


class GenerativeModel[T, Cond](nn.Module):
    def sample(self, key: jax.Array, shape: tuple[int, ...] = (), *, cond: Cond) -> T:
        raise NotImplementedError("GenerativeModel must implement sample method")

    def trainer(
        self,
        *args,
        **kwargs,
    ) -> Trainer[ModelTrainState[tp.Self, tp.Any], tp.Any]:
        raise NotImplementedError("GenerativeModel must implement trainer() method")


type GenTrainState[T, Cond] = ModelTrainState[GenerativeModel[T, Cond], tp.Any]


# Training plugins for periodically evaluating
# and logging generated samples for a generative model


@functools.partial(jax.vmap, in_axes=(None, 0, 0))
def _sample(model, key, cond):
    return model.sample(key, cond=cond)


class GeneratePlugin[T: Visualizable, Cond](Plugin[GenTrainState[T, Cond], tp.Any]):
    def __init__(
        self,
        data: DataSource[TrainSample[T, Cond]],
        batch_size: int,
        interval: int,
        final: bool,
        *,
        rngs: nn.Rngs,
    ):
        self.rngs = rngs
        self.batch_size = batch_size
        self.interval = interval
        self.final = final
        self.cond_data = data.batch((batch_size,))
        self.cond_iterator = self.cond_data.sampler(rngs.gen())

    @contextlib.contextmanager
    def __call__(
        self, _: Step[GenTrainState[T, Cond]]
    ) -> tp.Iterator[CallbackFn[GenTrainState[T, Cond], tp.Any]]:
        def update(step: Step[GenTrainState[T, Cond]]):
            if (self.interval and step.iteration % self.interval == 0) or (
                self.final and step.iteration == step.max_iterations - 1
            ):
                if not self.cond_iterator.has_next():
                    self.cond_iterator = self.cond_data.sampler(self.rngs.gen())
                data: TrainSample[T, Cond] = self.cond_iterator.next()

                model = step.state.eval_model()
                keys = jax.random.split(self.rngs.gen(), self.batch_size)
                step.add_results({"gt": data.value.visualize()})
                samples = _sample(model, keys, data.cond)
                step.add_results({"gen": samples.visualize(cond=data.cond)})

        yield update
