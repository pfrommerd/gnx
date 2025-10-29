import contextlib
import typing as tp

import logging
import jax
import jax.numpy as jnp


from ...core import nn
from ...core.dataclasses import dataclass
from ..datasource import DataSource, DataIterator

type Metrics = tp.Mapping[str, "jax.Array | Metrics"]

logger = logging.getLogger(__name__)


@dataclass
class Step[State: TrainState]:
    max_iterations: int
    max_epochs: int | None
    iterations_per_epoch: int | None

    iteration: int
    epoch: int
    epoch_iteration: int

    state: State

    metrics: Metrics
    extra_results: list[tp.Any]

    def add_results(self, results: tp.Any):
        self.extra_results.append(results)


@dataclass
class Checkpoint[State: TrainState]:
    iteration: int
    epoch: int
    epoch_iteration: int

    # The rngstream used for sampling the data
    shuffle_rng: nn.RngStream | None

    state: State


class CallbackFn[State: TrainState, Data](tp.Protocol):
    def __call__(self, step: Step[State], /): ...


class Plugin[State: TrainState, Data = tp.Any](tp.Protocol):
    def __call__(
        self, init_step: Step[State], /
    ) -> tp.ContextManager[CallbackFn[State, Data] | None]: ...


# A state that has an update function built-in.
class TrainState[Data](tp.Protocol):
    def update(self, data: Data, /) -> Metrics: ...


# A train state that is associated with a model
class ModelTrainState[Model: nn.Module, Data = tp.Any](TrainState[Data], tp.Protocol):
    # Should return the model that is being trained.
    # In particular this should be referentially equalent to the model
    # being trained by the optimizer.
    @property
    def model(self) -> Model: ...

    # Shoud always return a *duplicate* of the model to eval on
    # (i.e. referentially different from self.model)
    def eval_model(self) -> Model: ...


# For some reason pyright doesn't infer these should be covariant/contravariant
State = tp.TypeVar("State", bound=TrainState, covariant=True)
Data = tp.TypeVar("Data", default=tp.Any, covariant=True)


@jax.jit
def _data_next[T](data_iterator: DataIterator[T], shuffle_rng: nn.RngStream) -> T:
    batch = data_iterator.cyclic_next(shuffle_rng)
    logger.debug(f"Datasource yields: {jax.tree.map(jnp.shape, batch)}")
    return batch


@jax.jit
def _train_step(state, data) -> Metrics:
    return state.update(data)


class Trainer(tp.Generic[State, Data]):
    def __init__(
        self,
        state: State,
        *,
        data: DataSource[Data],
        shuffle_rng: nn.RngStream | None,
        plugins: tp.Sequence[Plugin] = (),
        logging_plugins: tp.Sequence[Plugin] = (),
        # Must be specified:
        iterations: int,
        # For with_state
        _data_iterator: DataIterator[Data] | None = None,
        _iterations_per_epoch: int | None = None,
        _iteration: int = 0,
        _epoch: int = 0,
        _epoch_iteration: int = 0,
    ):
        self.data = data
        self.shuffle_rng = shuffle_rng

        if _iterations_per_epoch is None:
            try:
                iterations_per_epoch = len(data)
                epochs = (iterations + iterations_per_epoch - 1) // iterations_per_epoch
            except (TypeError, NotImplementedError):
                iterations_per_epoch = None
                epochs = None
        else:
            iterations_per_epoch = _iterations_per_epoch
            epochs = (iterations + iterations_per_epoch - 1) // iterations_per_epoch

        self.data_iterator = (
            _data_iterator
            if _data_iterator is not None
            else data.sampler(shuffle_rng() if shuffle_rng is not None else None)
        )

        self.shuffle_rng = shuffle_rng

        self.plugins = plugins
        self.logging_plugins = logging_plugins

        self.iterations_per_epoch = iterations_per_epoch
        self.max_iterations = iterations
        self.max_epochs = epochs

        self._state = state
        self.iteration = _iteration
        self.epoch = _epoch
        self.epoch_iteration = _epoch_iteration

    @property
    def state(self) -> State:
        return self._state

    def replace_state[NewState: TrainState](
        self, state: NewState
    ) -> "Trainer[NewState, Data]":
        return Trainer(
            state,
            data=self.data,  # type: ignore
            _data_iterator=self.data.sampler(None),
            _iterations_per_epoch=self.iterations_per_epoch,
            shuffle_rng=self.shuffle_rng,
            plugins=self.plugins,
            logging_plugins=self.logging_plugins,
            iterations=self.max_iterations,
        )  # type: ignore

    @property
    def checkpoint(self) -> Checkpoint[State]:
        return Checkpoint(
            iteration=self.iteration,
            epoch=self.epoch,
            epoch_iteration=self.epoch_iteration,
            shuffle_rng=self.shuffle_rng,
            state=self.state,
        )

    def load_checkpoint(self, checkpoint: Checkpoint, reset_iterator: bool = True):
        self.iteration = checkpoint.iteration
        self.epoch = checkpoint.epoch
        self.epoch_iteration = checkpoint.epoch_iteration
        self.shuffle_rng = checkpoint.shuffle_rng
        # reset the data iterator to the correct state
        # copy the checkpoint state into the current trainer state
        if reset_iterator:
            if checkpoint.shuffle_rng is not None:
                self.data_iterator.reset(checkpoint.shuffle_rng.last())
            self.data_iterator.skip(self.epoch_iteration)
        nn.update(self._state, checkpoint.state)

    def steps(self, iterations: int | None = None) -> tp.Iterator[Step[State]]:
        if iterations is not None:
            self.max_iterations = self.max_iterations + iterations
            if self.iterations_per_epoch is not None:
                self.max_epochs = (
                    self.max_iterations + self.iterations_per_epoch - 1
                ) // self.iterations_per_epoch
            else:
                self.max_epochs = None

        with contextlib.ExitStack() as stack:
            pre_callbacks = None
            post_callbacks = None

            while self.iteration < self.max_iterations:
                data = _data_next(self.data_iterator, self.shuffle_rng)
                metrics = _train_step(self.state, data)
                current_step = Step(
                    max_iterations=self.max_iterations,
                    max_epochs=self.max_epochs,
                    iterations_per_epoch=self.iterations_per_epoch,
                    iteration=self.iteration,
                    epoch=self.epoch,
                    epoch_iteration=self.epoch_iteration,
                    state=self.state,
                    metrics=metrics,
                    extra_results=[],
                )
                if pre_callbacks is None:
                    pre_callbacks = [cb for cb in (
                            stack.enter_context(plugin(current_step))
                            for plugin in self.plugins
                    ) if cb is not None]  # fmt: skip

                for cb in pre_callbacks:
                    cb(current_step)

                yield current_step

                if post_callbacks is None:
                    post_callbacks = [cb for cb in (
                            stack.enter_context(plugin(current_step))
                            for plugin in self.logging_plugins
                    ) if cb is not None]  # fmt: skip

                for cb in post_callbacks:
                    cb(current_step)

                # Bump the iteration and epoch iteration counters
                self.iteration += 1
                if not self.data_iterator.has_next():
                    self.epoch += 1
                    self.epoch_iteration = 0
                else:
                    self.epoch_iteration += 1

    def run(self, iterations: int | None = None):
        for _ in self.steps(iterations):
            pass
