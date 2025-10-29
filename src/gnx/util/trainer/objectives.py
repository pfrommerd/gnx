import optax
import jax

import typing as tp

from ...core import nn, graph, graph_util, filters
from .. import optimizers
from . import Metrics, TrainState


# Associates a model for a given TrainState, including
# optional EMA tracking of parameters.


class TrackModel[Model: nn.Module, Data](graph.Object):
    def __init__(
        self,
        model: Model,
        state: TrainState[Data],
        tracker: optimizers.ModelTracker | None = None,
        track_wrt: filters.Filter = nn.Param,
    ):
        self.state = state
        self._model = model
        self.track = tracker.init(model, track_wrt) if tracker else None

    @property
    def model(self) -> Model:
        return self._model

    def eval_model(self) -> Model:
        if self.track is not None:
            model = self.track.current(self.model)
        else:
            model = graph_util.duplicate(self.model)
        # thaw the model to ensure we can
        # change the eval mode
        model = graph.thaw(model)
        model.eval_mode()
        return model

    def update(self, data: Data) -> Metrics:
        metrics = self.state.update(data)
        if self.track is not None:
            self.track.update(self.model)
        return metrics


class LossFn[Model, Data](tp.Protocol):
    def __call__(
        self,
        model: Model,
        batch: Data,
        /,
    ) -> tuple[jax.Array, Metrics]: ...


class ModelWithLoss[Model: nn.Module, Data](tp.Protocol):
    def loss(self: Model, data: Data, /) -> tuple[jax.Array, Metrics]: ...


# Force a donate of the associated buffers
# so that they are not accidentally reused elsewhere.
class Minimize[Model: nn.Module, Data](graph.Object):
    def __init__(
        self,
        loss: LossFn[Model, Data] | ModelWithLoss[Model, Data],
        optimizer: optimizers.Optimizer,
        model: Model | None = None,
        wrt: graph.Filter = nn.Param,
        **kwargs,
    ):
        if hasattr(loss, "loss"):
            l: LossFn[Model, *Data] = type(loss).loss  # type: ignore
            self.loss = l
            init_model: Model = loss  # type: ignore
        else:
            l: LossFn[Model, *Data] = loss  # type: ignore
            self.loss = l
            init_model: Model = model  # type: ignore

        self.model = init_model
        self.optimization = optimizer.init(self.model, wrt)
        self.extra_kwargs = kwargs

    @jax.jit
    def eval_model(self) -> Model:
        # Make a frozen duplicate of the model
        # to avoid modifying the original
        # set the Model to eval mode for convenience
        model = graph_util.duplicate(self.model)
        model.eval_mode()
        return model

    @jax.jit
    def update(self, data: Data) -> Metrics:
        def loss(
            graphdef,
            params,
            other,
            data,
            extra_args,
        ):
            model = graph.merge(graphdef, params, other)
            return self.loss(model, data, **extra_args)  # type: ignore

        graphdef, params, other = graph.split(self.model, self.optimization.wrt, ...)
        params = nn.pure(params)
        grads, metrics = jax.grad(
            loss,
            argnums=1,
            has_aux=True,
        )(graphdef, params, other, data, self.extra_kwargs)
        self.optimization.update(self.model, grads)
        return metrics
