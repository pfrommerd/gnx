"""Wraps optax-based optimizers and learning rate schedules
in a pytree-friendly way such that the optimizer parameters are part of the pytree.
This allows serializing the optimizer andoptimization state as part of a checkpoint.
"""

import optax
import chex
import jax

import jax.numpy as jnp
import typing as tp

from ..core import graph, graph_util, nn
from ..core.dataclasses import dataclass


type LRSchedule = optax.ScalarOrSchedule


class Optimization[Model](tp.Protocol):
    def update(self, model: Model, gradients: tp.Any): ...

    @property
    def wrt(self) -> graph.Filter: ...


class Optimizer(tp.Protocol):
    def init[Model](self, model: Model, wrt: graph.Filter) -> Optimization[Model]: ...


# For EMA-style model tracking


class ModelTrack[Model](tp.Protocol):
    # Will return a *copy* of the model with the tracked parameters replaced
    def update(self, model: Model) -> Model: ...

    # Will return a *copy* of the model with the tracked parameters replaced
    def current(self, model: Model) -> Model: ...

    @property
    def wrt(self) -> graph.Filter: ...


class ModelTracker(tp.Protocol):
    def init[Model](self, model: Model, wrt: graph.Filter) -> ModelTrack[Model]: ...


# Wrappers for optax-based learning schedules


@dataclass
class CosineDecaySchedule:
    init_value: float
    peak_value: float
    warmup_steps: int
    decay_steps: int
    end_value: float = 0.0
    exponent: float = 1.0

    def __call__(self, step: chex.Numeric) -> chex.Numeric:
        return jnp.array(
            optax.warmup_cosine_decay_schedule(
                self.init_value,
                self.peak_value,
                self.warmup_steps,
                self.decay_steps,
                self.end_value,
                self.exponent,
            )(jnp.array(step))
        )


@dataclass
class ExponentialDecaySchedule:
    init_value: float
    peak_value: float
    warmup_steps: int
    decay_interval_steps: int
    decay_rate: float
    decay_begin_steps: int = 0
    decay_stairscase: bool = True
    final_value: float | None = None

    def __call__(self, step: chex.Numeric) -> chex.Numeric:
        return jnp.array(
            optax.warmup_exponential_decay_schedule(
                self.init_value,
                self.peak_value,
                self.warmup_steps,
                self.decay_interval_steps,
                self.decay_rate,
                self.decay_begin_steps,
                self.decay_stairscase,
                self.final_value,
            )(jnp.array(step))
        )


def constant_schedule(value: jax.typing.ArrayLike) -> LRSchedule:
    return jnp.array(value, dtype=jnp.float32)


def warmup_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
    exponent: float = 1.0,
) -> LRSchedule:
    return CosineDecaySchedule(
        init_value, peak_value, warmup_steps, decay_steps, end_value, exponent
    )


def cosine_decay_schedule(
    peak_value: float,
    decay_steps: int,
    end_value: float = 0.0,
    exponent: float = 1.0,
) -> LRSchedule:
    return warmup_cosine_decay_schedule(
        peak_value, peak_value, 0, decay_steps, end_value, exponent
    )


def warmup_exponential_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_interval_steps: int,
    decay_rate: float,
    decay_begin_steps: int = 0,
    decay_staircase: bool = True,
    decay_final_value: float | None = None,
) -> LRSchedule:
    return ExponentialDecaySchedule(
        init_value,
        peak_value,
        warmup_steps,
        decay_interval_steps,
        decay_rate,
        decay_begin_steps,
        decay_staircase,
        decay_final_value,
    )


def exponential_decay_schedule(
    peak_value: float,
    decay_interval_steps: int,
    decay_rate: float,
    decay_begin_steps: int = 0,
    decay_staircase: bool = True,
) -> LRSchedule:
    return warmup_exponential_decay_schedule(
        peak_value,
        peak_value,
        0,
        decay_interval_steps,
        decay_rate,
        decay_begin_steps,
        decay_staircase,
    )


# Wrappers for optax-based optimizers


class OptaxOptimization[Model](graph.Object):
    def __init__(self, optimizer: "OptaxOptimizer", model: Model, wrt: graph.Filter):
        self.opt = optimizer

        optax_opt = optimizer._optax_optimizer()
        model_arrays = nn.variable_arrays(model, wrt)
        self.opt_state = jax.tree.map(
            lambda x: jax.new_ref(x), optax_opt.init(model_arrays)
        )
        self.wrt = wrt

    @jax.jit
    def update(self, model: Model, gradients: graph.GraphLeaves):
        # unpack any array refs or variables into arrays
        gradients = {k: x[...] for k, x in gradients.items()}
        model_refs = nn.variable_refs(model, self.wrt)
        model_arrays = nn.variable_arrays(model, self.wrt)

        optax_opt = self.opt._optax_optimizer()
        opt_state = jax.tree.map(lambda v: v[...], self.opt_state)
        updates, new_opt_state = optax_opt.update(gradients, opt_state, model_arrays)
        new_params = optax.apply_updates(model_arrays, updates)

        def update(v, u):
            v[...] = u

        jax.tree.map(update, self.opt_state, new_opt_state)
        jax.tree.map(update, model_refs, new_params)


class OptaxOptimizer(graph.Object):
    def _optax_optimizer(self) -> optax.GradientTransformation: ...

    def init[Model](self, model: Model, wrt: graph.Filter) -> Optimization[Model]:
        return OptaxOptimization(self, model, wrt)


# Optax-based tracker for model parameters


class OptaxTrack[Model](graph.Object):
    def __init__(self, tracker: "OptaxTracker", model: Model, wrt: graph.Filter):
        self.tracker = tracker

        optax_opt = tracker._optax_optimizer()
        model_arrays = nn.variable_arrays(model, wrt)
        self.track_state = jax.tree.map(
            lambda x: jax.new_ref(x), optax_opt.init(model_arrays)
        )
        self.wrt = wrt

    def update(self, model: Model):

        optax_opt = self.tracker._optax_optimizer()
        track_state = jax.tree.map(lambda v: v[...], self.track_state)
        model_arrays = nn.variable_arrays(model, self.wrt)
        _, new_opt_state = optax_opt.update(model_arrays, track_state)

        def update(v, u):
            v[...] = u

        jax.tree.map(update, self.track_state, new_opt_state)

    def current(self, model: Model) -> Model:
        optax_opt = self.tracker._optax_optimizer()
        track_state = jax.tree.map(lambda v: v[...], self.track_state)

        # extract the tracked parameters from the state
        updates, _ = optax_opt.update(None, track_state)  # type: ignore

        def update(v, u):
            v[...] = u

        model = graph_util.duplicate(model)
        jax.tree.map(update, nn.variable_refs(model, self.wrt), updates)
        return model


class OptaxTracker(ModelTracker):
    def _optax_optimizer(self) -> optax.GradientTransformation: ...

    def init[Model](self, model: Model, wrt: graph.Filter) -> OptaxTrack[Model]:
        return OptaxTrack(self, model, wrt)


# Optax-based adam, adamw, etc


@dataclass
class SGDOptimizer(OptaxOptimizer):
    lr: LRSchedule
    momentum: float | None = None
    nesterov: bool = False

    def _optax_optimizer(self) -> optax.GradientTransformation:
        return optax.sgd(self.lr, self.momentum, nesterov=self.nesterov)


@dataclass
class AdamOptimizer(OptaxOptimizer):
    lr: LRSchedule
    b1: float
    b2: float
    eps: float

    def _optax_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.lr, b1=self.b1, b2=self.b2, eps=self.eps)


@dataclass
class AdamWOptimizer(AdamOptimizer):
    weight_decay: float

    def _optax_optimizer(self) -> optax.GradientTransformation:
        return optax.adamw(
            self.lr,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


def sgd(
    lr: LRSchedule, momentum: float | None = None, nesterov: bool = False
) -> Optimizer:
    return SGDOptimizer(lr, momentum, nesterov)


def adam(
    lr: LRSchedule, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8
) -> Optimizer:
    return AdamOptimizer(lr, b1, b2, eps)


def adamw(
    lr: LRSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> Optimizer:
    return AdamWOptimizer(lr, b1, b2, eps, weight_decay)


@dataclass
class EMATracker(OptaxTracker):
    decay: float
    debias: bool = True
    accumulator_dtype: tp.Any | None = None

    def _optax_optimizer(self) -> optax.GradientTransformation:
        accumulator_dtype = canonicalize_dtype(self.accumulator_dtype)

        def init_fn(params):
            return optax.EmaState(
                count=jnp.zeros([], jnp.int32),
                ema=optax.tree.zeros_like(params, dtype=accumulator_dtype),
            )

        def update_fn(updates, state, params=None):
            del params
            # get the current ema state
            if updates is None:
                return state.ema, state
            updates = new_ema = optax.tree.update_moment(
                updates, state.ema, self.decay, order=1
            )
            count_inc = safe_increment(state.count)
            if self.debias:
                updates = optax.tree.bias_correction(new_ema, self.decay, count_inc)
            state_ema = optax.tree.cast(new_ema, accumulator_dtype)
            return updates, optax.EmaState(count=count_inc, ema=state_ema)

        return optax.GradientTransformation(init_fn, update_fn)


@dataclass
class AnnealedEMATracker(OptaxTracker):
    decay: float
    accumulator_dtype: tp.Any | None = None

    def _optax_optimizer(self) -> optax.GradientTransformation:
        accumulator_dtype = canonicalize_dtype(self.accumulator_dtype)

        def init_fn(params):
            return optax.EmaState(
                count=jnp.zeros([], jnp.int32),
                ema=jax.tree.map(
                    lambda x: jnp.astype(jnp.copy(x), accumulator_dtype), params
                ),
            )

        def update_fn(updates, state, params=None):
            del params
            # get the current ema state
            if updates is None:
                return state.ema, state
            # if state.count == 0, set the EMA to the current values,
            # otherwise perform an ema update
            count_inc = safe_increment(state.count)
            # compute the effective decay rate after increasing the count
            effective_decay = jnp.clip(
                self.decay, 0.0, (1.0 + count_inc) / (10.0 + count_inc)
            )
            updates = new_ema = optax.tree.update_moment(
                updates, state.ema, effective_decay, order=1
            )
            state_ema = optax.tree.cast(new_ema, accumulator_dtype)
            return updates, optax.EmaState(count=count_inc, ema=state_ema)

        return optax.GradientTransformation(init_fn, update_fn)


def ema_tracker(
    decay: float, debias: bool = True, accumulator_dtype: tp.Any | None = None
) -> ModelTracker:
    return EMATracker(decay, debias, accumulator_dtype)


def annealed_ema_tracker(
    decay: float, accumulator_dtype: tp.Any | None = None
) -> ModelTracker:
    return AnnealedEMATracker(decay, accumulator_dtype)


# Generic utilities...


def safe_increment(count: jax.Array) -> jax.Array:
    count_dtype = jnp.asarray(count).dtype
    if jnp.issubdtype(count_dtype, jnp.integer):
        max_value = jnp.iinfo(count_dtype).max
    elif jnp.issubdtype(count_dtype, jnp.floating):
        max_value = jnp.finfo(count_dtype).max
    else:
        raise ValueError(
            f"Cannot safely increment count with dtype {count_dtype},"
            ' valid dtypes are subdtypes of "jnp.integer" or "jnp.floating".'
        )
    max_value = jnp.array(max_value, count_dtype)
    one = jnp.array(1, count_dtype)
    return jnp.where(count < max_value, count + one, max_value)


def canonicalize_dtype(dtype: tp.Any | None):
    if dtype is not None:
        return jax.dtypes.canonicalize_dtype(dtype)
    return None
