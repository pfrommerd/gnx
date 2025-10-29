import typing as tp

import jax
import jax.numpy as jnp

from . import initializers, util

from .core import Module, RngStream, Rngs, Variable, Param
from .util import Axes, NamedAxes, DTypeLike


# def _split_axes(axes: Axes) -> tuple[str | None, Axes]:
#     if isinstance(axes, str):
#         return axes, ()
#     elif isinstance(axes, int):
#         return None, (axes,)
#     else:
#         named_axis = None
#         other_axes = []
#         for a in axes:
#             if isinstance(a, int):
#                 other_axes.append(a)
#             elif isinstance(a, str):
#                 assert named_axis is None, "only one named axis supported!"
#                 named_axis = a
#         return named_axis, other_axes


def _compute_stats(
    x: jax.Array,
    axes: NamedAxes,
    dtype: tp.Optional[DTypeLike],
    axis_index_groups: tp.Any = None,
    use_mean: bool = True,
    use_fast_variance: bool = True,
    mask: tp.Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array]:
    axis_name, axes = util.partition_axes(axes)
    if dtype is None:
        dtype = jnp.result_type(x)
    # promote x to at least float32, this avoids half precision computation
    # but preserves double or complex floating points
    dtype = jnp.promote_types(dtype, jnp.float32)
    x = jnp.asarray(x, dtype)
    axes = util.canonicalize_axes(x.ndim, axes)

    def maybe_distributed_mean(*xs, mask=None):
        mus = tuple(x.mean(axes, where=mask) for x in xs)
        if axis_name is None:
            return mus if len(xs) > 1 else mus[0]
        else:
            # In the distributed case we stack multiple arrays to speed comms.
            if len(xs) > 1:
                reduced_mus = jax.lax.pmean(
                    jnp.stack(mus, axis=0),
                    axis_name,
                    axis_index_groups=axis_index_groups,
                )
                return tuple(reduced_mus[i] for i in range(len(xs)))
            else:
                return jax.lax.pmean(
                    mus[0], axis_name, axis_index_groups=axis_index_groups
                )

    if use_mean:
        if use_fast_variance:
            mu, mu2 = maybe_distributed_mean(x, util.abs_sq(x), mask=mask)
            # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
            # to floating point round-off errors.
            var = jnp.maximum(0.0, mu2 - util.abs_sq(mu))
        else:
            mu = maybe_distributed_mean(x, mask=mask)
            var = maybe_distributed_mean(
                util.abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask  # type: ignore
            )
    else:
        var = maybe_distributed_mean(util.abs_sq(x), mask=mask)
        mu = jnp.zeros_like(var)  # type: ignore
    return mu, var  # type: ignore


def _normalize(
    x: jax.Array,
    mean: jax.Array,
    var: jax.Array,
    scale: tp.Optional[jax.Array],
    bias: tp.Optional[jax.Array],
    reduction_axes: NamedAxes,
    feature_axes: Axes,
    dtype: tp.Optional[DTypeLike],
    epsilon: float,
):
    _, reduction_axes = util.partition_axes(reduction_axes)
    reduction_axes = util.canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = util.canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
    y = x - mean
    mul = jax.lax.rsqrt(var + epsilon)
    args = [x]
    if scale is not None:
        scale = scale.reshape(feature_shape)
        mul *= scale
        args.append(scale)
    y *= mul
    if bias is not None:
        bias = bias.reshape(feature_shape)
        y += bias
        args.append(bias)
    dtype = util.canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, dtype)


class BatchStat(Variable):
    pass


class BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        *,
        spatial_dims: int,
        use_running_average: bool = False,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: initializers.Initializer = initializers.zeros,
        scale_init: initializers.Initializer = initializers.ones,
        use_fast_variance: bool = True,
        rngs: Rngs,
    ):
        feature_shape = (num_features,)
        self.mean = BatchStat(jnp.zeros(feature_shape, jnp.float32))
        self.var = BatchStat(jnp.ones(feature_shape, jnp.float32))

        if use_scale:
            key = rngs.params()
            self.scale = Param(scale_init(key, feature_shape, param_dtype))
        else:
            self.scale = None

        if use_bias:
            key = rngs.params()
            self.bias = Param(bias_init(key, feature_shape, param_dtype))
        else:
            self.bias = None

        self.num_features = num_features
        self.spatial_dims = spatial_dims
        self.use_running_average = use_running_average
        self.momentum = momentum
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.use_fast_variance = use_fast_variance

    def _set_training(self, train):
        self.use_running_average = not train

    def __call__(
        self,
        x,
        *,
        use_running_average: bool | None = None,
        batch_axis: str | int | None = None,
        mask: jax.Array | None = None,
    ):
        use_running_average = (
            use_running_average
            if use_running_average is not None
            else self.use_running_average
        )
        if batch_axis is None and x.ndim == self.spatial_dims + 2:
            batch_axis = 0

        assert batch_axis is not None, "batch_axis must be specified."
        assert use_running_average is not None, "use_running_average must be specified."
        assert (x.ndim == self.spatial_dims + 2) or (
            isinstance(batch_axis, str) and x.ndim == self.spatial_dims + 1
        ), f"got fewer dimensions than expected, expected {self.spatial_dims + 2}, got {x.ndim}"

        reduction_axes = tuple(range(x.ndim - 1 - self.spatial_dims, x.ndim - 1))
        reduction_axes = (batch_axis,) + reduction_axes

        if use_running_average:
            mean, var = self.mean[...], self.var[...]
        else:
            mean, var = _compute_stats(
                x,
                axes=reduction_axes,
                dtype=self.dtype,
                use_fast_variance=self.use_fast_variance,
                mask=mask,
            )
            self.mean[...] = self.momentum * self.mean[...] + (1 - self.momentum) * mean
            self.var[...] = self.momentum * self.var[...] + (1 - self.momentum) * var

        return _normalize(
            x,
            mean,
            var,
            self.scale[...] if self.scale else None,
            self.bias[...] if self.bias else None,
            reduction_axes,
            (x.ndim - 1,),
            self.dtype,
            self.epsilon,
        )


class GroupNorm(Module):
    def __init__(
        self,
        num_features: int,
        spatial_dims: int,
        num_groups: tp.Optional[int] = 32,
        group_size: tp.Optional[int] = None,
        *,
        epsilon: float = 1e-6,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: initializers.Initializer = initializers.zeros,
        scale_init: initializers.Initializer = initializers.ones,
        use_fast_variance: bool = True,
        rngs: Rngs,
    ):
        self.spatial_dims = spatial_dims
        if (num_groups is None and group_size is None) or (
            num_groups is not None and group_size is not None
        ):
            raise ValueError(
                "Either `num_groups` or `group_size` should be "
                "specified. If `group_size` is to be specified, "
                "pass `num_groups=None` as argument to override "
                "the default `num_groups` value of 32."
            )

        if group_size is not None:
            if num_features % group_size != 0:
                raise ValueError(
                    "Number of features ({}) is not multiple of the "
                    "group size ({}).".format(num_features, group_size)
                )
            self.num_groups = num_features // group_size
            self.group_size = group_size
        else:
            if (
                not isinstance(num_groups, int)
                or num_groups <= 0
                or (num_features % num_groups != 0)
            ):
                raise ValueError(
                    "Number of groups ({}) does not divide the number"
                    " of channels ({}).".format(num_groups, num_features)
                )
            self.num_groups = num_groups
            self.group_size = num_features // num_groups

        if use_scale:
            self.scale = Param(scale_init(rngs.params(), (num_features,), param_dtype))
        else:
            self.scale = None

        if use_bias:
            self.bias = Param(bias_init(rngs.params(), (num_features,), param_dtype))
        else:
            self.bias = None

        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.use_fast_variance = use_fast_variance

    def __call__(self, x, *, mask: tp.Optional[jax.Array] = None):
        assert x.ndim >= self.spatial_dims + 1, f"Expected at least {self.spatial_dims + 1} dimensions"
        group_shape = x.shape[:-1] + (self.num_groups, self.group_size)
        if mask is not None:
            mask = mask.reshape(group_shape)
        grouped = x.reshape(group_shape)
        # reduce over everything but the second to last axis
        start_axis = grouped.ndim - 2 - self.spatial_dims
        mean, var = _compute_stats(
            x.reshape(group_shape),
            axes=list(range(start_axis, grouped.ndim - 2)) + [grouped.ndim - 1],
            dtype=self.dtype,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )

        mean = jnp.repeat(mean, self.group_size, axis=-1)
        var = jnp.repeat(var, self.group_size, axis=-1)

        scale = self.scale[...] if self.scale is not None else None
        bias = self.bias[...] if self.bias is not None else None

        x = _normalize(
            x, mean, var, scale, bias,
            # we treat the stats as if they were computed per feature
            # rather than over the groups
            list(range(x.ndim - 1 - self.spatial_dims, x.ndim - 1)),
            (x.ndim - 1,),
            self.dtype, self.epsilon,
        ) # fmt: skip
        return x


class LayerNorm(GroupNorm):
    def __init__(
        self,
        num_features: int,
        spatial_dims: int,
        num_groups: tp.Optional[int] = 32,
        group_size: tp.Optional[int] = None,
        *,
        epsilon: float = 1e-6,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: initializers.Initializer = initializers.zeros,
        scale_init: initializers.Initializer = initializers.ones,
        use_fast_variance: bool = True,
        rngs: Rngs,
    ):
        super().__init__(
            num_features=num_features,
            spatial_dims=spatial_dims,
            num_groups=1,
            epsilon=epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=use_bias,
            use_scale=use_scale,
            bias_init=bias_init,
            scale_init=scale_init,
            use_fast_variance=use_fast_variance,
            rngs=rngs,
        )


class Dropout(Module):
    def __init__(
        self,
        rate: float,
        *,
        broadcast_dims: tp.Sequence[int] = (),
        deterministic: bool = False,
        rngs: Rngs | RngStream | None = None,
    ):
        self.rate = rate
        self.broadcast_dims = broadcast_dims
        self.deterministic = deterministic

        if isinstance(rngs, Rngs):
            self.rngs = rngs.dropout.fork()
        elif isinstance(rngs, RngStream):
            self.rngs = rngs.fork()
        elif rngs is None:
            self.rngs = None
        else:
            raise TypeError(
                f"rngs must be a Rngs, RngStream or None, but got {type(rngs)}."
            )

    def _set_training(self, train):
        self.deterministic = not train

    def __call__(
        self,
        inputs: jax.Array,
        *,
        deterministic: bool | None = None,
        rngs: Rngs | RngStream | jax.Array | None = None,
    ) -> jax.Array:
        """Applies a random dropout mask to the input.

        Args:
          inputs: the inputs that should be randomly masked.
          deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
            masked, whereas if true, no mask is applied and the inputs are returned
            as is. The ``deterministic`` flag passed into the call method will take
            precedence over the ``deterministic`` flag passed into the constructor.
          rngs: an optional key, RngStream, or Rngs object used to generate the dropout mask.
            If given it will take precedence over the rngs passed into the constructor.

        Returns:
          The masked inputs reweighted to preserve mean.
        """
        deterministic = self.deterministic if deterministic is None else deterministic
        assert (
            deterministic is not None
        ), """No `deterministic` argument was provided to Dropout"""

        if (self.rate == 0.0) or deterministic:
            return inputs

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.rate == 1.0:
            return jnp.zeros_like(inputs)

        rngs = rngs if rngs is not None else self.rngs
        assert rngs is not None, """No `rngs` argument was provided to Dropout"""

        if isinstance(rngs, Rngs):
            key = rngs.dropout()
        elif isinstance(rngs, RngStream):
            key = rngs()
        elif isinstance(rngs, jax.Array):
            key = rngs
        else:
            raise TypeError(
                f"rngs must be a Rngs, RngStream or jax.Array, but got {type(rngs)}."
            )

        keep_prob = 1.0 - self.rate
        broadcast_shape = list(inputs.shape)
        for dim in self.broadcast_dims:
            broadcast_shape[dim] = 1
        mask = jax.random.bernoulli(key, p=keep_prob, shape=broadcast_shape)
        mask = jnp.broadcast_to(mask, inputs.shape)
        return jax.lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))
