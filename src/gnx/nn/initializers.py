import math
import typing as tp
import jax
import jax.typing

from .util import has_keyword_arg

type Shape = tp.Sequence[int]
type OutShardingType = jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None


@tp.runtime_checkable
class Initializer(tp.Protocol):
    def __call__(
        self,
        key,
        shape: Shape,
        dtype: jax.typing.DTypeLike,
        out_sharding: OutShardingType = None,
    ) -> tp.Any: ...


# Allows for initializing the bias based on the kernel
@tp.runtime_checkable
class BiasInitializer(tp.Protocol):
    def __call__(
        self,
        key,
        shape: Shape,
        dtype: jax.typing.DTypeLike,
        out_sharding: OutShardingType = None,
        *,
        kernel_shape: Shape,
    ) -> tp.Any: ...


@tp.overload
def is_bias_initializer(fn: BiasInitializer) -> tp.Literal[True]: ...
@tp.overload
def is_bias_initializer(fn: Initializer) -> tp.Literal[False]: ...


def is_bias_initializer(fn: tp.Any) -> bool:
    return has_keyword_arg(fn, "kernel_shape")


def as_bias_initializer(fn: Initializer | BiasInitializer) -> BiasInitializer:
    if is_bias_initializer(fn):
        return fn  # type: ignore
    else:

        def wrapper(
            key,
            shape: Shape,
            dtype: jax.typing.DTypeLike,
            out_sharding: OutShardingType = None,
            *,
            kernel_shape: Shape,
        ) -> tp.Any:
            return fn(key, shape, dtype, out_sharding)  # type: ignore

        return wrapper


kaiming_uniform = jax.nn.initializers.kaiming_uniform
kaiming_normal = jax.nn.initializers.kaiming_normal
lecun_normal = jax.nn.initializers.lecun_normal
lecun_uniform = jax.nn.initializers.lecun_uniform
normal = jax.nn.initializers.normal
uniform = jax.nn.initializers.uniform
truncated_normal = jax.nn.initializers.truncated_normal
zeros = jax.nn.initializers.zeros
ones = jax.nn.initializers.ones
constant = jax.nn.initializers.constant
orthogonal = jax.nn.initializers.orthogonal


def ones_init():
    return ones


def zeros_init():
    return zeros

def pytorch_kernel_init(key, shape, dtype, out_sharding=None):

    in_axes, out_axes = len(shape) - 2, len(shape) - 1
    in_features, out_features = shape[in_axes], shape[out_axes]
    other_features = tuple(shape[i] for i in range(len(shape)) if i not in (in_axes, out_axes))
    other_prod = math.prod(other_features)

    bound = math.sqrt(1 / (in_features * other_prod))
    return jax.random.uniform(key, shape, dtype, -bound, bound)

def pytorch_bias_init(key, shape, dtype, out_sharding=None, *, kernel_shape):
    in_axes, out_axes = len(kernel_shape) - 2, len(kernel_shape) - 1
    in_features, out_features = kernel_shape[in_axes], kernel_shape[out_axes]
    other_features = tuple(kernel_shape[i] for i in range(len(kernel_shape)) if i not in (in_axes, out_axes))
    other_prod = math.prod(other_features)
    bound = math.sqrt(1 / (in_features * other_prod))
    return jax.random.uniform(key, shape, dtype, -bound, bound)

default_kernel_init = pytorch_kernel_init
default_bias_init = pytorch_bias_init
