import jax
import inspect
import typing as tp

import numpy as np
import jax.numpy as jnp

from .. import graph_util, asserts
from .core import Module


class Sequential(Module):
    def __init__(self, *layers: tp.Any):
        self.layers = list(layers)

    def __getitem__(self, idx: int) -> tp.Any:
        return self.layers[idx]

    def __call__(self, x, *args):
        for layer in self.layers:
            x, args = (layer(x, *args), ())
        return x

    def __iter__(self):
        return iter(self.layers)


class Flatten(Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        flat, _ = graph_util.ravel(x)
        return flat


type Shape = tp.Sequence[int]
type Axes = int | tp.Sequence[int]
type NamedAxes = str | int | tp.Sequence[str | int]

type PrecisionLike = jax.lax.PrecisionLike
type PaddingLike = str | int | tp.Sequence[int] | tp.Sequence[tuple[int, int]]
type DTypeLike = jax.typing.DTypeLike


def has_keyword_arg(func, name):
    return any(
        param.name == name
        and param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
        for param in inspect.signature(func).parameters.values()
    )


def canonicalize_dtype(
    *args, dtype: DTypeLike | None = None, inexact: bool = True
) -> DTypeLike:
    if dtype is None:
        args_filtered = [jnp.asarray(x) for x in args if x is not None]
        dtype = jnp.result_type(*args_filtered)
        if inexact and not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.promote_types(jnp.float32, dtype)
    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
        raise ValueError(f"Dtype must be inexact: {dtype}")
    return dtype


def canonicalize_padding(
    padding: PaddingLike, rank: int
) -> tp.Sequence[tuple[int, int]] | str:
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)] * rank
    if isinstance(padding, tp.Sequence) and len(padding) == rank:
        new_pad = []
        for p in padding:
            if isinstance(p, int):
                new_pad.append((p, p))
            elif isinstance(p, tuple) and len(p) == 2:
                new_pad.append(p)
            else:
                break
        if len(new_pad) == rank:
            return new_pad
    raise ValueError(
        f"Invalid padding format: {padding}, should be str, int,"
        f" or a sequence of len {rank} where each element is an"
        " int or pair of ints."
    )


def canonicalize_axes(rank: int, axes: int | tp.Iterable[int]) -> tuple[int, ...]:
    if not isinstance(axes, tp.Iterable):
        axes = (axes,)
    return tuple({rank + axis if axis < 0 else axis for axis in axes})


def partition_axes(named_axes: NamedAxes) -> tuple[str | None, Axes]:
    if isinstance(named_axes, (str, int)):
        return (named_axes, ()) if isinstance(named_axes, str) else (None, named_axes)
    else:
        axes = tuple(a for a in named_axes if isinstance(a, int))
        named_axis = next((a for a in named_axes if isinstance(a, str)), None)
        return named_axis, axes


class PromoteDTypeFn(tp.Protocol):
    def __call__[T](
        self, args: T, dtype: DTypeLike | None = None, inexact: bool = True
    ) -> T: ...


def promote_dtype[T](args: T, dtype=None, inexact=True) -> T:
    dtype = canonicalize_dtype(*args, dtype=dtype, inexact=inexact)  # type: ignore
    return [jnp.asarray(x, dtype) if x is not None else None for x in args]  # type: ignore


def pool(inputs, init, reduce_fn, window_shape, strides, padding):
    num_batch_dims = inputs.ndim - (len(window_shape) + 1)
    strides = strides or (1,) * len(window_shape)
    assert len(window_shape) == len(
        strides
    ), f"len({window_shape}) must equal len({strides})"
    strides = (1,) * num_batch_dims + strides + (1,)
    dims = (1,) * num_batch_dims + window_shape + (1,)

    is_single_input = False
    if num_batch_dims == 0:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        inputs = inputs[None]
        strides = (1,) + strides
        dims = (1,) + dims
        is_single_input = True

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    if not isinstance(padding, str):
        padding = tuple(map(tuple, padding))
        assert len(padding) == len(window_shape), (
            f"padding {padding} must specify pads for same number of dims as "
            f"window_shape {window_shape}"
        )
        assert all(
            [len(x) == 2 for x in padding]
        ), f"each entry in padding {padding} must be length 2"
        padding = ((0, 0),) + padding + ((0, 0),)
    y = jax.lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


def avg_pool(
    inputs, window_shape, strides=None, padding="VALID", count_include_pad=True
):
    """Pools the input by taking the average over a window.

    Args:
      inputs: input data with dimensions (batch, window dims..., features).
      window_shape: a shape tuple defining the window to reduce over.
      strides: a sequence of ``n`` integers, representing the inter-window
        strides (default: ``(1, ..., 1)``).
      padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
        of ``n`` ``(low, high)`` integer pairs that give the padding to apply before
        and after each spatial dimension (default: ``'VALID'``).
      count_include_pad: a boolean whether to include padded tokens
        in the average calculation (default: ``True``).
    Returns:
      The average for each window slice.
    """
    y = pool(inputs, 0.0, jax.lax.add, window_shape, strides, padding)
    if count_include_pad:
        y = y / np.prod(window_shape)
    else:
        div_shape = inputs.shape[:-1] + (1,)
        if len(div_shape) - 2 == len(window_shape):
            div_shape = (1,) + div_shape[1:]
        y = y / pool(
            jnp.ones(div_shape), 0.0, jax.lax.add, window_shape, strides, padding
        )
    return y


def max_pool(inputs, window_shape, strides=None, padding="VALID"):
    """Pools the input by taking the maximum of a window slice.

    Args:
      inputs: input data with dimensions (batch, window dims..., features).
      window_shape: a shape tuple defining the window to reduce over.
      strides: a sequence of ``n`` integers, representing the inter-window
        strides (default: ``(1, ..., 1)``).
      padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
        of ``n`` ``(low, high)`` integer pairs that give the padding to apply before
        and after each spatial dimension (default: ``'VALID'``).
    Returns:
      The maximum for each window slice.
    """
    y = pool(inputs, -jnp.inf, jax.lax.max, window_shape, strides, padding)
    return y


def min_pool(inputs, window_shape, strides=None, padding="VALID"):
    """Pools the input by taking the minimum of a window slice.
    Returns:
      The minimum for each window slice.
    """
    return pool(inputs, jnp.inf, jax.lax.min, window_shape, strides, padding)


def pmean(
    *xs,
    axes: Axes | None,
    axis_name: str | None = None,
    keepdims: bool = False,
    mask=None,
) -> tp.Any:
    if axes is not None:
        xs = tuple(
            jnp.mean(
                x, axis=canonicalize_axes(x.ndim, axes), keepdims=keepdims, where=mask
            )
            for x in xs
        )
    assert len(xs) > 0
    if axis_name is None:
        return xs if len(xs) > 1 else xs[0]
    if len(xs) > 1:
        xs = jnp.stack(xs)
        xs = jax.lax.pmean(xs, axis_name)
        xs = tuple(xs[i] for i in range(xs.shape[0]))
        return xs
    else:
        return jax.lax.pmean(xs[0], axis_name)


def abs_sq(x):
    if jnp.iscomplexobj(x):
        return jax.lax.square(jax.lax.real(x)) + jax.lax.square(jax.lax.imag(x))
    else:
        return jax.lax.square(x)


def maybe_broadcast(x: int | tp.Sequence[int] | None, dims: int) -> tuple[int, ...]:
    if x is None:
        x = 1
    if isinstance(x, int):
        return (x,) * dims
    return tuple(x)
