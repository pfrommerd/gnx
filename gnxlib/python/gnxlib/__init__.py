"""Symbolic arrays, graphs, and JIT tracing for Python."""

from . import _native
from . import array as array, dtype as dtype, graph as graph
from ._native import (
    Capture as Capture,
    Jit as Jit,
    Trace as Trace,
    Transformed as Transformed,
    VjpFunction as VjpFunction,
    cond as cond,
    expr as expr,
    hessian as hessian,
    jit as jit,
    jvp as jvp,
    scan as scan,
    stop_gradient as stop_gradient,
    switch as switch,
    vjp as vjp,
    while_loop as while_loop,
)
from .graph import Graph as Graph


def map(f, in_axes=0, out_axes=0, axis_size=None):
    """Vectorize f over mapped axes of its arguments, like jax.vmap.

    in_axes is an int or None for every argument, or a tuple with one entry
    (a pytree prefix of ints and None) per positional argument. Keyword
    arguments are always mapped over axis 0. out_axes is a pytree prefix of
    ints for the result.
    """
    return _native.map(f, in_axes, out_axes, axis_size)


def grad(f, argnums=0):
    """The gradient of a function returning one floating-point scalar array."""
    return _native.grad(f, argnums)


def value_and_grad(f, argnums=0):
    """Like grad, returning (value, gradient)."""
    return _native.value_and_grad(f, argnums)
