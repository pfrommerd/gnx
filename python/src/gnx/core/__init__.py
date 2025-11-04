from .config import USE_JAX

import typing as tp


@tp.overload
def jit[F](func: F) -> F: ...
@tp.overload
def jit[F](func: None = None) -> tp.Callable[[F], F]: ...


if USE_JAX:
    import flax.nnx as nnx
    import functools

    def jit[F](func: F | None = None) -> F | tp.Callable[[F], F]:
        if func is None:
            return functools.partial(jit)
        return nnx.jit(func)

    def cond(pred, true_fun, false_fun, *operands):
        nnx.cond(pred, true_fun, false_fun, *operands)
