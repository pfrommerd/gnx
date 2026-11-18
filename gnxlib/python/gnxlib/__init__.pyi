from collections.abc import Callable
from typing import Generic, NoReturn, ParamSpec, TypeVar, final

from . import array as array, backend as backend, dtype as dtype, graph as graph
from .backend import (
    Backend as Backend,
    Device as Device,
    backends as backends,
    devices as devices,
    enable_jax_backend as enable_jax_backend,
)
from .graph import Graph as Graph

_P = ParamSpec("_P")
_R = TypeVar("_R", covariant=True)

@final
class Trace:
    def __new__(cls, _unavailable: NoReturn, /) -> Trace: ...
    @property
    def input_count(self) -> int: ...
    @property
    def output_count(self) -> int: ...
    @property
    def capture_count(self) -> int: ...

@final
class Jit(Generic[_P, _R]):
    def __new__(cls, _unavailable: NoReturn, /) -> Jit[_P, _R]: ...
    @property
    def __wrapped__(self) -> Callable[_P, _R]: ...
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...
    def trace(self, *args: _P.args, **kwargs: _P.kwargs) -> Trace: ...

def jit(function: Callable[_P, _R]) -> Jit[_P, _R]: ...
