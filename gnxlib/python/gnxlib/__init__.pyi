from collections.abc import Callable, Sequence
from typing import Any, Generic, NoReturn, ParamSpec, TypeAlias, TypeVar, final

from . import array as array, dtype as dtype, graph as graph, jax as jax
from .graph import Graph as Graph

_P = ParamSpec("_P")
_R = TypeVar("_R", covariant=True)
_T = TypeVar("_T")
_C = TypeVar("_C")
_X = TypeVar("_X")
_Y = TypeVar("_Y")
_Binding = array.Array | array.ArrayRef
_Axes: TypeAlias = int | None | tuple[Any, ...] | list[Any] | dict[Any, Any]
_Argnums: TypeAlias = int | tuple[int, ...]

@final
class Capture:
    def __new__(cls, _unavailable: NoReturn, /) -> Capture: ...
    @property
    def closure(self) -> tuple[_Binding, ...]: ...
    @property
    def inputs(self) -> tuple[_Binding, ...]: ...
    @property
    def outputs(self) -> tuple[_Binding, ...]: ...

def expr(roots: object) -> Capture: ...

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

@final
class Transformed(Generic[_P, _R]):
    """A transformed function; calls are traced and cached like jit."""

    def __new__(cls, _unavailable: NoReturn, /) -> Transformed[_P, _R]: ...
    @property
    def __wrapped__(self) -> Callable[..., Any]: ...
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...
    def trace(self, *args: _P.args, **kwargs: _P.kwargs) -> Trace: ...

def map(
    f: Callable[_P, _R],
    in_axes: _Axes = 0,
    out_axes: _Axes = 0,
    axis_size: int | None = None,
) -> Transformed[_P, _R]: ...
def grad(f: Callable[_P, Any], argnums: _Argnums = 0) -> Transformed[_P, Any]: ...
def value_and_grad(
    f: Callable[_P, _R], argnums: _Argnums = 0
) -> Transformed[_P, tuple[_R, Any]]: ...
def hessian(f: Callable[_P, Any], argnums: int = 0) -> Transformed[_P, array.Array]: ...
def jvp(
    f: Callable[..., _T], primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> tuple[_T, _T]: ...

@final
class VjpFunction(Generic[_T]):
    """The pullback returned by vjp: maps an output cotangent to one
    cotangent per primal."""

    def __new__(cls, _unavailable: NoReturn, /) -> VjpFunction[_T]: ...
    def __call__(self, cotangent: _T, /) -> tuple[Any, ...]: ...

def vjp(f: Callable[..., _T], *primals: Any) -> tuple[_T, VjpFunction[_T]]: ...
def cond(
    pred: array.Array | bool,
    true_fun: Callable[..., _T],
    false_fun: Callable[..., _T],
    *operands: Any,
) -> _T: ...
def switch(
    index: array.Array | int, branches: Sequence[Callable[..., _T]], *operands: Any
) -> _T: ...
def while_loop(
    cond_fun: Callable[[_C], array.Array], body_fun: Callable[[_C], _C], init_val: _C
) -> _C: ...
def scan(
    f: Callable[[_C, _X], tuple[_C, _Y]],
    init: _C,
    xs: _X = ...,
    length: int | None = None,
) -> tuple[_C, _Y]: ...
