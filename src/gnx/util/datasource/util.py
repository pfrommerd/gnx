import typing as tp
import functools
import math
import json
import jax
import jax.numpy as jnp

from builtins import zip as _zip

from .common import DataIterator, DataSource
from .transform import DataTransform

from ...core import nn, graph_util, asserts


class GeneratorDataIterator[T, *Ts](DataIterator[T]):
    def __init__(
        self,
        key: jax.Array,
        generator: tp.Callable[[jax.Array, tuple[int, ...]], T],
        batch_shape: tuple[int, ...] = (),
    ):
        self.key = nn.RngKey(key, tag="sampler")
        self.count = nn.RngCount(
            jnp.zeros((), dtype=jnp.uint32),
            tag="sampler",
        )
        self.generator = generator
        self.batch_shape = batch_shape

    @property
    def instance(self) -> T:
        return self.generator(jax.random.key(0), self.batch_shape)

    def has_next(self) -> bool:
        return True

    @jax.jit
    def next(self) -> T:
        key = jax.random.fold_in(self.key[...], self.count[...])
        sample = self.generator(key, self.batch_shape)
        self.count[...] += 1
        return sample

    def skip(self, num: jax.typing.ArrayLike, /):
        self.count[...] += num

    def reset(self, key: jax.Array | None = None):
        self.count[...] = jnp.zeros(self.count.shape, self.count.dtype)
        if key is not None:
            self.key[...] = key


class GeneratorDataSource[T](DataSource[T]):
    def __init__(
        self,
        generator: tp.Callable[[jax.Array, tuple[int, ...]], T],
        batch_shape: tuple[int, ...] = (),
    ):
        self.generator = generator
        self.batch_shape = batch_shape

    @property
    def instance(self) -> T:
        return self.sample(jax.random.key(0), self.batch_shape)

    def batch(self, shape: tuple[int, ...]) -> DataSource[T]:
        return GeneratorDataSource(
            self.generator,
            shape + self.batch_shape,
        )

    def sampler(self, key: jax.Array | None = None) -> DataIterator[T]:
        key = jax.random.key(42) if key is None else key
        return GeneratorDataIterator(key, self.generator, self.batch_shape)

    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        if hasattr(self.generator, "sample"):
            generator: tp.Callable[[jax.Array, tuple[int, ...]], T] = self.generator.sample  # type: ignore
        else:
            generator: tp.Callable[[jax.Array, tuple[int, ...]], T] = self.generator  # type: ignore
        return generator(key, shape + self.batch_shape)


def generator[T](
    generator: tp.Callable[[jax.Array, tuple[int, ...]], T],
) -> GeneratorDataSource[T]:
    """Returns a data source that generates samples using the given generator."""
    return GeneratorDataSource(generator)


class EmptyIterator(DataIterator[tuple[()]]):
    @property
    def instance(self) -> tuple[()]:
        return ()

    def has_next(self) -> bool | jax.Array:
        return True

    def next(self) -> tuple:
        return ()

    def skip(self, num: jax.typing.ArrayLike):
        pass

    @tp.override
    def reset(self, key: jax.Array | None = None):
        pass


class EmptyDataSource(DataSource[tuple[()]]):
    @property
    def instance(self) -> tuple[()]:
        return ()

    def batch(self, shape: tuple[int, ...]) -> "EmptyDataSource":
        return EmptyDataSource()

    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> tuple[()]:
        return ()

    def sampler(self, key: jax.Array | None = None):
        return EmptyIterator()


def empty() -> DataSource[tuple[()]]:
    """Returns an empty data source."""
    return EmptyDataSource()


class NoneIterator(DataIterator[None]):
    @property
    def instance(self) -> None:
        return None

    def has_next(self):
        return True

    def next(self) -> None:
        return None

    def skip(self, num: jax.typing.ArrayLike):
        pass

    @tp.override
    def reset(self, key: jax.Array | None = None):
        pass


class NoneDataSource(DataSource[None]):
    @property
    def instance(self) -> None:
        return None

    def batch(self, shape: tuple[int, ...]) -> "NoneDataSource":
        return NoneDataSource()

    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> None:
        return None

    def sampler(self, key: jax.Array | None = None):
        return NoneIterator()


def none() -> DataSource[None]:
    """Returns an empty data source."""
    return NoneDataSource()


class RngIterator(DataIterator[jax.Array]):
    def __init__(self, key: jax.Array, shape: tuple[int, ...] = ()):
        self.key = nn.RngKey(key, tag="iterator")
        self.counter = nn.RngCount(
            jnp.zeros((), dtype=jax.numpy.uint32), tag="iterator"
        )
        self.batch_shape = shape

    @property
    def instance(self) -> jax.Array:
        keys = jax.random.split(jax.random.key(42), math.prod(self.batch_shape))
        return jnp.reshape(keys, self.batch_shape)

    def has_next(self) -> jax.Array | bool:
        return True

    @jax.jit
    def next(self) -> jax.Array:
        if self.batch_shape:
            N = math.prod(self.batch_shape)
            key, count = self.key[...], self.counter[...]
            keys = jax.vmap(lambda x: jax.random.fold_in(key, count + x))(jnp.arange(N))
            self.counter[...] = self.counter[...] + N
            return keys.reshape(self.batch_shape)
        else:
            key = jax.random.fold_in(self.key[...], self.counter[...])
            self.counter[...] = self.counter[...] + 1
            return key

    @jax.jit
    def skip(self, num: jax.typing.ArrayLike, /):
        N = math.prod(self.batch_shape)
        self.counter[...] = self.counter[...] + num * N

    @jax.jit
    def reset(self, key: jax.Array | None = None):
        self.counter[...] = jnp.zeros_like(self.counter[...])
        if key is not None:
            self.key[...] = key


class RngSource(DataSource[jax.Array]):
    def __init__(self, batch_shape: tuple[int, ...] = ()):
        self.batch_shape = batch_shape

    @property
    def instance(self) -> jax.Array:
        keys = jax.random.split(jax.random.key(0), math.prod(self.batch_shape))
        return jnp.reshape(keys, self.batch_shape)

    def batch(self, shape: tuple[int, ...]) -> "RngSource":
        return RngSource(shape + self.batch_shape)

    @functools.partial(jax.jit, static_argnames=("shape",))
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        N = math.prod(shape + self.batch_shape)
        keys = jax.random.split(key, N)
        return keys.reshape(shape + self.batch_shape)

    def sampler(self, key: jax.Array | None = None) -> DataIterator[jax.Array]:
        # use 42 as the starting seed if no initialization seed is provided
        key = jax.random.key(42) if key is None else key
        return RngIterator(key, self.batch_shape)


def rng() -> DataSource[jax.Array]:
    """Returns a data source that generates random keys."""
    return RngSource()


class ZipDataIterator[T](DataIterator[T]):
    def __init__(self, *iterators):
        self.iterators = iterators

    @property
    def instance(self) -> T:
        return tuple(it.instance for it in self.iterators)  # type: ignore

    @jax.jit
    def has_next(self) -> jax.Array:
        return jnp.all(jnp.array([it.has_next() for it in self.iterators]))

    @jax.jit
    def next(self) -> T:
        res = tuple(it.next() for it in self.iterators)
        return res  # type: ignore

    @jax.jit
    def skip(self, num: jax.typing.ArrayLike, /):
        for it in self.iterators:
            it.skip(num)

    @jax.jit
    def reset(self, key: jax.Array | None = None):
        if key is not None:
            for i, it in enumerate(self.iterators):
                it.reset(jax.random.fold_in(key, i))
        else:
            for it in self.iterators:
                it.reset()


class ZipDataSource[T](DataSource[T]):
    def __init__(self, *sources):
        self.sources = sources

    @property
    def instance(self) -> T:
        return tuple(s.instance for s in self.sources)  # type: ignore

    def batch(self, shape: tuple[int, ...]) -> "ZipDataSource[T]":
        return ZipDataSource(*(s.batch(shape) for s in self.sources))

    def sampler(self, key: jax.Array) -> DataIterator[T]:
        iterators = [
            source.sampler(jax.random.fold_in(key, i))
            for i, source in enumerate(self.sources)
        ]
        return ZipDataIterator(*iterators)

    @functools.partial(jax.jit, static_argnames=("shape",))
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        return tuple(
            source.sample(jax.random.fold_in(key, i), shape)
            for i, source in enumerate(self.sources)
        )  # type: ignore

    def __len__(self) -> int:
        def try_len(s):
            try:
                return len(s)
            except (TypeError, NotImplementedError):
                return None

        lens = tuple(len(s) for s in self.sources if try_len(s) is not None)
        if not lens:
            raise TypeError("No valid lengths found in sources")
        return min(lens)


@tp.overload
def zip() -> DataSource[tuple[()]]: ...
@tp.overload
def zip[A](a: DataSource[A], /) -> DataSource[tuple[A]]: ...
@tp.overload
def zip[A, B](a: DataSource[A], b: DataSource[B], /) -> DataSource[tuple[A, B]]: ...
@tp.overload
def zip[A, B, C](
    a: DataSource[A], b: DataSource[B], c: DataSource[C], /
) -> DataSource[tuple[A, B, C]]: ...
@tp.overload
def zip[A, B, C, D](
    a: DataSource[A], b: DataSource[B], c: DataSource[C], d: DataSource[D], /
) -> DataSource[tuple[A, B, C, D]]: ...


def zip(*sources) -> DataSource:
    return ZipDataSource(*sources)


class JoinDataIterator[T](DataIterator[T]):
    def __init__(self, *iterators):
        self.iterators = iterators

    @property
    def instance(self) -> T:
        return tuple(itertools.chain(it.instance for it in self.iterators))  # type: ignore

    def has_next(self) -> jax.Array:
        return jnp.all(jnp.array([it.has_next() for it in self.iterators]))

    @jax.jit
    def next(self) -> T:
        tuples = tuple(it.next() for it in self.iterators)
        for t in tuples:
            assert isinstance(t, (tuple, list)), (
                "Joined iterator does not return a tuple."
                "As a safety check, all joined DataSources must return tuples or lists"
            )
        return tuple(itertools.chain(s for t in tuples for s in t))  # type: ignore

    @jax.jit
    def skip(self, num: jax.typing.ArrayLike, /):
        for it in self.iterators:
            it.skip(num)

    @jax.jit
    def reset(self, key: jax.Array | None = None):
        if key is not None:
            for i, it in enumerate(self.iterators):
                it.reset(jax.random.fold_in(key, i))
        else:
            for it in self.iterators:
                it.reset()


class JoinDataSource[T](DataSource[T]):
    def __init__(self, *sources):
        self.sources = sources

    @property
    def instance(self) -> T:
        return tuple(itertools.chain(s.instance for s in self.sources))  # type: ignore

    def batch(self, shape: tuple[int, ...]) -> "JoinDataSource[T]":
        return JoinDataSource(*(s.batch(shape) for s in self.sources))

    def sampler(self, key: jax.Array | None = None) -> DataIterator[T]:
        if key is not None:
            keys = jax.random.split(key, len(self.sources))
            iterators = [
                source.sampler(key) for key, source in _zip(keys, self.sources)
            ]
        else:
            iterators = [source.sampler() for source in self.sources]
        return JoinDataIterator(*iterators)

    @functools.partial(jax.jit, static_argnames=("shape",))
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        keys = jax.random.split(key, len(self.sources))
        return tuple(
            s
            for key, source in _zip(keys, self.sources)
            for s in source.sample(key, shape)
        )  # type: ignore

    def __len__(self) -> int:
        def try_len(s):
            try:
                return len(s)
            except TypeError:
                return None

        lens = tuple(len(s) for s in self.sources if try_len(s) is not None)
        if not lens:
            raise TypeError("No valid lengths found in sources")
        return min(lens)


# One parameter
@tp.overload
def join[*Ts](a: DataSource[tuple[*Ts]], /) -> DataSource[tuple[*Ts]]: ...


# Two parameters
@tp.overload
def join[*Ts](
    a: DataSource[tuple[()]], b: DataSource[tuple[*Ts]], /
) -> DataSource[tuple[*Ts]]: ...
@tp.overload
def join[A, *Bs](
    a: DataSource[tuple[A]], b: DataSource[tuple[*Bs]], /
) -> DataSource[tuple[A, *Bs]]: ...
@tp.overload
def join[A, B, *Cs](
    a: DataSource[tuple[A, B]], b: DataSource[tuple[*Cs]], /
) -> DataSource[tuple[A, B, *Cs]]: ...
@tp.overload
def join[A, B, C, *Ds](
    a: DataSource[tuple[A, B, C]], b: DataSource[tuple[*Ds]], /
) -> DataSource[tuple[A, B, C, *Ds]]: ...


# Three parameters
@tp.overload
def join[*As](
    a: DataSource[tuple[()]], b: DataSource[tuple[()]], c: DataSource[tuple[*As]], /
) -> DataSource[tuple[*As]]: ...
@tp.overload
def join[A, *Bs](
    a: DataSource[tuple[A]], b: DataSource[tuple[()]], c: DataSource[tuple[*Bs]], /
) -> DataSource[tuple[*Bs]]: ...
@tp.overload
def join[A, *Bs](
    a: DataSource[tuple[()]], b: DataSource[tuple[A]], c: DataSource[tuple[*Bs]], /
) -> DataSource[tuple[*Bs]]: ...
@tp.overload
def join[A, B, *Cs](
    a: DataSource[tuple[A]], b: DataSource[tuple[B]], c: DataSource[tuple[*Cs]], /
) -> DataSource[tuple[A, B, *Cs]]: ...
@tp.overload
def join[A, B, C, *Ds](
    a: DataSource[tuple[A, B]], b: DataSource[tuple[C]], c: DataSource[tuple[*Ds]], /
) -> DataSource[tuple[A, B, C, *Ds]]: ...
@tp.overload
def join[A, B, C, *Ds](
    a: DataSource[tuple[A]], b: DataSource[tuple[B, C]], c: DataSource[tuple[*Ds]], /
) -> DataSource[tuple[A, B, C, *Ds]]: ...


@tp.overload
def join(*sources: DataSource) -> DataSource:
    """Joins multiple data sources into a single data source."""
    return JoinDataSource(*sources)


def join(*sources) -> DataSource:
    return JoinDataSource(*sources)
