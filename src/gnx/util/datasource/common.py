import math
import functools
import typing as tp
import abc
import hashlib

import jax
from jax._src.interpreters.partial_eval import PyTree
import jax.numpy as jnp

from pathlib import Path

from ...core import nn, asserts, graph, graph_util
from ...core.dataclasses import dataclass

from .transform import DataTransform


# Note: not necessarily an nn.Object
class DataCache(abc.ABC):
    @abc.abstractmethod
    def cached[T](self, source: "DataSource[T]") -> "DataSource[T]": ...


class DataIterator[T](graph.Object, abc.ABC):
    @property
    @abc.abstractmethod
    def instance(self) -> T: ...

    @abc.abstractmethod
    def has_next(self) -> jax.Array | bool: ...

    @abc.abstractmethod
    def next(self) -> T: ...

    # Returns None if has_next() is True,
    # otherwise returns the remaining elements in the batch
    @abc.abstractmethod
    def remainder(self) -> T | None: ...

    @abc.abstractmethod
    def skip(self, num: jax.typing.ArrayLike, /): ...

    # Reset the iterator, optionally with a different sampling key (as passed to sampler()).
    # If not specified, it should use the same key as was used to create the iterator.
    @abc.abstractmethod
    def reset(self, key: jax.Array | None = None, /): ...

    # Will reset the iterator using a key from the given
    # RngStream if has_next is False before calling next().
    def cyclic_next(self, rng: nn.RngStream | None = None) -> T:
        # if needed, reset the iterator
        jax.lax.cond(
            self.has_next(),
            lambda: None,
            lambda: self.reset(rng() if rng is not None else None),
        )
        return self.next()

    def __next__(self) -> T:
        if not self.has_next():
            raise StopIteration
        return self.next()

    def __iter__(self) -> tp.Iterator[T]:
        return self


# Data sources should be nn.compatible objects
# but may also be e.g. pytree-compatible objects
# and not necessarily extend nn.Object.
class DataSource[T](graph.Object, abc.ABC):
    @property
    @abc.abstractmethod
    def instance(self) -> T: ...

    @abc.abstractmethod
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T: ...

    @abc.abstractmethod
    def sampler(self, key: jax.Array | None = None) -> DataIterator[T]: ...

    @abc.abstractmethod
    def batch(self, shape: tuple[int, ...]) -> "DataSource[T]": ...

    # These may be overridden for efficiency
    def map[V](self, t: DataTransform[T, V]) -> "DataSource[V]":
        return MappedDataSource(self, t)

    def data(self) -> T:
        try:
            length = len(self)
        except TypeError:
            raise ValueError("Cannot read in a data source with unknown length.")
        iterator = self.sampler()

        def scan_fn(iterator, _):
            sample = iterator.next()
            return iterator, sample

        _, samples = jax.lax.scan(scan_fn, iterator, None, length=length)
        return samples

    # Do not override
    def cached(self, cache: DataCache) -> "DataSource[T]":
        return cache.cached(self)

    # For debugging purposes, allow python-style iteration over the data source
    def __iter__(self) -> tp.Iterator[T]:
        return self.sampler()

    def __len__(self) -> int:
        raise TypeError("Data source has no length")


class MappedDataIterator[T, V](DataIterator[V]):
    def __init__(
        self,
        key: jax.Array,
        source: DataIterator[T],
        transformation: DataTransform[T, V],
        batch_shape: tuple[int, ...] = (),
    ):
        self.key = nn.RngKey(key, tag="iterator")
        self.count = nn.RngCount(jnp.zeros((), dtype=jnp.uint32), tag="iterator")
        self.source = source
        self.transformation = transformation
        self.batch_shape = batch_shape

    @property
    def instance(self) -> V:
        sample = self.source.instance
        return self.transformation(jax.random.key(0), sample)

    def has_next(self) -> jax.Array | bool:
        return self.source.has_next()

    @jax.jit
    def next(self) -> V:
        sample = self.source.next()
        if not self.batch_shape:
            key = jax.random.fold_in(self.key[...], self.count[...])
            self.count[...] = self.count[...] + 1
            transformed = self.transformation(key, sample)
            return transformed
        else:
            N = math.prod(self.batch_shape)
            # flatten the sample
            sample = jax.tree.map(
                lambda x: jnp.reshape(x, (N,) + x.shape[len(self.batch_shape) :]),
                sample,
            )
            keys = jax.vmap(
                lambda c: jax.random.fold_in(self.key[...], self.count[...] + c)
            )(jnp.arange(N))
            self.count[...] = self.count[...] + N
            transformed = jax.vmap(
                lambda key, sample: self.transformation(key, sample),
            )(keys, sample)
            transformed = jax.tree.map(
                lambda x: jnp.reshape(x, self.batch_shape + x.shape[1:]), transformed
            )
            return transformed

    def remainder(self) -> V | None:
        if self.has_next():
            return None
        remainder = self.source.remainder()
        if remainder is None:
            return None
        N = graph_util.axis_size(remainder, 0)
        keys = jax.vmap(
            lambda c: jax.random.fold_in(self.key[...], self.count[...] + c)
        )(jnp.arange(N))
        transformed = jax.vmap(self.transformation)(keys, remainder)
        return transformed

    @jax.jit
    def skip(self, num: jax.typing.ArrayLike):
        self.source.skip(num)

    @jax.jit
    def reset(self, key: jax.Array | None = None):
        self.count[...] = jnp.zeros_like(self.count[...])
        if key is not None:
            # NOTE: Split the same as in sampler() below
            t_key, s_key = jax.random.split(key)
            self.source.reset(s_key)
            self.key[...] = t_key
        else:
            self.source.reset()


class MappedDataSource[T, V](DataSource[V]):
    def __init__(
        self,
        source: DataSource[T],
        transform: DataTransform[T, V],
        batch_shape: tuple[int, ...] = (),
    ):
        self.source = source
        self.transform = transform
        self.batch_shape = batch_shape

    @property
    def instance(self) -> V:
        return self.transform(jax.random.key(0), self.source.instance)

    def batch(self, shape: tuple[int, ...]) -> "MappedDataSource[T, V]":
        return MappedDataSource(
            self.source.batch(shape), self.transform, shape + self.batch_shape
        )

    def sampler(self, key: jax.Array) -> DataIterator[V]:
        t_key, s_key = jax.random.split(key)
        return MappedDataIterator(
            t_key, self.source.sampler(s_key), self.transform, self.batch_shape
        )

    @functools.partial(jax.jit, static_argnames=("shape",))
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> V:
        sample_key, transform_key = jax.random.split(key)
        # This will include any batch dimensions already
        samples = self.source.sample(sample_key, shape)
        shape = shape + self.batch_shape
        if len(shape) > 0:
            N = math.prod(shape)
            # Flatten the batch shape for transformation
            samples = jax.tree.map(
                lambda x: jnp.reshape(x, (N,) + x.shape[len(shape) :]), samples
            )
            keys = jax.random.split(transform_key, N)
            transformed = jax.vmap(
                lambda key, sample: self.transform(key, sample),
            )(keys, samples)
            transformed = jax.tree.map(
                lambda x: jnp.reshape(x, shape + x.shape[1:]), transformed
            )
            return transformed
        else:
            return self.transform(transform_key, samples)

    def __len__(self) -> int:
        return len(self.source)


class PyTreeDataIterator[T](DataIterator[T]):
    def __init__(
        self,
        data: T,
        indices: jax.Array,
        remainder_indices: jax.Array,
        batch_shape: tuple[int, ...] = (),
    ):
        self.data = data
        self.batch_shape = batch_shape
        self.indices = nn.Variable(indices)
        self.remainder_indices = nn.Variable(remainder_indices)
        self.count = nn.Variable(jnp.zeros((), jnp.uint32))

    @staticmethod
    def create_indices(
        key: jax.Array | None, data_size: int, batch_size: int
    ) -> tuple[jax.Array, jax.Array]:
        assert batch_size > 0, "Batch size must be positive"
        assert data_size > 0, "Data size must be positive"
        num_indices = batch_size * math.floor(data_size / batch_size)
        if key is not None:
            indices = jax.random.permutation(key, data_size)
        else:
            indices = jnp.arange(data_size)
        indices, remainder = indices[:num_indices], indices[num_indices:]
        indices = indices.reshape((num_indices // batch_size, batch_size))
        return indices, remainder

    @property
    def instance(self) -> T:
        return jax.tree.map(lambda x: x[0], self.data)

    @jax.jit
    def has_next(self) -> jax.Array | bool:
        return self.count[...] < self.indices.shape[0]

    @jax.jit
    def next(self) -> T:
        batch_indices = self.indices[self.count[...]]
        self.count[...] += 1
        return jax.tree.map(
            lambda x: jnp.reshape(
                jnp.take(x, batch_indices, axis=0, unique_indices=True),
                self.batch_shape + x.shape[1:],
            ),
            self.data,
        )

    def remainder(self) -> jax.Array | None:
        if self.has_next() or self.remainder_indices.shape[0] == 0:
            return None
        indices = self.remainder_indices[...]
        return jax.tree.map(
            lambda x: jnp.reshape(
                jnp.take(x, indices, axis=0, unique_indices=True),
                (indices.shape[0],) + x.shape[1:],
            ),
            self.data,
        )

    @jax.jit
    def skip(self, num: jax.typing.ArrayLike):
        self.count[...] = self.count[...] + num

    @jax.jit
    def reset(self, key: jax.Array | None = None):
        self.count[...] = jnp.zeros_like(self.count[...])
        if key is not None:
            data_size = graph_util.axis_size(self.data, 0)
            batch_size = math.prod(self.batch_shape)
            indices, remainder_indices = PyTreeDataIterator.create_indices(
                key, data_size, batch_size
            )
            self.indices[...] = indices
            self.remainder_indices[...] = remainder_indices


class PyTreeDataSource[T](DataSource[T]):
    def __init__(self, data: T, batch_shape: tuple[int, ...] = ()):
        self.pytree = data
        self.batch_shape = batch_shape

    @property
    def data(self) -> T:
        return self.pytree

    @property
    def instance(self) -> T:
        return jax.tree.map(lambda x: x[0], self.pytree)

    @functools.partial(jax.jit, static_argnames=("shape",))
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        N = graph_util.axis_size(self.pytree, 0)
        n = math.prod(shape + self.batch_shape)
        if n > N:
            idxs = jax.random.choice(key, N, (n,), replace=True)
        else:
            idxs = jax.random.choice(key, N, (n,), replace=False)
        samples = jax.tree.map(lambda x: x[idxs, ...], self.pytree)
        samples = jax.tree.map(
            lambda x: jnp.reshape(x, shape + self.batch_shape + x.shape[1:]), samples
        )
        return samples

    # Not jit'd since we need to return an ArrayRef
    def sampler(self, key: jax.Array | None = None) -> DataIterator[T]:
        N, n = graph_util.axis_size(self.data, 0), math.prod(self.batch_shape)
        indices, remainder_indices = PyTreeDataIterator.create_indices(key, N, n)
        return PyTreeDataIterator(
            self.data, indices, remainder_indices, self.batch_shape
        )

    def batch(self, shape: tuple[int, ...]) -> "PyTreeDataSource":
        return PyTreeDataSource(self.data, shape + self.batch_shape)

    def __len__(self) -> int:
        N = graph_util.axis_size(self.data, 0)
        n = math.prod(self.batch_shape)
        assert n > 0, "Batch shape must be non-empty"
        return N // n


def pytree[T](data: T) -> PyTreeDataSource[T]:
    return PyTreeDataSource(data)


# An in-memory cache
class PyTreeCache(DataCache):
    # The max_size is in bytes
    def __init__(self, max_nbytes: int | None, fallback: DataCache | None):
        self.max_nbytes = max_nbytes
        self.fallback = fallback

    def cached[T](self, source: DataSource[T]) -> DataSource[T]:
        try:
            length = len(source)
        except TypeError:
            raise ValueError("Cannot cache a data source with unknown length")
        data_size = graph_util.size_in_bytes(source.instance) * length
        if self.max_nbytes is not None and data_size >= self.max_nbytes:
            if self.fallback is None:
                raise ValueError(
                    "Data source too large to cache and no fallback provided"
                )
            return self.fallback.cached(source)
        # Materialize the entire data source
        return PyTreeDataSource(source.data())


def in_memory_cache(
    max_nbytes: int | None = None, fallback: DataCache | None = None
) -> PyTreeCache:
    return PyTreeCache(max_nbytes, fallback)
