import typing as tp

from ...core.dataclasses import dataclass
from ...core import graph_util

import json
import jax
import jax.numpy as jnp
import hashlib


class DataTransform[T, V](tp.Protocol):
    def __call__(self, key: jax.Array, x: T, /) -> V: ...


class Lambda[T, V](tp.Protocol):
    def __call__(self, key: jax.Array, x: T, /) -> V: ...


class LambdaTransform[T, V](DataTransform[T, V]):
    def __init__(self, lam: Lambda[T, V], id: str, kwargs):
        self.lam = lam
        self.lam_id = id
        self.lam_id_kwargs = kwargs

    def __call__(self, key: jax.Array, x: T) -> V:
        return self.lam(key, x)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            (f"{self.lam_id}-" + json.dumps(self.lam_id_kwargs)).encode()
        ).hexdigest()


def data_transform[T, V](
    id, **id_kwargs
) -> tp.Callable[[Lambda[T, V]], DataTransform[T, V]]:
    def decorator(func: Lambda[T, V]) -> DataTransform[T, V]:
        return LambdaTransform(func, id, id_kwargs)

    return decorator


# A transform that is linear over combinations of measures.
class LinearTransform[T, V](DataTransform[T, V]): ...


@dataclass(frozen=True)
class Scale[T](LinearTransform[T, T]):
    scale: T | jax.Array | float | int

    @property
    def sha256(self) -> str:
        return hashlib.sha256((f"scale-{self.scale}").encode()).hexdigest()

    def __call__(self, key: jax.Array, x: T) -> T:
        return jax.tree.map(lambda s, x: s * x, jax.tree.broadcast(self.scale, x), x)


@dataclass(frozen=True)
class Noise[T](LinearTransform[T, T]):
    sigma: T | jax.Array | float | int

    @property
    def sha256(self) -> str:
        return hashlib.sha256((f"noise-{self.sigma}").encode()).hexdigest()

    def __call__(self, key: jax.Array, x: T) -> T:
        sigma = jax.tree.broadcast(self.sigma, x)
        sigma = jax.tree.map(lambda s, x: jnp.broadcast_to(s, jnp.shape(x)), sigma, x)
        sigma_flat, _ = graph_util.ravel(sigma)
        x_flat, x_uf = graph_util.ravel(x)
        noise = jax.random.normal(key, shape=x_flat.shape, dtype=x_flat.dtype)
        return x_uf(x_flat + noise * sigma_flat)
