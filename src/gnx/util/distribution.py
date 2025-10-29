import math
import functools
import typing as tp
import jax
import jax.numpy as jnp

from ..core import graph, graph_util, graph, asserts
from ..core.dataclasses import dataclass

from .datasource import DataIterator, DataSource
from .datasource.util import GeneratorDataIterator
from .datasource.transform import DataTransform, LinearTransform, Scale, Noise


class Distribution[T](DataSource[T]):
    @property
    def instance(self) -> T:
        return self.sample(jax.random.key(0))

    def batch(self, shape: tuple[int, ...]) -> "Independent[T]":
        return Independent(self, shape)

    def sampler(self, key: jax.Array) -> DataIterator[T]:
        return GeneratorDataIterator(
            key,
            graph_util.Partial(type(self).sample, self),
        )

    def __len__(self) -> int:  # From DataSource, disabled for Distributions
        raise TypeError("Length not defined for Distribution.")

    # Distribution methods
    # Overridden by subclasses
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        raise NotImplementedError()

    def transform[V](self, t: DataTransform[T, V]) -> "Distribution[V]":
        raise NotImplementedError()

    # Continuous distribution methods

    def score(self, x: T) -> T:
        raise NotImplementedError()

    def log_potential(self, x: T) -> jax.Array:
        raise NotImplementedError()

    def log_pdf(self, x: T) -> jax.Array:
        raise NotImplementedError()


class Independent[T](Distribution[T]):
    def __init__(self, distribution: Distribution[T], batch_shape: tuple[int, ...]):
        self.distribution = distribution
        self.batch_shape = batch_shape

    def batch(self, shape: tuple[int, ...]) -> "Independent[T]":
        return Independent(self.distribution, shape + self.batch_shape)

    def sampler(self, key: jax.Array) -> DataIterator[T]:
        return GeneratorDataIterator(
            key,
            graph_util.Partial(type(self.distribution).sample, self.distribution),
            self.batch_shape,
        )

    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        return self.distribution.sample(key, shape + self.batch_shape)


# -------- Gaussian --------


@dataclass
class Gaussian[T](Distribution[T]):
    mean: T
    std: T

    @jax.jit
    def score(self, x: T) -> T:
        asserts.graphs_equal_shapes_and_dtypes(x, self.mean, self.std)
        mean_flat, std_flat, (x_flat, x_uf) = (
            graph_util.ravel(self.mean)[0],
            graph_util.ravel(self.std)[0],
            graph_util.ravel(x),
        )
        score = -(x_flat - mean_flat) / (std_flat * std_flat)
        return x_uf(score)

    @jax.jit
    def log_potential(self, x: T) -> jax.Array:
        asserts.graphs_equal_shapes_and_dtypes(x, self.mean, self.std)
        mean_flat, std_flat, x_flat = (
            graph_util.ravel(self.mean)[0],
            graph_util.ravel(self.std)[0],
            graph_util.ravel(x)[0],
        )
        return -jnp.sum(jnp.square(x_flat - mean_flat) / (std_flat * std_flat)) / 2

    @jax.jit
    def log_pdf(self, x: T) -> jax.Array:
        asserts.graphs_equal_shapes_and_dtypes(x, self.mean, self.std)
        mean_flat, std_flat, x_flat = (
            graph_util.ravel(self.mean)[0],
            graph_util.ravel(self.std)[0],
            graph_util.ravel(x)[0],
        )
        C = -(jnp.log(2 * jnp.pi) * x_flat.shape[0] / 2 + jnp.sum(jnp.log(std_flat)))
        log_pdf = (
            C - jnp.sum(jnp.square(x_flat - mean_flat) / (std_flat * std_flat)) / 2
        )
        return jax.vmap(lambda x, m, s: jax.scipy.stats.norm.logpdf(x, loc=m, scale=s))(
            x_flat, mean_flat, std_flat
        ).sum()
        return log_pdf

    @functools.partial(jax.jit, static_argnames=("shape",))
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        asserts.graphs_equal_shapes_and_dtypes(self.mean, self.std)
        (mean_flat, mean_uf), std_flat = (
            graph_util.ravel(self.mean),
            graph_util.ravel(self.std)[0],
        )
        samples = math.prod(shape)
        noise = jax.random.normal(
            key, shape=(samples,) + mean_flat.shape, dtype=mean_flat.dtype
        )
        samples = noise * std_flat[jnp.newaxis, :] + mean_flat[jnp.newaxis, :]
        samples = jax.vmap(mean_uf)(samples)
        return jax.tree.map(lambda x: jnp.reshape(x, shape + x.shape[1:]), samples)

    def transform[V](self, t: DataTransform[T, V]) -> Distribution[V]:
        if isinstance(t, Scale):
            scale = jax.tree.broadcast(t.scale, self.mean)
            return Gaussian(
                mean=jax.tree.map(lambda x, s: x * s, self.mean, scale),
                std=jax.tree.map(lambda x, s: x * s, self.std, scale),
            )
        elif isinstance(t, Noise):
            noise = jax.tree.broadcast(t.sigma, self.std)
            mean: tp.Any = self.mean
            std = jax.tree.map(lambda s, n: jnp.sqrt(s * s + n * n), self.std, noise)
            return Gaussian(mean=mean, std=std)
        return super().transform(t)


class Uniform[T](Distribution[T]):
    def __init__(self, min: T, max: T):
        self.min = min
        self.max = max

    @jax.jit
    def log_pdf(self, x: T) -> jax.Array:
        asserts.graphs_equal_shapes_and_dtypes(x, self.min, self.max)
        x_flat, _ = graph_util.ravel(x)[0]
        min_flat, _ = graph_util.ravel(self.min)[0]
        max_flat, _ = graph_util.ravel(self.max)[0]
        log_density = jnp.log(max_flat - min_flat + 1e-3).sum()
        return log_density * jnp.logical_and(
            jnp.all(x_flat >= min_flat), jnp.all(x_flat <= max_flat)
        )

    @jax.jit
    def log_potential(self, x: T) -> jax.Array:
        x_flat, _ = graph_util.ravel(x)[0]
        min_flat, _ = graph_util.ravel(self.min)[0]
        max_flat, _ = graph_util.ravel(self.max)[0]
        return 1.0 * jnp.logical_and(
            jnp.all(x_flat >= min_flat), jnp.all(x_flat <= max_flat)
        )

    @jax.jit
    def score(self, x: T) -> T:
        return jax.tree.map(lambda x: jnp.zeros_like(x), x)

    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        asserts.graphs_equal_shapes_and_dtypes(self.min, self.max)
        min_flat, min_uf = graph_util.ravel(self.min)
        max_flat = graph_util.ravel(self.max)[0]
        N = math.prod(shape)
        samples = jax.random.uniform(
            key,
            shape=(N,) + min_flat.shape,
            dtype=min_flat.dtype,
            minval=min_flat[None, :],
            maxval=max_flat[None, :],
        )
        samples = jax.vmap(min_uf)(samples)
        return jax.tree.map(lambda x: jnp.reshape(x, shape + x.shape[1:]), samples)


class Empirical[T](Distribution[T]):
    """A mixture of delta functions at the given samples."""

    def __init__(
        self,
        samples: T,
        *,
        sigma: jax.typing.ArrayLike | None = None,
        deterministic_sample: bool = False,
    ):
        self.samples = samples
        self.sigma = jnp.array(sigma) if sigma is not None else None
        self.deterministic_sample = deterministic_sample

    @jax.jit
    def log_pdf(self, x: T) -> jax.Array:
        raise NotImplementedError("Empirical distributions do not support log_pdf.")

    @jax.jit
    def log_potential(self, x: T) -> jax.Array:
        raise NotImplementedError(
            "Empirical distributions do not support log_potential."
        )

    @jax.jit
    def score(self, x: T) -> T:
        if self.sigma is None:
            raise NotImplementedError(
                "Unnoised empirical distributions do not support score()."
            )
        # compute the score
        x_flat, x_uf = graph_util.ravel(x)
        samples_flat = jax.vmap(lambda x: graph_util.ravel(x)[0])(self.samples)
        log_weights = -jnp.sum(
            jnp.square(samples_flat - x_flat[jnp.newaxis, :]), axis=-1
        ) / (2 * self.sigma**2)
        log_weights = jax.nn.log_softmax(log_weights)
        x0_flat = jnp.sum(samples_flat * jnp.exp(log_weights)[:, None], axis=0)
        score_flat = -(x_flat - x0_flat) / self.sigma**2
        return x_uf(score_flat)

    @functools.partial(jax.jit, static_argnames=("shape",))
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        N = math.prod(shape)
        S = graph_util.axis_size(self.samples, 0)
        i_key, n_key = jax.random.split(key)
        # If we are requesting all of the samples, we can just return them directly
        # if deterministic_sample is True.
        if N == S and self.deterministic_sample:
            samples = self.samples
        else:
            indices = jax.random.choice(i_key, S, shape=(N,), replace=N > S)
            samples = jax.tree.map(lambda x: jnp.take(x, indices, axis=0), self.samples)
        if self.sigma is not None:
            _, sample_uf = graph_util.ravel(jax.tree.map(lambda x: x[0], self.samples))
            samples_flat = jax.vmap(lambda x: graph_util.ravel(x)[0])(samples)
            samples_flat = samples_flat + self.sigma * jax.random.normal(
                n_key, shape=samples_flat.shape
            )
            samples = jax.vmap(sample_uf)(samples_flat)
        samples = jax.tree.map(lambda x: jnp.reshape(x, shape + x.shape[1:]), samples)
        return samples

    def transform[V](self, t: DataTransform[T, V]) -> Distribution[V]:
        sample = jax.tree.map(lambda x: x[0], self.samples)
        N = graph_util.axis_size(self.samples, 0)
        if isinstance(t, Scale):
            scale = t.scale
            sigma = self.sigma
            if sigma is not None:
                assert isinstance(
                    scale, (jax.Array, float)
                ), f"Scale must be a scalar or array, got {scale}"
                sigma = scale * sigma
            samples = jax.tree.map(
                lambda x, s: x * s[jnp.newaxis, ...],
                self.samples,
                jax.tree.broadcast(scale, sample),
            )
            return Empirical(
                samples,
                sigma=sigma,
                deterministic_sample=self.deterministic_sample,
            )
        elif isinstance(t, Noise):
            means: V = self.samples  # type: ignore
            sigma = jnp.array(t.sigma)
            if self.sigma is not None:
                sigma = jnp.sqrt(sigma**2 + self.sigma**2)
            return Empirical(
                means, sigma=sigma, deterministic_sample=self.deterministic_sample
            )
        return super().transform(t)


class Mixture[T](Distribution[T]):
    components: Distribution[T]
    log_weights: jax.Array | None = None

    def __init__(
        self, components: Distribution[T], log_weights: jax.Array | None = None
    ):
        self.components = components
        self.log_weights = log_weights

    @jax.jit
    def log_pdf(self, x: T) -> jax.Array:
        log_pdfs = jax.vmap(lambda c: c.log_pdf(x))(self.components)
        if self.log_weights is not None:
            log_weights = jax.nn.log_softmax(self.log_weights)
            log_pdfs = log_pdfs + log_weights
        log_pdfs = log_pdfs - jnp.log(
            log_pdfs.shape[0]
        )  # Normalize by number of components
        return jax.scipy.special.logsumexp(log_pdfs, axis=0)

    @jax.jit
    def log_potential(self, x: T) -> jax.Array:  # tpe: ignore
        # Use the log_pdf of each component to compute the log potential
        # to ensure the normalization is correct
        log_potentials = jax.vmap(lambda c: c.log_pdf(x))(self.components)
        if self.log_weights is not None:
            log_potentials = log_potentials + self.log_weights
        return jax.scipy.special.logsumexp(log_potentials, axis=0)

    @jax.jit
    def score(self, x: T) -> T:
        # The scores of the different components
        log_weights = jax.vmap(lambda c: c.log_pdf(x))(self.components)
        if self.log_weights is not None:
            log_weights = log_weights + self.log_weights
        log_weights = jax.nn.log_softmax(log_weights)
        weights = jnp.exp(log_weights)
        scores = jax.vmap(lambda c: c.score(x))(self.components)
        score = jax.tree.map(
            lambda s: jnp.sum(
                weights.reshape((weights.shape[0],) + (1,) * (s.ndim - 1)) * s, axis=0
            ),
            scores,
        )
        return score

    @functools.partial(jax.jit, static_argnames=("shape",))
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        i_key, r_key = jax.random.split(key)
        N = graph_util.axis_size(self.components, 0)
        component_idxs = jax.random.choice(
            i_key,
            N,
            shape,
            p=jnp.exp(self.log_weights) if self.log_weights is not None else None,
        )

        def _sample(idx: jax.Array, key: jax.Array) -> T:
            comp = jax.tree.map(lambda x: x[idx], self.components)
            return comp.sample(key)

        elts = jax.vmap(_sample)(
            component_idxs.reshape(-1),
            jax.random.split(r_key, component_idxs.size),
        )
        return jax.tree.map(lambda x: jnp.reshape(x, shape + x.shape[1:]), elts)

    def transform[V](self, t: DataTransform[T, V]) -> Distribution[V]:
        if isinstance(t, LinearTransform):
            components = jax.vmap(lambda c: c.transform(t))(self.components)
            # components = jax.vmap(lambda c: c.transform(t))(self.components)
            return Mixture(components, self.log_weights)
        return super().transform(t)


class Compose[T](Distribution[T]):
    def __init__(self, components: T):
        self.components = components

    @jax.jit
    def log_pdf(self, x: T) -> jax.Array:
        log_probs = jax.tree.map(
            lambda c, d: d.log_pdf(c),
            self.components,
            x,
            is_leaf=lambda x: isinstance(x, Distribution),
        )
        return jax.tree.reduce(jnp.sum, log_probs, jnp.zeros(()))

    @jax.jit
    def log_potential(self, x: T) -> jax.Array:
        log_potentials = jax.tree.map(
            lambda c, d: d.log_potential(c),
            self.components,
            x,
            is_leaf=lambda x: isinstance(x, Distribution),
        )
        return jax.tree.reduce(jnp.sum, log_potentials, jnp.zeros(()))

    @jax.jit
    def score(self, x: T) -> T:
        return jax.tree.map(
            lambda c, x: c.score(x),
            self.components,
            x,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

    @functools.partial(jax.jit, static_argnames=("shape",))
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        components, structure = jax.tree.flatten(
            self.components, is_leaf=lambda x: isinstance(x, Distribution)
        )
        keys = jax.random.split(key, len(components))
        samples = []
        for k, c in zip(keys, components):
            samples.append(c.sample(k, shape))
        return jax.tree.unflatten(structure, samples)
