import abc
import math
import typing as typ


import jax
import jax.numpy as jnp
import jax.typing as atp

from ..core import graph, graph_util, asserts
from ..core.dataclasses import dataclass

from ..util.distribution import Distribution, Gaussian, Noise, Scale


class NoiseSchedule(abc.ABC):
    @abc.abstractmethod
    def alpha(self, t: atp.ArrayLike) -> jax.Array: ...

    @abc.abstractmethod
    def sigma(self, t: atp.ArrayLike) -> jax.Array: ...

    @abc.abstractmethod
    def inverse(self, sigma_alpha_ratio: atp.ArrayLike) -> jax.Array: ...

    def alpha_dot(self, t: atp.ArrayLike) -> jax.Array:
        return jax.grad(self.alpha)(t)

    def sigma_dot(self, t: atp.ArrayLike) -> jax.Array:
        return jax.grad(self.sigma)(t)

    def alpha_dot_over_alpha(self, t: atp.ArrayLike) -> jax.Array:
        return jax.grad(lambda t: jnp.log(self.sigma(t)))(t)

    def sigma_dot_over_sigma(self, t: atp.ArrayLike) -> jax.Array:
        return jax.grad(lambda t: jnp.log(self.sigma(t)))(t)

    @staticmethod
    def linear() -> "NoiseSchedule":
        return Linear()

    @staticmethod
    def linear_noise(min_sigma: float, max_sigma: float = 1.0) -> "NoiseSchedule":
        return LinearNoise(min_sigma, max_sigma)

    @staticmethod
    def log_linear_noise(min_sigma: float, max_sigma: float) -> "NoiseSchedule":
        return LogLinearNoise(math.log(min_sigma), math.log(max_sigma))

    @staticmethod
    def ddpm_noise(
        num_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02
    ) -> "NoiseSchedule":
        return InterpolatedNoise.from_betas(
            jnp.linspace(beta_start, beta_end, num_steps)
        )

    @staticmethod
    def ldm_noise(
        num_steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012
    ) -> "NoiseSchedule":
        return InterpolatedNoise.from_betas(
            jnp.linspace(beta_start**0.5, beta_end**0.5, num_steps) ** 2
        )

    @staticmethod
    def sigmoid_noise(
        N: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02
    ) -> "NoiseSchedule":
        betas = (
            jax.nn.sigmoid(jnp.linspace(-6, 6, N)) * (beta_end - beta_start)
            + beta_start
        )
        return InterpolatedNoise.from_betas(betas)

    # Transforms on schedules

    def anneal_signal_linear(self) -> "NoiseSchedule":
        """Returns a schedule that linearly anneals the signal to 0 at t -> 1."""
        return AnnealedNoiseSchedule(self)

    def constant_variance(
        self, std: float | None = None, var: float | None = None
    ) -> "NoiseSchedule":
        std = std or (math.sqrt(var) if var else None) or 1.0
        return ConstantVariance(std, self)

    # Forward process methods

    @jax.jit
    def forward[T](self, key: jax.Array, x_0: T, /, t: jax.Array) -> T:
        x0_flat, x_uf = graph_util.ravel(x_0)
        dtype = (
            x0_flat.dtype
            if x0_flat.dtype in [jnp.float32, jnp.float16, jnp.float64]
            else jnp.float32
        )
        noise = jax.random.normal(key, shape=x0_flat.shape, dtype=dtype)
        return x_uf(self.sigma(t) * noise + self.alpha(t) * x0_flat)

    @jax.jit
    def forward_and_flow[T](
        self, key: jax.Array, x_0: T, /, t: jax.Array
    ) -> tuple[T, T]:
        """Both sample from the schedule and return the flow."""
        x0_flat, x_uf = graph_util.ravel(x_0)
        dtype = (
            x0_flat.dtype
            if x0_flat.dtype in [jnp.float32, jnp.float16, jnp.float64]
            else jnp.float32
        )
        noise = jax.random.normal(key, shape=x0_flat.shape, dtype=dtype)
        sigma, alpha, sigma_dot, alpha_dot = (
            self.sigma(t),
            self.alpha(t),
            self.sigma_dot(t),
            self.alpha_dot(t),
        )
        xt = x_uf(sigma * noise + alpha * x0_flat)
        xt_dot = x_uf(sigma_dot * noise + alpha_dot * x0_flat)
        return xt, xt_dot

    def transform[T](self, x0_dist: Distribution[T], t) -> Distribution[T]:
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        return x0_dist.transform(Scale(alpha)).transform(Noise(sigma))

    # Returns a parameterization of the flow associated with this schedule

    # Utilities for converting between different representations of the flow
    # for the denoising/noising process defined by the schedule.
    def parameterize(
        self,
        alpha_hat: typ.Callable[[jax.Array], jax.Array] | jax.typing.ArrayLike,
        sigma_hat: typ.Callable[[jax.Array], jax.Array] | jax.typing.ArrayLike,
    ) -> "NoiseScheduleFlowParam":
        if not callable(alpha_hat):
            alpha_hat = jnp.array(alpha_hat)
        if not callable(sigma_hat):
            sigma_hat = jnp.array(sigma_hat)
        return NoiseScheduleFlowParam(
            schedule=self,
            alpha_hat=alpha_hat,
            sigma_hat=sigma_hat,
        )

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        ax = ax if ax is not None else plt.gca()
        Ts = jnp.linspace(0.0, 1.0, 100)
        alphas = jax.vmap(self.alpha)(Ts)
        sigmas = jax.vmap(self.sigma)(Ts)
        alpha_dot = jax.vmap(self.alpha_dot)(Ts)
        sigma_dot = jax.vmap(self.sigma_dot)(Ts)
        ax.plot(Ts, alphas, label="alpha")
        ax.plot(Ts, sigmas, label="sigma")
        ax.plot(Ts, alpha_dot, label="alpha_dot")
        ax.plot(Ts, sigma_dot, label="sigma_dot")
        ax.legend()
        return ax


@dataclass(frozen=True)
class Linear(NoiseSchedule):
    def sigma(self, t: atp.ArrayLike) -> jax.Array:
        return jnp.array(t)

    def alpha(self, t: atp.ArrayLike) -> jax.Array:
        return 1.0 - jnp.array(t)

    def inverse(self, sigma_alpha_ratio: atp.ArrayLike) -> jax.Array:
        sigma_alpha_ratio = jnp.array(sigma_alpha_ratio)
        # 1 + s/a = (s + a) / a
        # thus (s/a) /(1 + s/a) = s / (s + a)
        # since sigma + alpha = 1, this is just s, which is t
        return sigma_alpha_ratio / (1.0 + sigma_alpha_ratio)


@dataclass(frozen=True)
class LinearNoise(NoiseSchedule):
    min_sigma: float
    max_sigma: float

    def sigma(self, t: atp.ArrayLike) -> jax.Array:
        return self.min_sigma + jnp.array(t) * (self.max_sigma - self.min_sigma)

    def alpha(self, t: atp.ArrayLike) -> jax.Array:
        return jnp.ones(())

    def inverse(self, sigma_alpha_ratio: atp.ArrayLike) -> jax.Array:
        return (jnp.array(sigma_alpha_ratio) - self.min_sigma) / (
            self.max_sigma - self.min_sigma
        )


@dataclass(frozen=True)
class LogLinearNoise(NoiseSchedule):
    log_min_sigma: float
    log_max_sigma: float

    def sigma(self, t: atp.ArrayLike) -> jax.Array:
        return jnp.exp(
            self.log_min_sigma + t * (self.log_max_sigma - self.log_min_sigma)
        )

    def alpha(self, t: atp.ArrayLike) -> jax.Array:
        return jnp.ones(())

    def inverse(self, sigma_alpha_ratio: atp.ArrayLike) -> jax.Array:
        log_sigma = jnp.log(sigma_alpha_ratio)
        return (log_sigma - self.log_min_sigma) / (
            self.log_max_sigma - self.log_min_sigma
        )


@dataclass
class ConstantVariance(NoiseSchedule):
    std: float
    base: NoiseSchedule

    def sigma(self, t):
        s, a = self.base.sigma(t), self.base.alpha(t)
        return self.std * s / jnp.sqrt(s * s + a * a)

    def alpha(self, t):
        s, a = self.base.sigma(t), self.base.alpha(t)
        return self.std * a / jnp.sqrt(s * s + a * a)

    def inverse(self, sigma_alpha_ratio: atp.ArrayLike) -> jax.Array:
        # the snr is the same as that for the base schedule
        return self.base.inverse(sigma_alpha_ratio)


@dataclass
class AnnealedNoiseSchedule(NoiseSchedule):
    base: NoiseSchedule

    def alpha(self, t: atp.ArrayLike) -> jax.Array:
        return (1.0 - jnp.array(t)) * self.base.alpha(t)

    def sigma(self, t: atp.ArrayLike) -> jax.Array:
        return self.base.sigma(t)

    def inverse(self, sigma_alpha_ratio: atp.ArrayLike) -> jax.Array:
        # TODO: Implement inversion for simple sub-schedules
        raise NotImplementedError("Unable to invert an AnnealedNoiseSchedule")


@dataclass
class InterpolatedNoise(NoiseSchedule):
    sigmas: jax.Array

    def sigma(self, t: atp.ArrayLike) -> jax.Array:
        """Linearly interpolates values in x at positions t ∈ [0, 1]."""
        return jnp.interp(t, jnp.linspace(0, 1, len(self.sigmas)), self.sigmas)

    def alpha(self, t: atp.ArrayLike) -> jax.Array:
        return jnp.ones(())

    @jax.jit
    def inverse(self, sigma_alpha_ratio: atp.ArrayLike) -> jax.Array:
        sigma = sigma_alpha_ratio
        # get the left and right indices
        sigma = jnp.clip(sigma, self.sigmas[0], self.sigmas[-1])
        right_idx = jnp.searchsorted(self.sigmas, sigma, side="left")
        right_idx = jnp.clip(right_idx, 1, len(self.sigmas) - 1)
        left_idx = right_idx - 1
        # convert to sigmas
        left_sigma, right_sigma = self.sigmas[left_idx], self.sigmas[right_idx]
        frac_t = (sigma - left_sigma) / (right_sigma - left_sigma)
        left_t = left_idx / (len(self.sigmas) - 1)
        t = left_t + frac_t
        return t

    @staticmethod
    def from_betas(betas: atp.ArrayLike) -> "InterpolatedNoise":
        return InterpolatedNoise(jnp.sqrt(1 / jnp.cumprod(1.0 - betas) - 1))

    # Prevent massive repr on print
    def __repr__(self) -> str:
        return (
            f"InterpolatedNoise(Array({self.sigmas.shape}, dtype={self.sigmas.dtype}))"
        )


# Samples time points from target according to the SNR distribution
# of the source noise schedule
class TimeSNRDistribution(Distribution[jax.Array]):
    def __init__(self, source: NoiseSchedule, target: NoiseSchedule):
        self.source = source
        self.target = target

    @jax.jit
    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        src_ts = jax.random.uniform(key, (math.prod(shape),), minval=0.0, maxval=1.0)

        def transform_t(src, tgt, src_t):
            sigma = src.sigma(src_t)
            alpha = src.alpha(src_t)
            tgt_t = tgt.inverse(sigma / alpha)
            return tgt_t

        transform_t = jax.vmap(transform_t, in_axes=(None, None, 0))

        tgt_t = transform_t(self.source, self.target, src_ts)
        return jnp.reshape(tgt_t, shape)


@dataclass
class NoiseScheduleFlowParam[T, Cond]:
    schedule: NoiseSchedule
    alpha_hat: jax.Array | typ.Callable[[jax.Array], jax.Array]
    sigma_hat: jax.Array | typ.Callable[[jax.Array], jax.Array]

    def output_to_flow(
        self, output: T, x_t: T, /, t: jax.Array, *, cond: Cond = None
    ) -> T:
        alpha_hat = self.alpha_hat(t) if callable(self.alpha_hat) else self.alpha_hat
        sigma_hat = self.sigma_hat(t) if callable(self.sigma_hat) else self.sigma_hat
        v = _flow_transform(
            v_t=output,
            x_t=x_t,
            alpha=self.schedule.alpha(t),
            sigma=self.schedule.sigma(t),
            alpha_dot=alpha_hat,
            sigma_dot=sigma_hat,
            alpha_dot_hat=self.schedule.alpha_dot(t),
            sigma_dot_hat=self.schedule.sigma_dot(t),
        )
        return v

    def flow_to_output(
        self, v_t: T, x_t: T, /, t: jax.Array, *, cond: Cond = None
    ) -> T:
        alpha_hat = self.alpha_hat(t) if callable(self.alpha_hat) else self.alpha_hat
        sigma_hat = self.sigma_hat(t) if callable(self.sigma_hat) else self.sigma_hat
        out = _flow_transform(
            v_t=v_t,
            x_t=x_t,
            alpha=self.schedule.alpha(t),
            sigma=self.schedule.sigma(t),
            alpha_dot=self.schedule.alpha_dot(t),
            sigma_dot=self.schedule.sigma_dot(t),
            alpha_dot_hat=alpha_hat,
            sigma_dot_hat=sigma_hat,
        )
        return out


def _flow_transform[T](
    v_t: T, x_t: T, alpha, sigma, alpha_dot, sigma_dot, alpha_dot_hat, sigma_dot_hat
) -> T:
    # We want to transform v from alpha_dot, sigma_dot to alpha_hat, sigma_hat
    M = jnp.array(
        [
            [alpha, sigma],
            [alpha_dot, sigma_dot],
        ]
    )
    dots = jnp.array([alpha_dot_hat, sigma_dot_hat])
    # M @ [x_0; x_1] = [x_t; v]
    # v_hat = [alpha_dot_hat, sigma_dot_hat] @ M^(-1) @ [x_t; v]
    # Let coeff^T := [alpha_dot_hat, sigma_dot_hat] @ M^(-1)
    # i.e.
    #  coeff^T @ M = [alpha_dot, sigma_dot]
    # or M^T @ coeff = [alpha_dot; sigma_dot]
    coeff = jnp.linalg.solve(M.T, dots)

    asserts.graphs_equal_shapes_and_dtypes(x_t, v_t)
    v_flat, v_uf = graph_util.ravel(v_t)
    x_t_flat, _ = graph_util.ravel(x_t)
    v_hat_flat = coeff[0] * x_t_flat + coeff[1] * v_flat
    return v_uf(v_hat_flat)
