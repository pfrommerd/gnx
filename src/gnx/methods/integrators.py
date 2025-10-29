import jax
import functools
import jax.numpy as jnp

from ..core import asserts, graph_util
from .noise_schedule import NoiseScheduleFlowParam
from .diffusion import (
    ForwardProcess,
    FlowParameterization,
    Integrator,
    Diffuser,
    NoisingForwardProcess,
)


# Euler integrator for diffusion ODEs.
class Euler[T, Cond](Integrator[T, Cond]):
    def step(
        self,
        x_s: T,
        output: T,
        /,
        s: jax.Array,
        t: jax.Array,
        *,
        cond: Cond,
        process: ForwardProcess[T, Cond],
        parameterization: FlowParameterization[T, Cond],
    ) -> T:
        flow = parameterization.output_to_flow(output, x_s, t=s, cond=cond)
        delta = jax.tree.map(lambda v: (t - s) * v, flow)
        x_t = jax.tree.map(lambda x, d: x + d, x_s, delta)
        return x_t

    @functools.partial(jax.jit, static_argnames=("nfe", "path"))
    def integrate(
        self,
        key: jax.Array,
        x_s: T,
        /,
        s: jax.Array,
        t: jax.Array,
        *,
        diffuser: Diffuser[T, Cond],
        process: ForwardProcess[T, Cond],
        cond: Cond,
        nfe: int,
        path: bool = False,
    ) -> T:

        def integrate(x, inputs):
            s, t = inputs
            output = diffuser(x, cond=cond, t=s)
            next = self.step(
                x, output,
                s=s, t=t,
                process=process,
                parameterization=diffuser.parameterization,
                cond=cond,
            ) # fmt: skip
            return next, x

        times = jnp.linspace(s, t, nfe + 1)
        s, t = times[:-1], times[1:]
        x_t, x_traj = jax.lax.scan(integrate, x_s, (s, t))
        x_traj = jax.tree.map(
            lambda traj, f: jnp.concatenate((traj, f[jnp.newaxis]), axis=0), x_traj, x_t
        )
        return x_traj if path else x_t


class GeneralizedDDPM[T, Cond](Integrator[T, Cond]):
    def __init__(self, gamma: float, mu: float):
        self.gamma = gamma
        self.mu = mu

    def step(
        self,
        key: jax.Array,
        x_s: T,
        output: T,
        eps_avg: T | None,
        /,
        s: jax.Array,
        t: jax.Array,
        *,
        cond: Cond,
        process: ForwardProcess[T, Cond],
        parameterization: FlowParameterization[T, Cond],
    ) -> tuple[T, T]:
        assert isinstance(process, NoisingForwardProcess)
        schedule = process.schedule
        eps_param = schedule.parameterize(0.0, 1.0)

        flow = parameterization.output_to_flow(output, x_s, t=s, cond=cond)
        eps = eps_param.flow_to_output(flow, x_s, t=s, cond=cond)
        eps = jax.tree.map(
            lambda e, ea: self.gamma * e + (1 - self.gamma) * ea, eps, eps_avg
        ) if eps_avg is not None else eps # fmt: skip
        asserts.graphs_equal_shapes_and_dtypes(x_s, eps)
        #
        sig = schedule.sigma(s) / schedule.alpha(s)
        sig_prev = schedule.sigma(t) / schedule.alpha(t)
        sig_p = (sig_prev / sig**self.mu) ** (1 / (1 - self.mu))  # sig_prev == sig**mu sig_p**(1-mu) # fmt: skip
        eta = jnp.sqrt(sig_prev**2 - sig_p**2)
        #
        x_flat, x_uf = graph_util.ravel(x_s)
        noise = x_uf(jax.random.normal(key, shape=x_flat.shape))
        # raw equations:
        # x = x / process.schedule.alpha(s)
        # x = x - (sig - sig_p) * eps_a + eta * noise
        # x = x * process.schedule.alpha(t)
        alpha_ratio = schedule.alpha(t) / schedule.alpha(s)
        x_t = jax.tree.map(
            lambda x, eps, noise: (
                alpha_ratio * x
                + ((sig_p - sig) * eps + eta * noise) * schedule.alpha(t)
            ),
            x_s,
            eps,
            noise,
        )
        return x_t, eps

    def integrate(
        self,
        key: jax.Array,
        x_s: T,
        /,
        s: jax.Array,
        t: jax.Array,
        *,
        diffuser: Diffuser[T, Cond],
        process: ForwardProcess[T, Cond],
        cond: Cond,
        nfe: int,
        path: bool = False,
    ) -> T:
        times = jnp.linspace(s, t, nfe + 1)
        s, t = times[:-1], times[1:]
        keys = jax.random.split(key, nfe)
        # do a first step to populate eps_avg
        output = diffuser(x_s, cond=cond, t=s[0])

        x_first, eps_avg = self.step(
            keys[0], x_s, output, None,
            s=s[0], t=t[0],
            process=process,
            parameterization=diffuser.parameterization,
            cond=cond
        ) # fmt: skip

        # do the rest
        def integrate(carry, input):
            key, s, t = input
            x, eps_avg = carry
            output = diffuser(x, cond=cond, t=s)
            next, eps_avg = self.step(
                key, x, output, eps_avg,
                s=s, t=t,
                process=process,
                parameterization=diffuser.parameterization,
                cond=cond,
            ) # fmt: skip
            return (next, eps_avg), next

        (x_t, eps_avg), traj = jax.lax.scan(
            integrate, (x_first, eps_avg), (keys[1:], s[1:], t[1:])
        )
        if path:
            traj = jax.tree.map(
                lambda i, f, t: jnp.concatenate((i[None], f[None], t), axis=0),
                x_s,
                x_first,
                traj,
            )
            return traj
        else:
            return x_t


class DDIM(GeneralizedDDPM):
    def __init__(self):
        super().__init__(gamma=1.0, mu=0.0)


class DDPM(GeneralizedDDPM):
    def __init__(self):
        super().__init__(gamma=1.0, mu=0.5)


class AccelDDIM(GeneralizedDDPM):
    def __init__(self):
        super().__init__(gamma=2.0, mu=0.0)
