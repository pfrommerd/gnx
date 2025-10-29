import math
import functools
import typing as tp


import jax
import jax.numpy as jnp

from ..core import graph, graph_util, nn, filters
from ..util import datasource, optimizers

from ..util.datasource import DataSource
from ..util.distribution import Distribution, Gaussian, Uniform
from ..util.trainer import Metrics, Trainer, Plugin, ModelTrainState
from ..util.trainer.objectives import TrackModel, Minimize

from .noise_schedule import NoiseSchedule
from . import GenerativeModel, TrainSample

type Time = jax.Array
type TimeLike = jax.typing.ArrayLike


class ForwardProcess[T, Cond = None](tp.Protocol):
    @property
    def approximate_final(self) -> Distribution[T]: ...

    def forward(self, key: jax.Array, x_0: T, /, t: Time, *, cond: Cond) -> T: ...

    def forward_and_flow(
        self, key: jax.Array, x_0: T, /, t: Time, *, cond: Cond
    ) -> tuple[T, T]: ...


class NoisingForwardProcess[T, Cond](graph.Object):
    # takes a noise schedule and an instance
    def __init__(self, schedule: NoiseSchedule, instance: T):
        self.schedule = schedule
        final_sigma = schedule.sigma(1.0)
        mean = jax.tree.map(jnp.zeros_like, instance)
        std = jax.tree.map(lambda x: final_sigma * jnp.ones_like(x), instance)
        self._approximate_final = Gaussian(mean, std)

    @property
    def approximate_final(self) -> Distribution[T]:
        return self._approximate_final

    def forward(self, key: jax.Array, x_0: T, /, t: Time, *, cond: Cond) -> T:
        return self.schedule.forward(key, x_0, t)

    def forward_and_flow(
        self, key: jax.Array, x_0: T, /, t: Time, *, cond: Cond
    ) -> tuple[T, T]:
        return self.schedule.forward_and_flow(key, x_0, t)


class FlowParameterization[T, Cond](tp.Protocol):
    def output_to_flow(self, output: T, x_t: T, /, t: Time, *, cond: Cond) -> T: ...
    def flow_to_output(self, flow: T, x_t: T, /, t: Time, *, cond: Cond) -> T: ...


class IdentityFlowParam[T, Cond](graph.Object):
    def output_to_flow(self, output: T, x_t: T, /, t: Time, *, cond: Cond) -> T:
        return output

    def flow_to_output(self, flow: T, x_t: T, /, t: Time, *, cond: Cond) -> T:
        return flow


class Diffuser[T, Cond = None](tp.Protocol):
    @property
    def parameterization(self) -> FlowParameterization[T, Cond]: ...

    # Will predict some parameterization of the flow
    def __call__(self, x_t: T, /, t: Time, *, cond: Cond) -> T: ...


# Compute the ideal denoiser, the diffuser that would perfectly denoise the forward process
# for a given noise schedule.
class IdealDiffuser[T](graph.Object):
    def __init__(
        self,
        distribution: Distribution[T],
        schedule: NoiseSchedule,
        parameterization: FlowParameterization[T, None],
    ):
        self.distribution = distribution
        self.schedule = schedule
        self.parameterization = parameterization

    @jax.jit
    def __call__(self, x_t: T, /, t: jax.Array, *, cond: None = None) -> T:
        t_distribution = self.schedule.transform(self.distribution, t)
        score = t_distribution.score(x_t)
        # more numerically stable way of computing dot(alpha)/alpha
        # as a log-linear schedule will have alpha(t) = exp(c t)
        alpha_dot_alpha = jax.grad(lambda t: jnp.log(self.schedule.alpha(t)))(t)
        sigma, sigma_dot = jax.value_and_grad(self.schedule.sigma)(t)
        gamma_1 = alpha_dot_alpha * sigma * sigma - sigma * sigma_dot
        gamma_2 = alpha_dot_alpha
        v = jax.tree.map(lambda x_t, score: gamma_1 * score + gamma_2 * x_t, x_t, score)
        return self.parameterization.flow_to_output(v, x_t, t=t, cond=cond)


# A forward process with a diffuser for computing the reverse flow
class DistilledForwardProcess[T, Cond](graph.Object):
    def __init__(
        self,
        forward_process: ForwardProcess[T, Cond],
        diffuser: Diffuser[T, Cond],
    ):
        self.forward_process = forward_process
        self.diffuser = diffuser

    @property
    def approximate_final(self) -> Distribution[T]:
        return self.forward_process.approximate_final

    def forward(self, key: jax.Array, x_0: T, /, t: jax.Array, *, cond: Cond) -> T:
        return self.forward_process.forward(key, x_0, t=t, cond=cond)

    def forward_and_flow(
        self, key: jax.Array, x_0: T, /, t: jax.Array, *, cond: Cond
    ) -> tuple[T, T]:
        x_t = self.forward(key, x_0, t=t, cond=cond)
        output = self.diffuser(x_t, t=t, cond=cond)
        v_t = self.diffuser.parameterization.output_to_flow(output, x_t, t=t, cond=cond)
        return x_t, v_t


class Integrator[T, Cond](graph.Object):
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
        raise NotImplementedError()


# --- The core diffusion model class ---
class DiffusionModel[T, Cond](GenerativeModel[T, Cond], nn.Module):
    def __init__(
        self,
        diffuser: Diffuser[T, Cond],
        forward_process: ForwardProcess[T, Cond],
        *,
        integrator: Integrator[T, Cond],
        nfe: int,
    ):
        self.diffuser = diffuser
        self.forward_process = forward_process
        self.integrator = integrator
        self.nfe = nfe

    @functools.partial(jax.jit, static_argnames=("nfe", "path"))
    def _sample(
        self,
        key: jax.Array,
        cond: Cond,
        *,
        path: bool = False,
        nfe: int | None = None,
        initial: T | None = None,
        start_time: jax.Array | None = None,
        final_time: jax.Array | None = None,
        integrator: Integrator[T, Cond] | None = None,
    ) -> T:
        s = jnp.ones(()) if start_time is None else start_time
        t = jnp.zeros(()) if final_time is None else final_time
        nfe = nfe or self.nfe
        integrator = integrator or self.integrator
        assert nfe > 0, "nfe must be positive"
        init_key, n_key = jax.random.split(key, 2)
        initial = (
            self.forward_process.approximate_final.sample(init_key)
            if initial is None
            else initial
        )
        sample = integrator.integrate(
            n_key,
            initial,
            s,
            t,
            diffuser=self.diffuser,
            process=self.forward_process,
            cond=cond,
            nfe=nfe,
            path=path,
        )
        return sample

    @functools.partial(jax.jit, static_argnames=("shape", "nfe", "path"))
    def sample(
        self,
        key: jax.Array,
        shape: tuple[int, ...] = (),
        *,
        cond: Cond,
        #
        path: bool = False,
        initial: T | None = None,
        # Sample an additional time step somewhere in the path
        start_time: jax.Array | None = None,
        final_time: jax.Array | None = None,
        #
        nfe: int | None = None,
        integrator: Integrator[T, Cond] | None = None,
    ) -> T:
        N = math.prod(shape)
        keys = jax.random.split(key, N)
        samples = jax.vmap(
            lambda self, initial, k: self._sample(
                k,
                cond,
                initial=initial,
                nfe=nfe,
                integrator=integrator,
                start_time=start_time,
                final_time=final_time,
                path=path,
            ),
            in_axes=(None, 0, 0),
        )(self, initial, keys)
        samples = jax.tree.map(lambda x: jnp.reshape(x, shape + x.shape[1:]), samples)
        return samples

    @functools.partial(jax.jit, static_argnames=("reference_loss",))
    def loss(
        self,
        data: tuple[jax.Array, TrainSample[T, Cond]],
        *,
        time_distribution: Distribution[Time] | None = None,
        reference_diffuser: Diffuser[T, Cond] | None = None,
        reference_loss: bool = False,
    ) -> tuple[jax.Array, Metrics]:
        key, batch = data
        keys = jax.random.split(key, graph_util.axis_size(batch))
        time_distribution = (
            time_distribution
            if time_distribution is not None
            else Uniform(min=jnp.zeros(()), max=jnp.ones(()))
        )

        @jax.vmap
        def sample_loss(
            key,
            sample: TrainSample[T, Cond],
        ):
            t_key, x_key = jax.random.split(key)
            t = time_distribution.sample(t_key)
            # Take the gradient of the forward process wrt t to get the stochastic interpolant
            xt, xt_dot = self.forward_process.forward_and_flow(
                x_key, sample.value, t=t, cond=sample.cond
            )
            ref_pred = (
                reference_diffuser(xt, cond=sample.cond, t=t)
                if reference_diffuser is not None
                else None
            )
            target = self.diffuser.parameterization.flow_to_output(
                xt_dot, xt, t=t, cond=sample.cond
            )
            if reference_loss:
                assert ref_pred is not None
                target = ref_pred
            pred = self.diffuser(xt, cond=sample.cond, t=t)
            mse = graph_util.mse(pred, target)
            losses = {"loss": mse}
            if reference_diffuser is not None:
                assert ref_pred is not None
                losses["ref_error"] = (
                    mse if reference_loss else graph_util.mse(pred, ref_pred)
                )
            return losses

        losses = sample_loss(keys, batch)
        losses = jax.tree.map(jnp.mean, losses)
        return losses["loss"], losses

    def trainer(
        self,
        data: DataSource[TrainSample[T, Cond]],
        *,
        time_distribution: Distribution[Time] | None = None,
        reference_diffuser: Diffuser[T, Cond] | None = None,
        # use the reference diffuser as the target for the loss
        reference_loss: bool = False,
        # Marked as optional, but must be provided!
        optimizer: optimizers.Optimizer,
        model_tracker: optimizers.ModelTracker | None = None,
        batch_size: int,
        # trainer arguments
        plugins: tp.Sequence[Plugin] = (),
        logging_plugins: tp.Sequence[Plugin] = (),
        iterations: int,
        shuffle_rng: nn.RngStream,
        **kwargs,
    ) -> Trainer[ModelTrainState[tp.Self, tp.Any], tp.Any]:
        assert not kwargs, "DiffusionModel.trainer does not accept additional arguments"

        combined = datasource.zip(
            datasource.rng(),
            data.batch((batch_size,)),
        )
        objective = Minimize(
            self,
            optimizer,
            wrt=filters.All(nn.Param, filters.PathPrefix("diffuser")),
            time_distribution=time_distribution,
            reference_diffuser=reference_diffuser,
            reference_loss=reference_loss,
        )
        objective = TrackModel(
            self,
            objective,
            tracker=model_tracker,
            track_wrt=filters.All(nn.Param, filters.PathPrefix("diffuser")),
        )
        return Trainer(
            objective,
            data=combined,
            shuffle_rng=shuffle_rng,
            plugins=plugins,
            logging_plugins=logging_plugins,
            iterations=iterations,
        )
