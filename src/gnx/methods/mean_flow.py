import typing as tp
import functools

import jax
import jax.numpy as jnp

from ..core import graph_util, nn, filters
from ..util import datasource, optimizers

from ..util.datasource import DataSource
from ..util.distribution import Distribution, Uniform
from ..util.trainer import Metrics, Plugin, Trainer, ModelTrainState
from ..util.trainer.objectives import TrackModel, Minimize
from . import TrainSample

from .diffusion import ForwardProcess, Diffuser, Integrator, Time, TimeLike
from .flow_map import FlowMap, FlowMapModel, UniformTimePairs


# Wraps a FlowMap to be used as a mean flow
# and exposes a FlowMap interface which uses the integration
class MeanFlowMap[T, Cond](nn.Module):
    def __init__(self, mean_flow: FlowMap[T, Cond, None]):
        self.mean_flow = mean_flow

    def __call__(
        self, x_s: T, /, s: TimeLike, t: TimeLike, *, cond: Cond, aux: None
    ) -> T:
        s, t = jnp.asarray(s), jnp.asarray(t)
        # Use the mean flow to compute the flow from s to t
        u_s = self.mean_flow(x_s, s=s, t=t, cond=cond, aux=aux)
        x_t = jax.tree.map(lambda x_s, u_s: x_s + (t - s) * u_s, x_s, u_s)
        return x_t


class IdealMeanFlow[T, Cond](nn.Module):
    def __init__(
        self,
        diffuser: Diffuser[T, Cond],
        forward_process: ForwardProcess[T, Cond],
        integrator: Integrator,
        nfe: int,
    ):
        self.forward_process = forward_process
        self.diffuser = diffuser
        self.integrator = integrator
        self.nfe = nfe

    def __call__(self, x_s: T, /, s: Time, t: Time, *, cond: Cond, aux: None):
        x_t = self.integrator.integrate(
            jax.random.key(42),
            x_s,
            s=s,
            t=t,
            cond=cond,
            process=self.forward_process,
            diffuser=self.diffuser,
            nfe=self.nfe,
        )
        return jax.tree.map(lambda x_t, x_s: (x_t - x_s) / (t - s), x_t, x_s)


class MeanFlowModel[T, Cond](FlowMapModel[T, Cond, None]):
    flow_map: MeanFlowMap[T, Cond]

    def __init__(
        self,
        mean_flow: FlowMap[T, Cond],
        forward_process: ForwardProcess[T, Cond],
        nfe: int = 1,
    ):
        super().__init__(
            flow_map=MeanFlowMap(mean_flow),
            forward_process=forward_process,
            flow_aux_data=datasource.none(),
            nfe=nfe,
        )

    @functools.partial(jax.jit, static_argnames=("stopgrad", "implicit_forward"))
    def loss(
        self,
        data: tuple[jax.Array, TrainSample[T, Cond]],
        *,
        stopgrad: bool = True,
        implicit_forward: bool = False,
        time_distribution: Distribution[tuple[Time, Time]] | None = None,
    ) -> tuple[jax.Array, Metrics]:
        key, batch = data
        if time_distribution is None:
            time_distribution = UniformTimePairs()

        # per-sample loss function
        @jax.vmap
        def sample_loss(key: jax.Array, x: T, cond: Cond):
            time_key, fwd_key = jax.random.split(key, 2)
            s, t = time_distribution.sample(time_key)
            # Note: the original meanflow paper uses
            # "r" for our target timestep "t"
            # and "t" for our starting timestep "s"
            # We use x_t = x_s + integral_s^t v_tau dtau
            # Thus x_t - x_s = integral_s^t v_tau dtau
            # or (t - s) * u(x_s, s, t) = integral_s^t v_tau dtau

            if implicit_forward:
                # Differentiating both sides with respect to t
                #  u(x_s, s, t) + (t - s) du(x_s, s, t)/dt = d/dt integral_s^t v_tau(x_tau, tau) dtau
                #  u(x_s, s, t) + (t - s) du(x_s, s, t)/dt = v(x_t, t)
                #  where x_s = x_s + (t - s) u(x_s, s, t)
                x_t, v_t = self.forward_process.forward_and_flow(
                    fwd_key, x, t, cond=cond
                )
                u_t = self.flow_map.mean_flow(x_t, s=t, t=s, cond=cond, aux=None)
                x_s = jax.tree.map(lambda x_t, u_t: x_t + (t - s) * u_t, x_t, u_t)
                u_s, du_s_dt = jax.jvp(
                    lambda t: self.flow_map.mean_flow(
                        x_s, s=s, t=t, cond=cond, aux=None
                    ),
                    (t,), (jnp.ones(()),), # fmt: skip
                )
                u_tgt = jax.tree.map(lambda v_t, du: v_t - (t - s) * du, v_t, du_s_dt)
                # stop the gradient through du_s/ds to avoid second derivatives
                if stopgrad:
                    u_tgt = jax.lax.stop_gradient(u_tgt)
                mse = graph_util.mse(u_tgt, u_s)
                return {"mse": mse}
            else:
                # Vanilla meanflow loss
                # Differentiating both sides with respect to s
                # and using u = u(x_s, s, t)
                #  - u + (t - s) du_ds = d/ds integral_s^t v_tau dtau
                #  - u + (t - s) du_ds = -v(x_s, s)
                # u = v(x_s, s) + (t - s) du_ds

                x_s, v_s = self.forward_process.forward_and_flow(
                    fwd_key, x, s, cond=cond
                )
                ds_ds, dt_ds = jnp.ones(()), jnp.zeros(())

                u_pred, du_ds = jax.jvp(
                    lambda x_s, s, t: self.flow_map.mean_flow(x_s, s=s, t=t, cond=cond, aux=None),
                    (x_s, s, t), (v_s, ds_ds, dt_ds),
                )  # fmt: skip
                u_tgt = jax.tree.map(
                    lambda v_s, du_ds: v_s + (t - s) * du_ds, v_s, du_ds
                )
                if stopgrad:
                    u_tgt = jax.lax.stop_gradient(u_tgt)

                mse = graph_util.mse(u_tgt, u_pred)
                return {"mse": mse}

        N = graph_util.axis_size(batch.value, 0)
        losses = sample_loss(
            jax.random.split(key, N),
            batch.value,
            batch.cond,
        )
        losses = jax.tree.map(jnp.mean, losses)
        return losses["mse"], losses

    def trainer(
        self,
        data: DataSource[TrainSample[T, Cond]],
        *,
        stopgrad: bool = True,
        implicit_forward: bool = False,
        time_distribution: Distribution[tuple[Time, Time]] | None,
        optimizer: optimizers.Optimizer,
        model_tracker: optimizers.ModelTracker | None = None,
        batch_size: int,
        #
        shuffle_rng: nn.RngStream | None = None,
        plugins: tp.Sequence[Plugin] = (),
        logging_plugins: tp.Sequence[Plugin] = (),
        iterations: int,
        #
        **kwargs,
    ) -> Trainer[ModelTrainState[tp.Self, tp.Any], tp.Any]:
        assert not kwargs, "No additional arguments expected for MeanFlowModel.trainer"

        combined = datasource.zip(
            datasource.rng(),
            data.batch((batch_size,)),
        )

        wrt = filters.All(nn.Param, filters.PathPrefix("flow_map"))
        objective = Minimize(
            self, optimizer,
            wrt=wrt,
            stopgrad=stopgrad,
            implicit_forward=implicit_forward,
            time_distribution=time_distribution,
        ) # fmt: skip
        objective = TrackModel(
            self, objective,
            tracker=model_tracker, track_wrt=wrt
        ) # fmt: skip
        return Trainer(
            objective,
            data=combined,
            shuffle_rng=shuffle_rng,
            plugins=plugins,
            logging_plugins=logging_plugins,
            iterations=iterations,
        )
