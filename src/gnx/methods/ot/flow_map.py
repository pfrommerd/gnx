import typing as tp
import jax
import jax.numpy as jnp
import optax


from ...core import graph, graph_util, nn
from ...util import datasource, optimizers

from ...util.datasource import DataSource
from ...util.distribution import Distribution, Uniform
from ...util.trainer import Trainer, ModelTrainState
from ...util.trainer.objectives import TrackModel, Minimize, Metrics
from .. import TrainSample
from ..flow_map import FlowMap, FlowMapModel, CouplingCond, ForwardProcess, Time

from . import (
    OTTrainSample,
    OTSamplerFactory,
    OTSample,
    OTCostFn,
    TrainableOTSampler,
)


class FlowOTCost[T, Cond, FlowAux](
    OTCostFn[tuple[T, FlowAux], T, CouplingCond[Cond]], graph.Object
):
    def __init__(
        self,
        flow_map: FlowMap[T, Cond, FlowAux],
    ):
        self.flow = flow_map

    @jax.jit
    def __call__(
        self,
        source_aux: tuple[T, FlowAux],
        target: T,
        /,
        cond: CouplingCond[Cond] | None = None,
    ) -> jax.Array:
        assert cond is not None, "Conditioning must be provided"
        source, aux = source_aux
        source_prop = self.flow(source, s=cond.s, t=cond.t, aux=aux, cond=cond.cond)
        return graph_util.mse(source_prop, target)


# Creates an OTSampler from a DataSource of training samples


# Wraps an OTSolver or a TrainableOTSampler and uses it to train a FlowModel
class OTFlowMapModel[T, Cond, FlowAux](FlowMapModel[T, Cond, FlowAux]):
    @jax.jit
    def loss(
        self, data: OTSample[tuple[T, FlowAux], T, CouplingCond[Cond]]
    ) -> tuple[jax.Array, Metrics]:
        source, aux = data.a
        target = data.b
        s, t, cond = data.cond.s, data.cond.t, data.cond.cond

        @jax.vmap
        def sample_loss(src, aux, tgt, s, t, cond):
            pred = self.flow_map(src, s=s, t=t, aux=aux, cond=cond)
            return graph_util.mse(pred, tgt)

        loss = sample_loss(source, aux, target, s, t, cond)
        loss = jnp.mean(loss)
        return loss, {"mse": loss}

    def trainer(
        self,
        data: DataSource[TrainSample[T, Cond]],
        *,
        shuffle_rng: nn.RngStream,
        solver: (
            TrainableOTSampler[tuple[T, FlowAux], T, CouplingCond[Cond]]
            | OTSamplerFactory[T, Cond, tp.Any, FlowAux]
        ),
        time_distribution: Distribution[tuple[Time, Time]] | None = None,
        model_tracker: optimizers.ModelTracker | None = None,
        #
        iterations: int,
        # more arguments to pass to the solver/trainer
        **kwargs,
    ) -> Trainer[ModelTrainState[tp.Self]]:
        if time_distribution is None:
            time_distribution = Uniform(
                min=(jnp.zeros(()), jnp.zeros(())), max=(jnp.ones(()), jnp.ones(()))
            )

        if not hasattr(solver, "trainer"):
            factory: OTSamplerFactory[T, Cond, tp.Any, FlowAux] = solver  # type: ignore
            optimizer: optimizers.Optimizer | None = kwargs.pop("optimizer", None)
            batch_size: int | None = kwargs.pop("batch_size", None)
            assert optimizer is not None, "Optimizer must be provided for OTSolver"
            assert batch_size is not None, "Batch size must be provided for OTSolver"
            data, ot_sampler = factory.create_flow_sampler(
                data,
                batch_shape=(batch_size,),
                flow_map=self.flow_map,
                forward_process=self.forward_process,
                time_distribution=time_distribution,
                flow_aux_data=self.flow_aux_data,
            )
            ot_data = data.map(lambda _, sample: ot_sampler.sample(sample))
            objective = Minimize(self, optimizer)
            objective = TrackModel(self, objective, tracker=model_tracker)

            return Trainer(
                objective,
                iterations=iterations,
                data=ot_data,
                shuffle_rng=shuffle_rng,
                **kwargs,
            )
        else:
            sampler: TrainableOTSampler[  # type: ignore
                tuple[T, FlowAux], T, CouplingCond[Cond]
            ] = solver  # type: ignore
            combined_data = datasource.zip(self.flow_aux_data, data)

            def process(
                key: jax.Array, data: tuple[FlowAux, TrainSample[T, Cond]]
            ) -> OTTrainSample[tuple[T, FlowAux], T, CouplingCond[Cond]]:
                flow_aux, sample = data
                rngs = nn.Rngs(key)
                s, t = time_distribution.sample(rngs.default())
                # Noise sample.value to times s and t
                xs = self.forward_process.forward(
                    rngs.default(), sample.value, t=s, cond=sample.cond
                )
                xt = self.forward_process.forward(
                    rngs.default(), sample.value, t=t, cond=sample.cond
                )
                return OTTrainSample(
                    a=(xs, flow_aux),
                    b=xt,
                    cond=CouplingCond(s, t, sample.cond),
                )

            proc_data = combined_data.map(process)
            cost_fn = FlowOTCost(self.flow_map)
            trainer = sampler.trainer(
                iterations=iterations,
                data=proc_data,
                cost=cost_fn,
                optimize_wrt_cost=True,
                **kwargs,
            )
            trainer: Trainer[ModelTrainState[tp.Self]] = trainer.replace_state(
                TrackModel(self, trainer.state, tracker=model_tracker)
            )
            return trainer
