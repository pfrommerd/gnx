import logging
import typing as tp

import jax
import jax.numpy as jnp

from ..core import nn
from ..core.dataclasses import dataclass
from ..util import optimizers, datasource, cli

from ..methods.noise_schedule import NoiseSchedule
from ..methods.flow_map import (
    DistilledFlowMapModel,
    EndpointTimePairs,
    FlowMapDistillSample,
    ReverseTimePairs,
    UniformTimePairs,
)
from ..methods.mean_flow import MeanFlowMap
from ..methods.diffusion import Diffuser, IdentityFlowParam, NoisingForwardProcess
from ..methods.integrators import DDIM

from ..datasets import Visualizable, TrainSample, Dataset
from ..models import FlowMapFactory
from . import common

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class DistilledFlowConfig(common.CommonConfig):
    method: str = "distilled_flow"

    model: FlowMapFactory
    eval_schedule: NoiseSchedule
    train_schedule: NoiseSchedule | None

    nfe: int
    #
    mean_flow: bool
    discrete_times: bool
    integrate_initial: bool
    forward_nfe: int
    #
    batch_size: int
    iterations: int
    optimizer: optimizers.Optimizer
    model_tracker: optimizers.ModelTracker | None


# fmt: off
@cli.option("--distilled_flow.forward_nfe", "forward_nfe", type=int, default=30)
@cli.option("--distilled_flow.nfe", "nfe", type=int, default=1)
@cli.option("--distilled_flow.integrate_initial", "integrate_initial", type=bool, default=False)
@cli.option("--distilled_flow.discrete_times", "discrete_times", type=bool, default=False)
@common.noise_schedule.options("--distilled_flow.eval_schedule", "eval_schedule")
@common.noise_schedule.options("--distilled_flow.train_schedule", "train_schedule", default=None)
@cli.option("--distilled_flow.mean_flow", "mean_flow", type=bool, default=False)
@common.model.flow_map_options("--model", "model")
@cli.option("--train.batch_size", "batch_size", type=int, default=256)
@cli.option("--train.iterations", "iterations", type=int, default=20_000)
@common.optimizer.options("--train", "optimizer")
@common.model_tracker.options("--train.tracker", "model_tracker")
@cli.group
# fmt: on
def distilled_flow_options(ctx: cli.Context): ...


@distilled_flow_options
@common.config_options(DistilledFlowConfig)
@cli.command
def train[T: Visualizable, Cond](config: DistilledFlowConfig):
    experiment = config.create_experiment("gnx.main.distilled_flow:train")
    rngs = nn.Rngs(config.seed)
    raw_dataset = config.create_dataset()
    dataset: Dataset[T, Cond] = config.preprocess_dataset(raw_dataset, rngs)

    time_dist = ReverseTimePairs(
        EndpointTimePairs() if config.discrete_times else UniformTimePairs()
    )
    forward_process = NoisingForwardProcess(
        config.eval_schedule, dataset.instance.value
    )
    gt_integrator = DDIM()
    gt_diffuser: Diffuser[T, Cond] = None  # type: ignore
    assert False
    flow_map = config.model.create_flow_map(
        value=dataset.instance.value,
        cond=dataset.instance.cond,
        aux=None,
        rngs=rngs,
    )
    if config.mean_flow:
        flow_map = MeanFlowMap(flow_map)

    model = DistilledFlowMapModel(
        flow_map=flow_map,
        forward_process=forward_process,
        flow_aux_data=datasource.none(),
    )
    config.initialize_model(experiment, model)
    model.train_mode()

    def transform(
        key, sample: TrainSample[T, Cond]
    ) -> FlowMapDistillSample[T, Cond, None]:
        t_key, n_key, i_key = jax.random.split(key, 3)
        s, t = time_dist.sample(t_key)  # t <= s
        # noise to time t (the smaller timestep), then integrate
        # *forwards* from t to s
        if config.integrate_initial:
            x_t = gt_integrator.integrate(
                n_key,
                sample.value,
                nfe=config.forward_nfe,
                s=jnp.zeros(()),
                t=t,
                diffuser=gt_diffuser,
                process=forward_process,
                cond=sample.cond,
            )
        else:
            x_t = forward_process.forward(n_key, sample.value, t, cond=sample.cond)
        x_s = gt_integrator.integrate(
            i_key,
            x_t,
            nfe=config.forward_nfe,
            s=t,
            t=s,
            diffuser=gt_diffuser,
            process=forward_process,
            cond=sample.cond,
        )
        return FlowMapDistillSample(
            x_s=x_s,
            x_t=x_t,
            s=s,
            t=t,
            cond=sample.cond,
            aux=None,
        )

    data = dataset.splits["train"].map(transform)

    trainer = model.trainer(
        data,
        shuffle_rng=rngs.data,
        optimizer=config.optimizer,
        model_tracker=config.model_tracker,
        batch_size=config.batch_size,
        iterations=config.iterations,
        plugins=config.create_train_plugins(raw_dataset, dataset, rngs=rngs),
        logging_plugins=config.create_logging_plugins(logger, experiment, rngs=rngs),
    )
    config.resume_training(experiment, trainer)
