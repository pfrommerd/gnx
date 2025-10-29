import logging
import typing as tp

import jax
import jax.numpy as jnp

from ..core import nn
from ..core.dataclasses import dataclass
from ..util import optimizers, cli

from ..methods.noise_schedule import NoiseSchedule
from ..methods.diffusion import NoisingForwardProcess
from ..methods.mean_flow import MeanFlowMap
from ..methods.flow_map import EndpointTimePairs, ReverseTimePairs, UniformTimePairs
from ..methods.ot import OTSamplerFactory
from ..methods.ot.flow_map import OTFlowMapModel
from ..methods.ot.sinkhorn import SinkhornOTSamplerFactory

from ..datasets import Visualizable, Dataset
from ..models import FlowMapFactory
from . import common

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class SinkhornFlowConfig(common.CommonConfig):
    method: str = "sinkhorn_flow"

    model: FlowMapFactory
    schedule: NoiseSchedule
    #
    aux_features: int
    rel_epsilon: float
    ot_batch_size: int
    ot_batches: int
    discrete_times: bool
    mean_flow: bool
    nfe: int
    #
    visualize_pairs: bool
    #
    batch_size: int
    iterations: int
    optimizer: optimizers.Optimizer
    model_tracker: optimizers.ModelTracker | None


# fmt: off
@cli.option("--sinkhorn_flow.aux_features", "aux_features", type=int, default=0)
@cli.option("--sinkhorn_flow.nfe", "nfe", type=int, default=1)
@cli.option("--sinkhorn_flow.visualize_pairs", "visualize_pairs", type=bool, default=True)
@cli.option("--sinkhorn_flow.discrete_times", "discrete_times", type=bool, default=False)
@cli.option("--sinkhorn_flow.mean_flow", "mean_flow", type=bool, default=False)
@cli.option("--sinkhorn_flow.rel_epsilon", "rel_epsilon", type=float, default=0.05)
@cli.option("--sinkhorn_flow.ot_batches", "ot_batches", type=int, default=1)
@cli.option("--sinkhorn_flow.ot_batch_size", "ot_batch_size", type=int, default=256)
@common.model.flow_map_options("--model", "model")
@common.noise_schedule.options("--sinkhorn_flow.schedule", "schedule")
@common.optimizer.options("--train", "optimizer")
@common.model_tracker.options("--train.tracker", "model_tracker")
@cli.option("--train.batch_size", "batch_size", type=int, default=256)
@cli.option("--train.iterations", "iterations", type=int, default=20_000)
@cli.group
# fmt: on
def sinkhorn_flow_options(ctx: cli.Context): ...


@sinkhorn_flow_options
@common.config_options(SinkhornFlowConfig)
@cli.command
def train[T: Visualizable](config: SinkhornFlowConfig):
    experiment = config.create_experiment("gnx.main.sinkhorn_flow:train")
    rngs = nn.Rngs(config.seed)
    raw_dataset = config.create_dataset()
    dataset: Dataset[T] = config.preprocess_dataset(raw_dataset, rngs)

    flow_map = config.model.create_flow_map(
        dataset.instance.value, dataset.instance.cond, None, rngs=rngs
    )
    if config.mean_flow:
        flow_map = MeanFlowMap(flow_map)

    logger.info(f"Model has {nn.num_params(flow_map)} parameters.")
    forward_process = NoisingForwardProcess(config.schedule, dataset.instance.value)
    model = OTFlowMapModel(flow_map=flow_map, forward_process=forward_process)
    config.initialize_model(experiment, model)
    model.train_mode()

    # Training options:

    ot_factory: OTSamplerFactory[T, None] = SinkhornOTSamplerFactory[T, tp.Any, None](
        ot_batch_size=config.ot_batch_size,
        ot_batches=config.ot_batches,
        rel_epsilon=config.rel_epsilon,
    )
    time_dist = ReverseTimePairs(
        EndpointTimePairs() if config.discrete_times else UniformTimePairs()
    )

    plugins = config.create_train_plugins(raw_dataset, dataset, rngs=rngs)
    logging_plugins = config.create_logging_plugins(logger, experiment, rngs=rngs)
    trainer = model.trainer(
        dataset.splits["train"],
        shuffle_rng=rngs.data,
        solver=ot_factory,
        time_distribution=time_dist,
        optimizer=config.optimizer,
        model_tracker=config.model_tracker,
        batch_size=config.batch_size,
        iterations=config.iterations,
        plugins=plugins,
        logging_plugins=logging_plugins,
    )

    config.resume_training(experiment, trainer)
