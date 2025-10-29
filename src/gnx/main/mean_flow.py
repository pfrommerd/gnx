import logging

from ..core import nn
from ..core.dataclasses import dataclass
from ..util import optimizers, cli

from ..methods import Visualizable
from ..methods.diffusion import (
    ForwardProcess,
    NoisingForwardProcess,
)

from ..methods.noise_schedule import NoiseSchedule
from ..methods.flow_map import ReverseTimePairs, UniformTimePairs
from ..methods.mean_flow import MeanFlowModel
from ..models import FlowMapFactory, FlowMap

from ..datasets import Dataset

from . import common

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class MeanFlowConfig(common.CommonConfig):
    method: str = "mean_flow"

    model: FlowMapFactory
    eval_schedule: NoiseSchedule
    train_schedule: NoiseSchedule
    nfe: int
    stopgrad: bool
    implicit_forward: bool

    batch_size: int
    iterations: int
    optimizer: optimizers.Optimizer
    model_tracker: optimizers.ModelTracker | None


@cli.option("--mean_flow.nfe", "nfe", type=int, default=1)
@cli.option("--mean_flow.stopgrad", "stopgrad", type=bool, default=True)
@cli.option(
    "--mean_flow.implicit_forward", "implicit_forward", type=bool, default=False
)
@common.noise_schedule.options("--mean_flow.eval_schedule", "eval_schedule")
@common.noise_schedule.options(
    "--mean_flow.train_schedule", "train_schedule", default=None
)
@common.model.flow_map_options("--model", "model")
@cli.option("--train.batch_size", "batch_size", type=int, default=64)
@cli.option("--train.iterations", "iterations", type=int, default=20000)
@common.optimizer.options("--train", "optimizer")
@common.model_tracker.options("--train.tracker", "model_tracker")
@cli.group
def mean_flow_options(ctx: cli.Context):
    pass


@mean_flow_options
@common.config_options(MeanFlowConfig)
@cli.command
def train[T: Visualizable, Cond](config: MeanFlowConfig):
    experiment = config.create_experiment("gnx.main.mean_flow:train")
    rngs = nn.Rngs(config.seed)
    raw_dataset = config.create_dataset()
    dataset: Dataset[T, Cond] = config.preprocess_dataset(raw_dataset, rngs)

    mean_flow: FlowMap[T, Cond] = config.model.create_flow_map(
        dataset.instance.value, dataset.instance.cond, None, rngs=rngs
    )
    forward_process: ForwardProcess[T, Cond] = NoisingForwardProcess(
        config.eval_schedule,
        dataset.instance.value,
    )

    logger.info(f"Model has {nn.num_params(mean_flow)} parameters.")
    model = MeanFlowModel(
        mean_flow=mean_flow,
        forward_process=forward_process,
        nfe=config.nfe,
    )
    config.initialize_model(experiment, model)
    model.train_mode()

    time_distribution = UniformTimePairs()

    plugins = config.create_train_plugins(raw_dataset, dataset, rngs=rngs)
    logging_plugins = config.create_logging_plugins(logger, experiment, rngs=rngs)
    trainer = model.trainer(
        dataset.splits["train"],
        stopgrad=config.stopgrad,
        implicit_forward=config.implicit_forward,
        time_distribution=time_distribution,
        optimizer=config.optimizer,
        model_tracker=config.model_tracker,
        batch_size=config.batch_size,
        shuffle_rng=rngs.data,
        plugins=plugins,
        logging_plugins=logging_plugins,
        iterations=config.iterations,
    )
    config.resume_training(experiment, trainer)
