import logging
import typing as tp

import jax

from ..core import nn
from ..core.dataclasses import dataclass
from ..util import optimizers, cli

from ..methods.noise_schedule import NoiseSchedule
from ..methods.diffusion import NoisingForwardProcess
from ..methods.ot.generator import OTGeneratorModel
from ..methods.ot.sinkhorn import SinkhornOTSamplerFactory

from ..datasets import Visualizable, Dataset
from ..models import GeneratorFactory
from . import common

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class SinkhornMapConfig(common.CommonConfig):
    method: str = "sinkhorn_map"

    model: GeneratorFactory
    #
    rel_epsilon: float
    ot_batch_size: int
    ot_batches: int
    #
    visualize_pairs: bool
    #
    batch_size: int
    iterations: int
    optimizer: optimizers.Optimizer
    model_tracker: optimizers.ModelTracker | None


# fmt: off
@cli.option("--sinkhorn_flow.nfe", "nfe", type=int, default=1)
@cli.option("--sinkhorn_flow.visualize_pairs", "visualize_pairs", type=bool, default=True)
@cli.option("--sinkhorn_flow.discrete_times", "discrete_times", type=bool, default=False)
@cli.option("--sinkhorn_flow.mean_flow", "mean_flow", type=bool, default=False)
@cli.option("--sinkhorn_flow.rel_epsilon", "rel_epsilon", type=float, default=0.05)
@cli.option("--sinkhorn_flow.ot_batches", "ot_batches", type=int, default=1)
@cli.option("--sinkhorn_flow.ot_batch_size", "ot_batch_size", type=int, default=256)
@common.model.flow_map_options("--model", "model")
@cli.option("--train.batch_size", "batch_size", type=int, default=256)
@cli.option("--train.iterations", "iterations", type=int, default=20_000)
@common.optimizer.options("--train", "optimizer")
@common.model_tracker.options("--train.tracker", "model_tracker")
@cli.group
# fmt: on
def sinkhorn_flow_options(ctx: cli.Context): ...


@sinkhorn_flow_options
@common.config_options(SinkhornMapConfig)
@cli.command
def train[T: Visualizable](config: SinkhornMapConfig):
    experiment = config.create_experiment("gnx.main.sinkhorn_flow:train")
    rngs = nn.Rngs(config.seed)
    raw_dataset = config.create_dataset()
    dataset: Dataset[T] = config.preprocess_dataset(raw_dataset, rngs)

    generator = config.model.create_generator(
        dataset.instance.value, dataset.instance.cond, rngs=rngs
    )

    model = OTGeneratorModel(generator)
    config.initialize_model(experiment, model)
    model.train_mode()

    # Training options:

    ot_factory = SinkhornOTSamplerFactory[T, tp.Any, None](
        ot_batch_size=config.ot_batch_size,
        ot_batches=config.ot_batches,
        rel_epsilon=config.rel_epsilon,
    )
    plugins = config.create_train_plugins(raw_dataset, dataset, rngs=rngs)
    logging_plugins = config.create_logging_plugins(logger, experiment, rngs=rngs)
    trainer = model.trainer(
        dataset.splits["train"],
        shuffle_rng=rngs.data,
        solver=ot_factory,
        optimizer=config.optimizer,
        model_tracker=config.model_tracker,
        batch_size=config.batch_size,
        iterations=config.iterations,
        plugins=plugins,
        logging_plugins=logging_plugins,
    )

    config.resume_training(experiment, trainer)
