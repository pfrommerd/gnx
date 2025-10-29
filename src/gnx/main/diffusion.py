import logging


from ..core import nn
from ..core.dataclasses import dataclass
from ..util.cli.command import UsageError
from ..util import optimizers, cli

from ..datasets import Dataset, Visualizable
from ..methods.diffusion import (
    Diffuser,
    DiffusionModel,
    Integrator,
    FlowParameterization,
    IdentityFlowParam,
    NoisingForwardProcess,
)
from ..methods.noise_schedule import NoiseSchedule, TimeSNRDistribution

from ..models import DiffuserFactory

from . import common

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class DiffusionConfig(common.CommonConfig):
    method: str = "diffusion"

    model: DiffuserFactory
    eval_schedule: NoiseSchedule
    train_schedule: NoiseSchedule | None
    flow_param: FlowParameterization
    use_reference: bool
    show_reference: bool

    integrator: Integrator
    nfe: int

    iterations: int
    batch_size: int

    optimizer: optimizers.Optimizer
    model_tracker: optimizers.ModelTracker | None


# fmt: off
@cli.option("--diffusion.flow_param", "flow_param", type=str, default="epsilon")
@cli.option("--diffusion.use_reference", "use_reference", type=bool, default=False)
@cli.option("--diffusion.show_reference", "show_reference", type=bool, default=False)
@cli.option("--diffusion.nfe", "nfe", type=int, default=32)  # fmt: skip
@common.integrator.options("--diffusion.integrator", "integrator")
@common.noise_schedule.options("--diffusion.eval_schedule", "eval_schedule")
@common.noise_schedule.options("--diffusion.train_schedule", "train_schedule", default=None)
#
@cli.option("--train.iterations", "iterations", type=int, default=20000)
@cli.option("--train.batch_size", "batch_size", type=int, default=256)
@common.optimizer.options("--train", "optimizer")
@common.model_tracker.options("--train.tracker", "model_tracker")
@common.model.diffuser_options("--model")
@cli.group
#  fmt: on
def diffusion_options(ctx: cli.Context):
    schedule: NoiseSchedule = ctx.eval_schedule
    match ctx.flow_param:
        case "epsilon":
            flow_param = schedule.parameterize(0., 1.)
        case "denoise":
            flow_param = schedule.parameterize(1., 0.)
        case "identity":
            flow_param = IdentityFlowParam()
        case _:
            raise UsageError(f"Unknown flow parameterization: {ctx.flow_param}")
    ctx.flow_param = flow_param

@diffusion_options
@common.config_options(DiffusionConfig)
@cli.command
def train[T: Visualizable, Cond](config: DiffusionConfig):
    experiment = config.create_experiment("gnx.main.diffusion:train")
    rngs = nn.Rngs(config.seed)
    logger.info("Loading data...")
    raw_dataset = config.create_dataset()
    dataset: Dataset[T, Cond] = config.preprocess_dataset(raw_dataset, rngs)

    logger.info("Creating model...")
    diffuser: Diffuser[T, Cond] = config.model.create_diffuser(
        config.flow_param,
        dataset.instance.value,
        dataset.instance.cond,
        rngs=rngs,
    )
    forward_process = NoisingForwardProcess(
        config.eval_schedule,
        dataset.instance.value,
    )
    model = DiffusionModel[T, Cond](
        diffuser,
        forward_process,
        nfe=config.nfe,
        integrator=config.integrator,
    )
    config.initialize_model(experiment, model)
    model.train_mode()

    # use the train schedule SNR to sample time points
    # from the associated eval schedule
    time_distribution = TimeSNRDistribution(
        config.train_schedule,
        config.eval_schedule
    ) if config.train_schedule is not None else None

    plugins = config.create_train_plugins(raw_dataset, dataset, rngs=rngs)
    logging_plugins = config.create_logging_plugins(logger, experiment, rngs=rngs)
    trainer = model.trainer(
        dataset.splits["train"],
        shuffle_rng=rngs.data,
        iterations=config.iterations,
        batch_size=config.batch_size,
        optimizer=config.optimizer,
        model_tracker=config.model_tracker,
        time_distribution=time_distribution,
        plugins=plugins,
        logging_plugins=logging_plugins,
    )
    config.resume_training(experiment, trainer)
