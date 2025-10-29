import typing as tp
import logging
import jax
import jax.numpy as jnp

from ..models.mlp.diffuser import MLPDiffuserFactory
from ..models.mlp.gan import MLPGANFactory
from ..models.mlp.flow import MLPFlowMapFactory
from ..models.unet.diffuser import UNetDiffuserFactory
from ..models.unet.flow import UNetFlowMapFactory

from ..core import nn, graph
from ..core.dataclasses import dataclass, replace

from ..datasets import Dataset, TrainSample
from ..datasets.image import Image

from ..methods import GeneratePlugin
from ..methods.noise_schedule import NoiseSchedule
from ..methods.integrators import Euler, DDIM, DDPM, AccelDDIM

from ..eval import EvaluatePlugin
from ..eval.fid import FidEvaluator
from ..eval.ot import OTEvaluator

from ..util import cli, optimizers
from ..util.cli.command import UsageError
from ..util.cli.option import Missing
from ..util.experiment import Experiment, ExperimentStatus, Artifact
from ..util.trainer import Checkpoint, Trainer, Plugin
from ..util.trainer.plugins import (
    CheckpointLogger,
    ConsoleLogger,
    ExperimentLogger,
    ProfileServer,
    RichProgressPlugin,
)


logger = logging.getLogger(__name__)


class EvalConfig(tp.Protocol):
    def create_plugin(
        self,
        name: str,
        raw_dataset: Dataset[tp.Any, tp.Any],
        dataset: Dataset[tp.Any, tp.Any],
        rngs: nn.Rngs,
    ) -> Plugin: ...


class DataPipelineConfig(tp.Protocol):
    def preprocess(
        self, dataset: Dataset[tp.Any, tp.Any], rngs: nn.Rngs
    ) -> Dataset[tp.Any, tp.Any]: ...


@dataclass(kw_only=True)
class CommonConfig:
    method: str
    seed: int
    experiment: str
    dataset: str

    init_model: str | None
    init_checkpoint: str | None

    # The evaluation configuration
    evaluations: dict[str, EvalConfig]
    data_pipeline: DataPipelineConfig

    generate_samples: int
    generate_interval: int
    generate_final: bool

    checkpoint_interval: int
    checkpoint_final: bool

    profiler_port: int | None

    def create_experiment(self, entrypoint: str) -> Experiment:
        experiment = Experiment.from_url(self.experiment)
        if experiment.status == ExperimentStatus.PAUSED:
            logger.info("Resuming experiment from a previous run...")
        else:
            # set the experiment to the exact url
            self.experiment = experiment.url
            logger.info(f"Running {self.method} with configuration: {self}")
            # initialize the experiment with the appropriate
            experiment.init(entrypoint, self)
        logger.info(
            f"Logging to experiment: [green]{experiment.name}[/green]",
            extra={"highlighter": None},
        )
        if experiment.link:
            logger.info(f"Experiment is accessible at: {experiment.link}")
        experiment.start()
        return experiment

    def create_dataset(self) -> Dataset[tp.Any, tp.Any]:
        return Dataset.from_url(self.dataset)

    def preprocess_dataset(
        self, raw_dataset: Dataset[tp.Any, tp.Any], rngs: nn.Rngs
    ) -> Dataset[tp.Any, tp.Any]:
        processed = self.data_pipeline.preprocess(raw_dataset, rngs)
        instance = jax.tree.map(jnp.shape, processed.instance)
        logger.info(f"Using samples of {instance.value} with condition {instance.cond}")
        return processed

    # Methods allow resuming already-started experiments
    # or (b) initializing from a particular model or checkpoint.
    def initialize_model(self, experiment: Experiment, model: tp.Any):
        loaded = None
        if self.init_model:
            logger.info(f"Loading model from {self.init_model}")
            model_artifact = Artifact.from_path_or_url(self.init_model)
            loaded = model_artifact.get()
            loaded_model = (
                loaded.model if hasattr(loaded, "model") else loaded  # type: ignore
            )
            nn.update(model, loaded_model)
        logger.info(f"Model has {nn.num_params(model)} parameters.")
        experiment.log_repr("model", model)

    def resume_training(self, experiment: Experiment, trainer: Trainer):
        if experiment.last_state is not None:
            checkpoint: Checkpoint = experiment.last_state.get()
            trainer.load_checkpoint(checkpoint)
        elif self.init_checkpoint:
            logger.info(f"Loading checkpoint from {self.init_checkpoint}")
            ckpt_artifact = Artifact.from_path_or_url(self.init_checkpoint)
            loaded: Checkpoint = ckpt_artifact.get()
            trainer.load_checkpoint(loaded)
        try:
            trainer.run()
        except KeyboardInterrupt as _:
            try:
                logger.info(
                    "Interrupted, saving checkpoint...interrupt again to exit without saving."
                )
            except KeyboardInterrupt as e:
                logger.info("Exiting without saving checkpoint.")

    def create_train_plugins(
        self,
        raw_dataset: Dataset[tp.Any, tp.Any],
        dataset: Dataset[tp.Any, tp.Any],
        rngs: nn.Rngs,
    ):
        plugins: list = [
            GeneratePlugin(
                dataset.splits["train"],
                self.generate_samples,
                self.generate_interval,
                self.generate_final,
                rngs=rngs,
            )
        ]
        if self.profiler_port is not None:
            plugins.append(ProfileServer(self.profiler_port))
        for name, eval_config in self.evaluations.items():
            plugins.append(
                eval_config.create_plugin(name, raw_dataset, dataset, rngs=rngs)
            )
        return plugins

    def create_logging_plugins(
        self, logger: logging.Logger, experiment: Experiment, *, rngs: nn.Rngs
    ):
        return [
            RichProgressPlugin(),
            ConsoleLogger(logger),
            ExperimentLogger(experiment),
            CheckpointLogger(
                experiment, self.checkpoint_interval, self.checkpoint_final
            ),
        ]


def config_options(Config: tp.Type) -> cli.Group:
    @cli.option("--seed", type=int, default=42)
    # Start a new experiment, with a particular
    # checkpoint or initial model from which to initialize training.
    @cli.option("--init-checkpoint", type=str, default=None)
    @cli.option("--init-model", type=str, default=None)
    #
    @cli.option("--experiment", type=str, default="none://")
    @cli.option("--dataset", type=str, default=None)
    @cli.option("--profiler-port", type=int, default=None)
    #
    @cli.option("--checkpoint-interval", type=int, default=0)
    @cli.option("--checkpoint-final", type=bool, default=True)
    #
    @cli.option("--generate-interval", type=int, default=0)
    @cli.option("--generate-samples", type=int, default=64)
    @cli.option("--generate-final", type=bool, default=True)
    #
    @data_pipeline.options("--data-pipeline")
    @eval.options("--eval", "evaluations")
    #
    @cli.group
    def options_group(ctx: cli.Context):
        kwargs = dict(ctx.flatten())
        if kwargs["dataset"] is None:
            raise UsageError("Must specify a dataset.")
        ctx.clear()
        config = Config(**kwargs)
        ctx.config = config

    return options_group


# Eval option groups


@dataclass(kw_only=True)
class FidEvalConfig(EvalConfig):
    batch_size: int
    samples: int
    interval: int

    def create_plugin(
        self, name: str, raw_dataset: Dataset, dataset: Dataset, rngs: nn.Rngs
    ) -> Plugin:
        return EvaluatePlugin(
            name=name,
            evaluator=FidEvaluator.from_dataset(
                raw_dataset, self.samples, self.batch_size
            ),
            interval=self.interval,
            rngs=rngs,
        )


@dataclass(kw_only=True)
class OTEvalConfig(EvalConfig):
    interval: int
    samples: int

    def create_plugin(
        self, name: str, raw_dataset: Dataset, dataset: Dataset, rngs: nn.Rngs
    ) -> Plugin:
        evalutor = OTEvaluator(dataset.splits, batch_size=self.samples)
        return EvaluatePlugin(name, evalutor, interval=self.interval, rngs=rngs)


@dataclass(kw_only=True)
class NoDataPipelineConfig(DataPipelineConfig):
    def preprocess(
        self, dataset: Dataset[tp.Any, tp.Any], rngs: nn.Rngs
    ) -> Dataset[tp.Any, tp.Any]:
        return dataset


@dataclass(kw_only=True)
class ImageDataPipelineConfig(DataPipelineConfig):
    flip_horizontal: bool = True

    def preprocess(
        self, dataset: Dataset[tp.Any, tp.Any], rngs: nn.Rngs
    ) -> Dataset[tp.Any, tp.Any]:
        if not self.flip_horizontal:
            return dataset
        image_dataset: Dataset[Image, tp.Any] = dataset  # type: ignore
        return image_dataset.map(self)

    @property
    def id(self) -> str:
        return "flipped" if self.flip_horizontal else "no_flip"

    def __call__(
        self, key: jax.Array, sample: TrainSample[Image, tp.Any]
    ) -> TrainSample[Image, tp.Any]:
        # Flip the image horizontally with a 50% chance
        flip = jax.random.bernoulli(key, p=0.5)
        new_pixels = jax.lax.cond(
            flip,
            lambda pixels: pixels[..., :, ::-1, :],
            lambda pixels: pixels,
            sample.value.pixels,
        )
        new_value = replace(sample.value, pixels=new_pixels)
        new_sample = replace(sample, value=new_value)
        return new_sample


# fmt: off

class data_pipeline:
    @staticmethod
    def options(option: str, param: str | None = None) -> cli.Group:
        param = cli.Option.infer_param(option, param)
        @cli.option(f"{option}.image.flip-horizontal", f"{param}.image.flip_horizontal", type=bool, default=False)
        @cli.group
        def option_group(ctx: cli.Context):
            params = ctx[param]
            if params.image.flip_horizontal:
                data_pipeline = ImageDataPipelineConfig(flip_horizontal=params.image.flip_horizontal)
            else:
                data_pipeline = NoDataPipelineConfig()
            del ctx[param]
            ctx[param] = data_pipeline

        return option_group

class eval:
    @staticmethod
    def fid_options(option: str, param: str | None = None) -> cli.Group:
        param = cli.Option.infer_param(option, param)
        @cli.option(f"{option}.enabled", f"{param}.enabled", type=bool, default=False)
        @cli.option(f"{option}.batch_size", f"{param}.batch_size", type=int, default=64)
        @cli.option(f"{option}.samples", f"{param}.samples", type=int, default=50000)
        @cli.option(f"{option}.interval", f"{param}.interval", type=int, default=2000)
        @cli.group
        def option_group(ctx: cli.Context):
            params = ctx[param]
            batch_size = params.pop("batch_size")
            samples = params.pop("samples")
            interval = params.pop("interval")
            if params.pop("enabled"):
                ctx[param] = FidEvalConfig(
                    batch_size=batch_size, samples=samples, interval=interval
                )
        return option_group

    @staticmethod
    def ot_options(option: str, param: str | None = None) -> cli.Group:
        param = cli.Option.infer_param(option, param)
        @cli.option(f"{option}.enabled", f"{param}.enabled", type=bool, default=False)
        @cli.option(f"{option}.interval", f"{param}.interval", type=int, default=2000)
        @cli.option(f"{option}.samples", f"{param}.samples", type=int, default=10000)
        @cli.group
        def option_group(ctx: cli.Context):
            params = ctx[param]
            interval = params.pop("interval")
            samples = params.pop("samples")
            if params.pop("enabled"):
                ctx[param] = OTEvalConfig(interval=interval, samples=samples)
        return option_group

    @staticmethod
    def options(option: str, param: str | None = None):
        param = cli.Option.infer_param(option, param)
        @eval.fid_options(f"{option}.fid", f"{param}.fid")
        @eval.ot_options(f"{option}.ot", f"{param}.ot")
        @cli.group
        def option_group(ctx: cli.Context):
            evaluations = dict(ctx[param].flatten())
            del ctx[param]
            ctx[param] = evaluations

        return option_group


# data pipeline options

# noise schedule options
class noise_schedule:
    @staticmethod
    def options(option: str, param: str | None = None,
                default: type[Missing] | NoiseSchedule | None = Missing) -> cli.Group:
        param = cli.Option.infer_param(option, param)
        @cli.option(f"{option}.type", f"{param}.type", type=str)
        #
        @cli.option(f"{option}.linear_noise.min_sigma", f"{param}.linear.min_sigma", type=float, default=0.01)
        @cli.option(f"{option}.linear_noise.max_sigma", f"{param}.linear.max_sigma", type=float, default=35)
        #
        @cli.option(f"{option}.log_linear_noise.min_sigma", f"{param}.log_linear.min_sigma", type=float, default=0.01)
        @cli.option(f"{option}.log_linear_noise.max_sigma", f"{param}.log_linear.max_sigma", type=float, default=35)
        #
        @cli.option(f"{option}.sigmoid.num_steps", f"{param}.sigmoid.num_steps", type=int, default=1000)
        @cli.option(f"{option}.sigmoid.beta_start", f"{param}.sigmoid.beta_start", type=float, default=0.0001)
        @cli.option(f"{option}.sigmoid.beta_end", f"{param}.sigmoid.beta_end", type=float, default=0.02)
        #
        @cli.option(f"{option}.ddpm.num_steps", f"{param}.ddpm.num_steps", type=int, default=1000)
        @cli.option(f"{option}.ddpm.beta_start", f"{param}.ddpm.beta_start", type=float, default=0.0001)
        @cli.option(f"{option}.ddpm.beta_end", f"{param}.ddpm.beta_end", type=float, default=0.02)
        #
        @cli.option(f"{option}.ldm.num_steps", f"{param}.ldm.num_steps", type=int, default=1000)
        @cli.option(f"{option}.ldm.beta_start", f"{param}.ldm.beta_start", type=float, default=0.00085)
        @cli.option(f"{option}.ldm.beta_end", f"{param}.ldm.beta_end", type=float, default=0.012)
        #
        @cli.group
        def option_group(ctx: cli.Context):
            params = ctx[param]
            schedule = None
            match params.type:
                case "log_linear_noise":
                    schedule = NoiseSchedule.log_linear_noise(
                        params.log_linear.min_sigma, params.log_linear.max_sigma
                    ).constant_variance()
                case "linear_noise":
                    schedule = NoiseSchedule.linear_noise(
                        params.linear.min_sigma, params.linear.max_sigma
                    ).constant_variance()
                case "linear":
                    schedule = NoiseSchedule.linear()
                case "sigmoid":
                    schedule = NoiseSchedule.sigmoid_noise(
                        params.sigmoid.num_steps, params.sigmoid.beta_start, params.sigmoid.beta_end
                    ).constant_variance()
                case "ddpm":
                    schedule = NoiseSchedule.ddpm_noise(
                        params.ddpm.num_steps, params.ddpm.beta_start, params.ddpm.beta_end
                    ).constant_variance()
                case "none":
                    if default is Missing:
                        raise UsageError(f"Must specify a noise schedule type for {option}.")
                case _:
                    if not params.type:
                        if default is Missing:
                            raise UsageError(f"Must specify a noise schedule type for {option}.")
                    else:
                        raise UsageError(f"Unknown noise schedule type: {params.type}")
            if schedule is None and default is not Missing:
                schedule = default
            del ctx[param]
            ctx[param] = schedule
        return option_group

class integrator:
    @staticmethod
    def options(option: str, param: str | None = None) -> cli.Group:
        param = cli.Option.infer_param(option, param)
        @cli.option(option, param, type=str, default="euler")
        @cli.group
        def option_group(ctx: cli.Context):
            type = ctx[param]
            match type:
                case "euler":
                    integrator = Euler()
                case "ddim":
                    integrator = DDIM()
                case "ddpm":
                    integrator = DDPM()
                case "accel":
                    integrator = AccelDDIM()
                case _:
                    raise UsageError(f"Unknown integrator type: {type}")
            del ctx[param]
            ctx[param] = integrator

        return option_group

# optimizer options
class optimizer:
    @staticmethod
    def options(option: str, param: str | None = None) -> cli.Group:
        param = cli.Option.infer_param(option, param)
        @cli.option(f"{option}.optimizer", f"{param}.optimizer", type=str, default="adamw")
        @cli.option(f"{option}.weight_decay", f"{param}.weight_decay", type=float, default=1e-4)
        @cli.option(f"{option}.lr", f"{param}.lr", type=float, default=1e-3)
        @cli.option(f"{option}.lr_schedule", f"{param}.lr_schedule", type=str, default="constant")
        #
        @cli.option(f"{option}.warmup_lr", f"{param}.warmup_lr", type=float, default=0.0)
        @cli.option(f"{option}.warmup_steps", f"{param}.warmup_steps", type=int, default=0)
        #
        @cli.option(f"{option}.cosine_decay.final", f"{param}.cosine_decay.final", type=float, default=0.0)
        @cli.option(f"{option}.cosine_decay.iterations", f"{param}.cosine_decay.iterations", type=int, default=0)
        #
        @cli.option(f"{option}.exp_decay.begin_steps", f"{param}.exp_decay.begin_steps", type=int, default=0)
        @cli.option(f"{option}.exp_decay.interval", f"{param}.exp_decay.interval", type=int, default=1000)
        @cli.option(f"{option}.exp_decay.rate", f"{param}.exp_decay.rate", type=float, default=0.9)
        @cli.option(f"{option}.exp_decay.staircase", f"{param}.exp_decay.staircase", type=bool, default=True)
        @cli.option(f"{option}.exp_decay.final", f"{param}.exp_decay.final", type=float, default=0.0)
        #
        @cli.group
        def option_group(ctx: cli.Context):
            params = ctx[param]
            match params.lr_schedule:
                case "constant":
                    assert params.warmup_steps == 0, "Warmup steps must be 0 for constant schedule."
                    lr = optimizers.constant_schedule(params.lr)
                case "cosine":
                    if params.cosine_decay.iterations <= 0:
                        raise UsageError("Must specify number of iterations for cosine decay.")
                    lr = optimizers.warmup_cosine_decay_schedule(
                        params.warmup_lr, params.lr, params.warmup_steps,
                        params.cosine_decay.iterations, params.cosine_decay.final
                    )
                case "exponential":
                    lr = optimizers.warmup_exponential_decay_schedule(
                        params.warmup_lr, params.lr, params.warmup_steps,
                        params.exp_decay.interval, params.exp_decay.rate,
                        params.exp_decay.begin_steps, params.exp_decay.staircase,
                        params.exp_decay.final
                    )
                case _:
                    raise UsageError(f"Unknown learning rate schedule: {params.lr_schedule}")

            match params.optimizer:
                case "sgd":
                    optimizer = optimizers.sgd(lr=lr)
                case "adam":
                    optimizer = optimizers.adam(lr=lr)
                case "adamw":
                    optimizer = optimizers.adamw(lr=lr, weight_decay=params.weight_decay)
                case _:
                    raise UsageError(f"Unknown optimizer: {params.optimizer}")

            del ctx[param]
            ctx[param] = optimizer

        return option_group

class model_tracker:
    @staticmethod
    def options(option: str, param: str | None = None) -> cli.Group:
        param = cli.Option.infer_param(option, param)

        @cli.option(f"{option}.type", f"{param}.type", type=str, default="none")
        @cli.option(f"{option}.ema.rate", f"{param}.ema.rate", type=float, default=0.9999)
        @cli.group
        def option_group(ctx: cli.Context):
            params = ctx[param]
            match params.type:
                case "none":
                    tracker = None
                case "annealed_ema":
                    tracker = optimizers.annealed_ema_tracker(params.ema.rate)
                case "ema":
                    tracker = optimizers.ema_tracker(params.ema.rate)
                case _:
                    raise UsageError(f"Unknown tracker type: {params.type}")
            del ctx[param]
            ctx[param] = tracker

        return option_group

# Model option groups
class model:
    @staticmethod
    def generic_options(option: str, param: str | None = None, *, MLPFactory, UNetFactory) -> cli.Group:
        param = cli.Option.infer_param(option, param)
        @cli.option(f"{option}.type", f"{param}.type", type=str, default=None)
        @cli.option(f"{option}.mlp.hidden_layers", f"{param}.mlp.hidden_layers", type=int, default=6)
        @cli.option(f"{option}.mlp.hidden_features", f"{param}.mlp.hidden_features", type=int, default=64)
        @cli.option(f"{option}.mlp.embed_features", f"{param}.mlp.embed_features", type=int, default=64)
        @cli.option(f"{option}.mlp.activation", f"{param}.mlp.activation", type=str, default="gelu")
        # UNet options
        @cli.option(f"{option}.unet.channels", f"{param}.unet.channels", type=int, default=128)
        @cli.option(f"{option}.unet.channel_mults", f"{param}.unet.channel_mults", type=tuple[int, ...], default=(1, 2, 2, 2))
        @cli.option(f"{option}.unet.attention_levels", f"{param}.unet.attention_levels", type=tuple[int, ...], default=(1, 2))
        @cli.option(f"{option}.unet.blocks_per_level", f"{param}.unet.blocks_per_level", type=int, default=2)
        @cli.option(f"{option}.unet.embed_features", f"{param}.unet.embed_features", type=int, default=256)
        @cli.option(f"{option}.unet.time_features", f"{param}.unet.time_features", type=int, default=256)
        @cli.option(f"{option}.unet.skip_every_block", f"{param}.unet.skip_every_block", type=bool, default=True)
        @cli.option(f"{option}.unet.film_conditioning", f"{param}.unet.film_conditioning", type=bool, default=True)
        @cli.option(f"{option}.unet.snr_time_embed", f"{param}.unet.snr_time_embed", type=bool, default=False)
        @cli.option(f"{option}.unet.dropout", f"{param}.unet.dropout", type=float, default=0.1)
        @cli.option(f"{option}.unet.activation", f"{param}.unet.activation", type=str, default="silu")
        @cli.group
        def option_group(ctx: cli.Context):
            params = ctx[param]
            mlp, unet = params.mlp, params.unet
            match params.type:
                case "mlp":
                    activation = getattr(jax.nn, mlp.pop("activation"))
                    model = MLPFactory(
                        **dict(mlp.flatten()),
                        activation=activation
                    )
                case "unet":
                    activation = getattr(jax.nn, unet.pop("activation"))
                    model = UNetFactory(
                        **dict(unet.flatten()),
                        activation=activation,
                    )
                case _:
                    raise UsageError(f"Unknown model type: {params.type}")
            del ctx[param]
            ctx[param] = model

        return option_group

    @staticmethod
    def diffuser_options(option: str, param: str | None = None) -> cli.Group:
        return model.generic_options(option, param, MLPFactory=MLPDiffuserFactory, UNetFactory=UNetDiffuserFactory)

    @staticmethod
    def flow_map_options(option: str, param: str | None = None) -> cli.Group:
        return model.generic_options(option, param, MLPFactory=MLPFlowMapFactory, UNetFactory=UNetFlowMapFactory)

    @staticmethod
    def gan_generator_options(option: str, param: str | None = None) -> cli.Group:
        param = cli.Option.infer_param(option, param)
        @cli.option(f"{option}.type", f"{param}.type", type=str, default=None)
        @cli.option(f"{option}.mlp.hidden_layers", f"{param}.mlp.hidden_layers", type=int, default=6)
        @cli.option(f"{option}.mlp.hidden_features", f"{param}.mlp.hidden_features", type=int, default=64)
        @cli.option(f"{option}.mlp.embed_features", f"{param}.mlp.embed_features", type=int, default=64)
        @cli.option(f"{option}.mlp.activation", f"{param}.mlp.activation", type=str, default="gelu")
        @cli.group
        def option_group(ctx: cli.Context):
            params = ctx[param]
            mlp = params.mlp
            match params.type:
                case "mlp":
                    activation = getattr(jax.nn, mlp.pop("activation"))
                    model = MLPGANFactory(
                        **dict(mlp.flatten()),
                        activation=activation
                    )
                case _:
                    raise UsageError(f"Unknown GAN type: {params.type}")
            del ctx[param]
            ctx[param] = model

        return option_group

    gan_discriminator_options = gan_generator_options


# The logging options
