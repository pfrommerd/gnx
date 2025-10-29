import logging
import typing as tp

from ..core import nn
from ..core.dataclasses import dataclass
from ..util import optimizers, cli

from ..datasets import Visualizable, Dataset
from ..methods.gan import AdversarialModel, Discriminator, Generator
from ..models import GeneratorFactory, DiscriminatorFactory
from ..models.mlp.gan import MLPGANFactory
from . import common

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class GANConfig(common.CommonConfig):
    method: str = "gan"

    batch_size: int
    update_ratio: int
    iterations: int
    disc_optimizer: optimizers.Optimizer
    gen_optimizer: optimizers.Optimizer

    generator: GeneratorFactory
    discriminator: DiscriminatorFactory


# fmt: off
@cli.option("--train.iterations", "iterations", type=int, default=20000)
@cli.option("--train.update_ratio", "update_ratio", type=int, default=6)
@cli.option("--train.batch_size", "batch_size", type=int, default=256)
@common.optimizer.options("--train.gen", "gen_optimizer")
@common.optimizer.options("--train.disc", "disc_optimizer")
@common.model.gan_generator_options("--model.gen", "generator")
@common.model.gan_discriminator_options("--model.disc", "discriminator")
@cli.group
# fmt: on
def gan_options(ctx: cli.Context): ...


@gan_options
@common.config_options(GANConfig)
@cli.command
def train[T: Visualizable, Cond](config: GANConfig):
    experiment = config.create_experiment("gnx.main.gan:train")
    rngs = nn.Rngs(config.seed)
    raw_dataset = config.create_dataset()
    dataset: Dataset[T, Cond] = config.preprocess_dataset(raw_dataset, rngs)

    train_data = dataset.splits["train"]

    generator: Generator[T, tp.Any, Cond] = config.generator.create_generator(
        dataset.instance.value, dataset.instance.cond, rngs=rngs
    )
    discriminator: Discriminator[T, Cond] = config.discriminator.create_discriminator(
        dataset.instance.value, dataset.instance.cond, rngs=rngs
    )
    model = AdversarialModel(generator, discriminator)
    config.initialize_model(experiment, model)
    # Set the model to training mode
    model.train_mode()
    plugins = config.create_train_plugins(raw_dataset, dataset, rngs=rngs)
    logging_plugins = config.create_logging_plugins(logger, experiment, rngs=rngs)
    trainer = model.trainer(
        train_data,
        shuffle_rng=rngs.data,
        iterations=config.iterations,
        batch_size=config.batch_size,
        update_ratio=config.update_ratio if config.update_ratio else None,
        gen_optimizer=config.gen_optimizer,
        disc_optimizer=config.disc_optimizer,
        plugins=plugins,
        logging_plugins=logging_plugins,
    )
    config.resume_training(experiment, trainer)
