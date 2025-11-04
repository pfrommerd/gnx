import logging
import typing as tp

from dataclasses import dataclass

from gnx import core, nn


logger = logging.getLogger(__name__)


@dataclass
class Step[State]:
    max_iterations: int
    max_epochs: int | None
    iterations_per_epoch: int | None

    iteration: int
    epoch: int
    epoch_iteration: int

    state: State


class TrainState(tp.Protocol):
    def step(self, data): ...


@dataclass
class Checkpoint[State: TrainState]:
    iteration: int
    epoch: int
    epoch_iteration: int

    shuffle_rng: nn.RngStream | None

    state: State


class Trainer[State: TrainState]:
    pass
