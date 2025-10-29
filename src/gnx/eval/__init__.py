import jax
import logging
import typing as tp
import contextlib
import numpy as np

from ..core import nn, graph
from ..methods import GenTrainState, GenerativeModel
from ..util.experiment import Scalar
from ..util.trainer import Plugin, Step, CallbackFn

logger = logging.getLogger(__name__)


class Evaluator[T, Cond = None](tp.Protocol):
    def __call__(
        self,
        key: jax.Array,
        model: GenerativeModel[T, Cond],
    ) -> tp.Any: ...


class EvaluatePlugin[T, Cond](Plugin[GenTrainState[T, Cond]]):
    def __init__(
        self,
        name: str,
        evaluator: Evaluator[T, Cond],
        interval: int | None = None,
        final: bool = True,
        *,
        rngs: nn.Rngs,
    ):
        self.name = name
        self.evaluator = evaluator
        self.interval = interval
        self.final = final
        self.rngs = rngs

    @contextlib.contextmanager
    def __call__(
        self, _: Step[GenTrainState[T, Cond]]
    ) -> tp.Iterator[CallbackFn[GenTrainState[T, Cond], tp.Any]]:
        def update(step: Step[GenTrainState[T, Cond]]):
            if (self.interval and step.iteration % self.interval == 0) or (
                self.final and step.iteration == step.max_iterations - 1
            ):
                model = step.state.eval_model()
                results = {self.name: self.evaluator(self.rngs.eval(), model)}
                step.add_results(results)

        yield update
