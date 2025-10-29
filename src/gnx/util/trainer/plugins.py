import contextlib
import logging
import math
import typing as typ
import sys

import jax.profiler

from ...core import graph_util
from ..experiment import Experiment, Scalar, flatten_results

from . import CallbackFn, Plugin, Step, ModelTrainState

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text as RichText


class MofNColumn(ProgressColumn):
    def __init__(self, min_width: int = 2):
        self.min_width = min_width
        super().__init__()

    def render(self, task) -> RichText:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        total_padding = max(0, self.min_width - total_width) * " "
        total_width = max(self.min_width, total_width)
        return RichText(
            f"{completed:{total_width}d}/{total}{total_padding}",
            style="progress.percentage",
        )


class SpeedColumn(ProgressColumn):
    def render(self, task) -> RichText:
        elapsed, completed = task.elapsed, task.completed
        if elapsed is None:
            return RichText("_ iters/s", style="progress.elapsed")
        speed = math.floor(completed / elapsed) if elapsed > 0 else 0
        return RichText(f"{speed:2} iters/s", style="progress.elapsed")


class RichProgressPlugin(Plugin):
    def __init__(self, show: bool | None = None):
        # if none, auto-detect based on the console type
        # whether to show the progress bar
        self.show = show

    @contextlib.contextmanager
    def __call__(self, train_state: Step) -> typ.Iterator[CallbackFn]:
        show = self.show if self.show is not None else sys.stdout.isatty()
        # if not show, yield a no-op callback function
        if not show:

            def update_fn(train_state: Step):
                pass

            yield update_fn

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNColumn(min_width=6),
            SpeedColumn(),
        )
        task = progress.add_task("Iteration", total=train_state.max_iterations)
        if train_state.max_epochs is not None:
            epoch_task = progress.add_task("Epoch", total=train_state.max_epochs)
            epoch_iter_task = progress.add_task(
                "Epoch Iteration", total=train_state.iterations_per_epoch
            )
        else:
            epoch_task = None
            epoch_iter_task = None

        def update_fn(train_state: Step):
            progress.update(task, completed=train_state.iteration + 1)
            if train_state.max_epochs is not None:
                assert epoch_iter_task is not None
                assert epoch_task is not None
                if train_state.epoch_iteration == 0:
                    progress.reset(epoch_iter_task)
                progress.update(
                    epoch_iter_task, completed=train_state.epoch_iteration + 1
                )
                progress.update(epoch_task, completed=train_state.epoch + 1)

        progress.start()
        with progress:
            yield update_fn


class ConsoleLogger(Plugin):
    def __init__(
        self,
        logger: logging.Logger,
        interval: int = 100,
    ):
        self.logger = logger
        self.interval = interval

    @contextlib.contextmanager
    def __call__(self, train_state: Step) -> typ.Iterator[CallbackFn]:
        def update_fn(train_state: Step):
            if (
                train_state.iteration % self.interval != 0
                and train_state.iteration != train_state.max_iterations - 1
            ):
                return
            items = []
            for k, v in graph_util.flatten_items(train_state.metrics):
                v = v.item()
                items.append(f"{k:>10}: {v:>7.4f}")
            self.logger.info(
                f"Iteration {train_state.iteration:>6} | " + "    ".join(items)
            )
            scalar_results = {}
            for results in train_state.extra_results:
                scalar_results.update(
                    dict(
                        (k, v)
                        for k, v in graph_util.flatten_items(results)
                        if isinstance(v, (int, float, jax.Array, Scalar))
                    )
                )
            if scalar_results:
                self.logger.info(
                    f" {' '*10} >>> "
                    + "    ".join(
                        f"{k:>10}: {v:>7.4f}" for k, v in scalar_results.items()
                    )
                )

        yield update_fn


class ExperimentLogger(Plugin):
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    @contextlib.contextmanager
    def __call__(self, _: Step) -> typ.Iterator[CallbackFn]:
        # things we have seen before the final iteration
        seen = set()

        def update_fn(train_state: Step):
            final = train_state.iteration == train_state.max_iterations - 1

            for k, v in graph_util.flatten_items(train_state.metrics):
                self.experiment.log_metric(k, v, train_state.iteration)

            for results in train_state.extra_results:
                for k, r in flatten_results(results):
                    if not final:
                        seen.add(k)
                    iter = train_state.iteration
                    # For the final iteration, any "new" metrics
                    # should be logged without a step.
                    if final and k not in seen:
                        iter = None
                    r.log(self.experiment, key=k, step=iter)

        yield update_fn


class CheckpointLogger(Plugin[ModelTrainState, typ.Any]):
    def __init__(self, experiment: Experiment, interval: int | None, final: bool):
        self.experiment = experiment
        self.interval = interval
        self.final = final

    @contextlib.contextmanager
    def __call__(self, _: Step) -> typ.Iterator:
        def update_fn(step: Step[ModelTrainState]):
            log_final = self.final and step.iteration == step.max_iterations - 1
            log_interval = self.interval and step.iteration % self.interval == 0
            if log_final or log_interval:
                state: nnx.Object = step.state  # type: ignore
                if log_interval:
                    self.experiment.create_artifact(
                        f"checkpoint-{step.iteration}", "model"
                    ).set(state).build() # fmt: skip
                if log_final:
                    self.experiment.create_artifact(
                        "final-checkpoint", "model"
                    ).set(state).build() # fmt: skip

        yield update_fn


class ProfileServer(Plugin):
    def __init__(self, port: int = 8000):
        self.port = port

    @contextlib.contextmanager
    def __call__(self, _: Step) -> typ.Iterator[CallbackFn | None]:
        jax.profiler.start_server(self.port)
        yield None
        jax.profiler.stop_server()
