import functools
import jax

import typing as tp
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

from ..core.dataclasses import dataclass

from ..util import datasource
from ..util.experiment import Figure
from ..util.distribution import Distribution, Empirical, Gaussian

from ..methods import TrainSample
from ..methods.noise_schedule import NoiseSchedule
from ..methods.diffusion import (
    Diffuser,
    FlowParameterization,
    IdealDiffuser,
)

from . import Dataset, Visualizable


@dataclass
class Point:
    loc: jax.Array

    def visualize(self, pair=None, cond=None) -> Figure:
        loc = jnp.atleast_2d(self.loc)
        x, y = np.array(loc).T
        traces = [go.Scatter(x=x, y=y, mode="markers")]
        if pair is not None:
            loc = jnp.atleast_2d(pair.loc)
            x_pair, y_pair = np.array(loc).T
            traces.append(go.Scatter(x=x_pair, y=y_pair, mode="markers"))
            traces.append(
                go.Scatter(
                    x=np.stack((x, x_pair, np.full_like(x, None)), axis=-1).flatten(),
                    y=np.stack((y, y_pair, np.full_like(y, None)), axis=-1).flatten(),
                    mode="lines",
                    line=dict(color="rgba(0, 0, 0, 0.5)", width=0.5),
                )
            )
        return Figure(go.Figure(traces))


def single_delta_distribution() -> Distribution[Point]:
    return Empirical(Point(jnp.array([[0.0, 0.0]])))


def two_gaussians_distribution() -> Distribution[Point]:
    return Empirical(Point(jnp.array([[-1.0, 1.0], [1.0, -1.0]])), sigma=0.01)


def spiral_distribution() -> Distribution[Point]:
    @jax.jit
    def _sample_spiral(key):
        r_key, n_key = jax.random.split(key)
        r = jax.random.uniform(key, (), minval=0.5, maxval=8)
        x = r * jnp.stack((jnp.cos(2 * r), jnp.sin(2 * r))) / 4
        return x + 0.02 * jax.random.normal(n_key, (2,))

    _spiral_samples = jax.vmap(_sample_spiral)(
        jax.random.split(jax.random.key(42), 2048)
    )
    # _gt_samples = jnp.array([[0., 1.], [0., 4.]])
    return Empirical(Point(_spiral_samples), sigma=0.05)


CSV_DATA = None


def csv_distribution(name) -> Distribution[Point]:
    global CSV_DATA
    if CSV_DATA is None:
        import csv
        import importlib.resources

        CSV_DATA = {}
        data = importlib.resources.files().joinpath("toy.csv").read_text()
        reader = csv.reader(data.split("\n"), delimiter=",")
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            CSV_DATA.setdefault(row[0], []).append((float(row[1]), float(row[2])))

        def proc(data):
            data = np.array(data, dtype=np.float32)
            data = data - np.mean(data, axis=0)  # center the data
            return data / np.std(data)

        CSV_DATA = {k: proc(v) for k, v in CSV_DATA.items()}
    if not name in CSV_DATA:
        raise ValueError(f"Dataset {name} not found.")
    data = jnp.array(CSV_DATA[name])
    return Empirical(Point(data))


# fmt: off
def distribution(name: str) -> Distribution[Point]:
    match name:
        case "spiral": return spiral_distribution()
        case "two_gaussians": return two_gaussians_distribution()
        case "single_delta": return single_delta_distribution()
        case x: return csv_distribution(name)
# fmt: on


def visualize_batch(a: jax.Array) -> Figure:
    scatter = go.Scatter(x=np.array(a[:, 0]), y=np.array(a[:, 1]), mode="markers")
    return Figure(go.Figure([scatter]))


def make_dataset[T: Visualizable](
    name, dist: Distribution[T]
) -> Dataset[TrainSample[T]]:
    data = TrainSample(dist.samples)  # type: ignore
    return Dataset(splits={"train": datasource.pytree(data)})


def dataset_wrapper[T: Visualizable](
    name,
    f: tp.Callable[[], Distribution[T]],
) -> tp.Callable[[], Dataset[TrainSample[T]]]:
    @functools.wraps(f)
    def wrapped() -> Dataset[TrainSample[T]]:
        return make_dataset(name, f())

    return wrapped


single_delta_dataset = dataset_wrapper("single_delta", single_delta_distribution)
two_gaussians_dataset = dataset_wrapper("two_gaussians", two_gaussians_distribution)
spiral_dataset = dataset_wrapper("spiral", spiral_distribution)


def dataset(name: str) -> Dataset[TrainSample[Point]]:
    return make_dataset(name, distribution(name))
