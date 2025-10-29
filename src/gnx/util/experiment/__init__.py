from __future__ import annotations

import io
import enum
import abc
import math
import typing as tp
import urllib.parse
import fsspec
import collections.abc as cabc


import jax
import numpy as np
import PIL.Image as PILImage

from ...core import graph_util
from ...core.dataclasses import dataclass


# A result is anything that knows how to log itself to an experiment.
class Result(abc.ABC):
    @abc.abstractmethod
    def log(
        self,
        experiment: Experiment,
        key: str,
        step: int | None = None,
    ): ...


class Scalar(Result):
    def __init__(self, value: jax.typing.ArrayLike):
        self.value = float(np.array(value).item())

    def log(
        self,
        experiment: Experiment,
        key: str,
        step: int | None = None,
    ):
        experiment.log_metric(key, self.value, step=step)


class Figure(Result):
    def __init__(self, figure):
        self.figure = figure

    def log(
        self,
        experiment: Experiment,
        key: str,
        step: int | None = None,
    ):
        experiment.log_figure(key, self.figure, step=step)

    def _display_(self):
        return self.figure


class Image(Result):
    def __init__(self, image, display_size: tuple[int, int] | None = None):
        if not isinstance(image, PILImage.Image):
            image = np.array(image)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            if image.ndim == 4:
                # reshape to a grid of images
                r, c, h, w, ch = (image.shape[0], 1, *image.shape[1:])
                for r in range(math.ceil(math.sqrt(image.shape[0])), 0, -1):
                    if image.shape[0] % r == 0:
                        c = image.shape[0] // r
                        break
                image = np.reshape(image, (r, c) + image.shape[1:])
                #  (r, c, h, w, ch) -> (r, h, c, w, ch)
                image = np.transpose(image, (0, 2, 1, 3, 4))
                image = np.reshape(image, (r, h, c * w, ch))
                # (r, h, c * w, ch) -> (r * h, c * w, ch)
                image = np.reshape(image, (r * h, c * w, ch))
            if image.dtype == np.float32:
                image = np.nan_to_num(
                    (image * 255).clip(0, 255), nan=0.0, posinf=255.0, neginf=0.0
                )
                image = image.astype(np.uint8)
            if image.shape[-1] == 1:
                image = np.squeeze(image, axis=-1)
            image = PILImage.fromarray(image)
        self.image = image
        self.display_size = display_size

    def log(
        self,
        experiment: Experiment,
        key: str,
        step: int | None = None,
    ):
        experiment.log_image(key, self.image, step=step)

    def _repr_mimebundle_(self, include=None, exclude=None):
        buf = io.BytesIO()
        if self.display_size is not None:
            image = self.image.resize(self.display_size)
        else:
            image = self.image
        image.save(buf, format="PNG")
        bytes = buf.getvalue()
        metadata = {}
        return {"image/png": bytes}, {"image/png": metadata}

    def _display_(self):
        return self.image


def flatten_results(results) -> tp.Iterator[tuple[str, Result]]:
    for key, value in graph_util.flatten_items(
        results, is_leaf=lambda x: isinstance(x, Result)
    ):
        if isinstance(value, (float, np.ndarray, jax.Array)):
            value = Scalar(value)
        else:
            assert isinstance(value, Result), f"Unexpected type: {type(value)}"
        yield key, value


@dataclass(frozen=True)
class ArtifactInfo:
    url: str
    name: str
    type: str
    version: str
    digest: str


class Artifact(abc.ABC):
    def __init__(self, url: str, name: str, type: str, version: str, digest: str):
        self.url = url
        self.name = name
        self.type = type
        self.version = version
        self.digest = digest

    @property
    def info(self) -> ArtifactInfo:
        return ArtifactInfo(self.url, self.name, self.type, self.version, self.digest)

    @abc.abstractmethod
    def keys(self) -> tp.Iterable[str]: ...

    @abc.abstractmethod
    def get(self, key: str = "data") -> tp.Any: ...

    @classmethod
    def from_path_or_url(cls, path_or_url: str) -> Artifact:
        url = path_or_url if "://" in path_or_url else f"file://{path_or_url}"
        return cls.from_url(url)

    @staticmethod
    def from_url(url: str) -> Artifact:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "wandb":
            from .wandb import WandbArtifact

            return WandbArtifact.from_url(url)
        elif parsed.scheme in ("file",):
            from .fs import FsArtifact

            return FsArtifact.from_url(url)
        else:
            raise ValueError(f"Unknown artifact URL scheme: {parsed.scheme}")


class ArtifactBuilder(abc.ABC):
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

    def set(self, obj: tp.Any) -> tp.Self:
        return self.put("data", obj)

    @abc.abstractmethod
    def put(self, key: str, obj: tp.Any, /) -> tp.Self: ...

    @abc.abstractmethod
    def remove(self, key: str) -> tp.Self: ...

    # Finalize the artifact
    @abc.abstractmethod
    def build(self) -> Artifact: ...


class ExperimentStatus(enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    STOPPED = "stopped"
    CRASHED = "crashed"


class Experiment(abc.ABC):
    @staticmethod
    def from_url(url: str) -> Experiment:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme in ("none", "mem"):
            from .memory import InMemoryExperiment

            return InMemoryExperiment.from_url(url)
        elif parsed.scheme == "wandb":
            from .wandb import WandbExperiment

            return WandbExperiment.from_url(url)
        elif parsed.scheme in fsspec.available_protocols():
            from .fs import FsExperiment

            return FsExperiment.from_url(url)
        else:
            raise ValueError(f"Unknown experiment URL scheme: {parsed.scheme}")

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    # A unqiue url identifier for the experiment, useful for tracking and resuming.
    @property
    @abc.abstractmethod
    def url(self) -> str: ...

    # A http link to an experiment dashboard for viewing, if available.
    @property
    @abc.abstractmethod
    def link(self) -> str | None: ...

    @abc.abstractmethod
    def init(self, entrypoint: str, config: tp.Any): ...

    @abc.abstractmethod
    def change_status(self, status: ExperimentStatus): ...

    def start(self):
        self.change_status(ExperimentStatus.RUNNING)

    def finish(self):
        self.change_status(ExperimentStatus.FINISHED)

    def pause(self):
        self.change_status(ExperimentStatus.PAUSED)

    # Get the current state of the experiment

    @property
    @abc.abstractmethod
    def entrypoint(self) -> str | None: ...

    @property
    @abc.abstractmethod
    def config(self) -> tp.Any | None: ...

    @property
    @abc.abstractmethod
    def status(self) -> ExperimentStatus: ...

    # We can associate an existing artifact with the experiment as the current state,
    # which may be used to resume the experiment later.
    @abc.abstractmethod
    def update_state(self, state: Artifact, /): ...

    # The last logged state of the experiment,
    # which may be used to resume the experiment later.
    @property
    @abc.abstractmethod
    def last_state(self) -> Artifact | None: ...

    @property
    @abc.abstractmethod
    def step(self) -> int: ...

    # Artifact
    @abc.abstractmethod
    def create_artifact(self, name: str, type: str) -> ArtifactBuilder: ...

    @property
    @abc.abstractmethod
    def results(self) -> cabc.Mapping[str, Result]: ...

    @abc.abstractmethod
    def logging_options(self, **kwargs): ...

    def log(
        self,
        results: tp.Any,
        key: str | None = None,
        step: int | None = None,
    ):
        for k, v in flatten_results(results):
            if key:
                k = f"{key}.{k}" if k else key
            v.log(self, key=k, step=step)

    @abc.abstractmethod
    def log_metric(
        self,
        key: str,
        value: jax.typing.ArrayLike,
        step: int | None = None,
    ): ...

    @abc.abstractmethod
    def log_image(
        self,
        key: str,
        image: PILImage.Image | jax.typing.ArrayLike,
        step: int | None = None,
    ): ...

    @abc.abstractmethod
    def log_figure(
        self,
        key: str,
        figure: tp.Any | dict,
        step: int | None = None,
    ): ...

    @abc.abstractmethod
    def log_repr(
        self,
        key: str,
        value: tp.Any,
        step: int | None = None,
    ): ...


# fmt: off
_NOUNS = [
    "spiral", "sunset", "book",
    "depths", "forest", "plain", "mountain", "sky",
    "lake", "dawn", "dusk", "valley", "circle", "square", "triangle"
]

_ADJECTIVES = [
    "alternating", "bright", "red", "blue", "green", "happy",
    "deep", "calm", "dazzling", "tall"
]
# fmt: on
