import random
import weakref
import hashlib
import jax
import typing as tp
import collections.abc as cabc
import urllib.parse
import PIL.Image as PIL

from ...core import graph_util

from . import (
    Experiment,
    ExperimentStatus,
    Artifact,
    ArtifactBuilder,
    Result,
    Scalar,
    _NOUNS,
    _ADJECTIVES,
)


class InMemoryArtifact(Artifact):
    def __init__(self, name: str, type: str, version: str, contents: dict[str, tp.Any]):
        digest = hashlib.sha256(random.randbytes(16)).hexdigest()
        id = "mem/" + digest
        super().__init__(name, id, type, version, digest)
        self._contents = contents

    def keys(self) -> list[str]:
        return list(self._contents.keys())

    def get(self, key: str = "") -> tp.Any:
        return self._contents.get(key, None)

    @staticmethod
    def from_url(url: str) -> "InMemoryArtifact":
        raise RuntimeError("InMemoryArtifact cannot be created from a URL.")


class InMemoryArtifactBuider(ArtifactBuilder):
    def __init__(self, name: str, type: str):
        super().__init__(name, type)
        self._contents = {}

    def put(self, key: str, object: tp.Any, /) -> "InMemoryArtifactBuider":
        object = graph_util.duplicate(object)
        self._contents[key] = object
        return self

    def remove(self, key: str) -> "InMemoryArtifactBuider":
        if key in self._contents:
            del self._contents[key]
        return self

    def build(self) -> InMemoryArtifact:
        return InMemoryArtifact(self.name, self.type, "latest", self._contents)


_EXPERIMENTS = weakref.WeakValueDictionary()


class InMemoryExperiment(Experiment):
    def __init__(self, *, name: str | None = None, history: bool = False):
        self._name = name
        self._entrypoint = None
        self._config = None
        self._results = {}
        self._status = ExperimentStatus.CREATED
        self._state = None
        self._step = 0

    @staticmethod
    def from_url(url: str) -> "InMemoryExperiment":
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "none":
            return InMemoryExperiment()
        elif parsed.scheme == "mem" and parsed.netloc in _EXPERIMENTS:
            return _EXPERIMENTS[parsed.netloc]
        elif parsed.scheme == "mem":
            name = parsed.netloc
            if not name:
                noun = random.choice(_NOUNS)
                adjective = random.choice(_ADJECTIVES)
                name = f"{noun}-{adjective}"
            # add numbers if name already exists
            base_name = name
            idx = 2
            while name in _EXPERIMENTS:
                name = f"{base_name}-{idx}"
                idx = idx + 1
            experiment = InMemoryExperiment(name=name, history=True)
            _EXPERIMENTS[name] = experiment
            return experiment
        else:
            raise ValueError(f"Invalid url: {url}")

    @property
    def name(self) -> str:
        return "none" if not self._name else self._name

    @property
    def url(self) -> str:
        return "none://" if not self._name else f"mem://{self._name}"

    @property
    def link(self) -> str | None:
        return None

    def init(self, entrypoint: str, config: dict):
        self._entrypoint = entrypoint
        self._config = config
        self._status = ExperimentStatus.RUNNING

    def change_status(self, status: ExperimentStatus):
        self._status = status

    @property
    def entrypoint(self) -> str | None:
        return self._entrypoint

    @property
    def config(self) -> tp.Any:
        return self._config

    @property
    def status(self) -> ExperimentStatus:
        return self._status

    def update_state(self, artifact: Artifact, /):
        self._state = artifact

    @property
    def last_state(self) -> Artifact | None:
        return self._state

    @property
    def step(self) -> int:
        return self._step

    def create_artifact(self, name: str, type: str) -> InMemoryArtifactBuider:
        return InMemoryArtifactBuider(name, type)

    @property
    def results(self) -> cabc.Mapping[str, Result]:
        return dict(self._results)

    def logging_options(self, **kwargs):
        pass

    def _check_step(self, step: int | None):
        if step is not None:
            assert step >= self._step
            self._step = max(self._step, step)

    def log_metric(
        self,
        key: str,
        value: jax.typing.ArrayLike,
        step: int | None = None,
    ):
        self._check_step(step)
        self._results[key] = Scalar(value)

    def log_figure(
        self,
        key: str,
        figure: tp.Any,
        step: int | None = None,
    ):
        self._check_step(step)

    def log_image(
        self,
        key: str,
        image: PIL.Image | jax.typing.ArrayLike,
        step: int | None = None,
    ):
        self._check_step(step)

    def log_repr(
        self,
        key: str,
        value: tp.Any,
        step: int | None = None,
    ):
        self._check_step(step)
