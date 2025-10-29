from ...core import graph
from ..fetch_util import md5_checksum_b64

from . import (
    Artifact,
    ArtifactBuilder,
    Experiment,
    ExperimentStatus,
    Result,
    _NOUNS,
    _ADJECTIVES,
)

from pathlib import Path

import urllib.parse
import numpy as np
import cloudpickle
import json
import jax
import random
import typing as tp
import PIL.Image as PIL


class FsArtifact(Artifact):
    def __init__(self, path: Path):
        self.path = path
        digest = md5_checksum_b64(*path.glob("**/*"))
        url = f"file://{path.absolute()}"
        assert path.is_dir(), f"Artifact {path} must be a directory."
        assert (
            path / "info.json"
        ).is_file(), "Artifact info file not found at expected path."
        with open(path / "info.json") as f:
            info = json.load(f)
            name, type, version = info["name"], info["type"], info["version"]
        super().__init__(url=url, name=name, type=type, version=version, digest=digest)

    def keys(self) -> list[str]:
        return [p.stem for p in self.path.glob("**/*.pkl")]

    def get(self, key: str = "data") -> tp.Any:
        combined_path = (self.path / f"{key}.pkl").resolve()
        if not combined_path.is_relative_to(self.path.resolve()):
            raise ValueError(
                f"Key '{key}' is not a valid subpath of the artifact path {self.path}"
            )
        with open(combined_path, "rb") as f:
            graphdef, values = cloudpickle.load(f)
            return graph.merge(graphdef, values)

    @staticmethod
    def from_url(url: str) -> "FsArtifact":
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("file",):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        return FsArtifact(Path(parsed.netloc + parsed.path))


class FsArtifactBuilder(ArtifactBuilder):
    def __init__(self, path: Path, name: str, type: str, version: str):
        self.path = path
        self.name = name
        self.type = type
        self.version = version
        self.path.mkdir(parents=True, exist_ok=True)
        info = {
            "name": name,
            "type": type,
            "version": version,
        }
        with open(self.path / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def put(self, key: str, object: tp.Any) -> "FsArtifactBuilder":
        assert isinstance(object, graph.Object), "Only graph.Object is supported."
        with open(str(self.path / key) + ".pkl", "wb") as f:
            cloudpickle.dump(graph.split(object), f)
        return self

    def remove(self, key: str) -> "FsArtifactBuilder":
        (self.path / key).unlink(missing_ok=True)
        return self

    def build(self) -> FsArtifact:
        return FsArtifact(self.path)


class FsExperiment(Experiment):
    def __init__(self, root: Path):
        self._root = root

        if (root / "experiment.json").is_file():
            with open(root / "experiment.json") as f:
                info = json.load(f)
                self._name = info["name"]
                self._status = ExperimentStatus(info["status"])
                self._state = info.get("state", None)
                self._step = info["step"]
                self._entrypoint = info.get("entrypoint", None)

            # read in the config from the config.pkl file
            if (root / "config.pkl").is_file():
                with open(root / "config.pkl", "rb") as f:
                    graphdef, graphstate = cloudpickle.load(f)
                    self._config = graph.merge(graphdef, graphstate)
            else:
                self._config = None

        else:
            self._name = root.name
            self._status = ExperimentStatus.CREATED
            self._step = 0
            self._state = None
            self._config = None
            self._entrypoint = None
            # write out the current experiment info
            self._write_info()

        self._results = {}
        self._metrics_file = open(self._root / "metrics.jsonl", "a")
        self._step_metrics = {}

    def _write_info(self):
        info = {
            "name": self._name,
            "status": self._status.value,
            "step": self._step,
        }
        if self._state is not None:
            info["state"] = self._state
        if self._entrypoint is not None:
            info["entrypoint"] = self._entrypoint
        with open(self._root / "experiment.json", "w") as f:
            json.dump(info, f, indent=2)

    @staticmethod
    def from_url(url: str) -> "FsExperiment":
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("file",):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        path = Path(parsed.netloc + parsed.path)

        if (path / "experiment.json").is_file():
            return FsExperiment(path)
        elif path.is_dir() or not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            # generate a new experiment in a subdirectory
            num = len(list(path.iterdir())) + 1
            name = f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"
            path = path / f"{num:02}-{name}"
            path.mkdir()
            return FsExperiment(path)
        else:
            raise ValueError("Path must a directory!")

    @property
    def name(self) -> str:
        return self._name

    @property
    def url(self) -> str:
        return f"file://{self._root.absolute()}"

    @property
    def link(self) -> str:
        return self.url

    @property
    def entrypoint(self):
        self._entrypoint

    @property
    def config(self) -> tp.Any | None:
        return self._config

    @property
    def status(self) -> ExperimentStatus:
        return self._status

    @property
    def step(self) -> int:
        return self._step

    @property
    def last_state(self) -> Artifact | None:
        if self._state is None:
            return None
        return FsArtifact(self._root / self._state)

    # Initialization, and status management

    def init(self, entrypoint: str, config: tp.Any):
        assert self._status == ExperimentStatus.CREATED
        self._entrypoint = entrypoint
        self._config = config
        self._status = ExperimentStatus.RUNNING
        # write the config to a file
        with open(self._root / "config.pkl", "wb") as f:
            graphdef, state = graph.split(config)
            cloudpickle.dump((graphdef, state), f)
        self._write_info()

    def change_status(self, status: ExperimentStatus):
        if status == self._status:
            return
        if status == ExperimentStatus.CREATED:
            raise ValueError("Cannot change status to CREATED.")
        if (
            self._status == ExperimentStatus.FINISHED
            or self._status == ExperimentStatus.CRASHED
        ):
            raise ValueError(
                "Cannot change status after the experiment has finished or crashed."
            )
        if status == ExperimentStatus.FINISHED:
            assert self._metrics_file is not None
            # Log the final step metrics if they exist
            if self._step_metrics:
                self._step_metrics["step"] = self._step
                # write out the current step metrics
                json.dump(self._step_metrics, self._metrics_file)
                self._step_metrics = {}
            self._metrics_file.close()
            self._metrics_file = None
        self._status = status
        self._write_info()

    def update_state(self, artifact: Artifact):
        assert isinstance(artifact, FsArtifact), "Only FsArtifact is supported."
        if not artifact.path.is_relative_to(self._root):
            raise ValueError(
                f"Artifact path {artifact.path} is not a subpath of the experiment root {self._root}"
            )
        relative_path = artifact.path.relative_to(self._root)
        self._state = str(relative_path)
        self._write_info()

    def create_artifact(self, name: str, type: str) -> FsArtifactBuilder:
        artifacts = self._root / "artifacts"
        artifacts.mkdir(exist_ok=True)
        version = str(
            len([a for a in artifacts.iterdir() if a.name == name.rsplit("-", 1)[0]])
            + 1
        )
        dir = artifacts / f"{name}-{version:02}"
        dir.mkdir()
        return FsArtifactBuilder(
            path=dir,
            name=name,
            type=type,
            version=version,
        )

    @property
    def results(self) -> dict[str, Result]:
        return self._results

    def logging_options(self, **kwargs):
        pass

    def _update_step(self, step: int | None):
        if step is not None:
            assert step >= self._step
            if step > self._step and self._step_metrics:
                if self._metrics_file is None:
                    raise ValueError("Cannot log metrics to a finished experiment.")
                self._step_metrics["step"] = self._step
                # write out the current step metrics
                json.dump(self._step_metrics, self._metrics_file)
                self._step_metrics = {}
            self._step = max(self._step, step)

    def log_metric(
        self,
        key: str,
        value: jax.typing.ArrayLike,
        step: int | None = None,
    ):
        self._update_step(step)
        self._step_metrics[key] = np.array(value).item()

    def log_figure(
        self,
        key: str,
        figure: tp.Any,
        step: int | None = None,
    ):
        self._update_step(step)

    def log_image(
        self,
        key: str,
        image: PIL.Image | jax.typing.ArrayLike,
        step: int | None = None,
    ):
        self._update_step(step)

    def log_repr(
        self,
        key: str,
        value: tp.Any,
        step: int | None = None,
    ):
        self._update_step(step)
