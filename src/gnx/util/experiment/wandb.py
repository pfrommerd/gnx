import os
import tempfile
import urllib.parse
import typing as tp
import collections.abc as cabc

from pathlib import Path

import jax
import PIL.Image as PIL
import plotly.graph_objects as go
import plotly.tools as tls
import treescope
import cloudpickle
import logging

import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from ...core import graph
from .. import fs
from . import (
    Artifact,
    ArtifactBuilder,
    ArtifactInfo,
    Experiment,
    ExperimentStatus,
    Result,
    Scalar,
)

CACHE_DIR = Path.home() / ".cache" / "gnx" / "wandb"


class WandbArtifact(Artifact):
    def __init__(self, wandb_artifact: wandb.Artifact):
        name = wandb_artifact.name
        if ":" in name:
            name, _ = name.split(":")
        url = f"wandb://{wandb_artifact.entity}/{wandb_artifact.project}/{name}#{wandb_artifact.version}"
        super().__init__(
            name=wandb_artifact.name,
            url=url,
            type=wandb_artifact.type,
            version=wandb_artifact.version,
            digest=wandb_artifact.digest,
        )
        self.wandb_artifact = wandb_artifact

    def keys(self) -> list[str]:
        entries = self.wandb_artifact.manifest.entries.keys()
        return [e.rsplit(".", 2)[0] for e in entries]

    def get(self, key: str = "data") -> graph.Object:
        entry = self.wandb_artifact.get_entry(f"{key}.pkl")
        target_key = CACHE_DIR / entry.digest
        if target_key.exists():
            target_md5 = fs
            if target_md5 != entry.digest:
                raise ValueError(
                    f"Artifact entry {key} has a different digest than the cached file: "
                    f"{target_md5} != {entry.digest}"
                )
        with open(target_key, "rb") as f:
            return cloudpickle.load(f)


class WandbArtifactBuilder(ArtifactBuilder):
    def __init__(self, wandb_artifact: wandb.Artifact, run: WandbRun):
        super().__init__(
            name=wandb_artifact.name,
            type=wandb_artifact.type,
        )
        self.wandb_artifact = wandb_artifact
        self.wandb_run = run
        self.queued = {}
        self.result = None

    def _cleanup_temp(self):
        for file in self.queued.values():
            os.remove(file)
        self.queued.clear()

    def put(self, key: str, object: graph.Object, /) -> tp.Self:
        if self.wandb_artifact is None:
            raise ValueError("WandbArtifact has already been built")
        local_key = tempfile.mktemp()
        with open(local_key, "wb") as f:
            (graphdef, graphstate) = graph.split(object)
            cloudpickle.dump((graphdef, graphstate), f)
        self.queued[key] = local_key
        return self

    def remove(self, key: str) -> tp.Self:
        del self.queued[key]
        return self

    def build(self) -> WandbArtifact:
        if self.wandb_artifact is not None:
            for key, local_key in self.queued.items():
                self.wandb_artifact.add_file(local_key, key, overwrite=True)
            artifact = self.wandb_run.log_artifact(self.wandb_artifact)
            artifact.wait()
            self._cleanup_temp()
            self.result = WandbArtifact(artifact)
            self.wandb_artifact = None
        assert self.result is not None
        return self.result


class WandbLoggingHandler(logging.Handler):
    def __init__(self, run):
        super().__init__()
        self.run = run

    def emit(self, record):
        log_entry = self.format(record)
        self.run._console_callback("stdout", log_entry + "\n")


class WandbExperiment(Experiment):
    def __init__(
        self,
        *,
        entity: str | None = None,
        project: str | None = None,
        id: str | None = None,
        run: WandbRun | None = None,
    ):
        # Make gnx the default project name if not specified
        project = project or "gnx"
        self.wandb_run: wandb.Run = (
            run
            if run is not None
            else wandb.init(
                project=project,
                entity=entity,
                id=id,
                resume="allow",
                reinit="finish_previous",
                settings=wandb.Settings(
                    console="off",
                    disable_git=True,
                    disable_code=True,
                ),
            )
        )
        self._log_handler = WandbLoggingHandler(self.wandb_run)
        self._status = (
            ExperimentStatus.PAUSED
            if self.wandb_run.resumed
            else ExperimentStatus.CREATED
        )
        self._state = None
        self._config = None

    @staticmethod
    def from_url(url: str) -> "WandbExperiment":
        parsed = urllib.parse.urlparse(url)
        entity = parsed.netloc if parsed.netloc else None
        key = parsed.path.split("/")
        project = key[1] if len(key) > 1 and key[1] else None
        id = key[2] if len(key) > 2 and key[2] else None
        return WandbExperiment(
            entity=entity,
            project=project,
            id=id,
        )

    @property
    def name(self) -> str:
        return self.wandb_run.name or self.wandb_run.id

    @property
    def url(self) -> str:
        return f"wandb://{self.wandb_run.entity}/{self.wandb_run.project}/{self.wandb_run.id}"

    @property
    def link(self) -> str | None:
        return self.wandb_run.url

    @property
    def entrypoint(self) -> str | None:
        return self.wandb_run.config.get("entrypoint")

    @property
    def config(self) -> tp.Any:
        return self._config

    @property
    def status(self) -> ExperimentStatus:
        return self._status

    @property
    def step(self) -> int:
        return self.wandb_run.step

    @property
    def last_state(self) -> Artifact | None:
        return self._state

    def init(self, entrypoint: str, config: tp.Any):
        assert self.wandb_run is not None
        self._config = config
        self.wandb_run.config.update(jsonify_config(config))
        self.wandb_run.config["entrypoint"] = entrypoint

    def change_status(self, status: ExperimentStatus):
        if status == self._status:
            return
        if status == ExperimentStatus.FINISHED:
            logging.root.removeHandler(self._log_handler)
            self.wandb_run.finish(exit_code=0)
        elif status == ExperimentStatus.PAUSED:
            logging.root.removeHandler(self._log_handler)
            self.wandb_run.mark_preempting()
        elif status == ExperimentStatus.RUNNING:
            logging.root.addHandler(self._log_handler)
            if self._status not in (ExperimentStatus.PAUSED, ExperimentStatus.CREATED):
                raise ValueError("Cannot change status to RUNNING unless it is PAUSED.")
        elif status == ExperimentStatus.CRASHED:
            logging.root.removeHandler(self._log_handler)
            self.wandb_run.finish(exit_code=1)
        else:
            raise ValueError(f"Cannot change status to: {status}")
        self._status = status

    def update_state(self, artifact: Artifact):
        self._state = artifact

    def create_artifact(self, name: str, type: str) -> WandbArtifactBuilder:
        wandb_artifact = wandb.Artifact(name, type)
        return WandbArtifactBuilder(wandb_artifact, self.wandb_run)

    @property
    def results(self) -> cabc.Mapping[str, Result]:
        def conv(v):
            return Scalar(v)

        return {k: conv(v) for k, v in self.wandb_run.summary._as_dict().items()}

    def logging_options(self, **kwargs):
        pass

    def log_metric(
        self,
        key: str,
        value: jax.typing.ArrayLike,
        step: int | None = None,
    ):
        assert self._status == ExperimentStatus.RUNNING
        if step is not None:
            assert step >= self.wandb_run.step
        key = "metrics/" + key
        if step is not None:
            self.wandb_run.log({key: value}, step=step)
        else:
            self.wandb_run.summary[key] = value

    def log_figure(
        self,
        key: str,
        figure: tp.Any,
        step: int | None = None,
    ):
        assert self._status == ExperimentStatus.RUNNING
        if not isinstance(figure, (go.Figure, dict)):
            figure = tls.mpl_to_plotly(figure)
        if step is not None:
            assert step >= self.wandb_run.step
        key = "figures/" + key
        return self.wandb_run.log({key: wandb.Plotly(figure)}, step=step)

    def log_image(
        self,
        key: str,
        image: PIL.Image | jax.typing.ArrayLike,
        step: int | None = None,
    ):
        assert self._status == ExperimentStatus.RUNNING
        if step is not None:
            assert step >= self.wandb_run.step
        key = "images/" + key
        self.wandb_run.log({key: wandb.Image(image)}, step=step)

    def log_repr(
        self,
        key: str,
        value: tp.Any,
        step: int | None = None,
    ):
        step = self.wandb_run.step if step is None else step
        assert self._status == ExperimentStatus.RUNNING
        assert step >= self.wandb_run.step
        key = "reprs/" + key
        html = treescope.render_to_html(value)
        self.wandb_run.log({key: wandb.Html(html)}, step=step)


# Convert a configuration object to a JSON-serializable format.
# This assumes the config is a jax-compatible datastructure
# We additionally have utilities that handle optax transformations
# and diffusion schedules to extract the parameters, even though
# they are not directly accessible.
def jsonify_config(config: tp.Any) -> dict:
    res = {}
    _, leaves = graph.split(config)
    for keys, v in leaves.items():
        keys = tuple(str(k) for k in keys)
        if not keys:
            return {"value": v}
        else:
            d = res
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v
    return res
