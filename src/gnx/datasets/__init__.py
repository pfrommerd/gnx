import typing as tp
import urllib.parse
import jax.random

from ..core import nn, graph, asserts
from ..core.dataclasses import dataclass
from ..util.datasource import DataSource, DataTransform
from ..util.experiment import Result


class Visualizable(tp.Protocol):
    def visualize(self, pair: tp.Any = None, cond=None) -> Result: ...


@dataclass
class TrainSample[T, Cond = None]:
    value: T
    cond: Cond = None  # type: ignore[assignment]


@dataclass
class Dataset[T, Cond = None]:
    splits: dict[str, DataSource[TrainSample[T, Cond]]]

    @property
    def instance(self) -> TrainSample[T, Cond]:
        instances = [ds.instance for ds in self.splits.values()]
        if len(instances) > 1:
            asserts.graphs_equal_shapes_and_dtypes(*instances)
        return instances[0]

    def map[V, NCond](
        self,
        transform: DataTransform[TrainSample[T, Cond], TrainSample[V, NCond]],
        splits: tp.Iterable[str] | None = None,
    ) -> "Dataset[V, NCond]":
        splits = None if splits is None else set(splits)
        new_splits: dict[str, DataSource[TrainSample[V, NCond]]] = {
            name: value.map(transform) if splits is None or name in splits else value
            for name, value in self.splits.items()
        }  # type: ignore
        new_instance = transform(jax.random.key(42), self.instance)
        return Dataset[V, NCond](splits=new_splits)

    @staticmethod
    def from_url(dataset: str) -> "Dataset":
        parsed = urllib.parse.urlparse(dataset)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        if parsed.scheme == "toy":
            from . import toy

            return toy.dataset(parsed.netloc)
        elif parsed.scheme == "hf":
            from . import huggingface

            entity = parsed.netloc
            parts = parsed.path.lstrip("/").split("/")
            if len(parts) < 1:
                raise ValueError(
                    "Hugging Face dataset URL must contain at least a repository name."
                )
            repo, *subset = parts
            repo = f"{entity}/{repo}"
            subset = None if not subset else "/".join(subset)
            rev = parsed.fragment if parsed.fragment else None
            data_limit = int(query.get("data_limit", "0")) or None
            ds = huggingface.HfDatasetBuilder.from_repo(repo, subset, rev, data_limit)
            return ds.fetch()
        else:
            raise ValueError(f"Unknown dataset URL scheme: {parsed.scheme}")
