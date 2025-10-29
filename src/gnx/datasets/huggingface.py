import json
import io
import os
import re
import typing as tp
import logging


import huggingface_hub as hf
import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path, PurePath
from fsspec.implementations.dirfs import DirFileSystem

from ..methods.diffusion import Diffuser, IdealDiffuser
from ..core import graph_util
from ..core.dataclasses import dataclass

from ..util.datasource.parquet import ParquetDataSource
from ..util.distribution import Empirical
from . import Dataset, TrainSample
from .image import Image

logger = logging.getLogger(__name__)


@dataclass
class HfDatasetBuilder:
    repo: str
    subset: str | None
    rev: str
    content_sha: str
    data_limit: int | None = None

    @staticmethod
    def from_repo(
        repo: str,
        subset: str | None = None,
        rev: str | None = None,
        data_limit: int | None = None,
    ) -> "HfDatasetBuilder":
        logger.debug("Getting dataset information from huggingface hub...")
        if rev is None:
            refs = hf.list_repo_refs(repo, repo_type="dataset")
            all_refs = refs.branches + refs.converts
            branches = [r for r in all_refs if r.ref == "refs/convert/parquet"]
            if not branches:
                raise ValueError(
                    f"Ref refs/convert/parquet not found in: {[r.ref for r in all_refs]}"
                )
            branch = branches[0]
            rev = branch.target_commit
        info = hf.dataset_info(repo, revision=rev)
        sha = info.sha
        assert sha is not None
        root_fs = hf.HfFileSystem()
        fs = DirFileSystem(PurePath("datasets") / f"{info.id}@{rev}", root_fs)
        splits = _hf_collect_split_files(fs, subset)
        if not splits:
            raise ValueError("Unable to find parquet files")
        logger.debug(f"Using {repo}@{rev} ({sha})")
        return HfDatasetBuilder(repo, subset, rev, sha, data_limit=data_limit)

    def fetch(self) -> Dataset:
        splits = {}
        return Dataset(
            splits=splits,
        )


FILENAME_REGEX = re.compile(r"(?P<split>[^.-]+)(?:-[\w\W]+)?(?:\.parquet)")


def _hf_collect_split_files(fs, subset):
    files = fs.ls("/", detail=False)
    if subset is None:
        if not files:
            raise ValueError("No files found in the repository")
        subset = files[0]
    if subset not in files:
        raise ValueError(
            f"Subset {subset} not found in repository. Valid subsets are: {files}"
        )
    splits = {}
    for f in fs.glob(f"{subset}/**/*.parquet", detail=True):
        split = f.split("/")[1]
        splits.setdefault(split, []).append(f)
    return splits
