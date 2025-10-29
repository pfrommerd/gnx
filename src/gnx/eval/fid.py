import logging
import time
import typing as tp


import jax
import jax.experimental
import jax.numpy as jnp

from pathlib import Path

from ..core import graph
from ..core.dataclasses import dataclass
from ..util.datasource import DataIterator
from ..datasets import Dataset
from ..datasets.image import Image
from ..methods import GenerativeModel
from ..models.common.inception import InceptionV3

logger = logging.getLogger(__name__)


@dataclass
class FidStats:
    mu: jax.Array
    sigma: jax.Array
    N: jax.Array | None = None  # number of samples computed over

    @jax.jit
    def combine(self, other: "FidStats") -> "FidStats":
        assert (
            self.N is not None and other.N is not None
        ), "Both FidStats must have a defined N to combine"
        new_mu = jnp.add(self.mu * self.N, other.mu * other.N) / (self.N + other.N)
        delta = self.mu - other.mu
        mean_corr = jnp.outer(delta, delta) * (self.N * other.N) / (self.N + other.N)
        new_sigma = (
            self.sigma * (self.N - 1) + other.sigma * (other.N - 1) + mean_corr
        ) / (self.N + other.N - 1)
        return FidStats(new_mu, new_sigma, self.N + other.N)

    @staticmethod
    def from_iterator(
        model: InceptionV3,
        data: DataIterator[jax.Array],
    ) -> "FidStats":
        """Compute the FID statistics (mean and covariance) for the given data."""
        activations = []
        for batch in data:
            activations.append(model(batch))
        activations = jnp.concatenate(activations, axis=0)
        return FidStats.from_activations(activations)

    @staticmethod
    def from_batch(model: InceptionV3, batch: jax.Array) -> "FidStats":
        activations = model(batch)
        return FidStats.from_activations(activations)

    @staticmethod
    def from_activations(activations: jax.Array) -> "FidStats":
        with jax.experimental.enable_x64():
            activations = jnp.asarray(activations, dtype=jnp.float64)
            mu = jnp.mean(activations, axis=0)
            sigma = jnp.cov(activations, rowvar=False, bias=False)
            return FidStats(mu, sigma, jnp.array(activations.shape[0], dtype=jnp.int32))

    def distance(self, stats: "FidStats") -> jax.Array:
        with jax.experimental.enable_x64():
            fid = compute_frechet_distance(self.mu, stats.mu, self.sigma, stats.sigma)
        return jnp.array(fid, dtype=jnp.float32)


@jax.jit
def compute_frechet_distance(mu1, mu2, sigma1, sigma2):
    mu1, mu2, sigma1, sigma2 = (
        jnp.asarray(x, dtype=jnp.float64) for x in (mu1, mu2, sigma1, sigma2)
    )
    nans = jnp.logical_or(
        jnp.logical_or(jnp.isnan(mu1).any(), jnp.isnan(mu2).any()),
        jnp.logical_or(jnp.isnan(sigma1).any(), jnp.isnan(sigma2).any()),
    )

    def compute_fid():
        U, S, V = jax.scipy.linalg.svd(sigma1)
        sigma1_sqrt = U @ jnp.diag(jnp.sqrt(S)) @ U.T
        M = sigma1_sqrt @ sigma2 @ sigma1_sqrt
        M = (M + M.T) / 2  # Ensure symmetric
        eigvals = jax.scipy.linalg.eigh(M, eigvals_only=True)
        M_sqrt_trace = jnp.sum(jnp.sqrt(jnp.abs(eigvals)))
        sigma1_tr = jnp.trace(sigma1)
        sigma2_tr = jnp.trace(sigma2)
        return jnp.sum(jnp.square(mu1 - mu2)) + sigma1_tr + sigma2_tr - 2 * M_sqrt_trace

    return jax.lax.cond(nans, lambda: jnp.nan, compute_fid)


class FidEvaluator(graph.Object):
    def __init__(
        self,
        inception: InceptionV3,
        references: dict[str, FidStats],
        samples: int = 10_000,
        batch_size: int = 256,
    ):
        self.inception = inception
        self.references = references
        self.samples = samples
        self.batch_size = batch_size

    def __call__(
        self,
        key: jax.Array,
        model: GenerativeModel[Image, None],
        **extra_kwargs: tp.Any,
    ):
        logger.info("Sampling images and computing FID score")
        t = time.time()

        num_batches = (self.samples + self.batch_size - 1) // self.batch_size
        activations = []
        report_interval = max(num_batches // 5, 1)
        for i in range(num_batches):
            key, sk = jax.random.split(key)
            samples = model.sample(sk, (self.batch_size,), cond=None).pixels
            samples = samples.reshape((-1, 64) + samples.shape[1:])
            for sub_batch in samples:
                activations.append(self.inception(sub_batch))
            del samples
            if (i + 1) % report_interval == 0:
                logger.info(
                    f"FID: [blue]{round(((i + 1) / num_batches) * 100)}%[/blue] complete..."
                )
        activations = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *activations)
        activations = jax.block_until_ready(activations)
        delta = time.time() - t
        logger.info(f"Sampling took {delta:.2f} seconds")
        stats = FidStats.from_activations(activations)
        scores = {
            split: stats.distance(ref_stats).item()
            for split, ref_stats in self.references.items()
        }
        return scores

    @staticmethod
    def from_dataset(dataset: Dataset, samples: int, batch_size: int) -> "FidEvaluator":
        logger.info("Loading inception model for FID evaluation...")
        inception = InceptionV3.load_fid_pretrained(precision=jax.lax.Precision.HIGHEST)

        references = {}
        cache_path = Path.home() / ".cache" / "gnx" / "fid" / dataset.id
        if not cache_path.exists():
            logger.info("Precomputing FID statistics for dataset splits...")
            cache_path.mkdir(parents=True, exist_ok=True)
            for split_name, split in dataset.splits.items():
                data = split.map(lambda _, x: x.value.pixels).batch((64,))
                stats = FidStats.from_iterator(
                    inception, data.sampler(jax.random.key(42))
                )
                split_dir = cache_path / split_name
                split_dir.mkdir(exist_ok=True)
                with open(split_dir / "mu.npy", "wb") as f:
                    jnp.save(f, stats.mu)
                with open(split_dir / "sigma.npy", "wb") as f:
                    jnp.save(f, stats.sigma)
                references[split_name] = stats
            logger.info("All FID statistics computed and saved to cache.")
        else:
            logger.info("Reading in precomputed FID statistics for dataset splits...")
            for split_name in dataset.splits:
                split_dir = cache_path / split_name
                mu_path = split_dir / "mu.npy"
                sigma_path = split_dir / "sigma.npy"
                with open(mu_path, "rb") as f:
                    mu = jnp.load(f)
                with open(sigma_path, "rb") as f:
                    sigma = jnp.load(f)
                references[split_name] = FidStats(mu, sigma)

        return FidEvaluator(inception, references, samples, batch_size)
