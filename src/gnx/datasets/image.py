import jax
import jax.numpy as jnp

from ..core import graph
from ..core.dataclasses import dataclass
from ..util.experiment import Image as ImageResult


@dataclass
class Image:
    # pixels, normalized to [-1, 1], shape (H, W, C)
    pixels: jax.Array

    def visualize(self, pair=None, cond=None) -> ImageResult:
        if pair is not None:
            assert (
                pair.pixels.shape == self.pixels.shape
            ), f"Pair pixels shape {pair.pixels.shape} does not match image pixels shape {self.pixels.shape}"
            pixels = jnp.concatenate(
                (self.pixels, pair.pixels), axis=-2
            )  # concatenate along columns
        else:
            pixels = self.pixels
        return ImageResult((pixels + 1.0) / 2)
