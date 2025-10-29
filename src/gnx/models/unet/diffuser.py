import itertools
import functools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from ...core import graph_util, nn
from ...core.dataclasses import dataclass

from ...methods.diffusion import FlowParameterization
from ...methods.noise_schedule import NoiseScheduleFlowParam

from ..common.embed import SinusoidalEmbedding, LogSNRSinusoidEmbed
from ..common.resnet import FiLMConditioner, ShiftConditioner, ActivationDef
from ..mlp import MLP
from .. import DiffuserFactory
from . import UNet


class UNetDiffuser[T, Cond](nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_dims: int,
        embed_features: int,
        time_embed: tp.Callable,
        *,
        parameterization: FlowParameterization[T, Cond],
        rngs: nn.Rngs,
        cond_embed: tp.Callable | None = None,
        **kwargs,
    ):
        self.parameterization = parameterization
        self.time_embed = time_embed
        self.cond_embed = cond_embed
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            cond_features=embed_features,
            spatial_dims=spatial_dims,
            rngs=rngs,
            **kwargs,
        )

    @jax.jit
    def __call__(self, x: T, /, t: jax.Array, *, cond: Cond = None) -> T:
        x_leaves, x_def = jax.tree.flatten(x)
        channels = np.cumsum([0] + [leaf.shape[-1] for leaf in x_leaves])
        input = jnp.concatenate(x_leaves, axis=-1)
        # Placeholder for the actual UNet implementation
        # This should include the forward pass through the UNet architecture
        embed = self.time_embed(t)
        if cond is not None:
            assert self.cond_embed is not None
            cond_embed = self.cond_embed(cond)
            embed = embed + cond_embed
        output = self.unet(input, embed)
        output_leaves = [
            output[..., cs:ce] for (cs, ce) in itertools.pairwise(channels)
        ]
        return jax.tree.unflatten(x_def, output_leaves)


@dataclass(frozen=True)
class UNetDiffuserFactory(DiffuserFactory):
    channels: int = 128
    channel_mults: tp.Sequence[int] = (1, 2, 2, 2)
    embed_features: int = 512
    time_features: int = 64
    attention_levels: tp.Sequence[int] = (1,)
    blocks_per_level: int = 2
    dropout: float = 0.1
    activation: ActivationDef = jax.nn.silu

    # use smalldiffusion-style sigma time embedding
    snr_time_embed: bool = False
    # use film conditioning or just shift conditioning
    film_conditioning: bool = True
    # whether to skip every block in the UNet or every level
    skip_every_block: bool = True

    def create_diffuser[T, Cond](
        self,
        parameterization: FlowParameterization[T, Cond],
        value: T,
        cond: Cond = None,
        *,
        precision: jax.lax.Precision | None = None,
        rngs: nn.Rngs,
    ) -> UNetDiffuser[T, Cond]:
        cond_flat, _ = graph_util.ravel(cond)
        if cond_flat.size > 0:
            cond_embed = nn.Sequential(
                nn.util.Flatten(),
                MLP(
                    cond_flat.shape[-1],
                    self.embed_features,
                    hidden_features=self.embed_features,
                    hidden_layers=2,
                    activation=self.activation,
                    rngs=rngs,
                ),
            )
        else:
            cond_embed = None

        if self.snr_time_embed:
            assert isinstance(parameterization, NoiseScheduleFlowParam)
            time_embed = nn.Sequential(
                LogSNRSinusoidEmbed(parameterization.schedule),
                MLP(
                    in_features=2,
                    out_features=self.embed_features,
                    hidden_features=self.embed_features,
                    hidden_layers=1,
                    activation=self.activation,
                    rngs=rngs,
                ),
            )
        else:
            time_embed = nn.Sequential(
                SinusoidalEmbedding(self.time_features),
                MLP(
                    in_features=self.time_features,
                    out_features=self.embed_features,
                    hidden_features=self.embed_features,
                    hidden_layers=1,
                    activation=self.activation,
                    rngs=rngs,
                ),
            )

        input = jnp.concatenate(jax.tree.leaves(value), axis=-1)
        return UNetDiffuser(
            in_channels=input.shape[-1],
            spatial_dims=input.ndim - 1,
            parameterization=parameterization,
            model_channels=self.channels,
            embed_features=self.embed_features,
            time_embed=time_embed,
            cond_embed=cond_embed,
            level_channel_mults=self.channel_mults,
            attention_levels=self.attention_levels,
            blocks_per_down_level=self.blocks_per_level,
            blocks_per_up_level=self.blocks_per_level + 1,
            skip_every_block=self.skip_every_block,
            dropout=self.dropout,
            activation=self.activation,
            precision=precision,
            Conditioner=(
                functools.partial(FiLMConditioner, activation=self.activation)
                if self.film_conditioning
                else functools.partial(ShiftConditioner, activation=self.activation)
            ),
            rngs=rngs,
        )
