import itertools
import typing as tp
import functools

import jax
import jax.numpy as jnp
import numpy as np

from ...core import nn, graph_util
from ...core.dataclasses import dataclass

from ..common.resnet import ActivationDef, FiLMConditioner, ShiftConditioner
from ..common.embed import DualSinusoidalEmbedding
from ..mlp import MLP
from .. import FlowMapFactory
from . import UNet


class UNetFlowMap[T, Cond = None, FlowAux = None](nn.Module):
    def __init__(
        self,
        in_channels: int,
        aux_channels: int,
        spatial_dims: int,
        embed_features: int,
        time_embed: tp.Callable,
        *,
        rngs: nn.Rngs,
        cond_embed: tp.Callable | None = None,
        **kwargs,
    ):
        self.time_embed = time_embed
        self.cond_embed = cond_embed
        self.unet = UNet(
            in_channels=in_channels + aux_channels,
            out_channels=in_channels,
            cond_features=embed_features,
            spatial_dims=spatial_dims,
            rngs=rngs,
            **kwargs,
        )

    @jax.jit
    def __call__(
        self,
        x: T,
        /,
        s: jax.Array,
        t: jax.Array,
        *,
        cond: Cond = None,
        aux: FlowAux = None,
    ) -> T:
        x_leaves, x_def = jax.tree.flatten(x)
        x_channels = np.cumsum([0] + [leaf.shape[-1] for leaf in x_leaves])
        aux_leaves = jax.tree.leaves(aux) if aux is not None else []
        input = jnp.concatenate(x_leaves + aux_leaves, axis=-1)
        embed = self.time_embed(s, t)
        if cond is not None:
            assert self.cond_embed is not None
            cond_flat, _ = graph_util.ravel(cond)
            cond_embed = self.cond_embed(cond_flat)
            embed = embed + cond_embed

        output = self.unet(input, cond)
        output_leaves = [
            output[..., cs:ce] for (cs, ce) in itertools.pairwise(x_channels)
        ]
        x_out = jax.tree.unflatten(x_def, output_leaves)
        return x_out


@dataclass(frozen=True)
class UNetFlowMapFactory(FlowMapFactory):
    channels: int = 128
    channel_mults: tp.Sequence[int] = (1, 2, 2, 2)
    embed_features: int = 256
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

    def create_flow_map[T, Cond, Aux](
        self, value: T, cond: Cond = None, aux: Aux = None, *, rngs: nn.Rngs
    ) -> UNetFlowMap[T, Cond, Aux]:
        cond_flat, _ = graph_util.ravel(cond)
        if cond_flat.size > 0:
            cond_embed = MLP(
                cond_flat.shape[-1],
                self.embed_features,
                hidden_features=self.embed_features,
                hidden_layers=2,
                activation=self.activation,
                rngs=rngs,
            )
        else:
            cond_embed = None
        time_embed = nn.Sequential(
            DualSinusoidalEmbedding(self.time_features),
            MLP(
                in_features=self.time_features,
                out_features=self.embed_features,
                hidden_features=self.embed_features,
                hidden_layers=2,
                activation=self.activation,
                rngs=rngs,
            ),
        )

        input = jnp.concatenate(jax.tree.leaves(value), axis=-1)
        assert aux is None

        return UNetFlowMap(
            in_channels=input.shape[-1],
            aux_channels=0,
            spatial_dims=input.ndim - 1,
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
            Conditioner=(
                functools.partial(FiLMConditioner, activation=self.activation)
                if self.film_conditioning
                else functools.partial(ShiftConditioner, activation=self.activation)
            ),
            rngs=rngs,
        )
