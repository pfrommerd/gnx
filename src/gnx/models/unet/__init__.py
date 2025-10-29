import functools
import itertools
import math
import typing as tp
import typing as typ

import jax
import jax.numpy as jnp

from ...core import nn

from ..common.resnet import (
    ActivationDef,
    BlockSequential,
    ConditionerDef,
    ConvDef,
    FiLMConditioner,
    NormDef,
    ResNetBlock,
)


def upsample(
    x: jax.Array,
    *,
    scale_factors: tp.Sequence[int] | None = None,
    target_shape: tp.Sequence[int] | None = None,
):
    if scale_factors is not None:
        assert len(scale_factors) == x.ndim - 1
        dest_shape = (
            tuple(s * f for s, f in zip(x.shape[1:], scale_factors)) + x.shape[-1:]
        )
    else:
        assert target_shape is not None
        assert len(target_shape) == x.ndim - 1
        dest_shape = tuple(target_shape) + x.shape[-1:]
    return jax.image.resize(x, dest_shape, method="nearest")


def downsample(x: jax.Array, scale_factors: tp.Sequence[int] | None = None):
    if scale_factors is None:
        scale_factors = (2,) * (x.ndim - 1)
    return nn.util.avg_pool(x, window_shape=scale_factors, strides=scale_factors)


class Upsample(nn.Module):
    def __init__(self, channels, spatial_dims: int, *, 
                 Conv: ConvDef = nn.Conv,
                 precision: jax.lax.Precision | None = None,
                 rngs: nn.Rngs):
        self.conv = Conv(
            channels, channels,
            kernel_size=(3,) * spatial_dims,
            precision=precision,
            rngs=rngs
        )

    def __call__(self, x: jax.Array, *, target_shape) -> jax.Array:
        x = upsample(x, target_shape=target_shape)
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, channels, spatial_dims: int, *,
                 Conv: ConvDef = nn.Conv,
                 precision: jax.lax.Precision | None = None,
                 rngs: nn.Rngs):
        self.conv = Conv(
            channels,
            channels,
            kernel_size=(3,) * spatial_dims,
            padding=((0, 1),) * spatial_dims,
            strides=2,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


# A groupnorm that also works on a batch size of 1


def _qkv_attention(q, k, v):
    assert q.shape == k.shape == v.shape
    *batch, length, heads, ch = q.shape
    scale = 1 / math.sqrt(math.sqrt(ch))
    weight = jnp.einsum(
        "...thc,...shc->...ths",
        q * scale, k * scale
    )
    weight = jax.nn.softmax(weight, axis=-1)
    a = jnp.einsum(
        "...ths,...shc->...thc",
        weight,
        v
    )
    a = a.reshape(*batch, length, heads*ch)
    return a


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_dims: int,
        heads: int | None = None,
        head_channels: int | None = None,
        qkv_bias: bool = False,
        *,
        Norm: NormDef = nn.GroupNorm,
        precision: jax.lax.Precision | None = None,
        rngs: nn.Rngs,
    ):
        self.spatial_dims = spatial_dims
        self.norm = Norm(channels, spatial_dims=spatial_dims, rngs=rngs)
        self.qkv = nn.Linear(
            channels, 3 * channels, use_bias=qkv_bias,
            precision=precision,
            rngs=rngs
        )
        if heads is not None:
            self.heads = heads
        else:
            assert head_channels is not None
            self.heads = channels // head_channels
        self.proj = nn.Linear(
            channels,
            channels,
            precision=precision,
            rngs=rngs
        )

    def __call__(self, x: jax.Array):
        batch_dims, spatial_dims, features = (
            math.prod(x.shape[:-1 - self.spatial_dims]),
            math.prod(x.shape[-1 - self.spatial_dims:-1]),
            x.shape[-1]
        )
        heads, head_channels = self.heads, features // self.heads

        a = self.norm(x)
        # Flatten the batch, spatial dimensions
        a = a.reshape((batch_dims, spatial_dims, features))
        q, k, v = jnp.moveaxis(self.qkv(a).reshape((
            batch_dims,
            spatial_dims,
            3,
            heads,
            head_channels
        )), -3, 0)
        a = _qkv_attention(q, k, v)
        assert a.shape == (batch_dims, spatial_dims, features)
        a = self.proj(a)
        # # reshape back to the original dimensions
        a = jnp.reshape(a, x.shape)
        x = x + a
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_features: int | None = None,
        *,
        Conditioner: ConditionerDef | None = None,
        Conv: ConvDef = nn.Conv,
        Norm: NormDef | None = None,
        #
        model_channels: int = 128,
        kernel_size: int | typ.Sequence[int] = 3,
        spatial_dims: int | None = None,
        num_heads_downsample: int = 1,
        num_heads_upsample: int = 1,
        level_channel_mults: typ.Sequence[int] = (1, 2, 2, 2),
        attention_levels: typ.Sequence[int] = (2,),
        blocks_per_down_level: int | typ.Sequence[int] = 2,
        blocks_per_up_level: int | typ.Sequence[int] = 3,
        dropout: float | None = 0.1,
        # Put a skip connection after every resblock
        skip_every_block: bool = False,
        activation: ActivationDef = jax.nn.silu,
        precision: jax.lax.Precision | None = None,
        rngs: nn.Rngs,
    ):
        if isinstance(kernel_size, int):
            if spatial_dims is None:
                raise ValueError(
                    "spatial_dims must be provided if kernel_size is an int"
                )
            kernel_size = (kernel_size,) * spatial_dims
        elif spatial_dims is not None:
            assert len(kernel_size) == spatial_dims
        else:
            spatial_dims = len(kernel_size)
        level_channel_mults = tuple(level_channel_mults)
        blocks_per_down_level = (
            (blocks_per_down_level,) * len(level_channel_mults)
            if isinstance(blocks_per_down_level, int)
            else blocks_per_down_level
        )
        blocks_per_up_level = (
            (blocks_per_up_level,) * len(level_channel_mults)
            if isinstance(blocks_per_up_level, int)
            else blocks_per_up_level
        )
        assert len(blocks_per_down_level) == len(level_channel_mults)
        assert len(blocks_per_up_level) == len(level_channel_mults)

        if Norm is None:
            Norm = functools.partial(
                nn.GroupNorm,
                num_groups=32,
            )
        if Conditioner is None:
            Conditioner = functools.partial(FiLMConditioner, activation=activation)

        ResBlock = functools.partial(
            ResNetBlock,
            cond_features=cond_features,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation,
            precision=precision,
            Conditioner=Conditioner,
            Conv=Conv,
            Norm=Norm,
        )
        AttenBlock = functools.partial(
            AttentionBlock,
            spatial_dims=spatial_dims,
            precision=precision,
            Norm=Norm
        )
        self.skip_every_block = skip_every_block
        self.input_conv = Conv(
            in_channels, model_channels, kernel_size,
            rngs=rngs, precision=precision
        )

        self.down_levels = []
        skip_channels: list[list[int]] = []

        down_channels = (model_channels,) + tuple(
            int(m * model_channels) for m in level_channel_mults
        )
        for level, ((in_ch, out_ch), num_blocks) in enumerate(
            zip(itertools.pairwise(down_channels), blocks_per_down_level)
        ):
            level_blocks, level_skips, level_skip_chs = [], [], []

            if skip_every_block:  # For block-wise skips, skip before
                level_skips.append(None)
                level_skip_chs.append(in_ch)
            for _ in range(num_blocks):
                level_blocks.append(ResBlock(in_ch, out_ch, rngs=rngs))
                in_ch = out_ch
                if level in attention_levels:
                    level_blocks.append(
                        AttenBlock(out_ch, heads=num_heads_downsample, rngs=rngs)
                    )
                level_skips.append(len(level_blocks) - 1)
                level_skip_chs.append(out_ch)
            if level < len(level_channel_mults) - 1:
                level_blocks.append(Downsample(out_ch, spatial_dims,
                        precision=precision, rngs=rngs))
            # If not skipping every block, only keep the last skip
            if not skip_every_block:
                level_skips = [level_skips[-1]]
                level_skip_chs = [level_skip_chs[-1]]
            skip_channels.append(level_skip_chs)
            self.down_levels.append(
                BlockSequential(*level_blocks, skip_out_layers=level_skips)
            )

        middle_channels = down_channels[-1]
        self.middle_block = BlockSequential(
            ResBlock(middle_channels, middle_channels, rngs=rngs),
            AttenBlock(middle_channels, heads=num_heads_downsample, rngs=rngs),
            ResBlock(middle_channels, middle_channels, rngs=rngs),
        )

        self.up_levels = []

        up_channels = (middle_channels,) + tuple(
            int(m * model_channels) for m in level_channel_mults[::-1]
        )
        for level, ((in_ch, out_ch), num_blocks) in enumerate(
            zip(itertools.pairwise(up_channels), blocks_per_up_level)
        ):
            blocks, skip_in_idxs, skip_chs = [], [], skip_channels.pop()
            if not skip_every_block:
                in_ch = in_ch + skip_chs.pop()
                skip_in_idxs.append(0)
            for _ in range(num_blocks):
                if skip_every_block:
                    in_ch = in_ch + skip_chs.pop()
                    skip_in_idxs.append(len(blocks))
                blocks.append(ResBlock(in_ch, out_ch, rngs=rngs))
                in_ch = out_ch
                if (len(level_channel_mults) - 1 - level) in attention_levels:
                    blocks.append(
                        AttenBlock(out_ch, heads=num_heads_upsample, rngs=rngs)
                    )
            if level < len(level_channel_mults) - 1:
                blocks.append(Upsample(out_ch, spatial_dims,
                    precision=precision, rngs=rngs))
            self.up_levels.append(BlockSequential(*blocks, skip_in_layers=skip_in_idxs))

        self.out_final = nn.Sequential(
            Norm(model_channels, spatial_dims=spatial_dims, rngs=rngs),
            activation,
            Conv(model_channels, out_channels, kernel_size, precision=precision, rngs=rngs),
        )

    @jax.jit
    def __call__(self, x, cond=None):
        x = self.input_conv(x)
        level_skips = []
        target_shapes: list = [None]

        for level in self.down_levels:
            target_shapes.append(x.shape[:-1])
            x, skips = level(x, cond=cond, return_skip_outputs=True)
            level_skips.append(skips)
        target_shapes.pop()  # Last level doesn't downsample


        x = self.middle_block(x, cond=cond)

        for level, skips, target_shape in zip(
            self.up_levels, level_skips[::-1], target_shapes[::-1]
        ):
            skips = skips[::-1]  # reverse the skips from the downsampling
            x = level(x, cond=cond, target_shape=target_shape, skip_inputs=skips)
        x = self.out_final(x)
        return x