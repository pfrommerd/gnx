import contextlib
import math
import typing as tp
from pathlib import Path

import warnings
import jax
import jax.numpy as jnp
import numpy as np

from ...core import nn
from ...util import fetch_util

# Type aliases
Array = tp.Any
Padding = str | tp.Sequence[tuple[int, int]]
Shape = tuple[int, ...]

INCEPTION_URL = (
    "https://www.dropbox.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle?dl=1"
)


class InceptionV3(nn.Module):
    def __init__(
        self,
        include_head: bool = True,
        num_classes: int = 1000,
        transform_input: bool = False,
        aux_logits: bool = False,
        resize_input: bool = True,
        *,
        weights: dict | None = None,
        precision: jax.lax.Precision | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: nn.Rngs,
    ):
        self.include_head = include_head
        self.num_classes = num_classes
        self.transform_input = transform_input
        self.aux_logits = aux_logits
        self.resize_input = resize_input

        use = lambda s: weights[s] if weights is not None else None

        self.block0 = nn.Sequential(
            BasicConv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), strides=(2, 2), padding="valid",
                rngs=rngs, precision=precision, dtype=dtype, weights=use("Conv2d_1a_3x3")),
            BasicConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding="valid",
                rngs=rngs, precision=precision, dtype=dtype, weights=use("Conv2d_2a_3x3")),
            BasicConv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same",
                rngs=rngs, precision=precision, dtype=dtype, weights=use("Conv2d_2b_3x3")
            )
        )  # fmt: skip
        self.block1 = nn.Sequential(
            BasicConv2d(in_channels=64, out_channels=80, kernel_size=(1, 1), padding="valid",
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Conv2d_3b_1x1")),
            BasicConv2d(in_channels=80, out_channels=192, kernel_size=(3, 3), padding="valid",
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Conv2d_4a_3x3")),
        )  # fmt: skip
        self.block2 = nn.Sequential(
            InceptionA(in_channels=192, pool_features=32,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_5b")),
            InceptionA(in_channels=256, pool_features=64,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_5c")),
            InceptionA(in_channels=288, pool_features=64,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_5d")),
            InceptionB(in_channels=288,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_6a")),
            InceptionC(in_channels=768, channels_7x7=128,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_6b")),
            InceptionC(in_channels=768, channels_7x7=160,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_6c")),
            InceptionC(in_channels=768, channels_7x7=160,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_6d")),
            InceptionC(in_channels=768, channels_7x7=192,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_6e"))
        )  # fmt: skip
        if aux_logits:
            self.aux = InceptionAux(
                in_channels=768, num_classes=num_classes,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("AuxLogits")
            )  # fmt: skip
        else:
            self.aux = None
        self.block3 = nn.Sequential(
            InceptionD(in_channels=768,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_7a")),
            InceptionE(pooling=avg_pool, in_channels=1280,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_7b")),
            InceptionE(pooling=nn.util.max_pool, in_channels=2048,
                dtype=dtype, precision=precision, rngs=rngs, weights=use("Mixed_7c"))
        )  # fmt: skip
        self.dropout = nn.Dropout(rate=0.5, rngs=rngs)
        self.fc = nn.Linear(
            in_features=2048,
            out_features=num_classes,
            precision=precision,
            dtype=dtype,
            rngs=rngs,
        )

    @staticmethod
    def load_fid_pretrained(
        quiet=False,
        device: str | None = None,
        precision: jax.lax.Precision | None = None,
        resize_input: bool = True,
        include_head: bool = False,
    ) -> "InceptionV3":
        PATH = (
            Path.home() / ".cache" / "fid-inception" / "inception_v3_weights_fid.pickle"
        )
        if not PATH.exists():
            fetch_util.download_url(INCEPTION_URL, PATH, quiet=quiet)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(PATH, "rb") as f:
                weights = np.load(f, allow_pickle=True)
        dev = (
            jax.default_device(device)
            if device is not None
            else contextlib.nullcontext()
        )
        if device:
            weights = jax.device_put(weights, device)
        with dev:
            model = InceptionV3(
                rngs=nn.Rngs(42),
                weights=weights,
                precision=precision,
                include_head=include_head,
                resize_input=resize_input,
            )
            model.eval_mode()  # set to eval mode
            # make the model pure (i.e. no array refs)
            model = nn.pure(model)
            return model

    @jax.jit
    def __call__(self, x) -> jax.Array:
        assert x.ndim == 3 or x.ndim == 4
        add_batch = x.ndim == 3
        x = jnp.expand_dims(x, axis=0) if add_batch else x
        if x.shape[1:] != (299, 299, 3) and self.resize_input:
            x = jax.image.resize(
                x, (x.shape[0], 299, 299, 3), jax.image.ResizeMethod.LINEAR
            )
        x = self._transform_input(x)
        x = self.block0(x)
        x = nn.util.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = self.block1(x)
        x = nn.util.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = self.block2(x)
        aux = self.aux(x) if self.aux_logits and self.aux is not None else None
        x = self.block3(x)
        x = jnp.mean(x, axis=(-2, -3), keepdims=True)
        if not self.include_head:
            if add_batch:
                x = jnp.squeeze(x, axis=0)
            # Get rid of the spatial dimensions
            x = x.squeeze(axis=(-2, -3))
            return x
        x = self.dropout(x)
        x = jnp.reshape(x, x.shape[:-3] + (math.prod(x.shape[-3:]),))
        x = self.fc(x)
        x = jnp.squeeze(x, axis=0) if add_batch else x
        if self.aux_logits and self.aux is not None:
            return x, aux  # type: ignore
        return x

    def _transform_input(self, x):
        if self.transform_input:
            scale = jnp.array([0.229, 0.224, 0.225]) / 0.5
            offset = (jnp.array([0.485, 0.456, 0.406]) - 0.5) / 0.5
            scale = jnp.reshape(scale, (1,) * (x.ndim - 1) + (3,))
            offset = jnp.reshape(offset, (1,) * (x.ndim - 1) + (3,))
            x = x * scale + offset
        return x


def _pretrained_init(value: jax.Array) -> jax.nn.initializers.Initializer:
    def initializer(key, shape, dtype=jnp.float32, out_sharding=None):
        if shape != value.shape:
            raise ValueError(f"Shape mismatch: expected {value.shape}, got {shape}")
        return value

    return initializer


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        strides: tuple[int, int] = (1, 1),
        padding: Padding = "valid",
        use_bias: bool = False,
        precision: jax.lax.Precision | None = None,
        dtype: jnp.dtype = jnp.float32,
        *,
        weights: dict[str, Array] | None = None,
        rngs: nn.Rngs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        conv_kernel_init = (
            nn.initializers.default_kernel_init
            if weights is None
            else _pretrained_init(weights["conv"]["kernel"])
        )
        conv_bias_init = (
            nn.initializers.default_bias_init
            if not use_bias or weights is None
            else _pretrained_init(weights["conv"]["bias"])
        )
        self.conv = nn.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            kernel_init=conv_kernel_init,
            bias_init=conv_bias_init,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            precision=precision,
            dtype=dtype,
            rngs=rngs,
        )
        bn_bias_init, bn_scale_init = (
            (nn.initializers.zeros_init(), nn.initializers.ones_init())
            if weights is None
            else (
                _pretrained_init(weights["bn"]["bias"]),
                _pretrained_init(weights["bn"]["scale"]),
            )
        )
        self.bn = nn.BatchNorm(
            num_features=out_channels,
            spatial_dims=2,
            epsilon=0.001,
            momentum=0.1,
            scale_init=bn_scale_init,
            bias_init=bn_bias_init,
            dtype=dtype,
            rngs=rngs,
        )
        # Update the batch statistics if weights are provided
        if weights is not None:
            self.bn.mean[...] = weights["bn"]["mean"]
            self.bn.var[...] = weights["bn"]["var"]

    def __call__(self, x: Array) -> Array:
        x = self.conv(x)
        x = self.bn(x)
        x = jax.nn.relu(x)
        return x


class InceptionA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        weights: dict | None = None,
        rngs: nn.Rngs,
    ):
        use = lambda s: weights[s] if weights is not None else None
        self.branch1x1 = BasicConv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch1x1"), rngs=rngs
        )  # fmt: skip
        self.branch5x5_1 = BasicConv2d(
            in_channels=in_channels, out_channels=48, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch5x5_1"), rngs=rngs
        )  # fmt: skip
        self.branch5x5_2 = BasicConv2d(
            in_channels=48, out_channels=64, kernel_size=(5, 5),
            padding="same", dtype=dtype, precision=precision, weights=use("branch5x5_2"), rngs=rngs
        )  # fmt: skip
        self.branch3x3dbl_1 = BasicConv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch3x3dbl_1"), rngs=rngs
        )  # fmt: skip
        self.branch3x3dbl_2 = BasicConv2d(
            in_channels=64, out_channels=96, kernel_size=(3, 3),
            padding="same", dtype=dtype, precision=precision, weights=use("branch3x3dbl_2"), rngs=rngs
        )  # fmt: skip
        self.branch3x3dbl_3 = BasicConv2d(
            in_channels=96, out_channels=96, kernel_size=(3, 3),
            padding="same", dtype=dtype, precision=precision, weights=use("branch3x3dbl_3"), rngs=rngs
        )  # fmt: skip
        self.branch_pool = BasicConv2d(
            in_channels=in_channels, out_channels=pool_features, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch_pool"), rngs=rngs
        )  # fmt: skip

    def __call__(self, x: Array) -> Array:
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding="same")
        branch_pool = self.branch_pool(branch_pool)
        output = jnp.concatenate(
            (branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=-1
        )
        return output


class InceptionB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        weights: dict | None = None,
        rngs: nn.Rngs,
    ):
        use = lambda s: weights[s] if weights is not None else None
        self.branch3x3 = BasicConv2d(
            in_channels=in_channels, out_channels=384, kernel_size=(3, 3),
            strides=(2, 2), padding="valid", dtype=dtype, precision=precision, weights=use("branch3x3"), rngs=rngs
        )  # fmt: skip
        self.branch3x3dbl_1 = BasicConv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch3x3dbl_1"), rngs=rngs
        )  # fmt: skip
        self.branch3x3dbl_2 = BasicConv2d(
            in_channels=64, out_channels=96, kernel_size=(3, 3),
            padding="same", dtype=dtype, precision=precision, weights=use("branch3x3dbl_2"), rngs=rngs
        )  # fmt: skip
        self.branch3x3dbl_3 = BasicConv2d(
            in_channels=96, out_channels=96, kernel_size=(3, 3),
            strides=(2, 2), padding="valid", dtype=dtype, precision=precision, weights=use("branch3x3dbl_3"), rngs=rngs
        )  # fmt: skip

    def __call__(self, x: Array) -> Array:
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = nn.util.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        output = jnp.concatenate((branch3x3, branch3x3dbl, branch_pool), axis=-1)
        return output


class InceptionC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        weights: dict | None = None,
        rngs: nn.Rngs,
    ):
        use = lambda s: weights[s] if weights is not None else None
        self.branch1x1 = BasicConv2d(
            in_channels=in_channels, out_channels=192, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch1x1"), rngs=rngs
        )  # fmt: skip
        self.branch7x7_1 = BasicConv2d(
            in_channels=in_channels, out_channels=channels_7x7, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch7x7_1"), rngs=rngs
        )  # fmt: skip
        self.branch7x7_2 = BasicConv2d(
            in_channels=channels_7x7, out_channels=channels_7x7, kernel_size=(1, 7), padding=((0, 0), (3, 3)),
            dtype=dtype, precision=precision, weights=use("branch7x7_2"), rngs=rngs
        )  # fmt: skip
        self.branch7x7_3 = BasicConv2d(
            in_channels=channels_7x7, out_channels=192, kernel_size=(7, 1), padding=((3, 3), (0, 0)),
            dtype=dtype, precision=precision, weights=use("branch7x7_3"), rngs=rngs
        )  # fmt: skip
        self.branch7x7dbl_1 = BasicConv2d(
            in_channels=in_channels, out_channels=channels_7x7, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch7x7dbl_1"), rngs=rngs
        )  # fmt: skip
        self.branch7x7dbl_2 = BasicConv2d(
            in_channels=channels_7x7, out_channels=channels_7x7, kernel_size=(7, 1), padding=((3, 3), (0, 0)),
            dtype=dtype, precision=precision, weights=use("branch7x7dbl_2"), rngs=rngs
        )  # fmt: skip
        self.branch7x7dbl_3 = BasicConv2d(
            in_channels=channels_7x7, out_channels=channels_7x7, kernel_size=(1, 7), padding=((0, 0), (3, 3)),
            dtype=dtype, precision=precision, weights=use("branch7x7dbl_3"), rngs=rngs
        )  # fmt: skip
        self.branch7x7dbl_4 = BasicConv2d(
            in_channels=channels_7x7, out_channels=channels_7x7, kernel_size=(7, 1), padding=((3, 3), (0, 0)),
            dtype=dtype, precision=precision, weights=use("branch7x7dbl_4"), rngs=rngs
        )  # fmt: skip
        self.branch7x7dbl_5 = BasicConv2d(
            in_channels=channels_7x7, out_channels=192, kernel_size=(1, 7), padding=((0, 0), (3, 3)),
            dtype=dtype, precision=precision, weights=use("branch7x7dbl_5"), rngs=rngs
        )  # fmt: skip
        self.branch_pool = BasicConv2d(
            in_channels=in_channels, out_channels=192, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch_pool"), rngs=rngs
        )  # fmt: skip

    def __call__(self, x: Array) -> Array:
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding="same")
        branch_pool = self.branch_pool(branch_pool)
        output = jnp.concatenate(
            (branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=-1
        )
        return output


class InceptionD(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        weights: dict | None = None,
        rngs: nn.Rngs,
    ):
        use = lambda s: weights[s] if weights is not None else None
        self.branch3x3_1 = BasicConv2d(
            in_channels=in_channels, out_channels=192, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch3x3_1"), rngs=rngs
        )  # fmt: skip
        self.branch3x3_2 = BasicConv2d(
            in_channels=192, out_channels=320, kernel_size=(3, 3),
            strides=(2, 2), padding="valid", dtype=dtype, precision=precision, weights=use("branch3x3_2"), rngs=rngs
        )  # fmt: skip
        self.branch7x7x3_1 = BasicConv2d(
            in_channels=in_channels, out_channels=192, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch7x7x3_1"), rngs=rngs
        )  # fmt: skip
        self.branch7x7x3_2 = BasicConv2d(
            in_channels=192, out_channels=192, kernel_size=(1, 7),
            padding=((0, 0), (3, 3)), dtype=dtype, precision=precision, weights=use("branch7x7x3_2"), rngs=rngs
        )  # fmt: skip
        self.branch7x7x3_3 = BasicConv2d(
            in_channels=192, out_channels=192, kernel_size=(7, 1),
            padding=((3, 3), (0, 0)), dtype=dtype, precision=precision, weights=use("branch7x7x3_3"), rngs=rngs
        )  # fmt: skip
        self.branch7x7x3_4 = BasicConv2d(
            in_channels=192, out_channels=192, kernel_size=(3, 3),
            strides=(2, 2), padding="valid", dtype=dtype, precision=precision, weights=use("branch7x7x3_4"), rngs=rngs
        )  # fmt: skip

    def __call__(self, x: Array) -> Array:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = nn.util.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        output = jnp.concatenate((branch3x3, branch7x7x3, branch_pool), axis=-1)
        return output


class InceptionE(nn.Module):
    def __init__(
        self,
        pooling: tp.Callable,
        in_channels: int,
        dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        weights: dict | None = None,
        rngs: nn.Rngs,
    ):
        use = lambda s: weights[s] if weights is not None else None
        self.branch1x1 = BasicConv2d(
            in_channels=in_channels, out_channels=320, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch1x1"), rngs=rngs
        )  # fmt: skip
        self.branch3x3_1 = BasicConv2d(
            in_channels=in_channels, out_channels=384, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch3x3_1"), rngs=rngs
        )  # fmt: skip
        self.branch3x3_2a = BasicConv2d(
            in_channels=384, out_channels=384, kernel_size=(1, 3),
            padding=((0, 0), (1, 1)), dtype=dtype, precision=precision, weights=use("branch3x3_2a"), rngs=rngs
        )  # fmt: skip
        self.branch3x3_2b = BasicConv2d(
            in_channels=384, out_channels=384, kernel_size=(3, 1),
            padding=((1, 1), (0, 0)), dtype=dtype, precision=precision, weights=use("branch3x3_2b"), rngs=rngs
        )  # fmt: skip
        self.branch3x3dbl_1 = BasicConv2d(
            in_channels=in_channels, out_channels=448, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch3x3dbl_1"), rngs=rngs
        )  # fmt: skip
        self.branch3x3dbl_2 = BasicConv2d(
            in_channels=448, out_channels=384, kernel_size=(3, 3),
            padding="same", dtype=dtype, precision=precision, weights=use("branch3x3dbl_2"), rngs=rngs
        )  # fmt: skip
        self.branch3x3dbl_3a = BasicConv2d(
            in_channels=384, out_channels=384, kernel_size=(1, 3),
            padding=((0, 0), (1, 1)), dtype=dtype, precision=precision, weights=use("branch3x3dbl_3a"), rngs=rngs
        )  # fmt: skip
        self.branch3x3dbl_3b = BasicConv2d(
            in_channels=384, out_channels=384, kernel_size=(3, 1),
            padding=((1, 1), (0, 0)), dtype=dtype, precision=precision, weights=use("branch3x3dbl_3b"), rngs=rngs
        )  # fmt: skip
        self.branch_pool = BasicConv2d(
            in_channels=in_channels, out_channels=192, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("branch_pool"), rngs=rngs
        )  # fmt: skip
        self.pooling = pooling

    def __call__(self, x: Array) -> Array:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3_a = self.branch3x3_2a(branch3x3)
        branch3x3_b = self.branch3x3_2b(branch3x3)
        branch3x3 = jnp.concatenate((branch3x3_a, branch3x3_b), axis=-1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl_a = self.branch3x3dbl_3a(branch3x3dbl)
        branch3x3dbl_b = self.branch3x3dbl_3b(branch3x3dbl)
        branch3x3dbl = jnp.concatenate((branch3x3dbl_a, branch3x3dbl_b), axis=-1)
        branch_pool = self.pooling(
            x, window_shape=(3, 3), strides=(1, 1), padding="same"
        )
        branch_pool = self.branch_pool(branch_pool)
        output = jnp.concatenate(
            (branch1x1, branch3x3, branch3x3dbl, branch_pool), axis=-1
        )
        return output


class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        *,
        weights: dict | None = None,
        rngs: nn.Rngs,
    ):
        use = lambda s: weights[s] if weights is not None else None
        self.conv0 = BasicConv2d(
            in_channels=in_channels, out_channels=128, kernel_size=(1, 1),
            dtype=dtype, precision=precision, weights=use("conv0"), rngs=rngs,
        )  # fmt: skip
        self.conv1 = BasicConv2d(
            in_channels=128, out_channels=768, kernel_size=(5, 5),
            dtype=dtype, precision=precision, weights=use("conv0"), rngs=rngs,
        )  # fmt: skip

        fc_kernel_init = (
            _pretrained_init(weights["fc"]["kernel"])
            if weights is not None
            else nn.initializers.default_kernel_init
        )
        fc_bias_init = (
            _pretrained_init(weights["fc"]["bias"])
            if weights is not None
            else nn.initializers.default_bias_init
        )
        self.dense = nn.Linear(
            in_features=768, out_features=num_classes, dtype=dtype, precision=precision, rngs=rngs,
            kernel_init=fc_kernel_init, bias_init=fc_bias_init
        )  # fmt: skip

    def __call__(self, x: Array) -> Array:
        x = avg_pool(x, window_shape=(5, 5), strides=(3, 3))
        x = self.conv0(x)
        x = self.conv1(x)
        x = jnp.mean(x, axis=(1, 2))
        x = x.reshape((x.shape[0], -1))
        x = self.dense(x)
        return x


def pool(
    inputs: Array,
    init: float,
    reduce_fn: tp.Callable,
    window_shape: Shape,
    strides: tp.Optional[Shape],
    padding: str | tp.Sequence[tuple[int, int]],
) -> Array:
    strides = strides or (1,) * len(window_shape)
    assert len(window_shape) == len(strides), f"len({window_shape}) == len({strides})"
    strides = (1,) + strides + (1,)
    dims = (1,) + window_shape + (1,)
    is_single_input = False
    if inputs.ndim == len(dims) - 1:
        inputs = inputs[None]
        is_single_input = True
    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    if not isinstance(padding, str):
        padding = tuple(padding)
        assert len(padding) == len(
            window_shape
        ), f"padding {padding} must specify pads for same number of dims as window_shape {window_shape}"
        assert all(
            len(x) == 2 for x in padding
        ), f"each entry in padding {padding} must be length 2"
        padding = ((0, 0),) + padding + ((0, 0),)
    y = jax.lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


def avg_pool(
    inputs: Array,
    window_shape: Shape,
    strides: tp.Optional[Shape] = None,
    padding: str = "VALID",
) -> Array:
    assert inputs.ndim == 4
    assert len(window_shape) == 2
    y = pool(inputs, 0.0, jax.lax.add, window_shape, strides, padding)
    ones = jnp.ones(shape=(1, inputs.shape[1], inputs.shape[2], 1)).astype(inputs.dtype)
    counts = jax.lax.conv_general_dilated(
        ones,
        jnp.expand_dims(jnp.ones(window_shape).astype(inputs.dtype), axis=(-2, -1)),
        window_strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=1,
    )
    y = y / counts
    return y
