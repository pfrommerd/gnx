import functools
import typing as tp
import inspect
import jax
import jax.numpy as jnp

from ..mlp import MLP

from ...core import nn


class ConvDef(tp.Protocol):
    def __call__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tp.Sequence[int],
        *,
        strides: int | tp.Sequence[int] = 1,
        padding: nn.PaddingLike = "SAME",
        precision: jax.lax.Precision | None = None,
        rngs: nn.Rngs,
    ) -> nn.Conv: ...


class NormDef(tp.Protocol):
    def __call__(
        self, num_features: int, *, spatial_dims: int, rngs: nn.Rngs
    ) -> nn.BatchNorm | nn.GroupNorm | nn.LayerNorm: ...


# A groupnorm that works on a batch size of 1

ActivationDef = tp.Callable[[jax.Array], jax.Array]


class ConditionerDef(tp.Protocol):
    def __call__(
        self, channels: int, cond_features: int, /, *, rngs: nn.Rngs
    ) -> tp.Callable[[jax.Array, jax.Array], jax.Array]: ...


class FiLMConditioner(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_features: int,
        *,
        activation: ActivationDef = jax.nn.relu,
        rngs: nn.Rngs,
    ):
        self.mlp = MLP(
            in_features=cond_features,
            out_features=2 * channels,
            hidden_features=cond_features,
            activation=activation,
            hidden_layers=1,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        film = self.mlp(cond)
        scale, shift = jnp.split(film, 2, axis=-1)
        assert x.shape[-1] == scale.shape[-1] == shift.shape[-1]
        scale = scale.reshape((1,) * (x.ndim - 1) + (-1,))
        shift = shift.reshape((1,) * (x.ndim - 1) + (-1,))
        return x * (1 + scale) + shift


class ShiftConditioner(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_features: int,
        *,
        activation: ActivationDef = jax.nn.relu,
        rngs: nn.Rngs,
    ):
        self.linear = nn.Linear(
            in_features=cond_features,
            out_features=channels,
            rngs=rngs,
        )
        self.activation = activation

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        cond = self.activation(cond)
        shift = self.linear(cond)
        shift = shift.reshape((1,) * (x.ndim - 1) + (-1,))
        return x + shift


def _has_keyword_arg(func: tp.Callable[..., tp.Any], name: str) -> bool:
    """Return True if func has keyword-only arguments with the given name."""
    return any(
        param.name == name
        and param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
        for param in inspect.signature(func).parameters.values()
    )


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tp.Sequence[int],
        strides: int | tp.Sequence[int] = 1,
        cond_features: int | None = None,
        *,
        operation: tp.Callable[[jax.Array], jax.Array] | None = None,
        dropout: float | None = None,
        skip_full_conv: bool = False,
        activation: ActivationDef = jax.nn.relu,
        precision: jax.lax.Precision | None = None,
        Conditioner: ConditionerDef | None = None,
        Conv: ConvDef = nn.Conv,
        Norm: NormDef = nn.BatchNorm,
        rngs: nn.Rngs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        if isinstance(strides, int):
            strides = (strides,) * len(kernel_size)
        self.activation = activation
        self.operation = operation
        if cond_features:
            assert (
                Conditioner is not None
            ), "Conditioner must be provided if cond_features is set"
            self.conditioner = Conditioner(out_channels, cond_features, rngs=rngs)
        self.norm_a = Norm(in_channels, spatial_dims=2, rngs=rngs)
        self.conv_a = Conv(
            in_channels,
            out_channels,
            kernel_size,
            strides=strides,
            precision=precision,
            rngs=rngs,
        )
        self.norm_b = Norm(out_channels, spatial_dims=2, rngs=rngs)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout, rngs=rngs)
        self.conv_b = Conv(
            out_channels, out_channels, kernel_size,
            strides=strides,
            precision=precision,
            rngs=rngs
        )
        if in_channels != out_channels or skip_full_conv:
            self.conv_proj = Conv(
                in_channels,
                out_channels,
                (3,) * len(kernel_size) if skip_full_conv else (1,) * len(kernel_size),
                strides=strides if skip_full_conv else (1,) * len(kernel_size),
                precision=precision,
                rngs=rngs,
            )
        else:
            self.conv_proj = None

    def __call__(self, x, cond=None, target_shape=None):
        assert (
            x.shape[-1] == self.conv_a.in_features
        ), f"Expected input with {self.conv_a.in_features} channels, got {x.shape[-1]}"
        residual = x
        x = self.norm_a(x)
        x = self.activation(x)

        # If an operation (upsampling/downsmapling) is provided,
        # apply it to h and the residual connection
        if self.operation is not None:
            op = (
                functools.partial(self.operation, target_shape=target_shape)  # type: ignore
                if _has_keyword_arg(self.operation, "target_shape")
                else self.operation
            )
            residual = op(residual)
            x = op(x)

        x = self.conv_a(x)

        if cond is not None:
            x = self.conditioner(x, cond)

        x = self.norm_b(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv_b(x)
        # If the input and output shapes are different,
        # we need to project the input
        if x.shape[-1] != residual.shape[-1]:
            assert self.conv_proj is not None
            residual = self.conv_proj(residual)
        x = x + residual

        return x


class BlockSequential(nn.Sequential):
    def __init__(
        self,
        *fns: tp.Callable[..., tp.Any],
        skip_in_layers: tp.Sequence[int] = (),
        skip_out_layers: tp.Sequence[int | None] = (),
    ):
        super().__init__(*fns)
        self.skip_in_layers = skip_in_layers
        self.skip_out_layers = skip_out_layers

    def __call__(
        self,
        *args,
        rngs: nn.Rngs | None = None,
        cond: jax.Array | None = None,
        # Whether to return the skips from the layers
        skip_inputs: tp.Sequence[tp.Any] = (),
        return_skip_outputs: bool = False,
        # For the last layer, we may want to pass a target shape
        target_shape: tp.Sequence[int] | None = None,
        **kwargs,
    ) -> tp.Any:
        (output,) = args
        skip_inputs = list(skip_inputs)
        skip_outputs = []
        # if None is in the skip_out_layers, skip the input directly
        if None in self.skip_out_layers:
            skip_outputs.append(output)
        for i, f in enumerate(self.layers):
            if not callable(f):
                raise TypeError(f"Sequence[{i}] is not callable: {f}")
            layer_kwargs = dict(kwargs)
            if rngs is not None and _has_keyword_arg(f, "rngs"):
                layer_kwargs["rngs"] = rngs
            if cond is not None and _has_keyword_arg(f, "cond"):
                layer_kwargs["cond"] = cond
            # For the last layer, we may want to pass a target shape
            if target_shape is not None and _has_keyword_arg(f, "target_shape"):
                layer_kwargs["target_shape"] = target_shape
            # Wether to consume a skip input
            if i in self.skip_in_layers:
                assert len(skip_inputs) > 0
                output = jax.tree.map(
                    lambda x, s: jnp.concatenate((x, s), axis=-1),
                    output,
                    skip_inputs.pop(0),
                )
            output = f(output, **layer_kwargs)
            if i in self.skip_out_layers:
                skip_outputs.append(output)
        assert not skip_inputs, f"Not all skip inputs were used: {skip_inputs}"
        if return_skip_outputs:
            return output, skip_outputs
        return output
