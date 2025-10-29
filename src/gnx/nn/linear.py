from .core import Module, Param, Rngs
from .util import (
    PrecisionLike,
    PaddingLike,
    DTypeLike,
    PromoteDTypeFn,
    promote_dtype,
)
from . import initializers, util

import jax
import typing as tp
import numpy as np
import jax.numpy as jnp


class Linear(Module):
    """A linear transformation applied over the last dimension of the input.

    Args:
      in_features: the number of input features.
      out_features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see ``jax.lax.Precision``
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
      dot_general: dot product function.
      promote_dtype: function to promote the dtype of the arrays to the desired
        dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
        and a ``dtype`` keyword argument, and return a tuple of arrays with the
        promoted dtype.
      rngs: rng key.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: initializers.Initializer = initializers.default_kernel_init,
        bias_init: (
            initializers.Initializer | initializers.BiasInitializer
        ) = initializers.default_bias_init,
        dot_general: tp.Any = jax.lax.dot_general,
        promote_dtype: PromoteDTypeFn = promote_dtype,
        rngs: Rngs,
    ):
        kernel_key = rngs.params()
        self.kernel = Param(
            kernel_init(kernel_key, (in_features, out_features), param_dtype)
        )
        if use_bias:
            bias_init = initializers.as_bias_initializer(bias_init)
            self.bias = Param(
                bias_init(
                    rngs.params(),
                    (out_features,),
                    param_dtype,
                    kernel_shape=self.kernel.value.shape,
                )
            )
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dot_general = dot_general
        self.promote_dtype = promote_dtype

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        kernel = self.kernel[...]
        bias = self.bias[...] if self.bias is not None else None

        if inputs.shape[-1] != kernel.shape[0]:
            raise ValueError(
                f"Invalid input shape: {inputs.shape}, expected {kernel.shape[0]} features"
            )

        inputs, kernel, bias = self.promote_dtype(
            (inputs, kernel, bias), dtype=self.dtype
        )
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        assert self.use_bias == (bias is not None)
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class Conv(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tp.Sequence[int],
        strides: tp.Union[None, int, tp.Sequence[int]] = 1,
        *,
        padding: PaddingLike = "SAME",
        input_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
        kernel_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        mask: tp.Optional[jax.Array] = None,
        dtype: tp.Optional[DTypeLike] = None,
        param_dtype: DTypeLike = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: initializers.Initializer = initializers.default_kernel_init,
        bias_init: (
            initializers.Initializer | initializers.BiasInitializer
        ) = initializers.default_bias_init,
        conv_general_dilated: tp.Callable = jax.lax.conv_general_dilated,
        promote_dtype: PromoteDTypeFn = util.promote_dtype,
        preferred_element_type: DTypeLike | None = None,
        rngs: Rngs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        else:
            kernel_size = tuple(kernel_size)

        kernel_shape = kernel_size + (
            in_features // feature_group_count,
            out_features,
        )
        kernel_key = rngs.params()
        self.kernel_shape = kernel_shape
        self.kernel = Param(kernel_init(kernel_key, kernel_shape, param_dtype))

        if use_bias:
            bias_shape = (out_features,)
            bias_key = rngs.params()
            self.bias = Param(
                initializers.as_bias_initializer(bias_init)(
                    bias_key,
                    bias_shape,
                    param_dtype,
                    kernel_shape=self.kernel[...].shape,
                )
            )
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.input_dilation = input_dilation
        self.kernel_dilation = kernel_dilation
        self.feature_group_count = feature_group_count
        self.use_bias = use_bias
        self.mask = mask
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.conv_general_dilated = conv_general_dilated
        self.promote_dtype = promote_dtype
        self.preferred_element_type = preferred_element_type

    def __call__(self, inputs: jax.Array) -> jax.Array:
        assert isinstance(self.kernel_size, tuple)
        kernel_size = self.kernel_size

        def maybe_broadcast(
            x: tp.Optional[tp.Union[int, tp.Sequence[int]]],
        ) -> tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        # Combine all input batch dimensions into a single leading batch axis.
        num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
        input_batch_shape = None
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            flat_input_shape = (-1,) + inputs.shape[num_batch_dimensions:]
            inputs = jnp.reshape(inputs, flat_input_shape)

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax = util.canonicalize_padding(self.padding, len(kernel_size))
        if padding_lax in ("CIRCULAR", "REFLECT"):
            assert isinstance(padding_lax, str)
            kernel_size_dilated = [
                (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
            ]
            zero_pad: list[tuple[int, int]] = [(0, 0)]
            pads = (
                zero_pad
                + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
                + [(0, 0)]
            )
            padding_mode = {"CIRCULAR": "wrap", "REFLECT": "reflect"}[padding_lax]
            inputs = jnp.pad(inputs, pads, mode=padding_mode)
            padding_lax = "VALID"
        elif padding_lax == "CAUSAL":
            if len(kernel_size) != 1:
                raise ValueError(
                    "Causal padding is only implemented for 1D convolutions."
                )
            left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
            pads = [(0, 0), (left_pad, 0), (0, 0)]
            inputs = jnp.pad(inputs, pads)
            padding_lax = "VALID"

        dimension_numbers = _conv_dimension_numbers(inputs.shape)

        # One shared convolutional kernel for all pixels in the output.
        assert self.in_features % self.feature_group_count == 0

        if self.mask is not None and self.mask.shape != self.kernel_shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. "
                f"Shapes are: {self.mask.shape}, {self.kernel_shape}"
            )

        kernel = self.kernel[...]

        if self.mask is not None:
            kernel *= self.mask

        bias = self.bias[...] if self.bias is not None else None

        inputs, kernel, bias = self.promote_dtype(
            (inputs, kernel, bias), dtype=self.dtype
        )

        # We use conv_kwargs for BC compatibility with
        # user custom self.conv_general_dilated method which may not have
        # preferred_element_type argument to avoid breaking
        # existing code
        conv_kwargs = {}
        if self.preferred_element_type is not None:
            conv_kwargs["preferred_element_type"] = self.preferred_element_type

        y = self.conv_general_dilated(
            inputs,
            kernel,
            strides,
            padding_lax,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
            **conv_kwargs,
        )

        if self.use_bias:
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)  # type: ignore
            y += bias

        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)
        return y
