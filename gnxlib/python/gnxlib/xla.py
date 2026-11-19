"""Native XLA execution (requires a build with the Cargo `xla` feature)."""

try:
    from ._native import XlaDevice as XlaDevice, xla_devices as devices
except ImportError as error:
    raise ImportError("gnxlib.xla requires a build with the 'xla' feature") from error
