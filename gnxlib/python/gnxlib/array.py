"""Create symbolic or concrete arrays and mutable array references."""

from ._native import (
    Array as Array,
    ArrayRef as ArrayRef,
    from_host_bytes as from_host_bytes,
    new_ref as new_ref,
    placeholder as placeholder,
)


def from_numpy(value):
    """Copy a NumPy-compatible value into immutable GNX host storage."""
    import numpy

    value = numpy.asarray(value)
    packed = value.astype(value.dtype.newbyteorder("<"), order="C", copy=False)
    return from_host_bytes(value.shape, value.dtype.name, packed.tobytes(order="C"))


def from_torch(value):
    """Copy a Torch tensor into immutable GNX host storage."""
    return from_numpy(value.detach().cpu().numpy())
