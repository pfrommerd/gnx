"""Create symbolic or concrete arrays and mutable array references."""

from ._native import (
    Array as Array,
    ArrayRef as ArrayRef,
    from_host_bytes as from_host_bytes,
    from_numpy_array as _from_numpy_array,
    new_ref as new_ref,
    placeholder as placeholder,
)


def from_numpy(value):
    """Copy a NumPy-compatible value into immutable GNX host storage."""
    import numpy

    return _from_numpy_array(numpy.asarray(value))


def from_torch(value):
    """Copy a Torch tensor into immutable GNX host storage."""
    return from_numpy(value.detach().cpu().numpy())
