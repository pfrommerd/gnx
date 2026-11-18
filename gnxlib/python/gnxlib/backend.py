"""Discover backends and devices."""

from ._native import (
    Backend as Backend,
    Device as Device,
    backends as backends,
    devices as devices,
    enable_jax_backend as enable_jax_backend,
)
