import sys
from types import ModuleType, SimpleNamespace
from uuid import UUID

import pytest

from gnxlib import jax


def native_device(*, process_index=0, local_hardware_id=2):
    return SimpleNamespace(
        platform="cpu",
        process_index=process_index,
        id=7,
        local_hardware_id=local_hardware_id,
        device_kind="test CPU",
    )


def test_devices_exposes_jax_metadata_and_native_handle(monkeypatch):
    calls = []
    native_devices = []
    fake_jax = ModuleType("jax")

    def devices(backend=None):
        calls.append(backend)
        native_devices.append(native_device())
        return [native_devices[-1]]

    fake_jax.devices = devices
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    (first,) = jax.devices()
    (repeated,) = jax.devices(backend="cpu")
    assert calls == [None, "cpu"]
    assert isinstance(first, jax.JaxDevice)
    assert first.native is native_devices[0]
    assert repeated.native is native_devices[1]
    assert isinstance(first.uuid, UUID)
    assert first == repeated
    assert hash(first) == hash(repeated)
    assert first.platform == "cpu"
    assert first.hardware_id == "2"
    assert first.hardware_kind == "test CPU"


def test_from_native_wraps_remote_jax_device():
    remote = native_device(process_index=1, local_hardware_id=None)
    device = jax.JaxDevice.from_native(remote)
    assert device.native is remote
    assert device.hardware_id == "7"
    assert device != jax.JaxDevice.from_native(native_device())


def test_jax_is_optional_until_discovery(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", None)
    with pytest.raises(ModuleNotFoundError):
        jax.devices()
