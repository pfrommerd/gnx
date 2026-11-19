import asyncio
import struct
import sys
from uuid import UUID

import pytest

from gnxlib import array


def test_cpu_xla_without_jax(monkeypatch):
    xla = pytest.importorskip("gnxlib.xla", exc_type=ImportError)
    monkeypatch.setitem(sys.modules, "jax", None)
    (device,) = xla.devices()
    assert device.platform == "cpu"
    assert isinstance(device.host_uuid, UUID)
    assert xla.devices()[0].host_uuid == device.host_uuid
    a = array.from_host_bytes([2], "float32", struct.pack("<2f", 1, 2))
    b = array.from_host_bytes([2], "float32", struct.pack("<2f", 3, 4))
    c = a + b
    with pytest.raises(TypeError):
        device.eval(c, allow_copy=True)
    asyncio.run(device.eval(c))
    d = c + c
    asyncio.run(device.eval(d))
    assert struct.unpack("<2f", device.to_host_bytes(d)) == (8, 12)


def test_mutable_array_import_copies():
    numpy = pytest.importorskip("numpy")
    source = numpy.arange(12, dtype=">f4").reshape(3, 4)[::-1, ::2]
    copied = array.from_numpy(source)
    source[:] = 99
    assert copied.shape == (3, 2)
    assert copied.dtype == "float32"
    assert "symbolic=False" in repr(copied)
    try:
        from gnxlib import xla
    except ImportError:
        return
    (device,) = xla.devices()
    assert struct.unpack("<6f", device.to_host_bytes(copied)) == (8, 10, 4, 6, 0, 2)
    result = copied + copied
    asyncio.run(device.eval(result))
    assert struct.unpack("<6f", device.to_host_bytes(result)) == (16, 20, 8, 12, 0, 4)


@pytest.mark.parametrize("values", [1.5, []])
def test_numpy_scalar_and_empty_snapshots(values):
    numpy = pytest.importorskip("numpy")
    xla = pytest.importorskip("gnxlib.xla", exc_type=ImportError)
    (device,) = xla.devices()
    source = numpy.asarray(values, dtype="float32")
    copied = array.from_numpy(source)
    result = copied + copied
    asyncio.run(device.eval(result))
    assert device.to_host_bytes(result) == (source + source).tobytes()


def test_torch_import_copies():
    torch = pytest.importorskip("torch")
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    copied = array.from_torch(source)
    assert copied.shape == (2,)
    assert copied.dtype == "float32"
    source.detach().fill_(10)
    try:
        from gnxlib import xla
    except ImportError:
        return
    (device,) = xla.devices()
    result = copied + copied
    asyncio.run(device.eval(result))
    assert struct.unpack("<2f", device.to_host_bytes(result)) == (2, 4)
