import asyncio
import gc
import struct
import weakref

import pytest

import gnxlib
from gnxlib import array


def scalar():
    return array.placeholder(())


def vector(n=2):
    return array.placeholder((n,))


# ----- map --------------------------------------------------------------------


def test_map_records_a_map_region_and_reuses_its_trace():
    calls = []

    def add(a, b):
        calls.append(1)
        assert a.shape == (2,) and b.shape == (2,)
        return a + b

    mapped = gnxlib.map(add, in_axes=(0, None))
    assert isinstance(mapped, gnxlib.Transformed)
    assert mapped.__wrapped__ is add
    x, w = array.placeholder((3, 2)), vector()
    assert mapped(x, w).shape == (3, 2)
    trace = mapped.trace(x, w)
    assert "map::map<in_axes: [0], out_axes: [0], size: 3>" in str(trace)
    assert trace.input_count == 2 and trace.output_count == 1
    assert mapped(x, w).shape == (3, 2)
    assert len(calls) == 1


def test_map_axes_specs():
    x = array.placeholder((2, 5))
    # Map over axis 1, stack along axis 1.
    assert gnxlib.map(lambda a: a + a, in_axes=1, out_axes=1)(x).shape == (2, 5)
    assert gnxlib.map(lambda a: a + a, in_axes=1)(x).shape == (5, 2)
    # Pytree prefixes per argument, and keyword arguments on axis 0.
    pair = {"a": x, "b": array.placeholder((2,))}
    mapped = gnxlib.map(lambda p, *, k: (p["a"] + p["a"], k), in_axes=({"a": 0, "b": None},))
    out, k = mapped(pair, k=array.placeholder((2, 3)))
    assert out.shape == (2, 5) and k.shape == (2, 3)
    # Unmapped arguments need an explicit axis size.
    broadcast = gnxlib.map(lambda a: a, in_axes=None, axis_size=4)
    assert broadcast(vector(3)).shape == (4, 3)


def test_map_spec_errors():
    x = array.placeholder((3, 2))
    with pytest.raises(ValueError, match="in_axes has 2 entries"):
        gnxlib.map(lambda a: a, in_axes=(0, 0))(x)
    with pytest.raises(ValueError, match="non-negative"):
        gnxlib.map(lambda a: a, in_axes=-1)
    with pytest.raises(TypeError, match="in_axes"):
        gnxlib.map(lambda a: a, in_axes="0")
    with pytest.raises(ValueError):
        gnxlib.map(lambda a: a, in_axes=2)(x)
    with pytest.raises(TypeError, match="callable"):
        gnxlib.map(1)


# ----- autodiff -----------------------------------------------------------------


def test_grad_follows_argnums():
    s = scalar()
    f = lambda a, b: a + a + b  # noqa: E731
    g0 = gnxlib.grad(f)
    assert isinstance(g0(s, s), array.Array)
    trace = g0.trace(s, s)
    assert "ad::vjp<of: [0], wrt: [0]>" in str(trace)
    both = gnxlib.grad(f, argnums=(1, 0))
    da = both(s, s)
    assert isinstance(da, tuple) and len(da) == 2
    assert "wrt: [0, 1]" in str(both.trace(s, s))
    # Keyword arguments are never differentiated.
    kw = gnxlib.grad(lambda a, *, b: a + b)
    assert kw(s, b=s).shape == ()


def test_grad_errors():
    s = scalar()
    with pytest.raises(TypeError, match="requires at least 2 positional"):
        gnxlib.grad(lambda a, b: a, argnums=1)(s)
    with pytest.raises(ValueError, match="one floating-point scalar"):
        gnxlib.grad(lambda a: a)(vector())
    with pytest.raises(ValueError, match="duplicate"):
        gnxlib.grad(lambda a: a, argnums=(0, 0))
    with pytest.raises(LookupError, match="body failed"):

        def fail(a):
            raise LookupError("body failed")

        gnxlib.grad(fail)(s)


def test_value_and_grad_and_hessian_shapes():
    s, v = scalar(), vector(3)
    value, grads = gnxlib.value_and_grad(lambda a, b: a + b, argnums=(0,))(s, s)
    assert value.shape == () and isinstance(grads, tuple) and grads[0].shape == ()
    # The Hessian of a scalar function of a (3,) argument has shape (3, 3);
    # here the scalar comes from a mapped-out argument.
    h = gnxlib.hessian(lambda a, b: a + a, argnums=0)(s, v)
    assert h.shape == ()
    trace = gnxlib.hessian(lambda a: a + a).trace(s)
    assert "ad::" in str(trace) and "map::map" in str(trace)


def test_jvp_and_vjp():
    v = vector()
    out, tangent = gnxlib.jvp(lambda a, b: a + b, (v, v), (v, v))
    assert out.shape == tangent.shape == (2,)
    with pytest.raises(TypeError, match="primals must be a tuple"):
        gnxlib.jvp(lambda a: a, [v], (v,))
    with pytest.raises(ValueError, match="same structure"):
        gnxlib.jvp(lambda a, b: a + b, (v, v), (v,))

    calls = []

    def f(a, b):
        calls.append(1)
        return {"sum": a + b}

    out, pullback = gnxlib.vjp(f, v, v)
    assert isinstance(pullback, gnxlib.VjpFunction)
    assert out["sum"].shape == (2,)
    ca, cb = pullback({"sum": v})
    assert ca.shape == cb.shape == (2,)
    pullback({"sum": v})
    assert len(calls) == 2  # primal + one pullback trace


def test_transformations_compose_and_nest_in_jit():
    s = scalar()

    @gnxlib.jit
    def outer(x):
        return gnxlib.grad(lambda a: a + a)(x)

    assert "ad::vjp" in str(outer.trace(s))
    per_example = gnxlib.map(gnxlib.grad(lambda w, x: w + x), in_axes=(None, 0))
    assert per_example(s, vector(4)).shape == (4,)


def test_transformed_is_collected_with_its_function():
    def f(a):
        return a

    mapped = gnxlib.map(f)
    ref = weakref.ref(f)
    f.cycle = mapped
    del f, mapped
    gc.collect()
    assert ref() is None


# ----- control flow ---------------------------------------------------------------


def test_cond_traces_both_branches():
    flag, v = array.placeholder((), dtype="bool"), vector()

    @gnxlib.jit
    def choose(p, a):
        return gnxlib.cond(p, lambda a: {"x": a + a}, lambda a: {"x": a}, a)

    result = choose(flag, v)
    assert result["x"].shape == (2,)
    text = str(choose.trace(flag, v))
    assert "cf::cond" in text and "array::add" in text
    assert gnxlib.cond(True, lambda: v, lambda: v + v).shape == (2,)
    with pytest.raises(TypeError, match="same structure"):
        gnxlib.cond(flag, lambda a: (a, a), lambda a: a, v)
    with pytest.raises(TypeError):
        gnxlib.cond(flag, lambda a: a, lambda a: array.placeholder((3,)), v)
    with pytest.raises(TypeError, match="branch selector type"):
        gnxlib.cond(array.placeholder(()), lambda a: a, lambda a: a, v)
    with pytest.raises(TypeError, match="scalar bool"):
        gnxlib.cond(1.0, lambda a: a, lambda a: a, v)


def test_switch():
    index, v = array.placeholder((), dtype="int32"), vector()

    @gnxlib.jit
    def pick(i, a):
        return gnxlib.switch(i, [lambda a: a, lambda a: a + a, lambda a: a + a + a], a)

    assert pick(index, v).shape == (2,)
    text = str(pick.trace(index, v))
    assert "cf::switch" in text
    assert gnxlib.switch(1, (lambda: v,)).shape == (2,)
    with pytest.raises(ValueError, match="at least one branch"):
        gnxlib.switch(index, [], v)


def test_while_loop():
    flag, v = array.placeholder((), dtype="bool"), vector()

    @gnxlib.jit
    def loop(p, a):
        return gnxlib.while_loop(lambda c: c[0], lambda c: (c[0], c[1] + c[1]), (p, a))

    p, a = loop(flag, v)
    assert p.dtype == "bool" and a.shape == (2,)
    assert "cf::while_loop" in str(loop.trace(flag, v))
    assert gnxlib.while_loop(lambda c: False, lambda c: c + c, v).shape == (2,)
    with pytest.raises(TypeError, match="structure of init_val"):
        gnxlib.while_loop(lambda c: False, lambda c: (c, c), v)
    with pytest.raises(TypeError):
        gnxlib.while_loop(lambda c: c, lambda c: c, v)


def test_scan():
    xs, init = array.placeholder((5, 2)), vector()
    seen = []

    @gnxlib.jit
    def cumulative(c, xs):
        def step(carry, x):
            seen.append(x.shape)
            total = carry + x
            return total, {"total": total}

        return gnxlib.scan(step, c, xs)

    carry, ys = cumulative(init, xs)
    assert carry.shape == (2,) and ys["total"].shape == (5, 2)
    assert seen == [(2,)]
    assert "cf::scan<carry: 1, length: 5>" in str(cumulative.trace(init, xs))
    carry, ys = gnxlib.scan(lambda c, x: (c + c, None), init, None, length=3)
    assert carry.shape == (2,) and ys is None
    with pytest.raises(TypeError, match=r"\(carry, y\) tuple"):
        gnxlib.scan(lambda c, x: c, init, xs)
    with pytest.raises(TypeError, match="structure of init"):
        gnxlib.scan(lambda c, x: ((c, c), x), init, xs)
    with pytest.raises(TypeError, match="requires a length"):
        gnxlib.scan(lambda c, x: (c, x), init, None)


# ----- numerics (XLA) ----------------------------------------------------------------


def f32(*values):
    return array.from_host_bytes([len(values)], "float32", struct.pack(f"<{len(values)}f", *values))


def f32_scalar(value):
    return array.from_host_bytes([], "float32", struct.pack("<f", value))


@pytest.fixture
def device():
    xla = pytest.importorskip("gnxlib.xla", exc_type=ImportError)
    (device,) = xla.devices()
    return device


def values(device, x):
    asyncio.run(device.eval(x))
    raw = device.to_host_bytes(x)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def test_numeric_map_and_autodiff(device):
    xs, w = f32(1, 2, 3, 4), f32(10, 20)
    mapped = gnxlib.map(lambda x, w: x + w, in_axes=(0, None))(
        array.from_host_bytes([2, 2], "float32", struct.pack("<4f", 1, 2, 3, 4)), w
    )
    assert values(device, mapped) == [11, 22, 13, 24]

    x = f32_scalar(3)
    assert values(device, gnxlib.grad(lambda a, b: a + a + a + b)(x, x)) == [3]
    db, da = gnxlib.grad(lambda a, b: a + a + b, argnums=(1, 0))(x, x)
    assert values(device, da) == [2] and values(device, db) == [1]
    value, g = gnxlib.value_and_grad(lambda a: a + a)(x)
    assert values(device, value) == [6] and values(device, g) == [2]
    assert values(device, gnxlib.hessian(lambda a: a + a)(x)) == [0]

    out, tangent = gnxlib.jvp(lambda a, b: a + a + b, (xs, xs), (f32(1, 1, 1, 1), f32(0, 1, 0, 1)))
    assert values(device, out) == [3, 6, 9, 12]
    assert values(device, tangent) == [2, 3, 2, 3]
    out, pullback = gnxlib.vjp(lambda a, b: a + a + b, w, w)
    assert values(device, out) == [30, 60]
    ca, cb = pullback(f32(1, 2))
    assert values(device, ca) == [2, 4] and values(device, cb) == [1, 2]


def test_numeric_control_flow(device):
    w = f32(1, 2)
    true = array.from_host_bytes([], "bool", b"\x01")
    false = array.from_host_bytes([], "bool", b"\x00")
    assert values(device, gnxlib.cond(true, lambda a: a + a, lambda a: a, w)) == [2, 4]
    assert values(device, gnxlib.cond(false, lambda a: a + a, lambda a: a, w)) == [1, 2]
    two = array.from_host_bytes([], "int32", struct.pack("<i", 2))
    branches = [lambda a: a, lambda a: a + a, lambda a: a + a + a]
    assert values(device, gnxlib.switch(two, branches, w)) == [3, 6]
    assert values(device, gnxlib.switch(7, branches, w)) == [3, 6]  # clamped
    flag, out = gnxlib.while_loop(lambda c: c[0], lambda c: (c[0], c[1] + c[1]), (false, w))
    assert values(device, out) == [1, 2]
    xs = array.from_host_bytes([3, 2], "float32", struct.pack("<6f", 1, 2, 3, 4, 5, 6))
    carry, ys = gnxlib.scan(lambda c, x: (c + x, c), w, xs)
    assert values(device, carry) == [10, 14]
    assert values(device, ys) == [1, 2, 2, 4, 5, 8]
