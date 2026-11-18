import concurrent.futures
import gc
import threading
import time
import traceback
import weakref

import gnxlib
import pytest


def test_trace_returns_higher_level_ir_and_reuses_the_call_cache():
    calls = []

    @gnxlib.jit
    def update(cell, *, value):
        calls.append(1)
        cell[:] = value
        return cell[:]

    x = gnxlib.array.placeholder((2,))
    cell = gnxlib.array.new_ref(x)
    trace = update.trace(cell, value=x)
    assert isinstance(trace, gnxlib.Trace)
    assert trace.input_count == 2
    assert trace.output_count == 1
    assert trace.capture_count == 0
    assert "[:] <-" in str(trace)
    assert update(cell, value=x).shape == (2,)
    assert str(update.trace(cell, value=x)) == str(trace)
    assert len(calls) == 1


def test_symbolic_metadata_and_graph_round_trip():
    x = gnxlib.array.placeholder((2, 3), dtype=gnxlib.dtype.int32)
    cell = gnxlib.array.new_ref(x)
    graph = gnxlib.graph.Graph.from_value({"cell": cell, "array": x})
    result = graph.to_value()
    assert isinstance(result["array"], gnxlib.array.Array)
    assert isinstance(result["cell"], gnxlib.array.ArrayRef)
    assert result["array"].shape == (2, 3)
    assert cell.dtype == gnxlib.dtype.int32
    assert result["cell"].shape == (2, 3)
    with pytest.raises(TypeError, match="truth value"):
        bool(x)
    with pytest.raises(ValueError, match="same shape and dtype"):
        cell[:] = gnxlib.array.placeholder((2, 3))


def test_array_types_have_no_direct_constructors():
    with pytest.raises(TypeError):
        gnxlib.array.Array((2,))
    x = gnxlib.array.placeholder((2,))
    with pytest.raises(TypeError):
        gnxlib.array.ArrayRef(x)
    assert not hasattr(gnxlib.array.new_ref(x), "aliases")


def test_array_ref_only_supports_whole_array_indexing():
    x = gnxlib.array.placeholder((2, 3), dtype=gnxlib.dtype.int32)
    cell = gnxlib.array.new_ref(x)
    cell[...] = x
    assert cell[...].shape == (2, 3)
    assert cell[:].dtype == gnxlib.dtype.int32
    with pytest.raises(IndexError, match="whole-array access"):
        cell[0]
    with pytest.raises(IndexError, match="whole-array access"):
        cell[1:] = x


def test_trace_cache_keys_use_metadata_static_values_and_dict_order():
    calls = []

    @gnxlib.jit
    def observe(value, *, config):
        calls.append(1)
        return [value, tuple(config), type(config["tag"]).__name__]

    for shape, config in [
        ((2,), {"tag": True, "other": b"x"}),
        ((2,), {"tag": True, "other": b"x"}),
        ((2,), {"tag": 1, "other": b"x"}),
        ((3,), {"tag": 1, "other": b"x"}),
        ((3,), {"other": b"x", "tag": 1}),
    ]:
        result = observe(gnxlib.array.placeholder(shape), config=config)
        assert result[0].shape == shape
        assert result[1] == tuple(config)
        assert result[2] == type(config["tag"]).__name__
    assert len(calls) == 4


def test_cells_with_different_alias_patterns_reuse_one_trace():
    calls = []

    @gnxlib.jit
    def write_and_read(a, b, value):
        calls.append(1)
        a[:] = value
        return b[:], a, b

    x = gnxlib.array.placeholder((4,))
    a, b = gnxlib.array.new_ref(x), gnxlib.array.new_ref(x)
    for left, right in [(a, b), (a, a)]:
        read, returned_left, returned_right = write_and_read(left, right, x)
        assert read.shape == (4,)
        assert returned_left[:].shape == returned_right[:].shape == (4,)
    assert len(calls) == 1


def test_nested_calls_can_return_cells_created_in_a_trace():
    calls = []

    @gnxlib.jit
    def inner(cell, value):
        calls.append("inner")
        cell[:] = value
        return cell

    @gnxlib.jit
    def outer(value):
        calls.append("outer")
        cell = inner(gnxlib.array.new_ref(value), value)
        return cell, cell, cell[:]

    x = gnxlib.array.placeholder((4,))
    for _ in range(2):
        cell, same_cell, value = outer(x)
        assert cell[:].shape == same_cell[:].shape == value.shape == (4,)
    assert calls == ["outer", "inner"]


def test_python_exception_keeps_traceback_and_does_not_fill_slot():
    attempts = []

    @gnxlib.jit
    def fail_once(value):
        attempts.append(1)
        if len(attempts) == 1:
            raise LookupError("first trace failed")
        return value

    with pytest.raises(LookupError, match="^first trace failed$") as exc:
        fail_once(gnxlib.array.placeholder((1,)))
    assert "fail_once" in [frame.name for frame in traceback.extract_tb(exc.value.__traceback__)]
    assert fail_once(gnxlib.array.placeholder((1,))).shape == (1,)
    assert fail_once(gnxlib.array.placeholder((1,))).shape == (1,)
    assert len(attempts) == 2


def test_threads_share_a_cold_trace():
    calls = []
    start = threading.Barrier(2)

    @gnxlib.jit
    def slow(value):
        calls.append(1)
        time.sleep(0.05)  # Give the other caller a chance to enter the cache.
        return value

    def call():
        value = gnxlib.array.placeholder((2,))
        start.wait(timeout=5)
        return slow(value).shape

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        first = pool.submit(call)
        second = pool.submit(call)
        assert first.result(timeout=5) == (2,)
        assert second.result(timeout=5) == (2,)
    assert len(calls) == 1


@pytest.mark.parametrize("parse", [gnxlib.graph.Graph.from_value, gnxlib.jit(lambda x: x)])
def test_graph_and_jit_reject_invalid_values(parse):
    with pytest.raises(TypeError, match="unsupported graph value"):
        parse(object())

    class ListSubclass(list):
        pass

    with pytest.raises(TypeError, match="unsupported graph value"):
        parse(ListSubclass())
    cycle = []
    cycle.append(cycle)
    with pytest.raises(TypeError, match="cyclic or shared"):
        parse(cycle)
    shared = [1]
    with pytest.raises(TypeError, match="cyclic or shared"):
        parse([shared, shared])
    with pytest.raises(TypeError, match="dict keys"):
        parse({True: 2})


def test_custom_graph_hooks_round_trip_and_reuse_jit_traces():
    def make_type():
        class Box:
            def __init__(self, value, label):
                self.value = value
                self.label = label

        def flatten(box):
            yield "value", box.value
            yield 1, box.label

        def unflatten(items):
            values = dict(items)
            return Box(values["value"], values[1])

        Box.__graph__ = (flatten, unflatten)
        return Box

    Box, OtherBox = make_type(), make_type()
    calls = []

    @gnxlib.jit
    def forward(box):
        calls.append(box.label)
        return box

    def value():
        return gnxlib.array.placeholder((2,))

    graph = gnxlib.graph.Graph.from_value(Box(value(), "graph"))
    rebuilt = graph.to_value()
    assert type(rebuilt) is Box
    assert rebuilt.label == "graph"
    assert rebuilt.value.shape == (2,)

    trace = forward.trace(Box(value(), "first"))
    assert trace.input_count == trace.output_count == 1
    for cls, label in [(Box, "first"), (Box, "second"), (OtherBox, "first")]:
        result = forward(cls(value(), label))
        assert type(result) is cls
        assert result.label == label
        assert result.value.shape == (2,)
    assert calls == ["first", "second", "first"]


def test_graph_hook_children_must_be_supported_and_acyclic():
    class Box:
        pass

    Box.__graph__ = (lambda box: [("self", box)], lambda items: Box())
    with pytest.raises(TypeError, match="cyclic or shared"):
        gnxlib.graph.Graph.from_value(Box())

    Box.__graph__ = (lambda box: [("opaque", object())], lambda items: Box())
    wrapped = gnxlib.jit(lambda: Box())
    with pytest.raises(TypeError, match="unsupported graph value"):
        wrapped()


def test_decorated_closure_cycles_are_collectable():
    def create():
        @gnxlib.jit
        def recursive(x):
            return recursive(x)

        return weakref.ref(recursive.__wrapped__)

    reference = create()
    gc.collect()
    assert reference() is None
