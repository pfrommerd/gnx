import gnxlib
import pytest


def test_expr_captures_graph_and_cell_home_without_executing():
    value = gnxlib.array.placeholder((2,))
    roots = {"cell": gnxlib.array.new_ref(value), "same": value}

    capture = gnxlib.expr(roots)
    assert isinstance(capture, gnxlib.Capture)
    assert capture.inputs == ()
    assert len(capture.closure) == len(capture.outputs) == 1
    assert isinstance(capture.outputs[0], gnxlib.array.Array)
    assert capture.outputs[0].shape == (2,)
    assert capture.outputs[0].dtype == "float32"
    assert "<f32>(2)" in str(capture)
    assert str(gnxlib.expr(gnxlib.Graph.from_value(roots))) == str(capture)


def test_expr_uses_graph_validation():
    with pytest.raises(TypeError, match="unsupported graph value"):
        gnxlib.expr(object())


def test_tracers_can_be_inspected_after_their_trace_exits():
    saved = []

    @gnxlib.jit
    def save(value):
        cell = gnxlib.array.new_ref(value)
        saved.append((cell, gnxlib.Graph.from_value(value), gnxlib.expr(value)))
        return value

    save.trace(gnxlib.array.placeholder((2,)))
    cell, graph, capture = saved[0]
    value = graph.to_value()
    assert cell[:].shape == (2,)
    assert gnxlib.array.new_ref(value)[:].shape == (2,)
    assert capture.outputs[0].shape == (2,)
    assert str(gnxlib.expr(value)) == str(gnxlib.expr(graph)) == str(capture)
