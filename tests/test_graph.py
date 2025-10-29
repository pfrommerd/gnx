import gnx.core.graph as graph

from gnx.core.dataclasses import dataclass

import jax
import jax.numpy as jnp


def test_dict_builtin():
    graphdef, leaves = graph.split({"a": 1, "b": "foo"})
    assert len(leaves) == 2
    assert leaves[("a",)] == 1
    assert leaves[("b",)] == "foo"
    combined = graph.merge(graphdef, leaves)
    assert combined == {"a": 1, "b": "foo"}

    frozen = graph.freeze(combined)
    assert frozen["a"] == 1
    assert frozen["b"] == "foo"

    graphdef, leaves = graph.split(frozen)
    assert len(leaves) == 2
    assert leaves[("a",)] == 1
    assert leaves[("b",)] == "foo"
    combined = graph.merge(graphdef, leaves)
    assert frozen == combined


class Simple(graph.Object):
    a: int | jax.Array
    b: str | jax.Array


def test_simple():
    test = Simple()
    test.a = 1
    test.b = "foo"
    graphdef, leaves = graph.split(test)

    assert ("a",) in leaves
    assert ("b",) in leaves

    combined = graph.merge(graphdef, leaves)
    print("combined", combined)

    assert type(combined) == Simple
    assert combined.a is test.a
    assert combined.b is test.b

    frozen = graph.freeze(combined)
    try:
        frozen.a = 2
        assert False, "Should not be able to set attribute on frozen object"
    except AttributeError:
        pass
    thawed = graph.thaw(frozen)
    thawed.a = 2
    assert thawed.a == 2


@dataclass
class DataClass:
    a: int
    b: str


def test_dataclass():
    test = DataClass(a=1, b="foo")

    graphdef, leaves = graph.split(test)
    combined = graph.merge(graphdef, leaves)

    assert combined.a == test.a
    assert combined.b == test.b


def test_jit():
    # test that the object is jittable
    test = Simple()
    test.a = 1
    test.b = "foo"
    assert jax.jit(lambda x: x.a + 1)(test) == 2
    res = jax.jit(lambda x: x)(test)
    assert res.a == test.a


def test_vmap():
    test = Simple()
    test.a = jnp.zeros((10, 5))
    jax.vmap(lambda x: x.a)(test)
    test.b = jnp.ones((10, 5))
    jax.vmap(lambda x: x.a + 2 * x.b)(test)
