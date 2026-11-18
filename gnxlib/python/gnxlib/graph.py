"""Convert between Python values and GNX graphs.

Graphs accept live GNX arrays and refs, None, exact built-in bool, int, float,
str, and bytes, and exact tuple, list, and dict containers. Keys must be
built-in str or int, and containers cannot be shared or cyclic.

A custom type can define ``__graph__ = (flatten, unflatten)``. ``flatten(obj)``
yields ordered ``(key, child)`` tuples containing supported values;
``unflatten(items)`` receives a list of those pairs and reconstructs the object.
JIT uses the same protocol, with the class and ordered children as part of its
cache key. Keep the class alive and its hooks stable, and make the hooks
deterministic: flatten can run even on a cache hit.
"""

from ._native import Graph as Graph
