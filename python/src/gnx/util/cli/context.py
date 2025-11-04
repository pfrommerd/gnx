import typing as tp


class Context:
    def __init__(self, *, root: dict[str, tp.Any] = {}, prefix: tuple[str, ...] = ()):
        self._root = root or {}
        self._prefix = prefix or ()

    def __contains__(self, name: str) -> bool:
        key = ".".join(self._prefix + (name,))
        prefix = key + "."
        return key in self._root or any(k.startswith(prefix) for k in self._root)

    def __setitem__(self, name: str, value: tp.Any, /):
        prefix = self._prefix + tuple(name.split("."))
        if not isinstance(value, Context):
            self._root[".".join(prefix)] = value
        else:
            prefix = ".".join(prefix)
            for k, v in value.flatten():
                self._root[f"{prefix}.{k}"] = v

    def __getitem__(self, name: str, /) -> tp.Any:
        prefix = self._prefix + tuple(name.split("."))
        key = ".".join(prefix)
        if key in self._root:
            return self._root[key]
        return Context(root=self._root, prefix=prefix)

    def __delitem__(self, name: str, /):
        key = ".".join(self._prefix + tuple((name,)))
        if key in self._root:
            del self._root[key]
        else:
            prefix = key + "."
            keys = list(k for k in self._root if k.startswith(prefix))
            for k in keys:
                del self._root[k]

    def __getattr__(self, name: str, /) -> tp.Any:
        return self[name]

    def __setattr__(self, name: str, value: tp.Any, /):
        if name == "_root" or name == "_prefix":
            super().__setattr__(name, value)
            return
        self[name] = value

    def __delattr__(self, name: str, /):
        del self[name]

    def flatten(self) -> tp.Iterable[tuple[str, tp.Any]]:
        if not self._prefix:
            return self._root.items()
        else:
            prefix = "".join(f"{p}." for p in self._prefix)
            return [
                (k[len(prefix) :], v)
                for k, v in self._root.items()
                if k.startswith(prefix)
            ]

    def update(self, other: "Context"):
        prefix = "".join(f"{p}." for p in self._prefix)
        new_items = ((prefix + k, v) for k, v in other.flatten())
        self._root.update(new_items)

    def pop(self, name: str, default: tp.Any = None) -> tp.Any:
        key = ".".join(self._prefix + (name,))
        return self._root.pop(key, default)

    def clear(self):
        if self._prefix:
            prefix = "".join(f"{p}." for p in self._prefix)
            keys = [k for k in self._root.keys() if k.startswith(prefix)]
            cleared = {}
            for k in keys:
                cleared[k] = self._root[k]
                del self._root[k]
            return Context(root=cleared, prefix=self._prefix)
        else:
            r = dict(self._root)
            self._root.clear()
            return Context(root=r)

    # Will turn a jax tree into a Context
    @staticmethod
    def from_tree(tree, map=lambda x: x, is_leaf=None):
        import jax.tree
        from ...core import graph

        items = (
            (".".join(graph.key_to_str(k) for k in path), map(v))
            for path, v in jax.tree.leaves_with_path(tree, is_leaf=is_leaf)
        )
        return Context(root=dict(items))

    @staticmethod
    def from_flat(mapping: tp.Mapping[str, tp.Any]):
        return Context(root=dict(mapping))
