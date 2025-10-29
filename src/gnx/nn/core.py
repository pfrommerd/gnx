import typing as tp
import treescope

import jax
import jax.numpy as jnp

from .. import asserts, graph, filters


class Module(graph.Object):
    def train_mode(self, enabled: bool = True):
        # get all children recursively
        for child in graph.nodes(self).values():
            if isinstance(child, Module) and hasattr(child, "_set_training"):
                child._set_training(enabled)  # type: ignore
        return self

    def eval_mode(self):
        self.train_mode(False)


type RngSeed = int | jax.Array


class RngStream(graph.Object):
    def __init__(self, key: RngSeed, *, tag: str):
        self.key = RngKey(jax.random.key(key) if isinstance(key, int) else key, tag=tag)
        self.count = RngCount(jax.numpy.array(0, dtype=jax.numpy.uint32), tag=tag)
        self.tag = tag

    def __call__(self) -> jax.Array:
        key = jax.random.fold_in(self.key[...], self.count[...])
        self.count[...] += 1
        return key

    def last(self) -> jax.Array:
        return jax.random.fold_in(self.key[...], self.count[...] - 1)

    def fork(self, *, split: int | tuple[int, ...] | None = None):
        key = self()
        if split is not None:
            key = jax.random.split(key, split)
        return type(self)(key, tag=self.tag)


class Rngs(graph.Object):
    def __init__(
        self,
        default: (
            RngSeed | RngStream | tp.Mapping[str, RngSeed | RngStream] | None
        ) = None,
        **rngs: RngSeed | RngStream,
    ):
        if default is not None:
            if isinstance(default, tp.Mapping):
                rngs = {**default, **rngs}
            else:
                rngs["default"] = default

        self.streams = {}
        for tag, key in rngs.items():
            if isinstance(key, RngStream):
                key = key.key.value[...]
            self.streams[tag] = RngStream(
                key=key,
                tag=tag,
            )

    def _get_stream(self, name: str, error_type: type[Exception]) -> RngStream:
        if name not in self.streams:
            if "default" not in self.streams:
                raise error_type(f"No RngStream named '{name}' found in Rngs.")
            stream = self.streams["default"]
        else:
            stream = self.streams[name]
        return stream

    def __getitem__(self, name: str):
        return self._get_stream(name, KeyError)

    def __getattr__(self, name: str):
        if name == "streams":
            super().__getattribute__(name)
        return self._get_stream(name, AttributeError)

    # override the repr

    def __treescope_repr__(self, path, subtree_renderer):
        attributes = {str(k): v for k, v in self.streams.items()}
        return treescope.repr_lib.render_object_constructor(
            self.__class__,
            attributes,
            path=path,
            subtree_renderer=subtree_renderer,
            color=treescope.formatting_util.color_from_string(str(self.__class__)),
        )

    def __rich_repr__(self):
        yield self.__class__.__name__
        for k, v in self.streams.items():
            yield str(k), v

    def __repr__(self):
        return treescope.render_to_text(self)

    def __call__(self):
        return self.default()

    def __contains__(self, name: tp.Any) -> bool:
        return name in vars(self)

    def items(self):
        for name, stream in vars(self).items():
            if isinstance(stream, RngStream):
                yield name, stream


# Variable


class Variable(graph.Object):
    def __init__(
        self,
        value: jax.typing.ArrayLike | jax.Ref,
        *,
        tag: str | None = None,
        mutable: bool = True,
    ):
        self.value: jax.Array | jax.Ref
        # convert to array_ref upon manual creation
        if mutable:
            self.value = jax.new_ref(jnp.array(value))
        elif isinstance(value, (jax.Array, jax.Ref)):
            self.value = value[...]
        else:
            self.value = jnp.array(value)
        self.is_mutable = mutable
        self.tag = tag

    def pure(self) -> tp.Self:
        return self.__class__(value=self.value, mutable=False, tag=self.tag)

    def mutable(self) -> tp.Self:
        return self.__class__(value=self.value, mutable=True, tag=self.tag)

    def __getitem__(self, idx: tp.Any) -> jax.Array:
        return self.value[idx]

    def __setitem__(self, idx: tp.Any, value: jax.typing.ArrayLike):
        self.value[idx] = value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.value.dtype


# Built-in variable types


class Param(Variable):
    pass


class RngKey(Variable):
    tag: str


class RngCount(Variable):
    tag: str


def variables(model: tp.Any, wrt: graph.Filter) -> dict[graph.Path, Variable]:
    return graph.leaves(model, filters.All(Variable, wrt))


def variable_refs(model: tp.Any, wrt: graph.Filter) -> dict[graph.Path, jax.Ref]:
    def assert_ref(value: jax.Array | jax.Ref) -> jax.Ref:
        return value  # type: ignore

    return {k: assert_ref(v.value) for k, v in variables(model, wrt).items()}


def variable_arrays(model: tp.Any, wrt: graph.Filter) -> dict[graph.Path, jax.Array]:
    return {k: v[...] for k, v in variables(model, wrt).items()}


def pure[Model](model: Model) -> Model:
    graphdef, variables, leaves = graph.split(model, Variable, ...)
    variables = {k: v.pure() for k, v in variables.items()}
    return graph.merge(graphdef, variables, leaves)


def mutable[Model](model: Model) -> Model:
    graphdef, variables, leaves = graph.split(model, Variable, ...)
    variables = {k: v.mutable() for k, v in variables.items()}
    return graph.merge(graphdef, variables, leaves)


def update[Model](target: Model, src: Model):
    asserts.graphs_equal_shapes_and_dtypes(target, src)
    refs = variable_refs(target, Variable)
    arrays = variable_arrays(src, Variable)
    for k, ref in refs.items():
        ref[...] = arrays[k]


def num_params(module: tp.Any) -> int:
    params = graph.leaves(module, Param).values()
    return sum(p[...].size for p in params)
