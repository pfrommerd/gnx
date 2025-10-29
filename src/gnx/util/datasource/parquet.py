import jax
import math
import numpy as np
import jax.numpy as jnp
import typing as tp
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import dataclasses

from ...core import nn, graph, graph_util
from ..fs import Resource
from .common import DataCache, DataIterator, DataSource


@dataclasses.dataclass
class ParquetInfo:
    arrow_type: pa.DataType
    metadata: dict | None = None


class ParquetConverter[T](tp.Protocol):
    @property
    def info(self) -> ParquetInfo: ...
    @property
    def instance(self) -> T: ...

    def serialize(self, data: T) -> pa.Array: ...
    def deserialize(self, data: pa.Array) -> T: ...


class ParquetFormat(tp.Protocol):
    def __init__(self) -> None: ...

    def from_instance[T](self, instance: T) -> ParquetConverter[T]: ...
    def from_arrow(
        self, info: ParquetInfo, sample: pa.Array
    ) -> ParquetConverter[tp.Any]: ...


class ParquetDataSource[T](DataSource[T]):
    def __init__(
        self,
        converter: ParquetConverter[T],
        dataset: ds.Dataset,
        batch_shape: tuple[int, ...] = (),
    ):
        self.converter = converter
        self.dataset = dataset
        self.batch_shape = batch_shape

    @property
    def instance(self) -> T:

        def expand(x):
            return jnp.tile(x, self.batch_shape + (1,) * x.ndim)

        return jax.tree.map(expand, self.converter.instance)

    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> T:
        return ParquetIterator(
            self.converter, self.dataset, shape + self.batch_shape, key
        ).next()

    def sampler(self, key: jax.Array | None = None) -> DataIterator[T]:
        return ParquetIterator(self.converter, self.dataset, self.batch_shape, key)

    def batch(self, shape: tuple[int, ...]) -> DataSource[T]:
        return ParquetDataSource(self.converter, self.dataset, shape + self.batch_shape)

    @staticmethod
    def write(
        format: ParquetFormat,
        data: DataSource[T],
        dest: Resource,
        batch_byte_size: int = 128 * 1024 * 1024,
        batches_per_file: int = 16,
    ) -> "ParquetDataSource[T]":
        try:
            len(data)
        except TypeError:
            raise ValueError(
                "Data source must have a known, finite length to write to Parquet"
            )
        converter = format.from_instance(data.instance)
        info = converter.info
        assert pa.types.is_struct(
            info.arrow_type
        ), "Top-level Parquet type must be a struct"
        schema = pa.schema(info.arrow_type.fields, metadata=info.metadata)
        item_size = graph_util.size_in_bytes(data.instance)
        # Load data in 128MB chunks
        batch_size = max(1, batch_byte_size // item_size)
        iterator = data.batch((batch_size,)).sampler()

        done = False
        parts = []
        while not done:
            part = dest / f"part-{len(parts):05d}.parquet"
            with part.writer() as f:
                with pq.ParquetWriter(f, schema) as writer:
                    for i in range(batches_per_file):
                        if iterator.has_next():
                            batch = iterator.next()
                            array = converter.serialize(batch)
                            batch = pa.RecordBatch.from_struct_array(array)
                            writer.write_batch(batch)
                        else:
                            remainder = iterator.remainder()
                            if remainder is not None:
                                array = converter.serialize(remainder)
                                batch = pa.RecordBatch.from_struct_array(array)
                                writer.write_batch(batch)
                            done = True
                            break
            parts.append(part)
        files, fs = PaFileSystem.convert(parts)
        dataset = ds.dataset(files, filesystem=fs)
        return ParquetDataSource(converter, dataset)


def parquet[T](
    format: ParquetFormat,
    *fragments: Resource,
) -> ParquetDataSource[T]:
    sources = list(itertools.chain(frag.glob("*.parquet") for frag in fragments))
    sources = [str(source) for source in sources]
    files, fs = PaFileSystem.convert(sources)
    dataset = ds.dataset(files, filesystem=fs)
    schema = dataset.schema
    fields = list(schema.field(n) for n in schema.names)
    type = pa.struct(fields)
    info = ParquetInfo(type, schema.metadata)
    sample = dataset.scanner().head(1)
    conveter = format.from_arrow(info, sample)
    return ParquetDataSource(converter, dataset, ())


# A cache backed by parquet data sources


class ParquetCache(DataCache):
    pass


# This is the "python-side" stream that reads parquet data into CPU memory
class ParquetStream[T]:
    def __init__(
        self,
        converter: ParquetConverter[T],
        dataset: ds.Dataset,
        batch_shape: tuple[int, ...],
        rng: jax.Array,
    ):
        self.converter = converter
        self.dataset = dataset
        self.batch_shape = batch_shape
        self.rng = nn.RngStream(rng, tag="iterator")
        self.scanner = dataset.scanner(
            batch_size=math.prod(batch_shape), batch_readahead=16
        )
        self.batches = self.scanner.to_batches()
        self.batch_iter = iter(self.batches)
        self.current_batch = None
        self.current_index = 0

    def next(self) -> T:
        pass


class ParquetIterator[T](DataIterator[T]):
    def __init__(
        self,
        converter: ParquetConverter[T],
        dataset: ds.Dataset,
        batch_shape: tuple[int],
        rng: jax.Array,
    ):
        self.instance = converter.instance
        self.stream = ParquetStream(converter, dataset, batch_shape, rng)

    @property
    def instance(self) -> T:
        def expand(x):
            return jnp.tile(x, self.batch_shape + (1,) * x.ndim)

        return jax.tree.map(expand, self.converter.instance)

    @jax.jit
    def next(self) -> T:
        return self.stream.next()


# Will convert a RecordBatch to a desired type
_DTYPE_MAP = {
    pa.float16: jnp.float16,
    pa.float32: jnp.float32,
    pa.float64: jnp.float64,
    pa.int8: jnp.int8,
    pa.int16: jnp.int16,
    pa.int32: jnp.int32,
    pa.int64: jnp.int64,
    pa.uint8: jnp.uint8,
    pa.uint16: jnp.uint16,
    pa.uint32: jnp.uint32,
    pa.uint64: jnp.uint64,
}


class RawArrayConverter(ParquetConverter[jax.Array]):
    def __init__(self, shape: tuple[int, ...], dtype: jnp.dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def info(self) -> ParquetInfo:
        dtype = pa.from_numpy_dtype(self.dtype)
        # if shape is not empty, use nested fixed size list
        if self.shape:
            for dim in reversed(self.shape):
                dtype = pa.list_(dtype, dim)
        return ParquetInfo(dtype)

    @property
    def instance(self) -> jax.Array:
        return jnp.zeros(self.shape, self.dtype)

    def serialize(self, data: jax.Array) -> pa.Array:
        data_np = np.array(data).flatten()
        pa_array = pa.array(data_np.flatten())
        for s in data_np.shape[1:][::-1]:
            pa_array = pa.FixedSizeListArray.from_arrays(pa_array, s)
        return pa_array

    def deserialize(self, data: pa.Array) -> jax.Array:
        shape = []
        while pa.types.is_fixed_size_list(data.type):
            shape.append(data.type.list_size)
            data = data.flatten()
        data = data.to_numpy(zero_copy_only=False)
        n = data.shape[0] // math.prod(shape)
        data = data.reshape((n, *shape))
        return jnp.array(data)

    @staticmethod
    def from_arrow_type(arrow_type: pa.DataType) -> "RawArrayConverter":
        shape = []
        data_type = arrow_type
        while pa.types.is_list(data_type):
            list_type = data_type
            shape.append(list_type.list_size)
            data_type = list_type.value_type
        dtype = _DTYPE_MAP[data_type]
        shape = tuple(shape)
        return RawArrayConverter(shape, dtype)


# Will serialize an array as a struct
# with a single field "data" containing the raw array
class WrappedArrayConverter(RawArrayConverter):
    @property
    def info(self) -> ParquetInfo:
        type = pa.struct([pa.field("data", super().info.arrow_type)])
        return ParquetInfo(type, {"type": "array"})

    def serialize(self, data: jax.Array) -> pa.Array:
        array = super().serialize(data)
        return pa.StructArray.from_arrays([array], ["data"])

    def deserialize(self, data: pa.Array) -> jax.Array:
        array = data.field("data")
        return super().deserialize(array)

    @staticmethod
    def from_arrow_type(arrow_type: pa.DataType) -> "WrappedArrayConverter":
        if not pa.types.is_struct(arrow_type):
            raise ValueError("Arrow type is not a struct")
        struct_type = arrow_type
        if len(struct_type) != 1 or struct_type[0].name != "data":
            raise ValueError("Struct type does not have a single field 'data'")
        data_type = struct_type[0].type
        # Unwrap nested fixed size lists to get shape and dtype
        shape = []
        while pa.types.is_list(data_type):
            list_type = data_type
            shape.append(list_type.list_size)
            data_type = list_type.value_type
        dtype = _DTYPE_MAP[data_type]
        shape = tuple(shape)
        return WrappedArrayConverter(shape, dtype)


class GraphConverter[T]:
    def __init__(
        self,
        type: tp.Type[T],
        children: dict[graph.Key, ParquetConverter[tp.Any]],
    ):
        self.type = type
        self.converters = children

    @property
    def info(self) -> ParquetInfo:
        fields = []
        for name, converter in self.converters.items():
            info = converter.info
            fields.append(pa.field(name, info.arrow_type, metadata=info.metadata))
        struct_type = pa.struct(fields)
        metadata = {"type": graph_util.qualified_type_name(self.type)}
        return ParquetInfo(struct_type, metadata)

    def serialize(self, data: T) -> pa.Array:
        assert False

    def deserialize(self, data: pa.Array) -> T:
        assert False


class SubFormatConverter[T]:
    def __init__(self, format: ParquetFormat, converter: ParquetConverter[T]):
        self.format = format
        self.converter = converter

    @property
    def info(self) -> ParquetInfo:
        info = self.converter.info
        if not info.metadata or "format" not in info.metadata:
            qualified_name = graph_util.qualified_type_name(type(self.format))
            metadata = info.metadata or {}
            metadata["format"] = qualified_name
            info = dataclasses.replace(info, metadata=metadata)
        return info

    @property
    def instance(self) -> T:
        return self.converter.instance

    def serialize(self, data: T) -> pa.Array:
        return self.converter.serialize(data)

    def deserialize(self, data: pa.Array) -> T:
        return self.converter.deserialize(data)


class ParquetAutoFormat(ParquetFormat):
    def from_instance[T](
        self, instance: T, is_child: bool = False
    ) -> ParquetConverter[T]:
        if isinstance(instance, jax.Array):
            if is_child:
                return RawArrayConverter(instance.shape, instance.dtype)  # type: ignore
            else:
                return WrappedArrayConverter(instance.shape, instance.dtype)  # type: ignore
        elif hasattr(type(instance), "__parquet_format__"):
            format: ParquetFormat = type(instance).__parquet_format__()  # type: ignore
            return SubFormatConverter(format, format.from_instance(instance))
        elif graph.is_graph_node_type(type(instance)):
            children = {}
            for key, child in graph.children(instance).items():
                child_converter = self.from_instance(child, is_child=True)
                children[key] = child_converter
            return GraphConverter(type(instance), children)
        else:
            raise ValueError(f"Cannot infer ParquetConverter for type {type(instance)}")

    def from_arrow(
        self, info: ParquetInfo, sample: pa.Array
    ) -> ParquetConverter[tp.Any]:
        metadata = info.metadata or {}
        format = metadata.get("format")
        if format is not None:
            del metadata["format"]
            # Resolve the format type
            format_type: type[ParquetFormat] = graph_util.resolve_qualified_type(format)  # type: ignore
            format = format_type()
            return SubFormatConverter(format, format.from_arrow(info, sample))
        type = metadata.get("type")
        if type == "array":
            return WrappedArrayConverter.from_arrow_type(info.arrow_type)
        elif type is None:
            return RawArrayConverter.from_arrow_type(info.arrow_type)
        # We have a graph node type
        type = graph_util.import_qualified_type(type)
        for field in info.arrow_type.fields:
            field_info = ParquetInfo(field.type, field.metadata)
            # Recursively get converters for each field
        assert False


# A wrapper around an origin to expose a pyarrow file system


class PaFile:
    def __init__(self, resource: Resource):
        self.reader = resource.reader().__enter__()
        self.closed = False

    def read(self, n: int = -1) -> bytes:
        return self.reader.read(n)

    def close(self):
        self.closed = True
        self.reader.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class PaFileSystem(pa.fs.FileSystemHandler):
    def __init__(self, origin):
        self.origin = origin

    def get_type_name(self) -> str:
        return "custom"

    def copy_file(self, src: str, dest: str) -> None:
        raise NotImplementedError()

    def create_dir(self, path: str, recursive: bool = True) -> None:
        raise NotImplementedError()

    def delete_dir_contents(self, path: str) -> None:
        raise NotImplementedError()

    def delete_dir(self, path: str) -> None:
        raise NotImplementedError()

    def delete_file(self, path: str) -> None:
        raise NotImplementedError()

    def delete_root_dir_contents(self) -> None:
        raise NotImplementedError()

    def get_file_info(self, path: str) -> pa.fs.FileInfo:
        raise NotImplementedError()

    def get_file_info_selector(self, selector):
        raise NotImplementedError()

    def move(self, src: str, dest: str) -> None:
        raise NotImplementedError()

    def normalize_path(self, path: str) -> str:
        return path

    def open_append_stream(self, path: str) -> pa.NativeFile:
        raise NotImplementedError()

    def open_input_stream(self, path: str) -> pa.NativeFile:
        resource = self.origin / path
        io = resource.reader().__enter__()
        return pa.PythonFile(io, mode="r")

    def open_input_file(self, path: str) -> pa.NativeFile:
        resource = self.origin / path
        io = resource.reader().__enter__()
        return pa.PythonFile(io, mode="r")

    def open_output_stream(self, path: str) -> pa.NativeFile:
        raise NotImplementedError()

    @staticmethod
    def convert(
        res: tp.Iterable[Resource],
    ) -> "tuple[list[Resource], pa.fs.FileSystem]":
        res = list(res)
        origin = res[0].origin
        paths = []
        for r in res:
            paths.append("/" + str(r.path))
        return paths, pa.fs.PyFileSystem(PaFileSystem(origin))
