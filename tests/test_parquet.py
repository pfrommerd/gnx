import jax.numpy as jnp

from gnx.util import datasource, fs
from gnx.util.datasource.parquet import ParquetAutoFormat, ParquetDataSource


def test_write():
    data = datasource.pytree(jnp.arange(10))
    resource = fs.InMemoryFileSystem() / "foo.ds"
    ds = ParquetDataSource.write(
        ParquetAutoFormat(),
        data,
        resource,
    )
    # read back the resource
    assert jnp.arange(10) == ds.data()
