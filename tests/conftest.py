import pytest
import jax
import os

from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        dest="full",
        default=False,
        help="enable full end-to-end tests",
    )
    parser.addoption(
        "--only-full",
        action="store_true",
        dest="onlyfull",
        default=False,
        help="run only full end-to-end tests",
    )


def pytest_configure(config):
    extramark = "not full" if not config.option.full else ""
    if config.option.onlyfull:
        extramark = "full"
    markexpr = config.option.markexpr if config.option.markexpr else ""
    markexpr = f"{markexpr} and {extramark}" if markexpr else extramark
    setattr(config.option, "markexpr", markexpr)


@pytest.fixture(scope="session", autouse=True)
def jax_setup():
    cache_dir = Path(os.environ.get("JAX_CACHE_DIR", Path.home() / ".cache" / "jax"))
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # make tests always run on cpu
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 256)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    yield
