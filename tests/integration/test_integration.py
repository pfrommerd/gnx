from pathlib import Path

import pytest
import logging
import jax
import os

from gnx.util.cli.option import OptionParser
from gnx.util.experiment import Experiment, Scalar
from gnx.main import setup_logging, shutdown_logging

CONFIGS_DIR = Path(__file__).parent
CONFIGS = list(CONFIGS_DIR.glob("*.toml"))
CONFIG_IDS = [config.stem for config in CONFIGS]

logger = logging.getLogger("tests.test_full")
logger.setLevel(logging.INFO)


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_read_configs(config):
    parser = OptionParser()
    parser.read_configs([config])


def get_test_fn(options):
    if "test-metric-lt" in options:
        target = float(options.pop("test-metric-lt"))

        def test(name, value):
            assert value < target, f"{name}={value} is not less than {target}"

        return test
    elif "test-metric-gt" in options:
        target = float(options.pop("test-metric-gt"))

        def test(name, value):
            assert value > target, f"{name}={value} is not greater than {target}"

        return test
    else:
        raise ValueError("No test metric target specified")


def run(config):
    setup_logging()

    parser = OptionParser()
    options = parser.read_configs([config])
    entrypoint = options.pop("entrypoint")
    metric_name = options.pop("test-metric-name")
    metric_test = get_test_fn(options)
    logger.info(f"Running end-to-end test for: {config.stem}")
    experiment = Experiment.from_url("mem://")
    options["experiment"] = experiment.url
    command = parser.import_entrypoint(entrypoint)
    assert command is not None
    command.run("test", options)
    end_value = experiment.results[metric_name]
    assert isinstance(end_value, Scalar)
    end_value = end_value.value

    shutdown_logging()

    metric_test(metric_name, end_value)


@pytest.mark.full
@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_run(config, capsys):
    # override the experiment to be an
    # in-memory experiment so that we can check the eval metrics
    with capsys.disabled():
        print()
        print("Running end-to-end test for: " + config.stem)
        run(config)


if __name__ == "__main__":
    cache_dir = Path(os.environ.get("JAX_CACHE_DIR", Path.home() / ".cache" / "jax"))
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 256)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )

    setup_logging()
    for config in CONFIGS:
        run(config)
