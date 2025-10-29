import jax
import rich
import sys
import os
import rich
import logging

from pathlib import Path

from ..util.cli.option import OptionParser
from ..util.cli.command import HelpRequested, UsageError, Command

from ..util.experiment import Experiment


class CustomLogRender(rich._log_render.LogRender):  # type: ignore
    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        if not self.show_path:
            output.expand = False
        return output


FORMAT = "%(name)s - %(message)s"


def setup_logging(show_path=False, verbose=False):
    # Disable XLA warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from rich.logging import RichHandler

    if rich.get_console().is_jupyter or not rich.get_console().is_interactive:
        return rich.reconfigure(force_jupyter=False, force_terminal=True)
    root_logger = logging.getLogger("gnx")
    root_logger.setLevel(logging.DEBUG)
    # setup jax loggers
    jax_logger = logging.getLogger("jax")
    jax_logger.setLevel(logging.INFO if verbose else logging.WARNING)
    if jax_logger.handlers:
        jax_logger.handlers.clear()
    jaxlib_logger = logging.getLogger("jaxlib")
    jaxlib_logger.setLevel(logging.INFO if verbose else logging.WARNING)
    if jaxlib_logger.handlers:
        jaxlib_logger.handlers.clear()

    console = rich.get_console()
    handler = RichHandler(
        markup=True, rich_tracebacks=True, show_path=show_path, console=console
    )
    renderer = CustomLogRender(
        show_time=handler._log_render.show_time,
        show_level=handler._log_render.show_level,
        show_path=handler._log_render.show_path,
        time_format=handler._log_render.time_format,
        omit_repeated_times=handler._log_render.omit_repeated_times,
    )
    # print(logging.getLogger("jax").handlers.pop(-1))
    handler._log_render = renderer
    logging.basicConfig(
        level=logging.WARNING,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )


def shutdown_logging():
    logging.root.handlers.clear()
    logging.getLogger("gnx").setLevel(logging.WARNING)


def resume(experiment_url):
    experiment = Experiment.from_url(experiment_url)
    config = experiment.config
    entrypoint = experiment.entrypoint
    assert entrypoint is not None, "Experiment has no entrypoint to resume!"
    parser = OptionParser()
    command = parser.import_entrypoint(entrypoint)
    # run the command with the config directly
    command(config=config)


def run():
    verbose = "false" != os.environ.get("GNX_VERBOSE", "false").lower()
    setup_logging(verbose=verbose)
    cache_dir = Path(os.environ.get("JAX_CACHE_DIR", Path.home() / ".cache" / "jax"))
    jax.config.update("jax_log_compiles", verbose)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 256)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )

    command = None
    try:
        parser = OptionParser()
        pos_args, options = parser.parse_args(sys.argv[1:])

        if pos_args[0] == "resume":
            # Special case for resuming an experiment
            if len(pos_args) != 2:
                print("Usage: gnx resume <experiment_url>")
                sys.exit(1)
            return resume(pos_args[1])

        config = parser.read_configs(arg for arg in pos_args if ":" not in arg)
        config.update(options)
        entrypoint = config.pop("entrypoint", None)
        # always override the entrypoint if specified in the positional args
        for arg in pos_args:
            if ":" in arg:
                entrypoint = arg
                break
        if not entrypoint:
            raise UsageError(
                "No entrypoint specified. Use --entrypoint, a positional argument, or a config file."
            )
        command = parser.import_entrypoint(entrypoint)
        command.run(sys.argv[0], config)
    except HelpRequested as e:
        assert command is not None
        command.print_help(sys.argv[0])
        sys.exit(1)
    except UsageError as e:
        rich.print(e)
        if command is not None:
            command.print_help(sys.argv[0])
        sys.exit(1)


if __name__ == "__main__":
    run()
