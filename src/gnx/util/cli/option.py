import importlib
import collections
import ast
import typing as tp
import tomllib

from pathlib import Path
from .context import Context

Missing = collections.namedtuple("Missing", ())


class Option:
    def __init__(
        self,
        *,
        name: str,
        short: str,
        param: str,
        type: tp.Type,
        default: tp.Any = Missing,
        help: str | None = None,
    ):
        self.param = param
        self.name = name
        self.short = short
        self.type = type
        self.default = default
        self.help = help

    def parse(self, value: str) -> tp.Any:
        """Parse the value from a string to the specified type."""
        if self.type is bool:
            return value.lower() in ("true", "1", "yes")
        elif self.type is int:
            return int(value)
        elif self.type is float:
            return float(value)
        elif self.type is str:
            return value
        else:
            # use ast.literal_eval, replacing variable names with
            # string literals of the same value
            tree = ast.parse(value, mode="eval")

            # map all names to string literals
            class NameReplacer(ast.NodeTransformer):
                def visit_Name(self, node: ast.Name) -> ast.Constant:
                    return ast.Constant(value=node.id, kind=None)

            tree = ast.fix_missing_locations(NameReplacer().visit(tree))
            try:
                v = ast.literal_eval(tree)
                if tp.get_origin(self.type) == tuple:
                    v = tuple(v)
                elif tp.get_origin(self.type) == list:
                    v = list(v)
                return v
            except Exception as e:
                raise ValueError(
                    f"Cannot parse value '{value}' for option '{self.param}'"
                )

    @staticmethod
    def infer_param(opt: str, param: str | None = None) -> str:
        if param is not None:
            return param
        if opt.startswith("--"):
            return opt[2:].replace("-", "_")
        elif opt.startswith("-"):
            return opt[1:].replace("-", "_")
        else:
            raise ValueError(f"Invalid option format: {opt}")

    @staticmethod
    def from_spec(opt: str, param: str | None = None, *, type, default) -> "Option":
        if not opt.startswith("--"):
            raise ValueError(f"Invalid option format: {opt}")
        name = opt[2:]
        param = param or opt[2:].replace("-", "_")
        return Option(name=name, short="", param=param, type=type, default=default)


class OptionParser:
    def parse_args(
        self, args: tp.Sequence[str], default: tp.Any = Missing
    ) -> tuple[list[str], dict[str, str]]:
        pos_args = []
        opts: dict[str, list[str]] = {}
        current_args = pos_args
        for arg in args:
            # if we see --, switch back to adding positional args
            if arg == "--":
                current_args = pos_args
            if arg.startswith("--"):
                current_args = []
                arg = arg[2:]
                if "=" in arg:
                    arg, value = arg.split("=", 1)
                    current_args.append(value)
                opts[arg] = current_args
            else:
                current_args.append(arg)

        options = {opt: " ".join(values) for opt, values in opts.items()}
        return pos_args, options

    def read_configs(self, config_paths: tp.Iterable[str | Path]):
        paths = [Path(p) for p in config_paths]
        # The queue is ordered by next
        # to be read in on the bottom
        queue = list(paths)
        visited = set()

        configs = []
        while queue:
            next_config = queue.pop()
            with open(next_config, "rb") as f:
                match next_config.suffix:
                    case ".toml":
                        contents = Context.from_tree(
                            tomllib.load(f),
                            is_leaf=lambda x: isinstance(x, (list, tuple)),
                        )
                    case _:
                        raise ValueError(
                            f"Unsupported config file format: {next_config}"
                        )
            if "include" in contents:
                for i in contents["include"]:
                    path = next_config.parent / i
                    queue.append(path)
                del contents.include
            #
            if "general" in contents:
                contents.update(contents.general)
                del contents.general
            #
            configs.append(dict(contents.flatten()))
            visited.add(next_config)
        # merge all the configs in reverse order
        config = {}
        for c in configs[::-1]:
            config.update(c)
        return config

    def import_entrypoint(self, entrypoint: str):
        from .command import Command

        module_name, func_name = entrypoint.rsplit(":", 1)
        module = importlib.import_module(module_name)
        cmd = getattr(module, func_name)
        assert isinstance(cmd, Command), "Entrypoint must be a Command instance"
        return cmd
