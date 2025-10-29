import inspect
import enum
import typing as tp

from rich.highlighter import RegexHighlighter
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.color import Color
from rich.console import Console

from pathlib import Path

from .option import Missing, Option
from .context import Context


class UsageError(Exception):
    """Raised when the command is used incorrectly, e.g., missing required options."""


class HelpRequested(Exception): ...


type Preprocessor = tp.Callable[[Context]]
type Entrypoint[Result] = tp.Callable[..., Result]


class Target(tp.Protocol):
    def add_option(self, option): ...

    # If first is False, will run this preprocessor last
    def add_preprocessor(self, preprocessor, first: bool = True): ...


class Group:
    def __init__(
        self,
        options: tp.Iterable[Option] = [],
        preprocessors: tp.Iterable[Preprocessor] = [],
        first: bool = True,
    ):
        self.options = {o.param: o for o in options}
        self.preprocessors = list(preprocessors) or []
        self.first = first

    def after(self) -> "Group":
        return Group(self.options.values(), self.preprocessors, first=False)

    def add_option(self, option):
        self.options[option.param] = option

    def add_preprocessor(self, preprocessor: Preprocessor, first: bool = True):
        if first:
            self.preprocessors.insert(0, preprocessor)
        else:
            self.preprocessors.append(preprocessor)

    def add_group(self, sub_group: "Group"):
        sub_group.add_to(self)

    def add_to(self, target: Target):
        for option in self.options.values():
            target.add_option(option)
        # If the group should run first, prepend the preprocessors otherwise append them
        if self.first:
            for preprocessor in self.preprocessors[::-1]:
                target.add_preprocessor(preprocessor, first=True)
        else:
            for preprocessor in self.preprocessors:
                target.add_preprocessor(preprocessor)

    def __call__[F: Target](self, target: F) -> F:
        self.add_to(target)
        return target


# A simple group decorator that removes the specified parameters from the context.
def remove_params(*params) -> Group:
    from .decorators import group

    @group
    def remove_params(ctx: Context):
        for param in params:
            del ctx[param]

    return remove_params


class Command[Result]:
    def __init__(
        self,
        main: Entrypoint[Result],
        options: tp.Iterable[Option] = [],
        preprocessors: tp.Iterable[Preprocessor] = [],
    ):
        self.main = main
        self.options = {o.param: o for o in options}
        self.preprocessors = list(preprocessors)

    def add_option(self, option):
        self.options[option.param] = option

    def add_preprocessor(self, preprocessor: Preprocessor, first: bool = True):
        if first:
            self.preprocessors.insert(0, preprocessor)
        else:
            self.preprocessors.append(preprocessor)

    def add_group(self, group: Group):
        group.add_to(self)

    def print_help(self, cmd: str | None):
        class OptionHighlighter(RegexHighlighter):
            highlights = [
                r"(?P<switch> \-\w)",
                r"(?P<option>\-\-[\.\w\-]+)",
            ]

        highlighter = OptionHighlighter()
        console = Console(
            theme=Theme(
                {"option": "bold", "switch": "bold green", "group": "bold blue"}
            ),
            highlighter=highlighter,
        )
        console.print(f"Usage: [b]{cmd}[/b] [b][OPTIONS][/]")

        options_table = Table(
            highlight=True, box=None, show_header=False, padding=(0, 0)
        )
        # group the options
        groups = {}
        for option in self.options.values():
            group = option.name.split(".")[0] if "." in option.name else "general"
            groups.setdefault(group, []).append(option)
        groups = dict(
            sorted(groups.items(), key=lambda x: x[0] if x[0] != "general" else "0")
        )
        for group_name, option_group in groups.items():
            options_table.add_row(Text(group_name.upper(), style="group"))
            for opt in option_group:
                if opt.short:
                    short = highlighter(" -" + opt.short)
                    long = highlighter("--" + opt.name)
                else:
                    short = Text("")
                    long = highlighter("--" + opt.name)
                help = highlighter(Text.from_markup(opt.help)) if opt.help else Text("")
                options_table.add_row(short, long, help)

        console.print()
        console.print(options_table)

    def run(self, prog_name: str, options: dict[str, str]) -> Result:
        # ctx contains a string-valued context of all options
        if "help" in options:
            raise HelpRequested()
        options = dict(options)
        parsed = {}
        for param, option in self.options.items():
            if option.name in options:
                value = options.pop(option.name)
                if isinstance(value, str):
                    parsed[param] = option.parse(value)
                else:
                    parsed[param] = value
            elif option.short in options:
                value = options.pop(option.short)
                if isinstance(value, str):
                    parsed[param] = option.parse(value)
                else:
                    parsed[param] = value
            elif option.default is not Missing:
                parsed[param] = option.default
        if options:
            raise UsageError(
                f"Unknown options: {', '.join(f'--{k}' for k in options.keys())}"
            )
        ctx = Context.from_flat(parsed)

        for p in self.preprocessors:
            p(ctx)

        params = dict(ctx.flatten())
        return self.main(**params)

    def __call__(self, **params: tp.Any) -> tp.Any:
        self.main(**params)
