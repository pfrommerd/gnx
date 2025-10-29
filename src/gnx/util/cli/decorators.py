import typing as tp

from .command import Target, Preprocessor, Group, Command
from .option import Option, Missing


class OptionDecorator:
    def __init__(self, option: Option):
        self.option = option

    def __call__[F: Target](self, target: F) -> F:
        target.add_option(self.option)
        return target


def option[F: Target](
    opt: str,
    param: str | None = None,
    *,
    type: tp.Type,
    default: tp.Any | type[Missing] = Missing,
    help: str | None = None,
) -> tp.Callable[[F], F]:
    option = Option.from_spec(opt, param, type=type, default=default)
    return OptionDecorator(option)


@tp.overload
def group(func: Preprocessor, /) -> Group: ...
@tp.overload
def group(func: None = None, /) -> tp.Callable[[Preprocessor], Group]: ...


def group(
    func: Preprocessor | None = None,
    /,
) -> Group | tp.Callable[[Preprocessor], Group]:
    if func is None:
        return group  # type: ignore
    else:
        g = Group()
        g.add_preprocessor(func)
        return g


def command[R](func: tp.Callable[..., R]) -> Command[R]:
    return Command(func)
