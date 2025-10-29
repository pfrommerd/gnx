# This represents a path on a generic file system.
import abc
import contextlib
import typing as tp
import io
import glob as globlib
import re
import pathlib
import shutil
import urllib.parse


class Path:
    def __init__(self, *parts: "str | Path"):
        split_parts = []
        for p in parts:
            if isinstance(p, Path):
                split_parts.extend(p.parts)
            elif "/" in p:
                split_parts.extend(p.split("/"))
            else:
                split_parts.append(p)
        self.parts = tuple(part for part in split_parts if part and part != ".")

    @tp.overload
    def __getitem__(self, index: int) -> "str": ...
    @tp.overload
    def __getitem__(self, index: slice) -> "Path": ...

    def __getitem__(self, index: int | slice) -> "str | Path":
        if isinstance(index, int):
            return self.parts[index]
        return Path(*self.parts[index])

    def __len__(self) -> int:
        return len(self.parts)

    def __truediv__(self, other: "str | Path") -> "Path":
        if isinstance(other, str):
            return Path(*self.parts, other)
        return Path(*self.parts, *other.parts)

    # Supports appending a path to a pathlib.Path for convenience (but not vice versa)

    @tp.overload
    def __rtruediv__(self, other: str) -> "Path": ...
    @tp.overload
    def __rtruediv__(self, other: pathlib.Path) -> pathlib.Path: ...
    @tp.overload
    def __rtruediv__(self, other: "Path") -> "Path": ...

    def __rtruediv__(self, other: "str | Path | pathlib.Path") -> "Path | pathlib.Path":
        if isinstance(other, str):
            return Path(other, *self.parts)
        elif isinstance(other, pathlib.Path):
            return other.joinpath(*self.parts)
        return Path(*other.parts, *self.parts)

    def __hash__(self) -> int:
        return hash(self.parts)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Path):
            return False
        return self.parts == other.parts

    def __lt__(self, other: "Path") -> bool:
        return self.parts < other.parts

    def __repr__(self) -> str:
        return "/".join(self.parts)

    def __str__(self):
        return "/".join(self.parts)


type PathLike = str | Path


class Origin(abc.ABC):
    def __truediv__(self, path: PathLike) -> "Resource":
        return Resource(self, Path(path))

    # Returns a new origin for the given netloc
    # (e.g. host:port for http, '.' or '/' for file, etc)
    @abc.abstractmethod
    def location(self, location: str) -> "Origin": ...

    @abc.abstractmethod
    def exists(self, path: Path) -> bool: ...
    @abc.abstractmethod
    def is_dir(self, path: Path) -> bool: ...
    @abc.abstractmethod
    def is_file(self, path: Path) -> bool: ...

    @abc.abstractmethod
    def iterdir(self, path: Path) -> tp.Iterator["Resource"]: ...
    @abc.abstractmethod
    def find(self, path: Path) -> tp.Iterator["Resource"]: ...
    @abc.abstractmethod
    def glob(self, path: Path, glob: str) -> tp.Iterator["Resource"]: ...

    @abc.abstractmethod
    def read(self, path: Path) -> tp.ContextManager[io.BufferedReader]: ...
    @abc.abstractmethod
    def write(self, path: Path) -> tp.ContextManager[io.BufferedWriter]: ...
    @abc.abstractmethod
    def mkdir(self, path: Path): ...
    @abc.abstractmethod
    def remove(self, path: Path, recursive: bool = False): ...


class Resource:
    def __init__(self, origin: Origin, path: Path):
        self.path = path
        self.origin = origin

    @property
    def exists(self) -> bool:
        return self.origin.exists(self.path)

    @property
    def is_dir(self) -> bool:
        return self.origin.is_dir(self.path)

    @property
    def is_file(self) -> bool:
        return self.origin.is_file(self.path)

    # return child resources
    def iterdir(self) -> tp.Iterator["Resource"]:
        return self.origin.iterdir(self.path)

    def glob(self, glob: str) -> tp.Iterator["Resource"]:
        return self.origin.glob(self.path, glob)

    # Open as read or write
    def reader(self) -> tp.ContextManager[io.BufferedReader]:
        return self.origin.read(self.path)

    def writer(self) -> tp.ContextManager[io.BufferedWriter]:
        return self.origin.write(self.path)

    def remove(self, recursive: bool = False):
        return self.origin.remove(self.path, recursive=recursive)

    def mkdir(self):
        self.origin.mkdir(self.path)

    def __truediv__(self, other: str | Path) -> "Resource":
        return Resource(self.origin, self.path / other)


class ResourceProvider:
    def __init__(self, **origins: Origin):
        self.origins = origins

    def from_url_or_path(self, url_or_path: str) -> Resource:
        if "://" not in url_or_path:
            path = pathlib.PurePath(url_or_path)
            if not path.root:
                path = "." / path
            origin = self.origins.get("file")
            if not origin:
                raise ValueError(
                    "No 'file' origin registered, cannot handle local paths"
                )
            return Resource(origin, Path(*path.parts))
        else:
            parsed = urllib.parse.urlparse(url_or_path)
            origin = self.origins.get(parsed.scheme)
            if not origin:
                raise ValueError(f"No origin registered for scheme '{parsed.scheme}'")
            origin = origin.location(parsed.netloc)
            path = Path(*pathlib.PurePosixPath(parsed.path).parts)
            return Resource(origin, path)


class LocalFileSystem(Origin):
    def __init__(self, base_path: pathlib.Path | None = None):
        self.base_path = base_path or pathlib.Path.cwd()

    def location(self, location: str) -> "Origin":
        return LocalFileSystem(self.base_path / location)

    def exists(self, path: Path) -> bool:
        return (self.base_path / path).exists()

    def is_dir(self, path: Path) -> bool:
        return (self.base_path / path).is_dir()

    def is_file(self, path: Path) -> bool:
        return (self.base_path / path).is_file()

    def read(self, path: Path) -> tp.ContextManager[io.BufferedReader]:
        return (self.base_path / path).open("rb")

    def iterdir(self, path: Path) -> tp.Iterator[Resource]:
        base = self.base_path / path
        for child in base.iterdir():
            yield Resource(self, Path(*child.parts[len(self.base_path.parts) :]))

    def find(self, path: Path) -> tp.Iterator[Resource]:
        base = self.base_path / path
        for child in base.rglob("*"):
            yield Resource(self, Path(*child.parts[len(self.base_path.parts) :]))

    def glob(self, path: Path, glob: str) -> tp.Iterator[Resource]:
        base = self.base_path / path
        for child in base.glob(glob):
            yield Resource(self, Path(*child.parts[len(self.base_path.parts) :]))

    def write(self, path: Path) -> tp.ContextManager[io.BufferedWriter]:
        (self.base_path / path).parent.mkdir(parents=True, exist_ok=True)
        return (self.base_path / path).open("wb")

    def mkdir(self, path: Path):
        (self.base_path / path).mkdir(parents=True, exist_ok=True)

    def remove(self, path: Path, recursive: bool = False):
        p = self.base_path / path
        if p.is_dir():
            if recursive:
                shutil.rmtree(p)
            else:
                p.rmdir()
        elif p.is_file():
            p.unlink()


class InMemoryFileSystem(Origin):
    def __init__(self):
        self.files: dict[Path, bytes] = {}
        self.dirs: dict[Path, list[str]] = {}

    def location(self, location: str) -> "Origin":
        return self

    def exists(self, path: Path) -> bool:
        return path in self.files or path in self.dirs

    def is_dir(self, path: Path) -> bool:
        return path in self.dirs

    def is_file(self, path: Path) -> bool:
        return path in self.files

    def iterdir(self, path: Path) -> tp.Iterator[Resource]:
        if path not in self.dirs:
            raise NotADirectoryError(f"Not a directory: {path}")
        children = self.dirs[path]
        for child in children:
            yield Resource(self, path / child)

    def find(self, path: Path) -> tp.Iterator[Resource]:
        if path in self.files:
            yield Resource(self, path)
        if path in self.dirs:
            for child in self.dirs[path]:
                yield from self.find(path / child)

    def glob(self, path: Path, glob: str) -> tp.Iterator[Resource]:
        if path not in self.dirs:
            raise NotADirectoryError(f"Not a directory: {path}")
        recursive = "**" in glob
        paths = self.find(path) if recursive else self.iterdir(path)
        regex = re.compile(globlib.translate(glob))
        for child in paths:
            part = str(child.path[len(path) :])
            if regex.match(part):
                yield child

    @contextlib.contextmanager
    def read(self, path: Path) -> tp.Iterator[io.BufferedReader]:
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        yield io.BytesIO(self.files[path])  # type: ignore

    @contextlib.contextmanager
    def write(self, path: Path) -> tp.Iterator[io.BufferedWriter]:
        buffer = io.BytesIO()
        # Make the parent directories
        if len(path) > 1:
            self.mkdir(path[:-1])
        # Initialize the file to empty
        if path not in self.files:
            self.dirs[path[:-1]].append(path.parts[-1])
            self.files[path] = bytes()
        try:
            yield buffer  # type: ignore
        finally:
            self.files[path] = buffer.getvalue()
            parent = Path(*path.parts[:-1])
            if parent not in self.dirs:
                self.dirs[parent] = []
            if path.parts[-1] not in self.dirs[parent]:
                self.dirs[parent].append(path.parts[-1])
                self.dirs[parent].sort()

    def mkdir(self, path: Path):
        if path in self.files:
            raise FileExistsError(f"File exists: {path}")
        if not path.parts:
            return
        if path not in self.dirs:
            self.dirs[path] = []
        # Make the parent directories
        for i in range(len(path) - 1):
            self.mkdir(path[:i])
        parent = Path(*path.parts[:-1])
        if parent not in self.dirs:
            self.dirs[parent] = []
        if path.parts[-1] not in self.dirs[parent]:
            self.dirs[parent].append(path.parts[-1])
            self.dirs[parent].sort()

    def remove(self, path: Path, recursive: bool = False):
        if path in self.files:
            del self.files[path]
            parent = Path(*path.parts[:-1])
            if parent in self.dirs and path.parts[-1] in self.dirs[parent]:
                self.dirs[parent].remove(path.parts[-1])
        elif path in self.dirs:
            if self.dirs[path] and not recursive:
                raise OSError(f"Directory not empty: {path}")
            if recursive:
                for child in list(self.dirs[path]):
                    self.remove(path / child, recursive=True)
            parent = path[:-1]
            if parent in self.dirs and path.parts[-1] in self.dirs[parent]:
                self.dirs[parent].remove(path.parts[-1])
        else:
            raise FileNotFoundError(f"File not found: {path}")
