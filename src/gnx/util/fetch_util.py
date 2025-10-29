import contextlib
import hashlib
import mmap
from pathlib import Path

import requests
import base64

from rich.progress import Progress


def download_url(url, destination: str | Path, quiet=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MB chunks

    if not quiet:
        progress = Progress()
        task = progress.add_task(f"Downloading {destination.name}", total=total_size)
        pbar = progress
    else:
        progress = None
        task = None
        pbar = contextlib.nullcontext()

    with open(destination, "wb") as file, pbar:
        for data in response.iter_content(block_size):
            file.write(data)
            if not quiet:
                assert progress is not None
                assert task is not None
                progress.update(task, advance=len(data))


_CHUNK_SIZE = 128 * 1024  # 128 KB chunks


def md5_checksum(*paths: str | Path) -> bytes:
    md5_hash = hashlib.md5()
    for path in sorted(map(Path, paths)):
        with open(path, "rb") as f:
            try:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    md5_hash.update(mm)
            except OSError:
                # If we can't memory-map the file, read it in chunks.
                chunk = f.read(_CHUNK_SIZE)
                while chunk:
                    md5_hash.update(chunk)
                    chunk = f.read(_CHUNK_SIZE)
            except ValueError:
                # The file is empty, continue to the next file.
                continue

    return md5_hash.digest()


def md5_checksum_hex(*paths: str | Path) -> str:
    return md5_checksum(*paths).hex()


def md5_checksum_b64(*paths: str | Path) -> str:
    return base64.b64encode(md5_checksum(*paths)).decode("utf-8")
