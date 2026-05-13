# AGENTS.md

## Cursor Cloud specific instructions

### Repository Overview

GNX ("Generate X") is a JAX/PyTorch-compatible generative modeling library. The core computation engine is in Rust (`rust/` workspace with 8 crates), exposed to Python via PyO3/maturin (`gnxlib` crate). There is also an optional Dioxus 0.7 web UI (`rust/gnx-ui/`).

### Environment Variables

- **`PYO3_PYTHON=python3.13`** must be set when building the Rust workspace or running `uv sync`, since the default `python3` is 3.12 but the project requires 3.13.
- **`PATH`** must include `$HOME/.local/bin` for `uv`.

### Rust Workspace

- **Build:** `cd rust && cargo build` (set `PYO3_PYTHON=python3.13` first)
- **Test:** `cd rust && cargo test` — runs 14 tests across gnx, gnx-expr, gnx-graph, gnx-io crates
- Edition 2024 crates require Rust 1.85+; the update script pins `rustup default stable` (currently 1.95).
- `gnxlib` is a `cdylib` that links against Python 3.13; it will fail to link if `PYO3_PYTHON` is not set.

### Python Package

- **Install/sync:** `cd python && uv sync` — installs all Python deps including building `gnxlib` from source
- **Run:** `cd python && uv run python main.py`
- **Type check:** `cd python && uv run pyright` (pyright is installed via `uv pip install pyright`; not in the lockfile)
- The Python venv lives at `python/.venv` and is created automatically by `uv sync`.
- `gnxlib` is referenced as a path dependency (`../rust/gnxlib`) in `python/pyproject.toml`; `uv sync` builds it via maturin.

### Dioxus Web UI (optional)

- **Serve:** `cd rust/gnx-ui && dx serve --platform web` — builds WASM + fullstack server, serves at `http://127.0.0.1:8080`
- `dx` (dioxus-cli) is installed via `cargo install dioxus-cli`; it is not part of the update script since it takes ~5 minutes to compile.
- There may be non-critical BundledAsset deserialization warnings due to a minor version gap between dioxus 0.7.3 (in Cargo.toml) and dioxus-cli 0.7.9; these do not affect functionality.
- See `rust/gnx-ui/AGENTS.md` for Dioxus 0.7 API reference.

### Gotchas

- The `deadsnakes` PPA is needed for Python 3.13 on Ubuntu 24.04 (Noble).
- `lld` and `libssl-dev` are required system packages for linking Rust crates.
- Pyright reports 2 pre-existing type errors in `python/src/gnx/`; these are not caused by environment setup.
