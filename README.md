# Generate X

A JAX- and PyTorch-compatible library for generative modeling.

## TorchScript backend (CPU and CUDA)

`gnx-torch` traces fixed-shape GNX expressions with `tch` 0.22.0. On Linux
x86_64 (GNU), Cargo downloads LibTorch 2.9.0: CPU by default, CUDA 13 with
the `gnx-torch/cuda` feature. The CUDA build uses the native SDK from the
PyTorch wheel (glibc 2.28+) without installing or running Python. On Apple
Silicon the managed runtime is CPU-only. The build verifies the pinned SHA-256
**before** extraction and reuses the verified install across builds. With a
C++17 compiler, `curl`, `unzip`, and `tar` (for CUDA):

```sh
cargo test --locked -p gnx-torch
```

Neither `LIBTORCH` nor `LD_LIBRARY_PATH` needs to be set for managed builds
and the Qwen binary. The backend's native bridge carries its own runtime paths;
keep both the Cargo build output and LibTorch cache when running a local binary. The cache defaults to `$XDG_CACHE_HOME/gnx/libtorch` (or
`~/.cache/gnx/libtorch`); set `GNX_LIBTORCH_CACHE` to relocate it before building.
Keep the shared CUDA libraries in `$XDG_CACHE_HOME/gnx/cuda` as well
(`~/.cache/gnx/cuda` by default; override with `GNX_CUDA_CACHE` at build time).
Allow several GB for the initial download and extraction. To use an
existing installation, or on an unlisted target, set `LIBTORCH` or use
`LIBTORCH_USE_PYTORCH=1` with matching PyTorch. These explicit overrides bypass
the managed download.

Compile once with `let program = TorchDevice::cpu(cache)?.compile(&expr).await?`, then
call `program.run(&args, opts)?` with captures followed by explicit inputs.
Use `ExecOpts { transfers: TransferPolicy::AllowCopy }` for host inputs; tensors
already imported on the device can be reused with the default zero-copy policy.
`device.download(&handle)?` reads a result back into a packed `HostBuffer`.

Use `TorchDevice::cuda(0, cache)?` for the first CUDA-visible GPU. This requires a
compatible NVIDIA driver and the CUDA-enabled runtime selected by `--features cuda`.
An unavailable GPU or invalid ordinal returns an error instead of falling back to CPU. To exercise
GPU import, device-buffer reuse, saved-module replay and readback, run
`cargo test --locked -p gnx-torch --features cuda --test cuda -- --include-ignored`.

See the [Torch operation matrix](gnx-torch/README.md) for arithmetic, comparisons,
selection, shape operations, and their dtype restrictions. The backend builds a
C++17 bridge against those same LibTorch headers to emit runtime conditionals
and loops. Saved modules execute using standard TorchScript operators. `program.modules()` exposes
one TorchScript module per distinct computed output for inspection or saving.
Shared intermediates can therefore be recomputed across distinct outputs.

## Native XLA backend

`gnx-xla` provides single-device execution through `gnx-xla-sys` and PJRT.
`XlaDevice::cpu(cache)` selects the CPU; `XlaDevice::cuda(0, cache)` selects the first
CUDA-visible GPU. `XlaDevice::compile` returns a reusable executable.
The default `lapack` feature adds CPU LAPACKE decompositions; Qwen omits it.

Creating a builder or client automatically routes XLA's Abseil logs to the
Rust `log` target `xla`. Install your application's logger before that point
to capture startup records. With no logger, ordinary records are discarded.
The bridge suppresses Abseil's process-wide stderr copy; fatal errors still
terminate. Earlier startup messages and plugins writing directly to stderr
can bypass the bridge.

With a C++17 compiler, `curl`, and `tar` on PATH, run:

```sh
cargo test --locked -p gnx-xla
```

On GNU/Linux and macOS (x86_64 or aarch64), `gnx-xla-sys` downloads the matching
[`xla_extension` v0.11.0](https://github.com/elixir-nx/xla/releases/tag/v0.11.0)
CPU archive with `curl`, verifies its pinned SHA-256 before extracting it with
`tar`, and caches it in Cargo's build output. On GNU/Linux x86_64 and aarch64,
`qwen-example` enables the managed CUDA 13 bundle. This additionally downloads
a SHA-256-pinned PJRT plugin, NVIDIA runtime libraries (including `libcudart`,
cuBLAS and cuDNN), and GPU compiler tools. Allow several GB for the first build.
No Python installation, system CUDA toolkit, or `LD_LIBRARY_PATH` setup is needed.
On GNU/Linux x86_64, the default managed CUDA 13 backends share the same NVIDIA
libraries, including cuDNN 9.13. You can keep `TorchDevice::cuda(0, cache)?` and
`XlaDevice::cuda(0, cache)?` alive in the same process. They still own separate buffers.
To test convolutions with both initialization orders, run:

```sh
cargo test --release --locked -p gnx --features qwen-example --test cuda_coexist -- --ignored
```

For XLA alone, use `XLA_EXTENSION_VARIANT=cuda12` for CUDA 12, or `cpu` to
skip XLA's GPU downloads. These overrides do not select Torch's runtime.
macOS builds are CPU-only.

The plugin and driver are opened only when a GPU client is requested. The same
managed binary runs on CPU with no NVIDIA driver installed; explicit `--cuda`
requires a compatible host driver and an addressable GPU. To test CUDA, run:

```sh
cargo test --locked -p gnx-xla --no-default-features --features cuda \
  --test cuda -- --ignored --nocapture
```

The tests execute a matrix multiplication and verify GNX evaluation, copies,
device-buffer reuse, and readback. On a host **without** the NVIDIA driver, use
`--test cuda_driverless -- --ignored` instead to check CPU startup and the
explicit CUDA error path. For arrays initially stored on the host:

```rust
use gnx::{Eval, ExecOpts, buffer::TransferPolicy, cache::Cache};
use gnx_xla::XlaDevice;

let gpu = XlaDevice::cuda(0, Cache::in_memory())?;
gpu.eval(&result, ExecOpts { transfers: TransferPolicy::AllowCopy }).await?;
let host = gpu.download(result.data().unwrap())?;
```

`ExecOpts::default()` requires zero-copy inputs. To upload a persistent weight
once, use `gpu.import(&data, TransferPolicy::AllowCopy)`; subsequent evaluations
on that GPU can reuse the returned handle under the default policy.

For an offline build or a custom runtime, set `XLA_EXTENSION_DIR` to the
unpacked `xla_extension` directory to bypass the managed XLA and CUDA downloads.
A custom extension retains its own driver and toolkit requirements. On Linux,
`gnx-xla-sys` links its C++ bindings as `libxla_rs.so`. Executables record the absolute
path to that library in Cargo's build output; the library carries the search
paths for XLA and its native dependencies. Native NixOS CUDA builds also search
`/run/opengl-driver/lib` for `libcuda.so.1`, so the NVIDIA driver directory does
not need to be in `LD_LIBRARY_PATH`. Keep the build output and runtime libraries in place; the
executable alone is not a portable bundle.

Bindgen uses a local libclang if available. On GNU/Linux and macOS build hosts
(x86_64 or aarch64), it otherwise downloads and caches a SHA-256-verified
libclang 18.1.1 from PyPI; Python is not required. To use a specific local
library, set `LIBCLANG_PATH` to its directory.

The native bridge first tries to compile and link against system LAPACKE. On
GNU/Linux x86_64 and aarch64 (glibc 2.27 or newer), if that fails, Cargo downloads a pinned,
SHA-256-verified `scipy-openblas32` wheel and uses its LAPACKE headers and
libraries. The download is cached in Cargo’s build output; neither Python nor
environment variables are required. On other targets, install system LAPACKE.
For an offline or custom installation, `LAPACKE_INCLUDE_DIR` and
`LAPACKE_LIB_DIR` override the header and library search paths.

Run `maturin develop --features xla` from `gnxlib/` to enable
`from gnxlib import xla`. `array.from_host_bytes`, `array.from_numpy`, and
`array.from_torch` copy into immutable, aligned host storage. NumPy arrays are
copied directly into that storage, including strided or byte-swapped inputs. CPU
PJRT borrows the resulting allocation, so `await device.eval(roots)` needs no
extra input copy and takes no `allow_copy` flag. In Rust, use
`HostBuffer::copy_from_slice` to create an aligned snapshot. Results already
owned by the same XLA client can also be reused without copying.

## Qwen text generation in Rust

`qwen-example` builds both XLA and Torch with their CUDA features enabled.
XLA is the default backend; execution is on CPU unless `--cuda` is passed.
Run the example on a local
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) checkpoint:

```sh
hf download Qwen/Qwen3.5-0.8B --local-dir Qwen3.5-0.8B \
  --include 'config.json' 'tokenizer.json' '*.safetensors'
cargo run --locked --release -p gnx --features qwen-example --example qwen -- \
  --model-dir Qwen3.5-0.8B --prompt 'The capital of France is' --max-new-tokens 4
```

To chat interactively with the same checkpoint:

```sh
cargo run --locked --release -p gnx --features qwen-example --example qwen -- \
  --model-dir Qwen3.5-0.8B --chat --max-new-tokens 64
```

On an NVIDIA host, pass `--cuda`:

```sh
cargo run --locked --release -p gnx --features qwen-example --example qwen -- \
  --model-dir Qwen3.5-0.8B --cuda --chat
```

`--cuda` selects ordinal 0; `--cuda 1` selects the second visible GPU. Weights
and layer caches stay on that device, so it must have enough VRAM for both.

Select Torch from the **same binary**, with no additional feature or environment setup:

```sh
cargo run --locked --release -p gnx --features qwen-example --example qwen -- \
  --backend torch --cuda --model-dir Qwen3.5-0.8B --chat
```

Omit `--cuda` to run on CPU with either backend. For a smaller Torch-only CPU
build, `--features qwen-torch` omits XLA and downloads the CPU LibTorch archive.

Enter one message per line. Replies stream as they are generated. `/reset` clears
history; `/quit` or Ctrl-D exits. The token limit applies to each reply. Chat
loads the weights once and reprocesses the conversation history on each turn.

For [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), substitute
`Qwen/Qwen3.8-27B` and `Qwen3.8-27B` in those commands. Use the original BF16
checkpoint and allow enough CPU RAM for the weights (roughly 54 GB for the 27B
text decoder), a checkpoint shard, and XLA execution buffers.

[`gnx/examples/qwen`](gnx/examples/qwen) runs greedy completion on raw text.
Prompt and generated tokens pass through the same single-token decoder. The
linear projections, feed-forward blocks, and recurrent updates use `#[gnx::jit]`;
`device.eval` materializes the next token and layer caches together. Only the
generated token ID is read back. This example favors clarity over throughput:
full-attention caches grow with the sequence, and evaluation compiles each step.

## Compilation cache

Every device takes a `gnx::cache::Cache`, in which compilation keeps the result
of each pass and the lowered program (HLO for XLA, TorchScript modules for
Torch). Compiling an expression again, on any device with the same cache, reads
the program back instead of lowering it; the native compilation of that
program still runs. `Cache::in_memory()` lasts as long as the process. To keep
programs between runs, give the cache a store:

```rust
use gnx::{cache::Cache, store::Store};

let cache = Cache::new(Store::local("/var/cache/gnx")?);
let cpu = XlaDevice::cpu(cache.clone())?;
```

Entries are files named by the hash of their contents and are memory-mapped
when read. `Store::mirrored(remote, dir)` puts a local directory in front of
any other `object_store`, so that its objects are mapped as well. The device
for a given CPU or CUDA ordinal is shared by the process and keeps the cache
of the first call that opened it.

Cached entries are **not validated** when read, and are not invalidated when
GNX changes: use a cache directory with one build of GNX, and clear it after
upgrading or after changing a pass.

## Array operations

`Array` supports lazy elementwise `+`, `-`, `*`, `/`, and `%` (remainder) on
owned or borrowed arrays. For example, `(&a - &b) * &scale` builds a trace;
`device.eval` executes it. Operands must have matching dtypes, and fixed shapes
broadcast from the right. Use `try_add`, `try_sub`, `try_mul`, `try_div`, or
`try_rem` to handle incompatible operands without a panic.

`gnx_common::ops` defines the core array dialect and its primitive constructors.
`gnx_math::ops` re-exports those primitives and adds modules for
elementwise math, bitwise operations, dtypes, creation, shape, indexing,
reductions, ordering, contractions, FFT, linear algebra, and special functions.
The same modules are available through `gnx::ops` in expression macros.
`reduction::reduce` and `reduction::reduce_window` accept a closed scalar
combiner, an identity per operand, and multiple results. `stop_gradient` is
available in `gnx-transforms` and the Python frontend.

## Concrete array storage

`gnx_math::DataHandle` carries shape and dtype, shard ranges and replicas, memory
location, and a storage reference. Storage may be owned packed host bytes, an
immutable local memory view, or an opaque allocation reference. A whole-array
reference can describe distributed data even if none of its shards is addressable
here. The owning runtime, not the descriptor, resolves opaque allocations. GNX
can forward these references without accessing their bytes.

Host UUIDs default to process scope; callers can supply an explicit `HostId`.
Physical device IDs use a vendor namespace and a hardware or partition UUID when
known. Runtime ordinals and allocation IDs are scoped to a runtime incarnation
and must not be treated as hardware identities.

DLPack adapts individual addressable shards. Local borrowing requires a valid
lifetime lease, ready immutable bytes, and a compatible importer. In particular,
borrowing NumPy or Torch storage from Rust requires the caller to guarantee no
writable aliases; the Python constructors instead copy. CPU XLA can import a
compatible local host view without copying; its own outputs remain same-client
references. XLA does not implement device-to-device DLPack interchange. Use
`XlaDevice::download` for explicit host readback. `RequireZeroCopy` never silently falls back to a transfer.
