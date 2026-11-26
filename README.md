# Generate X

A JAX- and PyTorch-compatible library for generative modeling.

## Random arrays and layer initialization

```rust
use gnx::{array::DType, nn::{Linear, Rngs}, random::{self, Algorithm}};

let root = random::key(42, Algorithm::ThreeFry2x32); // or Algorithm::Philox4x32
let (model_key, noise_key) = random::split2(&root);
let streams = Rngs::new(model_key);
let layer = Linear::init(128, 256, DType::BF16, &streams);
let noise = random::normal(&noise_key, (32, 256), DType::F64);
let mask_draw = random::uniform(&streams.next("dropout"), (32, 256), DType::F32);
```

A key has scalar shape `()` and a nonnumeric, algorithm-specific dtype.
Reusing it repeats a draw. `split` produces an array of keys; `fold_in` derives
one key from a scalar `U32` tag. Use `map` for draws from batched keys. For
interchange, `key_data` exposes two `U32` words per key and `wrap_key_data`
restores them with an explicit algorithm. GNX targets **JAX 0.12.1** for
`key`, `split`, `fold_in`, and `U32` bits with `threefry2x32` and `philox4x32`
(`jax_threefry_partitionable=True`, x64 enabled for 64-bit seeds).
This changes seeded Philox streams from the earlier GNX convention; Threefry streams are unchanged.
Floating samples do not promise bitwise JAX or cross-backend identity.

`uniform` and `normal` support `BF16`, `F16`, `F32`, and `F64`. `Rngs` draws
from independently named streams using traced counters. `Linear::init` uses two
keys from `"params"` to initialize weight and bias uniformly with PyTorch's
`1 / sqrt(in_features)` bound. Use `Linear::init_with(in_features, out_features,
false, dtype, &streams)` for a weight-only layer; it consumes just one key.
`layer.bias()` returns `Option<&Param>`. Shape arguments accept tuples through rank eight,
arrays of any rank, slices, vectors, and `Shape`; `()` denotes a scalar.

## Scalars, casts, and arithmetic

`Array::scalar(v)` creates a rank-zero array with the Rust value's dtype:
`Array::scalar(0.5)` is F64, `Array::scalar(0.5_f32)` is F32, and
`Array::scalar(u64::MAX)` retains every bit. `AsDType` describes the dtype;
`IntoScalar` also supplies the value. Both are implemented for `bool`, native
8–64-bit integers, pointer-sized integers, `f32`, `f64`, and `gnx::half::{f16, bf16}`.

```rust
use gnx::{Array, array::DType, half::bf16};

let x = Array::scalar(bf16::from_f32(4.0));
let y = 1.0 - &x / 2;       // literals adopt x's BF16 dtype in either position
let z = y.astype(DType::F32);
```

Use `lax::try_scalar(v, dtype)` when the dtype is known at runtime and the
value must be exactly representable. For intentional rounding, construct the
scalar in its native dtype and call `.astype(dtype)`. Arithmetic literals follow
the array dtype: floating literals may round to its floating precision; integer
literals must fit exactly, and float literals cannot be used with integer arrays.
`ArrayRef::astype` reads the cell and returns an immutable `Array`.

`array::constant` now stores typed values. Rebuild serialized expressions and
cache entries created with its old f64-only payload before loading them.

## Indexing arrays and references

Use `array.at(3)` for one row, `array.at((.., 1..4))` for a slice, or
`weights.at(&token_ids)` to gather rows for an integer array of token IDs.
Trailing axes are kept. `try_at` returns construction errors; tensor indices
must be in bounds when evaluated.

`ArrayRef::at` reads an immutable `Array`. `cell.set_at((1, ..), &row)` reads
the cell, replaces the selected window, and writes it back. For a functional
update on an `Array`, use `array.with_at((1, ..), &row)`. Writes accept integer
and range selectors and require the replacement to match the window shape and dtype.

## Neural-network layers and language models

`gnx::nn` exports `Conv`, `Linear`, `LayerNorm`, `GroupNorm`, `BatchNorm`, and
`RmsNorm`, plus token embeddings, rotary attention, SwiGLU and routed experts.
Convolution and normalization use channels-last inputs. `nn::activations`
contains the standalone activation functions.

`gnx-models` provides `Qwen3LM` (dense or MoE) and `KimiK2LM` (MLA and shared
and routed experts). `Qwen3Config::default()` describes Qwen3-0.6B;
`Qwen3Config::moe_30b_a3b()` and `KimiK2Config::default()` describe Qwen3-30B-A3B
and Kimi-K2-Base. Constructors initialize lazy floating-point parameters.

```rust
use gnx::{array::DType, lax, nn::Rngs, random::Algorithm};
use gnx_models::{DecoderCache, Qwen3Config, Qwen3LM};

let config = Qwen3Config { vocab: 128, hidden: 32, layers: 2, heads: 4,
    kv_heads: 2, head_dim: 8, intermediate: 64, ..Default::default() };
let model = Qwen3LM::init(&config, DType::F32, &Rngs::key(42, Algorithm::ThreeFry2x32));
let mut cache = DecoderCache::default();
let prompt = lax::iota([1, 3], 1, DType::I32);
let logits = model.forward_at(&prompt, 0, Some(&mut cache), None); // [1, 3, 128]
```

Use `forward(&ids)` for a full causal pass. For decoding, reuse the cache and
pass the absolute position of the next token. The optional boolean padding mask
covers the full key history (`[batch, key_time]`, `true` keeps a token).

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
  --model-dir Qwen3.5-0.8B --prompt $'<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n'
```

To chat interactively with the same checkpoint:

```sh
cargo run --locked --release -p gnx --features qwen-example --example qwen -- \
  --model-dir Qwen3.5-0.8B --chat
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

Enter one message per line. Chat prints each completed reply, excluding
`<|im_end|>`. `/reset` clears history; `/quit` or Ctrl-D exits. Chat loads the
weights once and reprocesses the conversation history on each turn.

For [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), substitute
`Qwen/Qwen3.8-27B` and `Qwen3.8-27B` in those commands. Use the original BF16
checkpoint and allow enough CPU RAM for the weights (roughly 54 GB for the 27B
text decoder), a checkpoint shard, and XLA execution buffers.

[`gnx/examples/qwen`](gnx/examples/qwen) runs greedy completion on raw text.
Prompt and generated tokens call the same `#[gnx::jit] fn step`. The cursor
and each layer’s two cache slots are `ArrayRef`s. The step updates KV storage
with `dynamic_update_slice`, masks attention using a shared symbolic prefix
length, then advances the cursor once. After prompt prefill, a compiled GNX
`while_loop` feeds each prediction into the next step and collects the reply
without a Rust readback per token. Only `<|im_end|>` completes a reply. If
the context fills first, generation reports an error rather than returning
partial text.
`--context-length` sets the storage bound (default 2048, capped by the model);
larger bounds reserve more memory and increase attention work. Add `--timings` to measure time to first
token and subsequent decode throughput separately.

With `--timings`, first-token time includes prefill but excludes checkpoint
loading. The first decode also compiles the loop. To measure warm decode, run
`--cuda --chat --timings`, send a message, `/reset`, then repeat the message.

## Compilation cache

Every device takes a `gnx::cache::Cache`, in which compilation keeps the result
of each pass and the lowered program (HLO for XLA, TorchScript modules for
Torch). Each device additionally caches the final runtime stage: the **loaded
PJRT executable** on XLA, or the loaded TorchScript module(s) on Torch. A hit
on the same runtime skips native compilation or module loading. A new process
can reuse the portable stages, but must create its own native handles. To keep
portable artifacts between runs, give the cache a store:

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

`gnx_base::ops` defines the core array dialect and its primitive constructors.
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
