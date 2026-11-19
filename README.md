# Generate X

A JAX- and PyTorch-compatible library for generative modeling.

## Native XLA backend

The `native` Cargo feature enables single-device CPU execution through
`xla-rs` and PJRT. `XlaDevice::compile` returns a reusable executable.
Matrix decompositions that use LAPACKE run as CPU XLA FFI operations.

Use clang, libclang, LAPACKE, and the matching
[`xla_extension` v0.11.0](https://github.com/elixir-nx/xla/releases/tag/v0.11.0)
build for `xla-rs` 0.5.5. On Linux, point Cargo and the loader at the unpacked
runtime:

```sh
export XLA_EXTENSION_DIR=/path/to/xla_extension
export CXX=clang++
export LD_LIBRARY_PATH="$XLA_EXTENSION_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test --locked -p gnx-xla --features native
```

Set `LIBCLANG_PATH` if libclang is outside the default search path. Set
`LAPACKE_INCLUDE_DIR` and `LAPACKE_LIB_DIR` for a custom LAPACKE installation.

Run `maturin develop --features xla` from `gnxlib/` to enable
`from gnxlib import xla`. `array.from_host_bytes`, `array.from_numpy`, and
`array.from_torch` copy into immutable, aligned host storage. NumPy arrays are
copied directly into that storage, including strided or byte-swapped inputs. CPU
PJRT borrows the resulting allocation, so `await device.eval(roots)` needs no
extra input copy and takes no `allow_copy` flag. In Rust, use
`HostBuffer::copy_from_slice` to create an aligned snapshot. Results already
owned by the same XLA client can also be reused without copying.

## Qwen text generation in Rust

With the native XLA environment above, run the `gnx` example on a local
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

## Array operations

`Array` supports lazy elementwise `+`, `-`, `*`, `/`, and `%` (remainder) on
owned or borrowed arrays. For example, `(&a - &b) * &scale` builds a trace;
`device.eval` executes it. Operands must have matching dtypes, and fixed shapes
broadcast from the right. Use `try_add`, `try_sub`, `try_mul`, `try_div`, or
`try_rem` to handle incompatible operands without a panic.

`gnx_array::ops` defines the core array dialect and its primitive constructors.
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
