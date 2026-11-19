# Generate X

A JAX- and PyTorch-compatible library for generative modeling.

## Native XLA backend

`gnx-xla` compiles GNX's lowered SSA directly with `xla-rs` and PJRT. It does not
link Python or require JAX. The `native` Cargo feature enables single-device CPU
execution, including `add`, `call`, `cond`, `switch`, `while_loop`, and `scan`.
`XlaDevice::compile` returns a reusable executable.

Use clang, libclang, and the matching
[`xla_extension` v0.11.0](https://github.com/elixir-nx/xla/releases/tag/v0.11.0)
build for `xla-rs` 0.5.5. On Linux, point Cargo and the loader at the unpacked
runtime:

```sh
export XLA_EXTENSION_DIR=/path/to/xla_extension
export CXX=clang++
export LD_LIBRARY_PATH="$XLA_EXTENSION_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test --locked -p gnx-xla --features native
```

If libclang is not on the search path, set `LIBCLANG_PATH` to its directory.
To build the Python frontend with the optional backend, run `maturin develop
--features xla` from `gnxlib/`, then `from gnxlib import xla`. `gnxlib.jax`
remains available separately. `array.from_host_bytes`, `array.from_numpy`, and
`array.from_torch` copy into immutable, aligned host storage. NumPy arrays are
copied directly into that storage, including strided or byte-swapped inputs. CPU
PJRT borrows the resulting allocation, so `await device.eval(roots)` needs no
extra input copy and takes no `allow_copy` flag. In Rust, use
`HostBuffer::copy_from_slice` to create an aligned snapshot. Results already
owned by the same XLA client can also be reused without copying.

## Distributed CPU mesh

Enable `gnx-xla`'s `mesh` feature on the controller and workers. On each worker
host, run `gnx-xla-worker [listen-address] [host-uuid]` (the default listen
address is `127.0.0.1:7440`). For example, to start one worker locally:

```sh
CXX=clang++ cargo run --locked -p gnx-xla --features mesh --bin gnx-xla-worker
```

One controller calls `Mesh::connect("data", worker_addresses).await` in a Tokio
runtime. Address order defines the mesh coordinates. `Mesh` implements GNX's
`Device` trait; `Mesh::compile` can retain executables for repeated calls, and
`Mesh::download` is the explicit readback operation. Workers retain their
shards, so a subsequent call using the same layout can pass references rather
than transferring bytes.

`gnx_math::shard::{constrain, replicate}` adds `shard.constraint` and
`shard.replicate` operations. Mesh lowering emits StableHLO with Shardy
annotations and runs OpenXLA's Shardy and SPMD passes before local PJRT
compilation. For example, `Sharding::new(vec![vec!["data".into()], vec![]])`
splits the leading dimension of a rank-two array along the `data` axis.

This first mesh supports one CPU device per worker, one mesh axis, replication
or even leading-axis partitioning, and `add` with nested `call`. It rejects
programs whose compiled partitions require cross-worker collectives. In Rust,
set `ExecOpts::transfers` to `AllowCopy` to permit uploads or resharding during
`eval` or `execute`. The tarpc TCP transport
is intended for trusted peers (no built-in authentication or TLS); use it on
loopback or a protected network. No coordinator process beyond the controller
is required.

### Mesh from Python

Build `gnxlib` with `maturin develop --features xla-mesh`. With two workers
already listening, run the following from an async function:

```python
import struct
from gnxlib import array, xla

mesh = await xla.Mesh.connect("data", ["127.0.0.1:7440", "127.0.0.1:7441"])
a = array.from_host_bytes([4], "float32", struct.pack("<4f", 1, 2, 3, 4))
result = xla.constrain(a + a, [["data"]])
await mesh.eval(result)
print(struct.unpack("<4f", await mesh.to_host_bytes(result)))
```

`mesh.eval` uploads host inputs or reshards inputs it can already read when the
compiled layout requires it. It reuses worker allocations that are already in the
right place. `mesh.upload` is optional: use it to place an array ahead of time
for reuse across runs; omit `axes` to replicate it. `mesh.to_host_bytes` explicitly
gathers a result.

## Concrete array storage

`gnx_math::DataHandle` carries shape and dtype, shard ranges and replicas, memory
location, and a storage reference. Storage may be owned packed host bytes, an
immutable local memory view, or an opaque allocation reference. A whole-array
reference can describe distributed data even if none of its shards is addressable
here. The owning runtime, not the descriptor, resolves opaque allocations. GNX
can forward these references without accessing their bytes. The descriptor by
itself does not create a connection; the optional mesh runtime resolves its own
workers' references.

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
