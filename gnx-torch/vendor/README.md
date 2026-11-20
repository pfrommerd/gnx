# Native binding build integration

`torch-sys/` contains the Rust bindings and `libtch/` sources from crates.io's
`torch-sys` 0.22.0 (`.crate` SHA-256: `aef40c585e342df95b66a1fa7c923188623999c2b657227befb481dfb03a6a42`). The workspace's `[patch.crates-io]` makes `tch` and `gnx-torch` use the same native binding crate. A separate workspace depending on this checkout must carry that patch in its own root manifest.

The upstream Rust and C++ binding sources are unchanged. Local build changes:

* `managed.rs` pins the official PyTorch archives, verifies downloads and publishes complete installations under a per-archive cache lock.
* `gnx-managed` resolves this install before compiling libtch. On GNU/Linux x86_64 `gnx-cuda` extracts the C++ headers and native libraries from the pinned PyTorch CUDA 13 wheel. Unlike the standalone LibTorch archive, its DSOs refer to NVIDIA libraries by their public SONAMEs. `gnx-cuda-build` makes those names resolve to the same files used by the managed XLA plugin, and installs the Torch-only dependencies beside LibTorch. Explicit `LIBTORCH` and `LIBTORCH_USE_PYTORCH` installations still use upstream discovery.
* For dynamic Linux and macOS builds, `build.rs` passes the native binding archive, libdir, headers and C++ ABI to `gnx-torch/build.rs`. The backend links both C++ bridges into one shared library with an absolute SONAME (install name on macOS) and its own runtime search paths. Consumer binaries need no build script. Keep the Cargo build output and the LibTorch installation when running a local binary.

When upgrading `tch`, refresh `torch-sys` and the LibTorch version, URLs and digests together. Recheck the managed CPU build, CUDA build, corrupt archive rejection and direct binary launch without a loader-path override.
