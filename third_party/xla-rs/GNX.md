Vendored from the crates.io xla 0.5.5 release by Laurent Mazare, licensed
MIT or Apache-2.0 as declared in Cargo.toml. GNX extends the operation
bindings against the same xla_extension release used by the workspace.

The `pjrt` feature exposes `xla::pjrt` for single-device CPU and CUDA clients,
leased host-buffer imports, compilation, execution, and checked readback. The independent `lapack`
feature supplies the Host FFI handler behind `xla::lapack::Kernel::call`.
Both C++ sources build with the same extension headers and ABI flags as xla-rs.
