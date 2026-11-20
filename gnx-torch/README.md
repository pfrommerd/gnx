# Torch operation support

`TorchDevice::cpu()` and `TorchDevice::cuda(ordinal)` use the same GNX lowering
rules. `cargo build -p gnx-torch` installs checksum-verified CPU LibTorch;
add `--features cuda` on Linux x86_64 (GNU) for the managed CUDA 13 SDK and the NVIDIA runtime shared with XLA.
CUDA execution requires a compatible NVIDIA driver. See the [build and cache
instructions](../README.md#torchscript-backend-cpu-and-cuda) for platforms and overrides.
Kernel and dtype availability depend on the installed runtime and GPU; the CPU
test suite is not an exhaustive CUDA compatibility matrix. Programs are traced
for the selected device. Extension lowerings can use `tracing_device()` when
creating tensors without operands. Host uploads require `AllowCopy`; imported
handles can be reused without copying on the same device. GPU smoke tests are
in [`tests/cuda.rs`](tests/cuda.rs).

All traced arrays require fixed dimensions. Supported element types are `bool`,
`u8`, `i8/i16/i32/i64`, `f16/bf16/f32/f64`, and `c64/c128`. **Numeric** below
means these types except `bool`; **ordered** excludes complex types. The table
lists primitive operations after transform expansion and dead-code elimination.

| Primitive | Contract |
| --- | --- |
| `zeros`, `ones` | Result shape and dtype come from the expression. |
| `reshape`, `transpose`, `broadcast_in_dim` | Preserve element type. Broadcast axes are strictly increasing. |
| `reduce` | Closed associative scalar reducer with one or more result lanes, potentially of different dtypes. Reduces arbitrary static axes; zero contributors return the supplied scalar initializers. The generic trace unrolls the reducer per contributing element. See [`tests/reduction.rs`](tests/reduction.rs). |
| `reduce_window` | Uses the same variadic scalar reducer over fixed-size sliding windows. Supports per-axis stride, signed padding, base dilation, and window dilation. Padding uses each lane's initializer; trace size grows with output size times window volume. |
| `bitcast_convert_type` | Preserves bits in little-endian element order. Equal widths retain shape; narrowing adds a trailing width-ratio axis, and widening requires and removes that axis. Complex-to-real and real-to-complex bitcasts are invalid. Non-boolean inputs cannot be bitcast to `bool` because GNX requires canonical 0/1 bytes. |
| `reduce_precision` | `f16`, `bf16`, `f32`, `f64`; retains the input dtype. Rounds mantissa ties to even before clipping the exponent. Underflow preserves the sign of zero; NaNs are preserved unless zero mantissa bits are requested, in which case they become positive infinity. |
| `reduce_add` | Sum over static axes, retaining dtype; an empty axis list is the identity. |
| `add`, `mul` | Numeric addition and multiplication; OR and AND for predicates. |
| `sub`, `div`, `neg` | Numeric. Integer division truncates toward zero. |
| `rem` | Ordered numeric. Remainder has the dividend's sign (`fmod` for floats). |
| `abs` | Numeric. Complex inputs return the corresponding real dtype. |
| `min`, `max` | Ordered. Propagate NaN; minimum prefers −0 and maximum prefers +0 when the signs of zero differ. |
| `eq_to`, `lt_to`, `le_to` | IEEE total order for real floats: distinguishes −0 from +0 and orders NaNs by sign, signaling/quiet bit, and payload. Integer and boolean operands use their ordinary order. |
| `eq`, `ne` | All supported types; boolean result. |
| `lt`, `le`, `gt`, `ge` | Ordered; boolean result. |
| `select` | Boolean predicate, either scalar or the same shape as the two alternatives. |

For `sub`, `div`, `rem`, extrema, comparisons, and `select` alternatives,
operands must match in shape and dtype. Use `broadcast_in_dim` explicitly when
needed. Integer division by zero returns all ones; integer remainder by zero
returns the dividend. Signed `MIN / -1` returns `MIN`, with remainder zero.
These cases are handled in the saved graph, not by inspecting trace examples.

### Additional operation families

| Family | Implemented operations and restrictions |
| --- | --- |
| Elementwise math | Exponential, logarithmic, trigonometric and hyperbolic functions; sign, powers, polynomial, round, square, clamp, nextafter, finite check. Kernel dtype support varies by function and device. |
| Special and complex | Erf/erfc/erfinv, lgamma, digamma, scaled Bessel I0/I1, igamma/igammac, zeta; real, imaginary, complex construction, conjugation, and element-type conversion. |
| Integer bits | AND/OR/XOR/NOT, shifts, leading zeros, population count, high-half multiply. |
| Construction and indexing | Scalar constant and bounds, eye, full, iota, reverse, concatenate, pad (including interior padding/cropping), slice, dynamic slice/update, select_n, gather and scatter. Dynamic indices remain tensor inputs. |
| Contractions and transforms | General, scaled and ragged dot; convolution with explicit layouts, dilation, padding and groups; FFT/IFFT/RFFT/IRFFT. |
| Ordering | Cumulative sum/product/min/max/logsumexp, lexicographic sort with payloads, top-k. Approximate top-k uses exact ranking. |
| Linear algebra | Cholesky/update, triangular and tridiagonal solves, Householder product, ORMQR, symmetric product, nonsymmetric eig (optional left and right vectors), Eigh, packed Hessenberg and Hermitian tridiagonal reductions, real and complex Schur, LU, pivots to permutation, QR (including pivoting), SVD. |
| Regions | `cond` and `switch` execute only the selected branch; `while` tests its predicate before each iteration. Calls and custom linear solve inline their supplied region. Fixed-length scan unrolls its body; stop-gradient detaches. Trace size grows with scan length and with the update count for scatter-apply. |

`polygamma` takes two matching real floating-point tensors (order and argument).
Orders are evaluated per element on replay; zero selects digamma, and negative,
nonintegral, or nonfinite orders yield NaN.

`igamma_grad_a(a, x)` differentiates the regularized lower incomplete gamma
function with respect to its shape. It accepts matching real floating-point
tensors and returns the input dtype. The trace uses a differentiated power
series for small arguments and 256-point quadrature of the differentiated
gamma integral for larger arguments, computing both in `f64`. For positive
finite shapes it returns zero at `x = 0` and `x = +∞`; negative arguments,
nonpositive shapes, and NaN inputs yield NaN.

`regularized_incomplete_beta(a, b, x)` takes three matching real floating-point
tensors. For positive finite shapes it returns exactly 0 and 1 at `x = 0` and
`x = 1`. Invalid shapes or arguments outside `[0, 1]` yield NaN. The portable
trace uses `f64` internally, chooses the tail per element, and evaluates a
512-point quadrature before casting to the input dtype.

`eig(left, right)` accepts square matrices or batches in `f32`, `f64`, `c64`,
or `c128`. It returns complex eigenvalues followed by the requested matrices
of column eigenvectors (left first). When both are requested, it solves for
`A` and `Aᴴ` separately and matches columns by eigenvalue on replay. Eigenvalue
order and vector phase are unspecified. See [`tests/eig.rs`](tests/eig.rs).

`hessenberg` accepts square `f32`, `f64`, `c64`, and `c128` matrices, including
batches. It returns a packed matrix and `max(n−1, 0)` Householder coefficients
per matrix. The upper triangle and first subdiagonal store `H`; entries below
them store the reflector tails for `Q`, with `Qᴴ A Q = H`. Saved-module
reconstruction tests are in [`tests/hessenberg.rs`](tests/hessenberg.rs).

`tridiagonal(lower)` reads the selected triangle of a real symmetric or complex
Hermitian matrix (or batch), ignoring the imaginary part of a complex diagonal.
It returns the packed reflector matrix, real diagonal, real off-diagonal, and
Householder coefficients. [`tests/tridiagonal.rs`](tests/tridiagonal.rs)
reconstructs `Q` for both storage conventions after TorchScript save/load.

`tridiagonal_solve(perturb_singular)` accepts three diagonals shaped `[..., n]`
(lower, main, upper) and a right-hand side shaped `[..., n, k]`. The first lower
and last upper entries are padding. Adjacent-row pivot choices vary by batch
and input at replay. Singular systems return NaNs unless perturbation is enabled
and a retry with `eps * max(1, max(abs(diag)))` added to the diagonal succeeds.
See [`tests/tridiagonal_solve.rs`](tests/tridiagonal_solve.rs).

`schur(vectors)` returns `T` and optionally `Q`, with `Qᴴ A Q = T`. Complex
`T` is upper triangular; real `T` is upper quasi-triangular with conjugate
pairs in 2×2 blocks. Block choices are tensor operations and can change between
batch members and replays. The portable lowering solves successively smaller
eigenproblems (`O(n⁴)` work per matrix). See [`tests/schur.rs`](tests/schur.rs).

`cond` takes a scalar boolean tensor. A C++17 tracing bridge emits both branch
regions inside a native TorchScript `prim::If`. Only the selected block runs on
replay. `switch` uses nested `prim::If` nodes and clamps its scalar `i32` index
to the first or last branch. `while` uses `prim::Loop`; its iteration count is
determined by live carry values and captures, not by the trace examples. No
GNX runtime operator is required to load or run the exported module. See
[`tests/control_flow.rs`](tests/control_flow.rs) for save/load replay tests.

The backend is single-device: replicated sharding annotations are identities;
named mesh axes require a multi-device runtime.

[`tests/primitives.rs`](tests/primitives.rs) exercises grouped convolution, non-leading
batch axes, scan, dynamic indexing, scatter collisions, special functions, FFT,
64-bit integer edge cases, pivoted QR reconstruction, and lexicographic sorting.
[`tests/elementwise.rs`](tests/elementwise.rs) covers integer edge cases, floating
special values, comparisons and selection, invalid signatures, and standalone
TorchScript save/load replay. [`tests/cpu.rs`](tests/cpu.rs) covers GNX device
execution, shape operations, transfers, and output routing.
