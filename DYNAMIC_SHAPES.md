# Dynamic and jagged shapes in GNX

**Status: design for a follow-up.** The current change supplies generic symbol binding in `gnx-expr`. The array types, JIT annotations, scan rules, and backend representation below are proposals, not supported APIs.

## Logical axes and storage

A fixed axis has one known length. A dynamic axis has one length per array, known at runtime. A jagged axis has a length **per coordinate of its dependency axes**. A bound is an inclusive upper limit, not a length: `0 ≤ T ≤ 128` allows an empty array and never authorizes truncation. A jagged bound applies independently to every row.

Keep the element dtype and logical shape in `ArrayInfo`. A shape has axis definitions and a public order: `axes[order[p]]` describes public position `p`. An `AxisId` indexes a definition within that shape; a name is an optional unique label, not an instruction to align operands by name. Dependencies refer to axis IDs and must be acyclic. A transpose permutes `order`; it does not renumber dependencies or repack data. Compare shapes by public structure, not by incidental allocation of axis IDs.

For example, in `[a=2, token=*[example], example=3]`, `token` may have lengths `[2, 0, 1]`. Store data in `example, token, a` order using offsets `[0, 2, 2, 3]` and dense values of shape `[3, 2]`. There is no `token=0` element for `example=1`. A view in `example, a, token` order uses the same buffers. Nesting uses another offsets table indexing the next level, with repeated offsets for empty rows. Offsets must be monotone, in bounds, and consistent with the child length. An omitted sparse entry, in contrast, is a value at an index that *does* exist; sparsity is not an extent kind.

## How shapes use expression symbols

Keep three identities separate: `AxisId` names an axis in one shape; `Symbol` identifies a declaration in one expression; a runtime **witness** supplies the actual integer length or partition. A concrete array need not remember the name of a JIT size parameter.

- The array dialect defines closed `SymbolInfo` sorts for uniform lengths and jagged partitions and interprets their bounds. `gnx-expr` has no `AxisId`, padding rule, or notion of an integer extent.
- An `ArrayInfo` payload refers to ordered metadata slots. `ValueInfo::with_symbols` binds those slots to the current region's symbols. A jagged axis records both its dependency axes and the slot identifying its partition: equal dependency lists and bounds alone do **not** imply equal row lengths.
- Both `HExpr` and `Expr` declare `symbol_closure` and `symbol_inputs`, alongside their value closure and inputs. Input symbols bind through input metadata. A child declares captured symbols with local IDs; its `HSubExpr` or `SubExpr` attachment maps those declarations to parent symbols positionally. Capture is possible even when no array value is captured. Lowering preserves symbol relationships while functionalizing cells.

For example, `[T: Size(max=128)] (x: f32[T], mask: bool[T])` has **one** binder referenced by two inputs. At a concrete call, the array boundary binds `T` from `x`, checks the bound, then checks `mask` against the same binding. With independent `T` and `U`, `(17, 23)` is valid. This is a signature-wide check; checking each `ArrayInfo` independently is insufficient. Names such as `T` are diagnostic; substitution and inlining use scoped IDs.

## JIT promotion

Allow the caller to promote selected input axes before tracing, for example `dynamic_axes={"x": {"time": Dynamic("T", max=128)}}` (illustrative syntax). Resolve the selection against public axes, replace the selected fixed extents in placeholder metadata, and build the trace key from that abstract signature. Preserve unselected extents, bounds, and which slots share a binder. Inputs with actual `T=17` and `T=93` should reuse a trace; `T=129` must fail. Executable reuse depends on backend lowering. Querying a bound is distinct from querying the actual extent: an unresolved length is not zero, and a valid empty row has length zero.

## Scan: iteration axes, lengths, and output inference

`cf.scan` should retain the resolved `in_axes` and `out_axes` **on the operation** so verification, batching, differentiation, and lowering agree about which axes are sliced and stacked. Keep the trip-count mode and bounds in metadata; runtime lengths are operands or projections from values, not op attributes. Scanned inputs must agree on actual iteration lengths, per row for a jagged scan.

Trace the body once. For each output leaf and each of its axes, ask where the extent is bound:

| Step result extent | Stacked result over `S` steps |
| --- | --- |
| Fixed `64` | `[S, 64]` |
| `T`, captured from outside the scan or otherwise proven invariant | `[S, T]`: rectangular, possibly dynamic on both axes |
| `K`, produced by filtering each step's slice | `[S, *[step]]`: jagged, with lengths `K[step]` |

If a step returns `(f32[K], i32[K])` using one produced witness, both stacked outputs share **one partition**, even when the lengths differ between steps. Distinct witnesses must not be unified just because sample lengths or bounds coincide. Infer uniformity from the extent, not the data: a result may depend on a slice while its length comes from a captured template. Never infer a shape by inspecting the first iteration. With zero iterations, captured `T` is still known and a step-varying result has an empty partition.

Input and closure symbols cover invariant sizes. A fresh, data-dependent output size is **not** another `symbol_input`. A follow-up needs a producer/result binder or an explicit length witness plus a result-slot association. The same applies when lifting a sequence of per-step witnesses into a partition. Keep actual witness computations in SSA and avoid hiding raw SSA variable references inside `ArrayInfo`.

## Bounded lowering and validation

For XLA, a bounded uniform scan can allocate at `max` and loop to the actual `T`. A bounded jagged scan can take a length vector, iterate to its maximum (zero for an empty batch), and update each row only while active. With lengths `[5, 0, 2]`, the middle row returns its initial carry and emits no elements. Preserve logical lengths on results; padding is backend storage, not user-visible data. Start with pure bodies and independent per-row carries: masking a value does not mask side effects. Reverse-mode differentiation must reverse each valid prefix, not the padded buffer.

Acceptance cases: bounds of zero and overflow; inputs sharing a size versus independent sizes; equal bounds with unequal actual lengths; a later-position jagged dependency surviving transpose; nested and empty rows; nonleading scan axes; uniform versus per-step witnesses, including zero steps; and per-row results and gradients compared with independent unpadded scans. Defer scoped `map` axes, data-dependent carry shapes, and sparse kernels to separate changes.

## References

- [Henriksen & Elsman, *Towards Size-Dependent Types for Array Programming*](https://doi.org/10.1145/3460944.3464310): shared size variables and existential sizes for results.
- [StableHLO dynamism](https://openxla.org/stablehlo/dynamism): bounded dimensions, input shape polymorphism, and data-dependent dynamism.
- [Arrow variable-size list layout](https://arrow.apache.org/docs/format/Columnar.html#variable-size-list-layout): offsets and nested child arrays.
