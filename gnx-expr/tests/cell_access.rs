use gnx_expr::{CellAccess, CellTrace, Generic, TraceContext, Capture, Tracer, ValueInfo};

#[test]
fn cell_access_union_covers_all_access_pairs() {
    use CellAccess::{Read, ReadWrite, Unused, Write};
    let accesses = [Unused, Read, Write, ReadWrite];
    let expected = [
        [Unused, Read, Write, ReadWrite],
        [Read, Read, ReadWrite, ReadWrite],
        [Write, ReadWrite, Write, ReadWrite],
        [ReadWrite, ReadWrite, ReadWrite, ReadWrite],
    ];
    for (i, first) in accesses.into_iter().enumerate() {
        for (j, second) in accesses.into_iter().enumerate() {
            assert_eq!(first.combine(second), expected[i][j]);
        }
    }
}

#[test]
fn summaries_are_derived_only_from_structural_cell_accesses() {
    let guard = TraceContext::enter();
    let info = ValueInfo::new(0usize);
    let unused = CellTrace::placeholder(info.clone());
    let read_only = CellTrace::placeholder(info.clone());
    let write_only = CellTrace::placeholder(info.clone());
    let read_write = CellTrace::placeholder(info.clone());
    let value = Tracer::<Generic>::placeholder(info);
    let read = read_only.get();
    read_write.set(value.trace_ref().clone());
    let rw_read = read_write.get();
    write_only.set(value.trace_ref().clone());
    let captured = Capture::from_context(
        guard.context(),
        &[
            unused.into(),
            read_only.into(),
            write_only.into(),
            read_write.into(),
            value.clone().into(),
        ],
        &[read.into(), rw_read.into(), value.into()],
    )
    .unwrap();
    let expr = captured.expr();
    for (index, access) in [
        CellAccess::Unused,
        CellAccess::Read,
        CellAccess::Write,
        CellAccess::ReadWrite,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(expr.cell_access(&expr.inputs()[index].0), Some(access));
    }
    // Immutable values have ownership in Expr, not resource-access effects.
    assert_eq!(expr.cell_access(&expr.inputs()[4].0), None);
}
