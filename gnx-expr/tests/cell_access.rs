use gnx_expr::{
    Capture, CellAccess, CellTrace, Dialect, Generic, HEqn, Op, Operation, TraceContext, Tracer,
    ValueInfo, Var, hexpr,
};

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
fn structural_cell_instructions_report_reads_and_writes() {
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

struct TestDialect;

impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "test"
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DefaultRegions;

impl std::fmt::Display for DefaultRegions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
    }
}

impl Operation for DefaultRegions {
    fn name(&self) -> &'static str {
        "default_regions"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &TestDialect
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct SelectedRegion(usize);

impl std::fmt::Display for SelectedRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
    }
}

impl Operation for SelectedRegion {
    fn name(&self) -> &'static str {
        "selected_region"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &TestDialect
    }

    fn cell_access(&self, eqn: &HEqn, cell: Var) -> CellAccess {
        eqn.subexprs()[self.0].captured_cell_access(cell)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DirectRead;

impl std::fmt::Display for DirectRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
    }
}

impl Operation for DirectRead {
    fn name(&self) -> &'static str {
        "direct_read"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &TestDialect
    }

    fn cell_access(&self, eqn: &HEqn, cell: Var) -> CellAccess {
        if eqn.inputs().contains(&cell) {
            CellAccess::Read
        } else {
            CellAccess::Unused
        }
    }
}

mod ops {
    pub mod test {
        use super::super::{DefaultRegions, DirectRead, Op, SelectedRegion};

        pub fn default_regions() -> Op {
            Op::new(DefaultRegions)
        }

        pub fn selected_region(index: usize) -> Op {
            Op::new(SelectedRegion(index))
        }

        pub fn direct_read() -> Op {
            Op::new(DirectRead)
        }
    }

    pub use gnx_transforms::ops::cf;
}

#[test]
fn operation_controls_cell_effects_of_its_regions_and_operands() {
    let info = ValueInfo::new(0usize);
    let body = hexpr!(&info; |all[:], selected[:], direct[:], aliased[:], x| => {
        test::default_regions [|| => {
            read <- all[:];
            in ()
        }] [|| => {
            all[:] <- x;
            in ()
        }];
        test::selected_region<1> [|| => {
            selected[:] <- x;
            in ()
        }] [|| => {
            read <- selected[:];
            in ()
        }];
        test::direct_read direct[:];
        cf::call [|formal[:]: &info| => {
            read <- aliased[:];
            formal[:] <- x;
            in ()
        }] aliased[:];
        in ()
    });
    let accesses: Vec<_> = body
        .inputs()
        .iter()
        .map(|(var, _)| body.cell_access(var))
        .collect();
    assert_eq!(
        accesses,
        [
            Some(CellAccess::ReadWrite),
            Some(CellAccess::Read),
            Some(CellAccess::Read),
            Some(CellAccess::ReadWrite),
            None,
        ],
    );
}
