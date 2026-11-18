use gnx_expr::{CellAccess, HInstr, InputId, InputSpec, Kind, Ownership, ValueInfo, expr, hexpr};
use std::sync::Arc;

#[test]
fn cell_syntax_preserves_metadata_and_lowers_without_dialect_cell_ops() {
    let default = ValueInfo::new(0usize);
    let cell_info = ValueInfo::new(42usize);
    let body = hexpr!({
        lambda captured[:]: &cell_info; input: &default =>
        old <- captured[:];
        fresh[:] = old;
        captured[:] <- old;
        in (fresh[:], input)
    });
    let captured = body.closure()[0].0;
    assert_eq!(body.cell_access(&captured), Some(CellAccess::ReadWrite));
    let [
        HInstr::ReadCell {
            cell,
            output,
            kind: read_kind,
        },
        HInstr::CreateCell {
            cell: fresh,
            initial,
            kind: fresh_kind,
        },
        HInstr::WriteCell {
            cell: written,
            value,
        },
    ] = body.instructions()
    else {
        panic!("expected structural cell instructions")
    };
    assert_eq!(*cell, captured);
    assert_eq!(*written, captured);
    assert_eq!(output, initial);
    assert_eq!(output, value);
    assert_eq!(read_kind, &Kind::Value(cell_info.clone()));
    assert_eq!(fresh_kind, &Kind::Cell(cell_info.clone()));
    assert_eq!(body.kind(output), Some(&Kind::Value(cell_info.clone())));
    assert_eq!(body.kind(fresh), Some(&Kind::Cell(cell_info.clone())));
    let lowered = body
        .lower(&InputSpec {
            closure: vec![(
                InputId::Closure(0),
                Kind::Cell(cell_info),
                Ownership::Borrow,
            )],
            inputs: vec![(InputId::Input(0), Kind::Value(default), Ownership::Borrow)],
        })
        .unwrap();
    assert!(lowered.expr.eqns().is_empty());
    assert_eq!(lowered.results().len(), 2);
    assert_eq!(lowered.updates.len(), 1);
}

#[test]
fn hexpr_subexpressions_have_no_ownership_annotations() {
    let info = ValueInfo::new(0usize);
    let passthrough = hexpr!({ lambda value: &info => in (value) });
    let inner = Arc::new(hexpr!({ lambda x: &info =>
        y: &info = builtins::call[passthrough => ()] x;
        in (y)
    }));
    let body = hexpr!({
        lambda input: &info =>
        result: &info = builtins::call[inner.clone() => ()] input;
        in (result)
    });
    let eqn = body.instructions()[0].as_apply().unwrap();
    assert_eq!(
        body.to_string(),
        concat!(
            "{ lambda input: 0 =>\n",
            "  result: 0 = builtins::call[hexpr!({ lambda x: 0 =>\n",
            "    y: 0 = builtins::call[hexpr!({ lambda value: 0 =>\n",
            "      in (value) }) => ()] x;\n",
            "    in (y) }) => ()] input;\n",
            "  in (result) }",
        ),
    );
    assert_eq!(eqn.op(), &gnx_expr::builtins::call_op());
    assert!(Arc::ptr_eq(eqn.subexprs()[0].expr(), &inner));
    assert_eq!(
        body.kind(&eqn.outputs()[0].0),
        Some(&Kind::Value(info.clone()))
    );
    // Only the outer boundary chooses ownership.
    let lowered = body
        .lower(&InputSpec {
            closure: vec![],
            inputs: vec![(InputId::Input(0), Kind::Value(info), Ownership::Move)],
        })
        .unwrap();
    lowered.expr.check().unwrap();
    assert!(lowered.expr.eqns().is_empty());
}

#[test]
fn closureless_headers_accept_cell_inputs_and_zero_parameters() {
    let info = ValueInfo::new(0usize);
    let body = hexpr!({ lambda cell[:]: &info, x: info => in (cell[:], x) });
    assert!(body.closure().is_empty());
    assert_eq!(body.inputs().len(), 2);
    assert_eq!(
        body.to_string(),
        "{ lambda cell[:]: 0, x: 0 =>\n  in (cell[:], x) }"
    );

    let high = hexpr!({ lambda => in () });
    let low = expr!({ lambda => in () });
    assert_eq!(high.to_string(), "{ lambda =>\n  in () }");
    assert_eq!(low.to_string(), high.to_string());
}

#[test]
fn arrow_headers_accept_field_access_and_default_metadata_forms() {
    struct Config {
        info: ValueInfo,
    }
    let config = Config {
        info: ValueInfo::new(7usize),
    };
    let low = expr!({ lambda x: &config.info => in (x) });
    let high = hexpr!({ lambda x: &config.info => in (x) });
    assert_eq!(low.to_string(), "{ lambda x: 7 =>\n  in (x) }");
    assert_eq!(high.to_string(), low.to_string());
    assert_eq!(
        expr!(&config.info; { lambda x => in (x) }).to_string(),
        low.to_string()
    );
    assert_eq!(
        hexpr!(&config.info; { lambda x => in (x) }).to_string(),
        high.to_string()
    );
}
