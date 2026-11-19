use gnx_expr::{CellAccess, HInstr, InputId, InputSpec, Kind, ValueInfo, expr, hexpr};
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
            closure: vec![(InputId::Closure(0), Kind::Cell(cell_info))],
            inputs: vec![(InputId::Input(0), Kind::Value(default))],
        })
        .unwrap();
    assert!(lowered.expr.eqns().is_empty());
    assert_eq!(lowered.results().len(), 2);
    assert_eq!(lowered.updates.len(), 1);
}

#[test]
fn hexpr_prints_nested_bodies_inline() {
    let info = ValueInfo::new(0usize);
    let inner = Arc::new(hexpr! {
        passthrough = |value: &info| => (value)
        in |x: &info| => {
            y: &info = cf::call [@passthrough] x;
            in (y)
        }
    });
    let body = hexpr!({
        lambda input: &info =>
        result: &info = cf::call[inner.clone() => ()] input;
        in (result)
    });
    let eqn = body.instructions()[0].as_apply().unwrap();
    assert_eq!(
        body.to_string(),
        concat!(
            "|input: 0| => {\n",
            "  result: 0 = cf::call [|x: 0| => {\n",
            "    y: 0 = cf::call [|value: 0| => (value)] x;\n",
            "    in (y)\n  }] input;\n",
            "  in (result)\n}",
        ),
    );
    assert_eq!(eqn.op(), &gnx_transforms::cf::call_op());
    assert!(Arc::ptr_eq(eqn.subexprs()[0].expr(), &inner));
    assert_eq!(
        body.kind(&eqn.outputs()[0].0),
        Some(&Kind::Value(info.clone()))
    );
    let lowered = body
        .lower(&InputSpec {
            closure: vec![],
            inputs: vec![(InputId::Input(0), Kind::Value(info))],
        })
        .unwrap();
    lowered.expr.check().unwrap();
    assert!(lowered.expr.eqns().is_empty());
}

#[test]
fn inline_hexpr_captures_cells_and_values_across_regions() {
    let info = ValueInfo::new(0usize);
    let _ = gnx_transforms::cf::call_op();
    let body = hexpr!(&info; |state[:], replacement| => {
        result = cf::call [|| => {
            old = cf::call [|| => {
                read <- state[:];
                state[:] <- replacement;
                in (read)
            }];
            in (old)
        }];
        in (result)
    });
    let middle = &body.instructions()[0].as_apply().unwrap().subexprs()[0];
    assert_eq!(middle.closure(), &[body.inputs()[0].0, body.inputs()[1].0]);
    let inner = &middle.expr().instructions()[0]
        .as_apply()
        .unwrap()
        .subexprs()[0];
    assert_eq!(
        inner.closure(),
        &[middle.expr().closure()[0].0, middle.expr().closure()[1].0]
    );
    assert_eq!(inner.expr().closure()[0].1, Kind::Cell(info.clone()));
    assert_eq!(inner.expr().closure()[1].1, Kind::Value(info.clone()));
    assert_eq!(
        body.to_string(),
        concat!(
            "|state[:]: 0, replacement: 0| => {\n",
            "  result: 0 = cf::call [|| => {\n",
            "    old: 0 = cf::call [|| => {\n",
            "      read <- state[:];\n",
            "      state[:] <- replacement;\n",
            "      in (read)\n    }];\n",
            "    in (old)\n  }];\n",
            "  in (result)\n}",
        ),
    );
    let lowered = body
        .lower(&InputSpec {
            closure: vec![],
            inputs: vec![
                (InputId::Input(0), Kind::Cell(info.clone())),
                (InputId::Input(1), Kind::Value(info)),
            ],
        })
        .unwrap();
    assert!(lowered.expr.eqns().is_empty());
    assert_eq!(
        lowered.expr.outputs(),
        &[lowered.expr.inputs()[0].0, lowered.expr.inputs()[1].0]
    );
    assert_eq!(lowered.updates.len(), 1);
}

#[test]
fn hexpr_hoists_shared_bodies_with_positional_cell_captures() {
    let info = ValueInfo::new(0usize);
    let body = hexpr!(&info;
        step = |state[:]; replacement| => {
            old <- state[:];
            state[:] <- replacement;
            in (old)
        }
        in |a[:], b[:], x| => {
            first = cf::call [@step{a[:]}] x;
            second = cf::call [@step{b[:]}] x;
            in (first, second)
        }
    );
    assert_eq!(
        body.pretty().max_inline_eqns(1).to_string(),
        concat!(
            "step = |state[:]: 0; replacement: 0| => {\n",
            "  old <- state[:];\n  state[:] <- replacement;\n  in (old)\n}\n",
            "in\n|a[:]: 0, b[:]: 0, x: 0| => {\n",
            "  first: 0 = cf::call [@step{a[:]}] x;\n",
            "  second: 0 = cf::call [@step{b[:]}] x;\n",
            "  in (first, second)\n}",
        )
    );
}

#[test]
fn closureless_headers_accept_cell_inputs_and_zero_parameters() {
    let info = ValueInfo::new(0usize);
    let body = hexpr!(|cell[:]: &info, x: info| => (cell[:], x));
    assert!(body.closure().is_empty());
    assert_eq!(body.inputs().len(), 2);
    assert_eq!(body.to_string(), "|cell[:]: 0, x: 0| => (cell[:], x)");

    let high = hexpr!(|| => ());
    let low = expr!(|| => ());
    assert_eq!(high.to_string(), "|| => ()");
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
    assert_eq!(low.to_string(), "|x: 7| => (x)");
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
