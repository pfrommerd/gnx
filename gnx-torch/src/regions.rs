//! Expand static regions into tensor equations before checking Torch support.
use gnx_expr::{
    Expr, Var,
    pass::{Pass, PassError, rewrite},
};
use gnx_math::ops::{linalg::CustomLinearSolveOperation, shape, zeros_op};
use gnx_transforms::cf::{CallOperation, ScanOperation};
use std::io;

pub(super) struct StaticRegions;
impl Pass for StaticRegions {
    fn name(&self) -> &'static str {
        "torch.static_regions"
    }
    fn run(&self, expr: Expr) -> Result<Expr, PassError> {
        rewrite(&expr, &mut |b, e| -> io::Result<Option<Vec<Var>>> {
            let region = if e.op().downcast_ref::<CallOperation>().is_some() {
                Some(0)
            } else if e
                .op()
                .downcast_ref::<CustomLinearSolveOperation>()
                .is_some()
            {
                Some(1)
            } else {
                None
            };
            if let Some(i) = region {
                let r = &e.subexprs()[i];
                return Ok(Some(
                    b.inline(r.expr(), &r.symbol_captures, r.closure(), e.inputs())
                        .map_err(io::Error::other)?,
                ));
            }
            let Some(scan) = e.op().downcast_ref::<ScanOperation>() else {
                return Ok(None);
            };
            let p = scan.parameters();
            let n = p.physical_carry_len().map_err(io::Error::other)?;
            let (carry, xs) = e.inputs().split_at(n);
            let mut carry = carry.to_vec();
            let r = &e.subexprs()[0];
            let ys = &e.outputs()[n..];
            if p.length == 0 {
                carry.extend(
                    ys.iter()
                        .map(|(_, info)| b.emit1(zeros_op(), &[], info.clone())),
                );
                return Ok(Some(carry));
            }
            let xs = xs
                .iter()
                .map(|&x| shape::unstack(b, x, 0))
                .collect::<Result<Vec<_>, _>>()
                .map_err(io::Error::other)?;
            let mut stacked = vec![Vec::new(); ys.len()];
            for i in 0..p.length {
                carry.extend(xs.iter().map(|v| v[i]));
                let mut next = b
                    .inline(r.expr(), &r.symbol_captures, r.closure(), &carry)
                    .map_err(io::Error::other)?;
                for (list, y) in stacked.iter_mut().zip(next.drain(n..)) {
                    list.push(y)
                }
                carry = next;
            }
            for parts in stacked {
                carry.push(shape::stack(b, &parts, 0).map_err(io::Error::other)?)
            }
            Ok(Some(carry))
        })
        .map_err(|e| PassError::new(self.name(), e))
    }
}
