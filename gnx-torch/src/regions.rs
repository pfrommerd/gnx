//! Expand static regions into tensor equations before checking Torch support.
use gnx_base::pass::{Pass, PassError, rewrite};
use gnx_expr::{RootExpr, Var};
use gnx_math::ops::{linalg::CustomLinearSolveOperation, shape, zeros_op};
use gnx_transforms::cf::{CallOperation, ScanOperation};
use gnx_transforms::util::region_body;
use std::{io, sync::Arc};

pub(super) struct StaticRegions;
impl Pass for StaticRegions {
    fn name(&self) -> &'static str {
        "torch.static_regions"
    }
    fn run(&self, expr: Arc<RootExpr>) -> Result<Arc<RootExpr>, PassError> {
        rewrite(expr.view(), &mut |b, e| -> io::Result<Option<Vec<Var>>> {
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
                let body = region_body(r).map_err(io::Error::other)?;
                return Ok(Some(
                    b.inline(
                        body.view(),
                        r.provided_symbols(),
                        r.provided_vars(),
                        e.inputs(),
                    )
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
            let body = region_body(r).map_err(io::Error::other)?;
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
                    .inline(body.view(), r.provided_symbols(), r.provided_vars(), &carry)
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
        .map_err(|e| PassError::new(self.name(), e))?
        .build()
        .map_err(|e| PassError::new(self.name(), e))
    }
}

impl gnx_base::anyhash::Hash for StaticRegions {
    fn hash<H: gnx_base::anyhash::HasherWrite>(&self, _: &mut H) {}
}
