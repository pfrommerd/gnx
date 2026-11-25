mod ad;
mod map;

use gnx_expr::{Eqn, SubExprBuilder};

/// The regions of an equation, to emit with another equation.
fn copy_regions(eqn: Eqn<'_>) -> Vec<SubExprBuilder> {
    eqn.subexprs()
        .map(|sub| {
            SubExprBuilder::new(sub.to_ref())
                .with_vars(sub.provided_vars().iter().copied())
                .with_symbols(sub.provided_symbols().iter().copied())
        })
        .collect()
}
