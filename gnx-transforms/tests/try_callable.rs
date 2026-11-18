use gnx_transforms::{Callable, Jitable, TryCallable, TryJitable, jit};
use std::cell::Cell;
use std::sync::Arc;

#[derive(Debug, PartialEq, Eq)]
struct Zero;

fn divide(dividend: u32, divisor: u32) -> Result<u32, Zero> {
    dividend.checked_div(divisor).ok_or(Zero)
}

#[test]
fn result_returning_functions_support_both_entry_points() {
    assert_eq!(divide.try_invoke((12, 3)), Ok(4));
    assert_eq!(divide.try_invoke((12, 0)), Err(Zero));
    assert_eq!(divide.try_invoke_owned((12, 3)), Ok(4));
    assert_eq!(divide.try_invoke_owned((12, 0)), Err(Zero));
}

struct BorrowedDivisor;

impl TryCallable<(u32, &u32)> for BorrowedDivisor {
    type Output = u32;
    type Error = Zero;

    fn try_invoke(&self, (dividend, divisor): (u32, &u32)) -> Result<u32, Zero> {
        divide(dividend, *divisor)
    }
}

impl TryJitable<(u32, &u32)> for BorrowedDivisor {
    type OwnedOutput = u32;

    fn try_invoke_owned(&self, (dividend, divisor): (u32, Arc<u32>)) -> Result<u32, Zero> {
        self.try_invoke((dividend, &divisor))
    }
}

#[test]
fn owned_input_can_be_adapted_to_borrowed_arguments() {
    assert_eq!(BorrowedDivisor.try_invoke((12, &3)), Ok(4));
    assert_eq!(BorrowedDivisor.try_invoke_owned((12, Arc::new(3))), Ok(4));
    assert_eq!(
        BorrowedDivisor.try_invoke_owned((12, Arc::new(0))),
        Err(Zero)
    );
}

#[test]
fn fallible_jit_retries_failed_traces_and_caches_successes() {
    let attempts = Cell::new(0);
    let function = jit(|value: u32| {
        attempts.set(attempts.get() + 1);
        if attempts.get() == 1 {
            Err(Zero)
        } else {
            Ok(value + 1)
        }
    });

    let caller = gnx_expr::TraceContext::enter();
    assert_eq!(function.try_invoke((5,)), Err(Zero));
    assert_eq!(gnx_expr::TraceContext::current_id(), caller.context().id());
    assert_eq!(function.try_invoke((5,)), Ok(6));
    assert_eq!(function.try_invoke((5,)), Ok(6));
    assert_eq!(attempts.get(), 2);
}

#[test]
fn fallible_jit_uses_the_owned_input_adapter() {
    let divide = jit(BorrowedDivisor);
    assert_eq!(divide.try_invoke((12, &3)), Ok(4));
    assert_eq!(divide.try_invoke((12, &0)), Err(Zero));
}

impl Callable<(u32, &u32)> for BorrowedDivisor {
    type Output = u32;

    fn invoke(&self, (dividend, divisor): (u32, &u32)) -> u32 {
        divide(dividend, *divisor).unwrap_or(0)
    }
}

impl Jitable<(u32, &u32)> for BorrowedDivisor {
    type OwnedOutput = u32;

    fn invoke_owned(&self, (dividend, divisor): (u32, Arc<u32>)) -> u32 {
        self.invoke((dividend, &divisor))
    }
}

#[test]
fn fallible_and_infallible_calls_do_not_share_a_trace() {
    let divide = jit(BorrowedDivisor);
    assert_eq!(divide.invoke((12, &0)), 0);
    assert_eq!(divide.try_invoke((12, &0)), Err(Zero));
}
