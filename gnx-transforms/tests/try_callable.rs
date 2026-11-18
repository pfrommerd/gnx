use gnx_transforms::{Callable, Function, Jitable, TryCallable, TryJitable, jit};
use std::cell::Cell;
use std::convert::Infallible;
use std::sync::Arc;

#[derive(Debug, PartialEq, Eq)]
struct Zero;

fn divide(dividend: u32, divisor: u32) -> Result<u32, Zero> {
    dividend.checked_div(divisor).ok_or(Zero)
}

#[test]
fn result_returning_functions_support_both_try_entry_points() {
    assert_eq!(divide.try_invoke((12, 3)), Ok(4));
    assert_eq!(divide.try_invoke((12, 0)), Err(Zero));
    assert_eq!(divide.try_invoke_owned((12, 3)), Ok(4));
    assert_eq!(divide.try_invoke_owned((12, 0)), Err(Zero));
}

struct BorrowedDivisor;

impl Function<(u32, &u32)> for BorrowedDivisor {
    type Output = u32;
}

impl TryCallable<(u32, &u32)> for BorrowedDivisor {
    type Error = Zero;

    fn try_invoke(&self, (dividend, divisor): (u32, &u32)) -> Result<u32, Zero> {
        divide(dividend, *divisor)
    }
}

impl TryJitable<(u32, &u32)> for BorrowedDivisor {
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
    assert_eq!(TryCallable::try_invoke(&divide, (12, &3)), Ok(4));
    assert_eq!(divide.try_invoke((12, &0)), Err(Zero));
}

#[test]
fn both_interfaces_cache_their_own_traces() {
    let traces = Cell::new(0);
    let square = |value: u32| {
        traces.set(traces.get() + 1);
        value * value
    };
    assert_eq!(square.invoke_owned((1,)), 1);
    assert_eq!(square.try_invoke_owned((2,)), Ok(4));
    let square = jit(square);
    assert_eq!(square.invoke((3,)), 9);
    assert_eq!(square.invoke((3,)), 9);
    assert_eq!(square.try_invoke((3,)), Ok(9));
    assert_eq!(TryCallable::try_invoke(&square, (3,)), Ok(9));
    assert_eq!(traces.get(), 4);
}

#[test]
fn jittable_macro_adapts_fallible_borrowed_functions() {
    let function = gnx_transforms::jittable!(fn(&u32) -> Result<u32, Zero>, |value: &u32| divide(
        12, *value
    ));
    let function = jit(function);
    assert_eq!(function.try_invoke((&3,)), Ok(4));
    assert_eq!(function.try_invoke((&0,)), Err(Zero));
}

#[test]
fn jit_can_trace_a_hand_written_jitable_without_a_try_jitable_impl() {
    struct Double;
    impl Function<(u32,)> for Double {
        type Output = u32;
    }

    impl Callable<(u32,)> for Double {
        fn invoke(&self, (value,): (u32,)) -> u32 {
            value * 2
        }
    }

    impl Jitable<(u32,)> for Double {
        fn invoke_owned(&self, input: (u32,)) -> u32 {
            self.invoke(input)
        }
    }

    assert_eq!(jit(Double).invoke((5,)), 10);
}

#[test]
fn closure_macros_choose_the_return_shape_explicitly() {
    let divide_by = gnx_transforms::try_jit_fn!(|divisor: u32| divide(12, divisor));
    assert_eq!(divide_by(3), Ok(4));
    assert_eq!(divide_by(0), Err(Zero));

    let wrapped = gnx_transforms::jit_fn!(|value: u32| Ok::<_, Infallible>(value));
    assert_eq!(wrapped(3), 3);

    let fallible = gnx_transforms::try_jit_fn!(|value: u32| value + 1);
    let result: Result<u32, Infallible> = fallible(3);
    assert_eq!(result, Ok(4));
}

#[test]
fn jit_attribute_preserves_a_fallible_borrowed_signature() {
    #[gnx_transforms::jit]
    fn divide_by(divisor: &u32) -> Result<u32, Zero> {
        divide(12, *divisor)
    }

    let mut divisor = 3;
    assert_eq!(divide_by(&divisor), Ok(4));
    divisor = 0;
    assert_eq!(divide_by(&divisor), Err(Zero));
}
