use gnx_graph::OwnedGraph;
use gnx_transforms::Jitable;
use gnx_transforms::jit;
use gnx_transforms::jit_fn;
use gnx_transforms::jittable;
use std::sync::Arc;

#[jit]
fn f(x: u32) -> u32 {
    x + 1
}

fn g(x: u32) -> u32 {
    let jitted = jit_fn!(|y| f(y + x));
    let z = 2 * x;
    jitted(z)
}

#[jit]
fn f_ref(x: &u32) -> u32 {
    *x + 1
}

fn g_ref<T: OwnedGraph>(x: T) -> T {
    let jitted = jit_fn!(|_: &_| x.duplicate());
    jitted(&x)
}

#[test]
fn test_jit() {
    // Check that the type of f is unaltered
    // by the jit transform.
    let _: fn(u32) -> u32 = f;
    assert_eq!(f(1), 2);
    assert_eq!(g(1), 4);
}

#[test]
fn test_jit_refs() {
    let _: fn(&u32) -> u32 = f_ref;
    assert_eq!(f_ref(&1), 2);
    assert_eq!(g_ref(2), 2);
}

#[test]
fn test_jit_fn_wraps_function_items() {
    let jitted = jit_fn!(f);
    assert_eq!(jitted(3), 4);
}

#[test]
fn test_jittable_invoke_owned() {
    let callable = jittable!(fn(&u32) -> u32, |x: &u32| *x + 1);
    assert_eq!(Jitable::invoke_owned(&callable, (Arc::new(1),)), 2);
}
