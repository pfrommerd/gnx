use gnx::Callable;

#[gnx::jit]
fn increment(value: u32) -> u32 {
    value + 1
}

#[gnx::jit]
fn increment_ref(value: &u32) -> u32 {
    *value + 1
}

#[test]
fn attribute_and_closure_macros_are_reexported() {
    let _: fn(u32) -> u32 = increment;
    let _: fn(&u32) -> u32 = increment_ref;
    let offset = 2;
    let callable = gnx::jit_fn!(|value: u32| increment(value) + offset);
    assert_eq!(callable(3), 6);
    assert_eq!(increment_ref(&3), 4);

    let fallible = gnx::try_jit_fn!(|value: u32| value.checked_sub(1).ok_or("zero"));
    assert_eq!(fallible(3), Ok(2));
    assert_eq!(fallible(0), Err("zero"));
}

#[test]
fn runtime_jit_function_keeps_its_name() {
    let callable = gnx::jit(|value: u32| value + 1);
    assert_eq!(Callable::invoke(&callable, (3,)), 4);
}
