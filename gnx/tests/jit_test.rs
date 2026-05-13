use gnx::transforms::*;

#[transform(jit)]
fn f(x: u32) -> u32 {
    x + 1
}

fn g(x: u32) -> u32 {
    let jitted = jit!(|y| f(y + x));
    let z = 2*x;
    jitted(z)
}

#[test]
fn test_jit() {
    // Check that the type of f is unaltered
    // by the jit transform.
    let _: fn(u32) -> u32 = f;
    assert_eq!(f(1), 2);
    assert_eq!(g(1), 4);
}