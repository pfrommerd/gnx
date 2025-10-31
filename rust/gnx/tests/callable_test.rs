use gnx::graph::Callable;

fn f(input: u8) -> u8 {
    input
}

fn invoke<F: Callable<(u8,), u8>>(func: F) {}

#[test]
fn test_callable() {
    invoke(f)
}
