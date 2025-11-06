use gnx::util::Callable;

fn f(input: u8) -> u8 {
    input + 1
}

fn invoke<F: Callable<(u8,), u8>>(func: F) -> u8 {
    func.call((8,))
}

#[test]
fn test_callable() {
    let res = invoke(f);
    assert_eq!(res, 9);
}
