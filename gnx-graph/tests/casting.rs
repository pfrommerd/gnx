use gnx_graph::util::{try_specialize, LifetimeFree};

#[test]
fn try_specialize_primitives() {
    assert_eq!(try_specialize!(0u8, u16), Err(0u8));
    assert_eq!(try_specialize!(1u8, u8), Ok(1u8));
    assert_eq!(try_specialize!(2u8, &'static u8), Err(2u8));
    assert_eq!(try_specialize!(2u8, &u8), Err(2u8));

    static VALUE: u8 = 2u8;
    assert_eq!(try_specialize!(&VALUE, &u8), Ok(&2u8));
    assert_eq!(try_specialize!(&VALUE, &'static u8), Ok(&2u8));
    assert_eq!(try_specialize!(&VALUE, &u16), Err(&2u8));
    assert_eq!(try_specialize!(&VALUE, &i8), Err(&2u8));

    let value = 2u8;

    fn inner<'a>(value: &'a u8) {
        assert_eq!(try_specialize!(value, &u8), Ok(&2u8));
        assert_eq!(try_specialize!(value, &'a u8), Ok(&2u8));
        assert_eq!(try_specialize!(value, &u16), Err(&2u8));
        assert_eq!(try_specialize!(value, &i8), Err(&2u8));
    }

    inner(&value);

    let mut slice = [1u8; 2];

    fn inner2<'a>(value: &'a [u8]) {
        assert_eq!(try_specialize!(value, &[u8]), Ok(&[1, 1][..]));
        assert_eq!(try_specialize!(value, &'a [u8]), Ok(&[1, 1][..]));
        assert_eq!(try_specialize!(value, &'a [u16]), Err(&[1, 1][..]));
        assert_eq!(try_specialize!(value, &'a [i8]), Err(&[1, 1][..]));
    }

    inner2(&slice);

    fn inner3<'a>(value: &'a mut [u8]) {
        assert_eq!(try_specialize!(value, &mut [u8]), Ok(&mut [1, 1][..]));
    }
    inner3(&mut slice);
}

#[test]
fn try_specialize_type_inference() {
    let result: Result<u8, u8> = try_specialize!(0u8);
    assert_eq!(result, Ok(0u8));

    let result: Result<u8, u16> = try_specialize!(0u16);
    assert_eq!(result, Err(0u16));
}

#[test]
fn try_specialize_lifetime_free_unsized() {
    fn can_cast<T>(value: &[T]) -> bool {
        try_specialize!(value, &[u8]).is_ok()
    }

    let value = 42i32;
    assert!(can_cast(&[1_u8, 2, 3, 4]));
    assert!(!can_cast(&[1_i8, 2, 3, 4]));
    assert!(!can_cast(&[&value, &value]));
}

#[test]
fn try_specialize_lifetime_free_without_static_bound() {
    fn is_u8<T>(value: T) -> bool {
        try_specialize!(value, u8).is_ok()
    }

    assert!(is_u8(0u8));
    assert!(!is_u8(0u16));
}

struct Container<T>(T);

unsafe impl LifetimeFree for Container<u32> {}

#[test]
fn lifetime_free_custom_impl() {
    fn cast_container<T>(value: T) -> Result<Container<u32>, T> {
        try_specialize!(value, Container<u32>)
    }

    assert!(cast_container(Container(1u32)).is_ok());
    assert!(cast_container(1u32).is_err());
}
