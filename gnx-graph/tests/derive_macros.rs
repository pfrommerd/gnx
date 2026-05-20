use gnx_graph::util::{LifetimeFree, impl_lifetime_free, cast};
use gnx_graph::Leaf;

#[derive(LifetimeFree)]
#[allow(dead_code)]
struct LifetimeFreeNewtype(u32);

#[derive(LifetimeFree)]
#[allow(dead_code)]
struct StaticRefField(&'static u8);

#[derive(LifetimeFree)]
#[allow(dead_code)]
struct GenericPair<A, B>(A, B);

#[derive(Clone, Leaf)]
#[allow(dead_code)]
struct LeafNewtype(u32);

struct ContainerA<T>(T);
struct ContainerB<T>(T);

impl_lifetime_free!(ContainerA<u32>, for<A> ContainerB<A>);

#[test]
fn derive_lifetime_free() {
    fn cast_newtype<T: 'static>(value: T) -> Result<LifetimeFreeNewtype, T> {
        cast!(value, LifetimeFreeNewtype)
    }

    assert!(cast_newtype(LifetimeFreeNewtype(1)).is_ok());
    assert!(cast_newtype(1u32).is_err());
}

#[test]
fn derive_leaf_includes_lifetime_free() {
    fn cast_leaf<T: 'static>(value: T) -> Result<LeafNewtype, T> {
        cast!(value, LeafNewtype)
    }

    assert!(cast_leaf(LeafNewtype(42)).is_ok());
    assert!(cast_leaf(42u32).is_err());
}

#[test]
fn derive_lifetime_free_static_ref_field() {
    fn cast<T: 'static>(value: T) -> Result<StaticRefField, T> {
        cast!(value, StaticRefField)
    }

    assert!(cast(StaticRefField(&42u8)).is_ok());
    assert!(cast(42u8).is_err());
}

#[test]
fn derive_lifetime_free_generic_bounds() {
    fn cast_pair<T: 'static>(value: T) -> Result<GenericPair<u32, u32>, T> {
        cast!(value, GenericPair<u32, u32>)
    }

    assert!(cast_pair(GenericPair(1u32, 2u32)).is_ok());
    assert!(cast_pair(1u32).is_err());
}

#[test]
fn impl_lifetime_free_concrete() {
    fn cast_container_a<T: 'static>(value: T) -> Result<ContainerA<u32>, T> {
        cast!(value, ContainerA<u32>)
    }
    fn cast_container_b<T: 'static, V: LifetimeFree>(value: T) -> Result<ContainerB<V>, T> {
        cast!(value, ContainerB<V>)
    }

    assert!(cast_container_a(ContainerA(1u32)).is_ok());
    assert!(cast_container_a(1u32).is_err());
    assert!(cast_container_b::<_, u32>(ContainerB(1u32)).is_ok());
    assert!(cast_container_b::<_, bool>(ContainerB(false)).is_ok());
}

#[test]
fn impl_lifetime_free_for_syntax() {
    struct Pair<A, B>(A, B);
    impl_lifetime_free!(for<A, B> Pair<A, B>);

    fn cast_pair<T: 'static>(value: T) -> Result<Pair<u32, u32>, T> {
        cast!(value, Pair<u32, u32>)
    }

    assert!(cast_pair(Pair(1u32, 2u32)).is_ok());
    assert!(cast_pair(1u32).is_err());
}