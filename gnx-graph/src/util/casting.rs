// Portions vendored from castaway 0.2.4 (MIT)
// https://github.com/sagebind/castaway
//
// Copyright (c) 2020 Stephen M. Coakley

mod utils {
    use core::{
        any::{type_name, TypeId},
        marker::PhantomData,
        mem,
        ptr,
    };

    #[inline(always)]
    pub(crate) fn type_eq<T: 'static, U: 'static>() -> bool {
        mem::size_of::<T>() == mem::size_of::<U>()
            && mem::align_of::<T>() == mem::align_of::<U>()
            && mem::needs_drop::<T>() == mem::needs_drop::<U>()
            && TypeId::of::<T>() == TypeId::of::<U>()
            && type_name::<T>() == type_name::<U>()
    }

    #[inline(always)]
    pub(crate) fn type_eq_non_static<T: ?Sized, U: ?Sized>() -> bool {
        non_static_type_id::<T>() == non_static_type_id::<U>()
    }

    fn non_static_type_id<T: ?Sized>() -> TypeId {
        trait NonStaticAny {
            fn get_type_id(&self) -> TypeId
            where
                Self: 'static;
        }

        impl<T: ?Sized> NonStaticAny for PhantomData<T> {
            fn get_type_id(&self) -> TypeId
            where
                Self: 'static,
            {
                TypeId::of::<T>()
            }
        }

        let phantom_data = PhantomData::<T>;
        NonStaticAny::get_type_id(unsafe {
            mem::transmute::<&dyn NonStaticAny, &(dyn NonStaticAny + 'static)>(&phantom_data)
        })
    }

    #[inline(always)]
    pub(crate) unsafe fn transmute_unchecked<T, U>(value: T) -> U {
        assert!(
            mem::size_of::<T>() == mem::size_of::<U>(),
            "cannot transmute_unchecked if Dst and Src have different size"
        );

        let dest = unsafe { ptr::read(&value as *const T as *const U) };
        mem::forget(value);
        dest
    }
}

#[doc(hidden)]
pub mod internal {
    use super::utils::{transmute_unchecked, type_eq, type_eq_non_static};
    use super::LifetimeFree;
    use core::marker::PhantomData;

    pub struct CastToken<T: ?Sized>(PhantomData<T>);

    impl<T: ?Sized> CastToken<T> {
        pub const fn of_val(_value: &T) -> Self {
            Self::of()
        }

        pub const fn of() -> Self {
            Self(PhantomData)
        }
    }

    pub trait TryCastMutLifetimeFree<'a, T: ?Sized, U: LifetimeFree + ?Sized> {
        #[inline(always)]
        fn try_cast(&self, value: &'a mut T) -> Result<&'a mut U, &'a mut T> {
            if type_eq_non_static::<T, U>() {
                Ok(unsafe { transmute_unchecked::<&mut T, &mut U>(value) })
            } else {
                Err(value)
            }
        }
    }

    impl<'a, T: ?Sized, U: LifetimeFree + ?Sized> TryCastMutLifetimeFree<'a, T, U>
        for &&&&&&&(CastToken<&'a mut T>, CastToken<&'a mut U>)
    {
    }

    pub trait TryCastRefLifetimeFree<'a, T: ?Sized, U: LifetimeFree + ?Sized> {
        #[inline(always)]
        fn try_cast(&self, value: &'a T) -> Result<&'a U, &'a T> {
            if type_eq_non_static::<T, U>() {
                Ok(unsafe { transmute_unchecked::<&T, &U>(value) })
            } else {
                Err(value)
            }
        }
    }

    impl<'a, T: ?Sized, U: LifetimeFree + ?Sized> TryCastRefLifetimeFree<'a, T, U>
        for &&&&&&(CastToken<&'a T>, CastToken<&'a U>)
    {
    }

    pub trait TryCastOwnedLifetimeFree<T, U: LifetimeFree> {
        #[inline(always)]
        fn try_cast(&self, value: T) -> Result<U, T> {
            if type_eq_non_static::<T, U>() {
                Ok(unsafe { transmute_unchecked::<T, U>(value) })
            } else {
                Err(value)
            }
        }
    }

    impl<T, U: LifetimeFree> TryCastOwnedLifetimeFree<T, U> for &&&&&(CastToken<T>, CastToken<U>) {}

    pub trait TryCastSliceMut<'a, T: 'static, U: 'static> {
        #[inline(always)]
        fn try_cast(&self, value: &'a mut [T]) -> Result<&'a mut [U], &'a mut [T]> {
            if type_eq::<T, U>() {
                Ok(unsafe { &mut *(value as *mut [T] as *mut [U]) })
            } else {
                Err(value)
            }
        }
    }

    impl<'a, T: 'static, U: 'static> TryCastSliceMut<'a, T, U>
        for &&&&(CastToken<&'a mut [T]>, CastToken<&'a mut [U]>)
    {
    }

    pub trait TryCastSliceRef<'a, T: 'static, U: 'static> {
        #[inline(always)]
        fn try_cast(&self, value: &'a [T]) -> Result<&'a [U], &'a [T]> {
            if type_eq::<T, U>() {
                Ok(unsafe { &*(value as *const [T] as *const [U]) })
            } else {
                Err(value)
            }
        }
    }

    impl<'a, T: 'static, U: 'static> TryCastSliceRef<'a, T, U>
        for &&&(CastToken<&'a [T]>, CastToken<&'a [U]>)
    {
    }

    pub trait TryCastMut<'a, T: 'static, U: 'static> {
        #[inline(always)]
        fn try_cast(&self, value: &'a mut T) -> Result<&'a mut U, &'a mut T> {
            if type_eq::<T, U>() {
                Ok(unsafe { &mut *(value as *mut T as *mut U) })
            } else {
                Err(value)
            }
        }
    }

    impl<'a, T: 'static, U: 'static> TryCastMut<'a, T, U>
        for &&(CastToken<&'a mut T>, CastToken<&'a mut U>)
    {
    }

    pub trait TryCastRef<'a, T: 'static, U: 'static> {
        #[inline(always)]
        fn try_cast(&self, value: &'a T) -> Result<&'a U, &'a T> {
            if type_eq::<T, U>() {
                Ok(unsafe { &*(value as *const T as *const U) })
            } else {
                Err(value)
            }
        }
    }

    impl<'a, T: 'static, U: 'static> TryCastRef<'a, T, U>
        for &(CastToken<&'a T>, CastToken<&'a U>)
    {
    }

    pub trait TryCastOwned<T: 'static, U: 'static> {
        #[inline(always)]
        fn try_cast(&self, value: T) -> Result<U, T> {
            if type_eq::<T, U>() {
                Ok(unsafe { transmute_unchecked::<T, U>(value) })
            } else {
                Err(value)
            }
        }
    }

    impl<T: 'static, U: 'static> TryCastOwned<T, U> for (CastToken<T>, CastToken<U>) {}
}

/// Marker trait for types that do not contain any lifetime parameters.
///
/// # Safety
///
/// When implementing this trait for a type, you must ensure that the type is
/// free of any lifetime parameters. Failure to meet **all** of the requirements
/// below may result in undefined behavior.
///
/// - The type must be `'static`.
/// - The type must be free of lifetime parameters.
/// - All contained fields must also be `LifetimeFree`.
pub unsafe trait LifetimeFree {}

unsafe impl LifetimeFree for () {}
unsafe impl LifetimeFree for bool {}
unsafe impl LifetimeFree for char {}
unsafe impl LifetimeFree for f32 {}
unsafe impl LifetimeFree for f64 {}
unsafe impl LifetimeFree for i8 {}
unsafe impl LifetimeFree for i16 {}
unsafe impl LifetimeFree for i32 {}
unsafe impl LifetimeFree for i64 {}
unsafe impl LifetimeFree for i128 {}
unsafe impl LifetimeFree for isize {}
unsafe impl LifetimeFree for str {}
unsafe impl LifetimeFree for u8 {}
unsafe impl LifetimeFree for u16 {}
unsafe impl LifetimeFree for u32 {}
unsafe impl LifetimeFree for u64 {}
unsafe impl LifetimeFree for u128 {}
unsafe impl LifetimeFree for usize {}

unsafe impl LifetimeFree for core::num::NonZeroI8 {}
unsafe impl LifetimeFree for core::num::NonZeroI16 {}
unsafe impl LifetimeFree for core::num::NonZeroI32 {}
unsafe impl LifetimeFree for core::num::NonZeroI64 {}
unsafe impl LifetimeFree for core::num::NonZeroI128 {}
unsafe impl LifetimeFree for core::num::NonZeroIsize {}
unsafe impl LifetimeFree for core::num::NonZeroU8 {}
unsafe impl LifetimeFree for core::num::NonZeroU16 {}
unsafe impl LifetimeFree for core::num::NonZeroU32 {}
unsafe impl LifetimeFree for core::num::NonZeroU64 {}
unsafe impl LifetimeFree for core::num::NonZeroU128 {}
unsafe impl LifetimeFree for core::num::NonZeroUsize {}

unsafe impl<T: LifetimeFree> LifetimeFree for [T] {}
unsafe impl<T: LifetimeFree, const SIZE: usize> LifetimeFree for [T; SIZE] {}
unsafe impl<T: LifetimeFree> LifetimeFree for Option<T> {}
unsafe impl<T: LifetimeFree, E: LifetimeFree> LifetimeFree for Result<T, E> {}
unsafe impl<T: LifetimeFree> LifetimeFree for core::num::Wrapping<T> {}
unsafe impl<T: LifetimeFree> LifetimeFree for core::cell::Cell<T> {}
unsafe impl<T: LifetimeFree> LifetimeFree for core::cell::RefCell<T> {}

macro_rules! tuple_impls {
    ($( $( $name:ident )+, )+) => {
        $(
            unsafe impl<$($name: LifetimeFree),+> LifetimeFree for ($($name,)+) {}
        )+
    };
}

tuple_impls! {
    T0,
    T0 T1,
    T0 T1 T2,
    T0 T1 T2 T3,
    T0 T1 T2 T3 T4,
    T0 T1 T2 T3 T4 T5,
    T0 T1 T2 T3 T4 T5 T6,
    T0 T1 T2 T3 T4 T5 T6 T7,
    T0 T1 T2 T3 T4 T5 T6 T7 T8,
    T0 T1 T2 T3 T4 T5 T6 T7 T8 T9,
    T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10,
    T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11,
}

unsafe impl LifetimeFree for String {}
unsafe impl<T: LifetimeFree> LifetimeFree for Box<T> {}
unsafe impl<T: LifetimeFree> LifetimeFree for Vec<T> {}

#[cfg(target_has_atomic = "ptr")]
unsafe impl<T: LifetimeFree> LifetimeFree for std::sync::Arc<T> {}

/// Attempt to cast the result of an expression into a given concrete type.
///
/// Vendored from castaway's [`cast!`](https://docs.rs/castaway/latest/castaway/macro.cast.html).
#[macro_export]
macro_rules! cast {
    ($value:expr, $T:ty) => {{
        #[allow(unused_imports)]
        use $crate::util::casting::internal::*;

        let value = $value;
        let src_token = CastToken::of_val(&value);
        let dest_token = CastToken::<$T>::of();

        let result: ::core::result::Result<$T, _> = (&&&&&&&(src_token, dest_token)).try_cast(value);

        result
    }};

    ($value:expr) => {
        $crate::cast!($value, _)
    };
}
