use core::any::TypeId;
use core::fmt;
use core::marker::PhantomData;
use core::ops::Deref;
use std::collections::HashMap;

use rkyv::munge::munge;
use rkyv::ptr_meta::{self, DynMetadata, Pointee};
use rkyv::rancor::{BoxedError, Fallible, Source, Strategy};
use rkyv::rend::u32_le;
use rkyv::ser::{Allocator, Positional, Writer};
use rkyv::vec::{ArchivedVec, VecResolver};
use rkyv::with::{ArchiveWith, AsString, SerializeWith};
use rkyv::{Archive, Place, Portable, RawRelPtr, Serialize, SerializeUnsized};

// ---------------------------------------------------------------------------
// Type names and dialects
// ---------------------------------------------------------------------------

/// Names a concrete implementor of a trait object: a dialect, and the name
/// of the type within that dialect.
///
/// Names are not stored with each value. An archive holds one type table per
/// trait object (see `TypeTable`), with one archived name per distinct type,
/// and each archived pointer holds an index into that table.
#[derive(Archive, Serialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeName {
    #[rkyv(with = AsString)]
    pub dialect: &'static str,
    #[rkyv(with = AsString)]
    pub name: &'static str,
}

impl ArchivedTypeName {
    pub fn dialect(&self) -> &str {
        &self.dialect
    }
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Gives the name of a concrete type. Implemented by `dialect!`, so a type
/// can be named in the dialects of one trait object only.
///
/// SAFETY: no dialect of the trait object identified by `trait_object` may
/// map the returned name to a different type.
pub unsafe trait Typed {
    fn type_name(&self) -> TypeName;
    /// The `TypeId` of the trait object whose dialects name this type.
    fn trait_object(&self) -> TypeId;
}

/// A named set of implementors of the trait object `T`. Dialects are defined
/// and registered with `dialect!`; each trait object has its own registry.
pub struct Dialect<T: ?Sized + 'static> {
    name: &'static str,
    types: &'static [DialectType<T>],
}

/// An entry of a `Dialect`: a type name and the implementation it maps to.
pub struct DialectType<T: ?Sized + 'static> {
    name: &'static str,
    metadata: fn() -> DynMetadata<T>,
}

impl<T: ?Sized + 'static> Dialect<T> {
    /// SAFETY: every entry of `types` must be the vtable of the concrete
    /// type that reports this dialect and that entry's name.
    pub const unsafe fn new(name: &'static str, types: &'static [DialectType<T>]) -> Self {
        Self { name, types }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn type_names(&self) -> impl Iterator<Item = &'static str> {
        self.types.iter().map(|ty| ty.name)
    }
}

impl<T: ?Sized + 'static> DialectType<T> {
    pub const fn new(name: &'static str, metadata: fn() -> DynMetadata<T>) -> Self {
        Self { name, metadata }
    }
}

/// Finds the implementation registered for `name` among the dialects of `T`.
///
/// Panics if the name is registered more than once, since data written by
/// one type could then be read as another.
pub fn lookup_type<T: ?Sized + 'static>(name: &ArchivedTypeName) -> Option<DynMetadata<T>>
where
    Dialect<T>: inventory::Collect,
{
    let mut matches = inventory::iter::<Dialect<T>>
        .into_iter()
        .filter(|dialect| dialect.name == name.dialect())
        .flat_map(|dialect| dialect.types)
        .filter(|ty| ty.name == name.name());
    let found = matches.next()?;
    assert!(
        matches.next().is_none(),
        "multiple types are registered as {}.{}",
        name.dialect(),
        name.name(),
    );
    Some((found.metadata)())
}

/// Defines a dialect of a trait object, as a mapping of names to types, and
/// registers it so that archives containing those types can be loaded.
///
/// ```ignore
/// dialect! {
///     dyn Operation, "math" => {
///         "add" => Add,
///         "scale" => Scale,
///     }
/// }
/// ```
#[macro_export]
macro_rules! dialect {
    ($obj:ty, $dialect:literal => { $($name:literal => $ty:ty),* $(,)? }) => {
        $(
            // SAFETY: the dialect registered below maps this name to this type.
            unsafe impl $crate::Typed for $ty {
                fn type_name(&self) -> $crate::TypeName {
                    $crate::TypeName { dialect: $dialect, name: $name }
                }
                fn trait_object(&self) -> ::core::any::TypeId {
                    ::core::any::TypeId::of::<$obj>()
                }
            }
        )*

        $crate::__private::inventory::submit! {{
            const TYPES: &[$crate::DialectType<$obj>] = &[
                $($crate::DialectType::new($name, || {
                    $crate::__private::rkyv::ptr_meta::metadata(
                        ::core::ptr::null::<$ty>() as *const $obj,
                    )
                })),*
            ];
            // SAFETY: each name is paired with the vtable of the type that
            // reports it, per the `Typed` impls above.
            unsafe { $crate::Dialect::<$obj>::new($dialect, TYPES) }
        }}
    };
}

/// Implemented for trait objects, e.g. `dyn Operation`, by `dyn_archive!`.
///
/// SAFETY: `lookup` must return the vtable of the concrete type that reports
/// `name`, or `None`.
pub unsafe trait DynTyped:
    Portable + Typed + DynSerialize + Pointee<Metadata = DynMetadata<Self>> + 'static
{
    /// Looks up the implementation for an entry of an archive's type table.
    /// Called once per entry when the table is resolved, not on every access.
    fn lookup(name: &ArchivedTypeName) -> Option<DynMetadata<Self>>;
}

// ---------------------------------------------------------------------------
// Type tables
// ---------------------------------------------------------------------------

/// An index into the type table of an archive.
#[derive(
    Archive, Serialize, Portable, Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord,
)]
#[rkyv(as = Self)]
#[repr(transparent)]
pub struct TypeIdx(u32_le);

impl From<TypeIdx> for usize {
    fn from(idx: TypeIdx) -> usize {
        idx.0.to_native() as usize
    }
}

/// Implemented by serializers that build the type table for `T`. Each trait
/// object has its own table.
pub trait TypeTableContext<T: ?Sized> {
    /// Returns the index of `name` in the table, adding it if it is new.
    fn intern_type(&mut self, name: TypeName) -> TypeIdx;
    /// The names interned so far, in index order.
    fn interned_types(&self) -> &[TypeName];
}

/// The serializer-side state behind a `TypeTableContext`.
pub struct TypeTableBuilder<T: ?Sized> {
    indices: HashMap<TypeName, TypeIdx>,
    names: Vec<TypeName>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ?Sized> Default for TypeTableBuilder<T> {
    fn default() -> Self {
        Self {
            indices: HashMap::new(),
            names: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized> TypeTableContext<T> for TypeTableBuilder<T> {
    fn intern_type(&mut self, name: TypeName) -> TypeIdx {
        if let Some(&idx) = self.indices.get(&name) {
            return idx;
        }
        let idx = u32::try_from(self.names.len()).expect("type index exceeds u32::MAX");
        let idx = TypeIdx(idx.into());
        self.names.push(name);
        self.indices.insert(name, idx);
        idx
    }

    fn interned_types(&self) -> &[TypeName] {
        &self.names
    }
}

/// The type table for `T`. The unarchived version is a zero-sized
/// placeholder: serializing it writes out the names the serializer has
/// interned so far, so it must be serialized after every pointer to `T`.
pub struct TypeTable<T: ?Sized>(PhantomData<fn() -> T>);

impl<T: ?Sized> Default for TypeTable<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// The archived type table for `T`: one archived type name per index.
#[derive(Portable)]
#[repr(transparent)]
pub struct ArchivedTypeTable<T: ?Sized> {
    names: ArchivedVec<ArchivedTypeName>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ?Sized> ArchivedTypeTable<T> {
    pub fn names(&self) -> &[ArchivedTypeName] {
        &self.names
    }

    /// Looks up the implementation of every entry.
    pub fn resolve(&self) -> TypeImpls<T>
    where
        T: DynTyped,
    {
        TypeImpls(self.names.iter().map(T::lookup).collect())
    }
}

impl<T: ?Sized> Archive for TypeTable<T> {
    type Archived = ArchivedTypeTable<T>;
    type Resolver = (usize, VecResolver);

    fn resolve(&self, (len, resolver): Self::Resolver, out: Place<Self::Archived>) {
        munge!(let ArchivedTypeTable { names, _marker: _ } = out);
        ArchivedVec::resolve_from_len(len, resolver, names);
    }
}

impl<T, S> Serialize<S> for TypeTable<T>
where
    T: ?Sized,
    S: Fallible + Allocator + Writer + TypeTableContext<T> + ?Sized,
    S::Error: Source,
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let names = serializer.interned_types().to_vec();
        let resolver = ArchivedVec::serialize_from_slice(&names, serializer)?;
        Ok((names.len(), resolver))
    }
}

/// The implementations for a type table, resolved at load time. Entries
/// whose type is unknown are `None`.
pub struct TypeImpls<T: ?Sized>(Vec<Option<DynMetadata<T>>>);

impl<T: ?Sized> TypeImpls<T> {
    pub fn get(&self, idx: TypeIdx) -> Option<DynMetadata<T>> {
        self.0.get(usize::from(idx)).copied().flatten()
    }

    /// The indices of the entries with no known implementation.
    pub fn missing(&self) -> impl Iterator<Item = usize> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, m)| m.is_none())
            .map(|(i, _)| i)
    }
}

// ---------------------------------------------------------------------------
// Type-erased serialization of the value behind a trait object
// ---------------------------------------------------------------------------

/// Dyn-compatible view of a writer.
pub trait ErasedWriter {
    fn erased_pos(&self) -> usize;
    fn erased_write(&mut self, bytes: &[u8]) -> Result<(), BoxedError>;
}

impl Positional for dyn ErasedWriter + '_ {
    fn pos(&self) -> usize {
        self.erased_pos()
    }
}

impl Writer<BoxedError> for dyn ErasedWriter + '_ {
    fn write(&mut self, bytes: &[u8]) -> Result<(), BoxedError> {
        self.erased_write(bytes)
    }
}

/// A serializer that is not generic over the underlying serializer, so it
/// can appear in a trait object's method signature. It can only write
/// bytes. Its error type is `BoxedError`, which implements `Source`, so
/// `rancor::fail!` and friends work as usual.
pub type DynSerializer<'a> = Strategy<dyn ErasedWriter + 'a, BoxedError>;

/// Added as a supertrait by `dyn_archive!`. Implemented automatically for
/// every type that archives as itself.
pub trait DynSerialize {
    /// Writes `self` in archived form and returns its position.
    fn serialize_dyn(&self, serializer: &mut DynSerializer<'_>) -> Result<usize, BoxedError>;
}

impl<T> DynSerialize for T
where
    T: Archive<Archived = T> + for<'a> Serialize<DynSerializer<'a>>,
{
    fn serialize_dyn(&self, serializer: &mut DynSerializer<'_>) -> Result<usize, BoxedError> {
        // rkyv's standard path: zeroed output, then `resolve` field by field.
        SerializeUnsized::serialize_unsized(self, serializer)
    }
}

/// Placeholder sent through the erased call when the underlying serializer
/// fails; `serialize_erased` swaps the original error back in.
#[derive(Debug)]
struct Stashed;

impl fmt::Display for Stashed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the underlying serializer failed")
    }
}

impl core::error::Error for Stashed {}

struct Adapter<'a, S: Fallible + ?Sized> {
    inner: &'a mut S,
    stashed: Option<S::Error>,
}

impl<S: Fallible + Writer + ?Sized> ErasedWriter for Adapter<'_, S> {
    fn erased_pos(&self) -> usize {
        self.inner.pos()
    }

    fn erased_write(&mut self, bytes: &[u8]) -> Result<(), BoxedError> {
        self.inner.write(bytes).map_err(|e| {
            self.stashed = Some(e);
            BoxedError::new(Stashed)
        })
    }
}

/// Runs `f` against a type-erased view of `serializer`.
fn serialize_erased<S>(
    serializer: &mut S,
    f: impl FnOnce(&mut DynSerializer<'_>) -> Result<usize, BoxedError>,
) -> Result<usize, S::Error>
where
    S: Fallible + Writer + ?Sized,
    S::Error: Source,
{
    let mut adapter = Adapter {
        inner: serializer,
        stashed: None,
    };
    let erased: &mut dyn ErasedWriter = &mut adapter;
    f(Strategy::wrap(erased)).map_err(|e| match adapter.stashed.take() {
        // The serializer itself failed: return its error untouched.
        Some(original) => original,
        // An implementor created the error: carry it over.
        None => S::Error::new(e),
    })
}

// ---------------------------------------------------------------------------
// Archived pointers
// ---------------------------------------------------------------------------

/// The archived form of a pointer to a trait object: where the value is,
/// plus the index of its type in the archive's type table for `T`.
#[derive(Portable)]
#[repr(C)]
pub struct ArchivedDyn<T: ?Sized> {
    ptr: RawRelPtr,
    type_idx: TypeIdx,
    _marker: PhantomData<T>,
}

impl<T: DynTyped + ?Sized> ArchivedDyn<T> {
    pub fn type_idx(&self) -> TypeIdx {
        self.type_idx
    }

    /// Returns the value, or `None` if its type has no known implementation.
    ///
    /// SAFETY: `impls` must be resolved from the type table of the archive
    /// this value is in, and the archive must be valid.
    pub unsafe fn get(&self, impls: &TypeImpls<T>) -> Option<&T> {
        let metadata = impls.get(self.type_idx)?;
        // SAFETY: the pointer targets a value of the concrete type that
        // `metadata` describes, per the `DynTyped` contract.
        unsafe {
            let ptr: *const T = ptr_meta::from_raw_parts(self.ptr.as_ptr().cast(), metadata);
            Some(&*ptr)
        }
    }
}

pub struct DynResolver {
    pos: usize,
    type_idx: TypeIdx,
}

/// Shared by every pointer type; see `dyn_archive!` and `AsDyn`.
pub fn serialize_dyn<T, S>(value: &T, serializer: &mut S) -> Result<DynResolver, S::Error>
where
    T: DynTyped + ?Sized,
    S: Fallible + Writer + TypeTableContext<T> + ?Sized,
    S::Error: Source,
{
    let name = value.type_name();
    // A name only identifies a type among the dialects it was registered in.
    assert!(
        value.trait_object() == TypeId::of::<T>(),
        "{}.{} is not in a dialect of this trait object",
        name.dialect,
        name.name,
    );
    let type_idx = serializer.intern_type(name);
    let pos = serialize_erased(serializer, |erased| value.serialize_dyn(erased))?;
    Ok(DynResolver { pos, type_idx })
}

/// Shared by every pointer type; see `dyn_archive!` and `AsDyn`.
pub fn resolve_dyn<T: ?Sized>(resolver: DynResolver, out: Place<ArchivedDyn<T>>) {
    munge!(let ArchivedDyn { ptr, type_idx: TypeIdx(type_idx), _marker: _ } = out);
    RawRelPtr::emplace(resolver.pos, ptr);
    type_idx.write(resolver.type_idx.0);
}

/// `#[rkyv(with = AsDyn)]` for pointers that `dyn_archive!` cannot cover
/// directly, such as `Rc<dyn Trait>` and `Arc<dyn Trait>`.
pub struct AsDyn;

impl<F> ArchiveWith<F> for AsDyn
where
    F: Deref + ?Sized,
    F::Target: DynTyped,
{
    type Archived = ArchivedDyn<F::Target>;
    type Resolver = DynResolver;

    fn resolve_with(_: &F, resolver: Self::Resolver, out: Place<Self::Archived>) {
        resolve_dyn(resolver, out)
    }
}

impl<F, S> SerializeWith<F, S> for AsDyn
where
    F: Deref + ?Sized,
    F::Target: DynTyped,
    S: Fallible + Writer + TypeTableContext<F::Target> + ?Sized,
    S::Error: Source,
{
    fn serialize_with(field: &F, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        serialize_dyn(&**field, serializer)
    }
}

/// Wrap a trait definition to make pointers to `dyn Trait` archivable.
///
/// ```ignore
/// dyn_archive! {
///     pub trait Operation { ... }
/// }
/// ```
///
/// `Box<dyn Trait>` and `&dyn Trait` then implement `Archive`/`Serialize`,
/// archiving as `ArchivedDyn<dyn Trait>`, and `dyn Trait` gets its own
/// dialect registry. Implementors of the trait must archive as themselves
/// and be named in a `dialect!`.
macro_rules! dyn_archive {
    ($(#[$meta:meta])* $vis:vis trait $tr:ident $($rest:tt)*) => {
        $crate::util::dyn_archive!(@trait [$(#[$meta])* $vis] $tr $($rest)*);
    };

    // Re-emit the trait with the supertraits that make it archivable.
    (@trait [$($head:tt)*] $tr:ident : $($rest:tt)*) => {
        #[::rkyv::ptr_meta::pointee(crate = ::rkyv::ptr_meta)]
        $($head)* trait $tr:
            ::rkyv::Portable
            + $crate::util::Typed
            + $crate::util::DynSerialize
            + $($rest)*

        $crate::util::dyn_archive!(@impls $tr);
    };
    (@trait [$($head:tt)*] $tr:ident { $($body:tt)* }) => {
        #[::rkyv::ptr_meta::pointee(crate = ::rkyv::ptr_meta)]
        $($head)* trait $tr:
            ::rkyv::Portable + $crate::util::Typed + $crate::util::DynSerialize
        {
            $($body)*
        }

        $crate::util::dyn_archive!(@impls $tr);
    };

    (@impls $tr:ident) => {
        // The dialects of this trait object
        ::inventory::collect!($crate::util::Dialect<dyn $tr>);

        const _: () = {
            use ::rkyv::ptr_meta::DynMetadata;
            use ::rkyv::rancor::{Fallible, Source};
            use ::rkyv::ser::Writer;
            use ::rkyv::{Archive, Place, Serialize};
            use $crate::util::{
                ArchivedDyn, ArchivedTypeName, DynResolver, DynTyped, TypeTableContext,
                lookup_type, resolve_dyn, serialize_dyn,
            };

            // SAFETY: dialects map each name to the type that reports it.
            unsafe impl DynTyped for dyn $tr {
                fn lookup(name: &ArchivedTypeName) -> Option<DynMetadata<Self>> {
                    lookup_type::<Self>(name)
                }
            }

            // `Box` and `&` are the pointer types a downstream crate may
            // implement foreign traits for; others use `with = AsDyn`.
            impl Archive for ::std::boxed::Box<dyn $tr> {
                type Archived = ArchivedDyn<dyn $tr>;
                type Resolver = DynResolver;

                fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
                    resolve_dyn(resolver, out)
                }
            }

            impl<S> Serialize<S> for ::std::boxed::Box<dyn $tr>
            where
                S: Fallible + Writer + TypeTableContext<dyn $tr> + ?Sized,
                S::Error: Source,
            {
                fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
                    serialize_dyn::<dyn $tr, S>(&**self, serializer)
                }
            }

            impl Archive for &(dyn $tr + 'static) {
                type Archived = ArchivedDyn<dyn $tr>;
                type Resolver = DynResolver;

                fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
                    resolve_dyn(resolver, out)
                }
            }

            impl<S> Serialize<S> for &(dyn $tr + 'static)
            where
                S: Fallible + Writer + TypeTableContext<dyn $tr> + ?Sized,
                S::Error: Source,
            {
                fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
                    serialize_dyn::<dyn $tr, S>(&**self, serializer)
                }
            }
        };
    };
}

pub(crate) use dyn_archive;

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use rkyv::rancor::Error;
    use rkyv::ser::Serializer;
    use rkyv::util::AlignedVec;

    use super::*;

    dyn_archive! {
        trait Shape: Send {
            fn describe(&self) -> String;
        }
    }

    dyn_archive! {
        trait Other {}
    }

    #[derive(Archive, Serialize, Portable)]
    #[rkyv(as = Self)]
    #[repr(C)]
    struct Square(u32_le);
    impl Shape for Square {
        fn describe(&self) -> String {
            format!("square({})", self.0)
        }
    }
    // Also an `Other`, but only named in a dialect of `dyn Shape`.
    impl Other for Square {}

    // Has 3 padding bytes after the `u8`.
    #[derive(Archive, Serialize, Portable)]
    #[rkyv(as = Self)]
    #[repr(C)]
    struct Padded(u8, u32_le);
    impl Shape for Padded {
        fn describe(&self) -> String {
            format!("padded({}, {})", self.0, self.1)
        }
    }

    // Its `Serialize` constructs an error.
    #[derive(Archive, Portable)]
    #[rkyv(as = Self)]
    #[repr(transparent)]
    struct Refuses(u32_le);

    #[derive(Debug)]
    struct Refused;
    impl fmt::Display for Refused {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("refused")
        }
    }
    impl core::error::Error for Refused {}

    impl<S: Fallible + ?Sized> Serialize<S> for Refuses
    where
        S::Error: Source,
    {
        fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
            rkyv::rancor::fail!(Refused)
        }
    }
    impl Shape for Refuses {
        fn describe(&self) -> String {
            "refuses".into()
        }
    }

    crate::dialect! {
        dyn Shape, "shapes" => {
            "square" => Square,
            "padded" => Padded,
            "refuses" => Refuses,
        }
    }

    // ---- a serializer with a type table for each trait object ----

    #[derive(Default)]
    struct Tables {
        shapes: TypeTableBuilder<dyn Shape>,
        others: TypeTableBuilder<dyn Other>,
    }

    impl<W, A, E> TypeTableContext<dyn Shape> for Strategy<Serializer<W, A, Tables>, E> {
        fn intern_type(&mut self, name: TypeName) -> TypeIdx {
            self.sharing.shapes.intern_type(name)
        }
        fn interned_types(&self) -> &[TypeName] {
            self.sharing.shapes.interned_types()
        }
    }
    impl<W, A, E> TypeTableContext<dyn Other> for Strategy<Serializer<W, A, Tables>, E> {
        fn intern_type(&mut self, name: TypeName) -> TypeIdx {
            self.sharing.others.intern_type(name)
        }
        fn interned_types(&self) -> &[TypeName] {
            self.sharing.others.interned_types()
        }
    }

    fn to_bytes<T>(value: &T) -> Result<AlignedVec, Error>
    where
        T: for<'a> Serialize<
            Strategy<Serializer<AlignedVec, rkyv::ser::allocator::ArenaHandle<'a>, Tables>, Error>,
        >,
    {
        rkyv::util::with_arena(|arena| {
            let mut serializer =
                Serializer::new(AlignedVec::new(), arena.acquire(), Tables::default());
            rkyv::api::serialize_using::<_, Error>(value, &mut serializer)?;
            Ok(serializer.writer)
        })
    }

    #[derive(Portable)]
    #[repr(C)]
    struct ArchivedHolder {
        boxed: ArchivedDyn<dyn Shape>,
        borrowed: ArchivedDyn<dyn Shape>,
        shared: ArchivedDyn<dyn Shape>,
        rest: ArchivedVec<ArchivedDyn<dyn Shape>>,
        shapes: ArchivedTypeTable<dyn Shape>,
    }

    #[derive(Archive, Serialize)]
    #[rkyv(
        as = ArchivedHolder,
        archive_bounds(&'a (dyn Shape + 'static): Archive<Archived = ArchivedDyn<dyn Shape>>),
    )]
    struct Holder<'a> {
        boxed: Box<dyn Shape>,
        borrowed: &'a (dyn Shape + 'static),
        #[rkyv(with = AsDyn)]
        shared: Rc<dyn Shape>,
        rest: Vec<Box<dyn Shape>>,
        shapes: TypeTable<dyn Shape>,
    }

    fn holder<'a>(borrowed: &'a (dyn Shape + 'static), rest: Vec<Box<dyn Shape>>) -> Holder<'a> {
        Holder {
            boxed: Box::new(Padded(7, 9.into())),
            borrowed,
            shared: Rc::new(Square(5.into())),
            rest,
            shapes: TypeTable::default(),
        }
    }

    #[test]
    fn round_trip() {
        let borrowed = Square(4.into());
        let rest: Vec<Box<dyn Shape>> =
            vec![Box::new(Square(1.into())), Box::new(Padded(2, 3.into()))];
        let bytes = to_bytes(&holder(&borrowed, rest)).unwrap();
        let archived = unsafe { rkyv::access_unchecked::<ArchivedHolder>(&bytes) };

        // One table entry per distinct type, in first-use order.
        let names: Vec<_> = archived
            .shapes
            .names()
            .iter()
            .map(|n| (n.dialect(), n.name()))
            .collect();
        assert_eq!(names, [("shapes", "padded"), ("shapes", "square")]);

        let impls = archived.shapes.resolve();
        assert_eq!(impls.missing().count(), 0);
        let describe =
            |shape: &ArchivedDyn<dyn Shape>| unsafe { shape.get(&impls) }.unwrap().describe();
        assert_eq!(describe(&archived.boxed), "padded(7, 9)");
        assert_eq!(describe(&archived.borrowed), "square(4)");
        assert_eq!(describe(&archived.shared), "square(5)");
        let rest: Vec<_> = archived.rest.iter().map(describe).collect();
        assert_eq!(rest, ["square(1)", "padded(2, 3)"]);
    }

    #[test]
    fn padding_is_zeroed() {
        // Build a `Padded` whose padding bytes are 0xFF.
        let mut slot = core::mem::MaybeUninit::<Padded>::uninit();
        let padded = unsafe {
            slot.as_mut_ptr()
                .cast::<u8>()
                .write_bytes(0xFF, size_of::<Padded>());
            (&raw mut (*slot.as_mut_ptr()).0).write(7);
            (&raw mut (*slot.as_mut_ptr()).1).write(9.into());
            slot.assume_init()
        };
        let boxed: Box<dyn Shape> = Box::new(padded);
        let bytes = to_bytes(&boxed).unwrap();
        assert_eq!(&bytes[..8], &[7, 0, 0, 0, 9, 0, 0, 0]);
    }

    #[test]
    fn implementors_can_construct_errors() {
        let borrowed = Square(4.into());
        let err = to_bytes(&holder(&borrowed, vec![Box::new(Refuses(0.into()))])).unwrap_err();
        assert!(err.to_string().contains("refused"), "{err}");
    }

    struct FailingWriter;
    impl Positional for FailingWriter {
        fn pos(&self) -> usize {
            0
        }
    }
    impl<E: Source> Writer<E> for FailingWriter {
        fn write(&mut self, _: &[u8]) -> Result<(), E> {
            Err(E::new(std::io::Error::other("disk full")))
        }
    }

    #[test]
    fn writer_errors_propagate() {
        let boxed: Box<dyn Shape> = Box::new(Square(1.into()));
        let mut serializer = Serializer::new(FailingWriter, (), Tables::default());
        let err = rkyv::api::serialize_using::<_, Error>(&boxed, &mut serializer).unwrap_err();
        assert!(err.to_string().contains("disk full"), "{err}");
    }

    #[test]
    #[should_panic(expected = "shapes.square is not in a dialect of this trait object")]
    fn names_are_scoped_to_their_trait_object() {
        let boxed: Box<dyn Other> = Box::new(Square(1.into()));
        let _ = to_bytes(&boxed);
    }

    #[test]
    fn unknown_names_do_not_resolve() {
        let name = TypeName {
            dialect: "shapes",
            name: "square",
        };
        let bytes = rkyv::to_bytes::<Error>(&name).unwrap();
        let archived = unsafe { rkyv::access_unchecked::<ArchivedTypeName>(&bytes) };
        assert!(<dyn Shape as DynTyped>::lookup(archived).is_some());
        // Each trait object has its own registry.
        assert!(<dyn Other as DynTyped>::lookup(archived).is_none());
    }
}
