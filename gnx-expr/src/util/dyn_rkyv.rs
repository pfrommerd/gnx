use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::Deref;
use std::collections::HashMap;
use std::sync::Arc;

use rkyv::api::high::HighSerializer;
use rkyv::munge::munge;
use rkyv::ptr_meta::{self, DynMetadata, Pointee};
use rkyv::rancor::{self, Fallible, Source};
use rkyv::rend::u32_le;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::ser::{Allocator, Writer, WriterExt};
use rkyv::util::AlignedVec;
use rkyv::vec::{ArchivedVec, VecResolver};
use rkyv::with::AsString;
use rkyv::{Archive, Place, Portable, RawRelPtr, Serialize};

// Trait objects in an archive.
//
// A trait such as Operation is implemented by archived types. A value is
// made from a "builder": any type that archives as an implementor. The
// builder is serialized right away, into a small buffer of its own, so
// the same `&dyn Trait` is available before and after the value is
// stored in a larger archive. Storing it there copies the buffer.

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

impl fmt::Display for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.dialect, self.name)
    }
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
/// SAFETY: no dialect of the trait object the type is named in may map
/// the returned name to a different type.
pub unsafe trait Typed {
    fn type_name(&self) -> TypeName;
}

/// Implemented by `dialect!` for a type named in a dialect of the trait
/// object `T`.
///
/// SAFETY: `NAME` is what `Typed::type_name` returns, a dialect of `T`
/// maps it to this type, and `metadata` is the vtable of this type for `T`.
pub unsafe trait TypedIn<T: ?Sized + 'static>: Typed + Sized + 'static {
    const NAME: TypeName;
    fn metadata() -> DynMetadata<T>;
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
pub fn lookup_type<T: ?Sized + 'static>(name: &ArchivedTypeName) -> Option<TypeImpl<T>>
where
    Dialect<T>: inventory::Collect,
{
    let mut matches = inventory::iter::<Dialect<T>>
        .into_iter()
        .filter(|dialect| dialect.name == name.dialect())
        .flat_map(|dialect| dialect.types.iter().map(move |ty| (dialect, ty)))
        .filter(|(_, ty)| ty.name == name.name());
    let (dialect, found) = matches.next()?;
    assert!(
        matches.next().is_none(),
        "multiple types are registered as {}.{}",
        name.dialect(),
        name.name(),
    );
    Some(TypeImpl {
        name: TypeName {
            dialect: dialect.name,
            name: found.name,
        },
        metadata: (found.metadata)(),
    })
}

/// Defines a dialect of a trait object, as a mapping of names to the
/// (archived) types that implement the trait, and registers it so that
/// archives containing those types can be loaded.
///
/// ```ignore
/// dialect! {
///     dyn Operation, "math" => {
///         "add" => Add,
///         "scale" => ArchivedScale,
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
                    <$ty as $crate::util::TypedIn<$obj>>::NAME
                }
            }
            // SAFETY: as above, and the metadata is that of this type.
            unsafe impl $crate::util::TypedIn<$obj> for $ty {
                const NAME: $crate::TypeName =
                    $crate::TypeName { dialect: $dialect, name: $name };
                fn metadata() -> $crate::__private::rkyv::ptr_meta::DynMetadata<$obj> {
                    $crate::__private::rkyv::ptr_meta::metadata(
                        ::core::ptr::null::<$ty>() as *const $obj,
                    )
                }
            }
        )*

        $crate::__private::inventory::submit! {{
            const TYPES: &[$crate::DialectType<$obj>] = &[
                $($crate::DialectType::new(
                    $name,
                    <$ty as $crate::util::TypedIn<$obj>>::metadata,
                )),*
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
    Portable + Typed + Pointee<Metadata = DynMetadata<Self>> + 'static
{
    /// Looks up the implementation for an entry of an archive's type table.
    /// Called once per entry when the table is resolved, not on every access.
    fn lookup(name: &ArchivedTypeName) -> Option<TypeImpl<Self>>;
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
pub trait TypeTableContext<T: ?Sized + DynTyped> {
    /// Returns the index of `name` in the table, adding it if it is new.
    fn intern_type(&mut self, name: TypeName) -> TypeIdx;
    /// The names interned so far, in index order.
    fn interned_types(&self) -> &[TypeName];
    /// Where a value equal to `value` has been written, if anywhere.
    fn written(&self, value: &OwnedDyn<T>) -> Option<usize>;
    /// Records that `value` has been written at `pos`.
    fn set_written(&mut self, value: &OwnedDyn<T>, pos: usize);
}

/// The serializer-side state behind a `TypeTableContext`.
pub struct TypeTableBuilder<T: ?Sized + DynTyped> {
    indices: HashMap<TypeName, TypeIdx>,
    names: Vec<TypeName>,
    // Equal values are written once.
    written: HashMap<OwnedDyn<T>, usize>,
}

impl<T: ?Sized + DynTyped> Default for TypeTableBuilder<T> {
    fn default() -> Self {
        Self {
            indices: HashMap::new(),
            names: Vec::new(),
            written: HashMap::new(),
        }
    }
}

impl<T: ?Sized + DynTyped> TypeTableContext<T> for TypeTableBuilder<T> {
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

    fn written(&self, value: &OwnedDyn<T>) -> Option<usize> {
        self.written.get(value).copied()
    }

    fn set_written(&mut self, value: &OwnedDyn<T>, pos: usize) {
        self.written.insert(value.clone(), pos);
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
    T: ?Sized + DynTyped,
    S: Fallible + Allocator + Writer + TypeTableContext<T> + ?Sized,
    S::Error: Source,
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let names = serializer.interned_types().to_vec();
        let resolver = ArchivedVec::serialize_from_slice(&names, serializer)?;
        Ok((names.len(), resolver))
    }
}

/// The implementation of one type: its vtable, and its name.
pub struct TypeImpl<T: ?Sized> {
    name: TypeName,
    metadata: DynMetadata<T>,
}

impl<T: ?Sized> Clone for TypeImpl<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ?Sized> Copy for TypeImpl<T> {}

/// The implementations for a type table, resolved at load time. Entries
/// whose type is unknown are `None`.
pub struct TypeImpls<T: ?Sized>(Vec<Option<TypeImpl<T>>>);

impl<T: ?Sized> TypeImpls<T> {
    pub fn get(&self, idx: TypeIdx) -> Option<TypeImpl<T>> {
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
// Owned values
// ---------------------------------------------------------------------------

// Every buffer is aligned to this, and is copied to positions aligned
// to this, so that the alignment of everything in it is preserved.
const BLOCK_ALIGN: usize = AlignedVec::<16>::ALIGNMENT;

/// An owned, cheaply cloned value of a type that implements the trait
/// object `T`, held in archived form.
///
/// Two values are equal if they are of the same type and their archived
/// bytes are the same.
pub struct OwnedDyn<T: ?Sized + DynTyped> {
    inner: Arc<Inner<T>>,
}

struct Inner<T: ?Sized> {
    // The value, archived, at `pos`: everything it points to is in here.
    bytes: AlignedVec,
    pos: usize,
    ty: TypeImpl<T>,
}

impl<T: ?Sized + DynTyped> OwnedDyn<T> {
    /// Archives `builder`, whose archived type implements `T` and is named
    /// in a dialect of `T`.
    ///
    /// # Panics
    /// If `builder` fails to serialize.
    pub fn new<B>(builder: B) -> Self
    where
        B: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
        B::Archived: TypedIn<T>,
    {
        let bytes = rkyv::to_bytes::<rancor::Error>(&builder)
            .unwrap_or_else(|err| panic!("failed to archive {}: {err}", B::Archived::NAME));
        let pos = rkyv::api::root_position::<B::Archived>(bytes.len());
        let ty = TypeImpl {
            name: B::Archived::NAME,
            metadata: B::Archived::metadata(),
        };
        Self {
            inner: Arc::new(Inner { bytes, pos, ty }),
        }
    }

    pub fn get(&self) -> &T {
        let inner = &*self.inner;
        // SAFETY: bytes holds a valid archived value at pos, of the type
        // that the metadata is the vtable of. Past the end of an empty
        // buffer is where a zero-sized value is.
        unsafe {
            let ptr = inner.bytes.as_ptr().add(inner.pos);
            &*ptr_meta::from_raw_parts::<T>(ptr.cast(), inner.ty.metadata)
        }
    }

    /// The dialect and name of the type of the value.
    pub fn type_name(&self) -> TypeName {
        self.inner.ty.name
    }

    /// The value, if it is of the type `A`.
    pub fn downcast<A: TypedIn<T>>(&self) -> Option<&A> {
        // SAFETY: a name identifies one type among the dialects of T
        (self.type_name() == A::NAME).then(|| unsafe { &*(self.get() as *const T).cast::<A>() })
    }

    pub fn is<A: TypedIn<T>>(&self) -> bool {
        self.type_name() == A::NAME
    }

    /// The archived bytes of the value and everything it points to.
    pub fn bytes(&self) -> &[u8] {
        &self.inner.bytes
    }

    /// Whether these are the same allocation.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T: ?Sized + DynTyped> Clone for OwnedDyn<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: ?Sized + DynTyped> Deref for OwnedDyn<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.get()
    }
}

impl<T: ?Sized + DynTyped> PartialEq for OwnedDyn<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other)
            || (self.type_name() == other.type_name()
                && self.inner.pos == other.inner.pos
                && self.bytes() == other.bytes())
    }
}
impl<T: ?Sized + DynTyped> Eq for OwnedDyn<T> {}

impl<T: ?Sized + DynTyped> Hash for OwnedDyn<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_name().hash(state);
        self.bytes().hash(state);
    }
}

impl<T: ?Sized + DynTyped> fmt::Debug for OwnedDyn<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.type_name(), f)
    }
}

// ---------------------------------------------------------------------------
// Archived values
// ---------------------------------------------------------------------------

/// The archived form of an `OwnedDyn`: where the value is, the index of
/// its type in the archive's type table for `T`, and the extent of the
/// bytes that were copied in for it.
#[derive(Portable)]
#[repr(C)]
pub struct ArchivedDyn<T: ?Sized> {
    ptr: RawRelPtr,
    type_idx: TypeIdx,
    // The value's bytes start this far before the value,
    block_pos: u32_le,
    // and are this long.
    block_len: u32_le,
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
        let ty = impls.get(self.type_idx)?;
        // SAFETY: the pointer targets a value of the concrete type that
        // the metadata describes, per the `DynTyped` contract.
        unsafe {
            let ptr: *const T = ptr_meta::from_raw_parts(self.ptr.as_ptr().cast(), ty.metadata);
            Some(&*ptr)
        }
    }

    /// The bytes of the value and everything it points to.
    ///
    /// SAFETY: the archive this value is in must be valid.
    pub unsafe fn block(&self) -> &[u8] {
        let (pos, len) = (self.block_pos.to_native(), self.block_len.to_native());
        // SAFETY: these are the bytes serializing an OwnedDyn wrote
        unsafe {
            let start = self.ptr.as_ptr().cast::<u8>().sub(pos as usize);
            core::slice::from_raw_parts(start, len as usize)
        }
    }
}

pub struct DynResolver {
    block: usize,
    pos: usize,
    len: usize,
    type_idx: TypeIdx,
}

impl<T: ?Sized + DynTyped> Archive for OwnedDyn<T> {
    type Archived = ArchivedDyn<T>;
    type Resolver = DynResolver;

    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        munge!(let ArchivedDyn { ptr, type_idx: TypeIdx(type_idx), block_pos, block_len, _marker: _ } = out);
        RawRelPtr::emplace(resolver.block + resolver.pos, ptr);
        type_idx.write(resolver.type_idx.0);
        let small = |n: usize| u32::try_from(n).expect("value exceeds u32::MAX bytes");
        block_pos.write(small(resolver.pos).into());
        block_len.write(small(resolver.len).into());
    }
}

impl<T, S> Serialize<S> for OwnedDyn<T>
where
    T: ?Sized + DynTyped,
    S: Fallible + Writer + TypeTableContext<T> + ?Sized,
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let type_idx = serializer.intern_type(self.type_name());
        let block = match serializer.written(self) {
            Some(block) => block,
            None => {
                // Relative pointers within the bytes stay valid wherever
                // the bytes are copied to, given the same alignment.
                let block = serializer.align(BLOCK_ALIGN)?;
                serializer.write(self.bytes())?;
                serializer.set_written(self, block);
                block
            }
        };
        Ok(DynResolver {
            block,
            pos: self.inner.pos,
            len: self.bytes().len(),
            type_idx,
        })
    }
}

/// A value of a type that implements the trait object `T`, in an archive.
pub struct DynRef<'a, T: ?Sized + DynTyped> {
    impls: &'a TypeImpls<T>,
    types: &'a ArchivedTypeTable<T>,
    value: &'a ArchivedDyn<T>,
}

impl<T: ?Sized + DynTyped> Clone for DynRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ?Sized + DynTyped> Copy for DynRef<'_, T> {}

impl<'a, T: ?Sized + DynTyped> DynRef<'a, T> {
    /// SAFETY: value and types must be part of the same valid archive, and
    /// impls resolved from types, with an implementation for every type.
    pub unsafe fn new(
        impls: &'a TypeImpls<T>,
        types: &'a ArchivedTypeTable<T>,
        value: &'a ArchivedDyn<T>,
    ) -> Self {
        Self {
            impls,
            types,
            value,
        }
    }

    fn ty(self) -> TypeImpl<T> {
        (self.impls.get(self.value.type_idx))
            .expect("every type in the archive has an implementation")
    }

    /// The value. Unlike `Deref`, the result outlives this reference.
    pub fn get(self) -> &'a T {
        // SAFETY: guaranteed by the caller of new
        unsafe { self.value.get(self.impls) }
            .expect("every type in the archive has an implementation")
    }

    /// The dialect and name of the type of the value.
    pub fn type_name(self) -> TypeName {
        self.ty().name
    }

    /// The dialect and name of the type of the value, as archived.
    pub fn archived_type_name(self) -> &'a ArchivedTypeName {
        &self.types.names()[usize::from(self.value.type_idx)]
    }

    /// The value, if it is of the type `A`.
    pub fn downcast<A: TypedIn<T>>(self) -> Option<&'a A> {
        // SAFETY: a name identifies one type among the dialects of T
        (self.type_name() == A::NAME).then(|| unsafe { &*(self.get() as *const T).cast::<A>() })
    }

    pub fn is<A: TypedIn<T>>(self) -> bool {
        self.type_name() == A::NAME
    }

    fn block(self) -> &'a [u8] {
        // SAFETY: guaranteed by the caller of new
        unsafe { self.value.block() }
    }

    /// Copies the value out of the archive.
    pub fn to_owned(self) -> OwnedDyn<T> {
        let block = self.block();
        let mut bytes = AlignedVec::with_capacity(block.len());
        bytes.extend_from_slice(block);
        OwnedDyn {
            inner: Arc::new(Inner {
                bytes,
                pos: self.value.block_pos.to_native() as usize,
                ty: self.ty(),
            }),
        }
    }
}

impl<'a, T: ?Sized + DynTyped> Deref for DynRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.get()
    }
}

impl<T: ?Sized + DynTyped> PartialEq for DynRef<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.type_name() == other.type_name()
            && self.value.block_pos == other.value.block_pos
            && self.block() == other.block()
    }
}
impl<T: ?Sized + DynTyped> Eq for DynRef<'_, T> {}

impl<T: ?Sized + DynTyped> PartialEq<OwnedDyn<T>> for DynRef<'_, T> {
    fn eq(&self, other: &OwnedDyn<T>) -> bool {
        self.type_name() == other.type_name()
            && self.value.block_pos.to_native() as usize == other.inner.pos
            && self.block() == other.bytes()
    }
}

impl<T: ?Sized + DynTyped> PartialEq<DynRef<'_, T>> for OwnedDyn<T> {
    fn eq(&self, other: &DynRef<'_, T>) -> bool {
        other == self
    }
}

impl<T: ?Sized + DynTyped> Hash for DynRef<'_, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_name().hash(state);
        self.block().hash(state);
    }
}

impl<T: ?Sized + DynTyped> fmt::Debug for DynRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.type_name(), f)
    }
}

/// Wrap a trait definition to make values of `dyn Trait` archivable.
///
/// ```ignore
/// dyn_archive! {
///     pub trait Operation { ... }
/// }
/// ```
///
/// `OwnedDyn<dyn Trait>` then implements `Archive`/`Serialize`, archiving
/// as `ArchivedDyn<dyn Trait>`, and `dyn Trait` gets its own dialect
/// registry. Implementors of the trait are archived types, named in a
/// `dialect!`. They can be shared between threads.
macro_rules! dyn_archive {
    ($(#[$meta:meta])* $vis:vis trait $tr:ident { $($body:tt)* }) => {
        $(#[$meta])*
        #[::rkyv::ptr_meta::pointee(crate = ::rkyv::ptr_meta)]
        $vis trait $tr:
            ::rkyv::Portable + $crate::util::Typed + Send + Sync + 'static
        {
            $($body)*
        }

        // The dialects of this trait object
        ::inventory::collect!($crate::util::Dialect<dyn $tr>);

        // SAFETY: dialects map each name to the type that reports it.
        unsafe impl $crate::util::DynTyped for dyn $tr {
            fn lookup(
                name: &$crate::util::ArchivedTypeName,
            ) -> Option<$crate::util::TypeImpl<Self>> {
                $crate::util::lookup_type::<Self>(name)
            }
        }
    };
}

pub(crate) use dyn_archive;

#[cfg(test)]
mod tests {
    use rkyv::rancor::{Error, Strategy};
    use rkyv::ser::Serializer;

    use super::*;

    dyn_archive! {
        trait Shape {
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

    // A builder that archives as another type, with data out of line.
    #[derive(Archive, Serialize)]
    #[rkyv(archived = PolygonData)]
    struct Polygon {
        sides: Vec<u32>,
        label: String,
    }
    impl Shape for PolygonData {
        fn describe(&self) -> String {
            format!("{}{:?}", self.label, self.sides.as_slice())
        }
    }

    // Zero-sized
    #[derive(Archive, Serialize, Portable)]
    #[rkyv(as = Self)]
    #[repr(C)]
    struct Point;
    impl Shape for Point {
        fn describe(&self) -> String {
            "point".into()
        }
    }

    crate::dialect! {
        dyn Shape, "shapes" => {
            "square" => Square,
            "polygon" => PolygonData,
            "point" => Point,
        }
    }

    fn polygon(sides: &[u32]) -> OwnedDyn<dyn Shape> {
        OwnedDyn::new(Polygon {
            sides: sides.to_vec(),
            label: "a polygon with a long label, stored out of line ".into(),
        })
    }

    // ---- a serializer with a type table ----

    #[derive(Default)]
    struct Tables {
        shapes: TypeTableBuilder<dyn Shape>,
    }

    impl<W, A, E> TypeTableContext<dyn Shape> for Strategy<Serializer<W, A, Tables>, E> {
        fn intern_type(&mut self, name: TypeName) -> TypeIdx {
            self.sharing.shapes.intern_type(name)
        }
        fn interned_types(&self) -> &[TypeName] {
            self.sharing.shapes.interned_types()
        }
        fn written(&self, value: &OwnedDyn<dyn Shape>) -> Option<usize> {
            self.sharing.shapes.written(value)
        }
        fn set_written(&mut self, value: &OwnedDyn<dyn Shape>, pos: usize) {
            self.sharing.shapes.set_written(value, pos)
        }
    }

    #[derive(Archive, Serialize)]
    #[rkyv(archived = HolderData)]
    struct Holder {
        // Shifts what follows off its alignment
        pad: u8,
        first: OwnedDyn<dyn Shape>,
        rest: Vec<OwnedDyn<dyn Shape>>,
        shapes: TypeTable<dyn Shape>,
    }

    fn to_bytes(value: &Holder) -> AlignedVec {
        rkyv::util::with_arena(|arena| {
            let mut serializer =
                Serializer::new(AlignedVec::new(), arena.acquire(), Tables::default());
            rkyv::api::serialize_using::<_, Error>(value, &mut serializer).unwrap();
            serializer.writer
        })
    }

    #[test]
    fn owned_values() {
        let square: OwnedDyn<dyn Shape> = OwnedDyn::new(Square(4.into()));
        assert_eq!(square.describe(), "square(4)");
        assert_eq!(square.type_name().to_string(), "shapes::square");
        assert_eq!(square.downcast::<Square>().unwrap().0, 4);
        assert!(square.downcast::<Point>().is_none());

        let poly = polygon(&[3, 4, 5]);
        assert!(poly.describe().ends_with("[3, 4, 5]"));
        assert_eq!(poly.downcast::<PolygonData>().unwrap().sides.len(), 3);

        let point: OwnedDyn<dyn Shape> = OwnedDyn::new(Point);
        assert_eq!(point.describe(), "point");

        // Equality is by type and contents.
        assert!(poly == polygon(&[3, 4, 5]));
        assert!(poly != polygon(&[3, 4]));
        assert!(square != point);
    }

    #[test]
    fn round_trip() {
        let holder = Holder {
            pad: 1,
            first: polygon(&[1, 2, 3]),
            rest: vec![
                OwnedDyn::new(Square(1.into())),
                polygon(&[9]),
                OwnedDyn::new(Point),
                polygon(&[1, 2, 3]),
            ],
            shapes: TypeTable::default(),
        };
        let bytes = to_bytes(&holder);
        let archived = unsafe { rkyv::access_unchecked::<HolderData>(&bytes) };

        // One table entry per distinct type, in first-use order.
        let names: Vec<_> = (archived.shapes.names().iter())
            .map(|n| (n.dialect(), n.name()))
            .collect();
        assert_eq!(
            names,
            [
                ("shapes", "polygon"),
                ("shapes", "square"),
                ("shapes", "point")
            ]
        );

        let impls = archived.shapes.resolve();
        assert_eq!(impls.missing().count(), 0);
        let view = |shape| unsafe { DynRef::new(&impls, &archived.shapes, shape) };
        let first = view(&archived.first);
        assert!(first.get().describe().ends_with("[1, 2, 3]"));
        assert!(first.downcast::<PolygonData>().is_some());
        assert!(first.downcast::<Square>().is_none());
        let rest: Vec<_> = archived.rest.iter().map(view).collect();
        assert_eq!(rest[0].describe(), "square(1)");
        assert!(rest[1].describe().ends_with("[9]"));
        assert_eq!(rest[2].describe(), "point");

        // Equal values are written once.
        assert!(std::ptr::eq(first.get(), rest[3].get()));
        assert!(first == rest[3]);
        assert!(first != rest[1]);

        // Values can be copied back out.
        for (view, owned) in rest.iter().zip(&holder.rest) {
            assert!(view == owned);
            let copy = view.to_owned();
            assert!(&copy == owned);
            assert_eq!(copy.describe(), owned.describe());
        }
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
