// What Expr and HExpr have in common: a root is an archive of one
// expression and every expression it refers to, identified by its hash.
//
// This module is generic over the kind of expression, by way of its
// builder B, which is also the unarchived form of its data.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use rkyv::{
    Archive, Place, Portable, Serialize,
    bytecheck::{self, CheckBytes},
    munge::munge,
    rancor::{self, Fallible, Source, Strategy},
    rend::i32_le,
    ser::{Allocator, Serializer, Writer, WriterExt, allocator::ArenaHandle},
    util::{AlignedVec, with_arena},
    vec::{ArchivedVec, VecResolver},
};

use crate::check::BuildError;
use crate::op::{OpData, OpRef, OpTypeCtx, OpTypeImpls, OpTypeTable, Operation};
use crate::util::{
    ArchivedTypeTable, Blake3Hash, DynTyped, OwnedArchive, OwnedDyn, SharedBytes, TypeIdx,
    TypeImpls, TypeName, TypeTableContext,
};
use crate::var::{Info, InfoTypeCtx, InfoTypeImpls, InfoTypeTable, VarInfoData, VarInfoRef};

/// Implemented by the builders of the kinds of expression a root can hold:
/// `ExprBuilder` and `HExprBuilder`.
//
// Not a subtrait of Archive: the builders are only archivable given that
// references to them are, which in turn requires this trait.
pub trait ExprSource: Sized {
    /// The archived form of an expression of this kind.
    type Data: Portable;

    /// A view of an archived expression of this kind.
    type View<'a>: Copy
    where
        Self: 'a;

    /// SAFETY: data must be part of the data of root.
    #[doc(hidden)]
    unsafe fn view<'a>(root: &'a Root<Self>, data: &'a Self::Data) -> Self::View<'a>;

    /// Checks expr and the internal expressions it refers to, skipping
    /// those already in `checked`. External expressions were checked when
    /// their root was built.
    #[doc(hidden)]
    fn check(expr: &Rc<Self>, checked: &mut HashSet<*const Self>) -> Result<(), BuildError>;
}

// ---------------------------------------------------------------------------
// Roots and handles
// ---------------------------------------------------------------------------

/// A root: an immutable archive of one expression and every expression it
/// refers to. Every operation and info in its data has an implementation:
/// views rely on this.
pub struct Root<B: ExprSource> {
    // An owned handle of the underlying root data
    data: OwnedArchive<RootData<B>>,
    hash: Blake3Hash,
    // The associated operation lookup table, resolved
    op_impls: OpTypeImpls,
    // The associated info lookup table, resolved
    info_impls: InfoTypeImpls,
    // The external expressions, by slot
    external: Vec<Handle<B>>,
}

impl<B: ExprSource> Root<B> {
    /// The hash of the archived data, which identifies this root.
    pub fn hash(&self) -> Blake3Hash {
        self.hash
    }

    pub fn view(&self) -> B::View<'_> {
        self.expr_at(self.data.root)
    }

    // Get a handle to the root expr in self
    pub fn handle(self: Arc<Self>) -> Handle<B> {
        let idx = self.data.root;
        Handle { root: self, idx }
    }

    // The expression an index in the data of this root refers to
    pub(crate) fn expr_at(&self, idx: ExprIdx) -> B::View<'_> {
        let slot: usize = idx.into();
        if idx.is_internal() {
            // SAFETY: the expression is part of the data of self
            unsafe { B::view(self, &self.data.exprs.internal[slot]) }
        } else {
            self.external[slot].view()
        }
    }

    // The handle an index in the data of this root refers to, if the
    // index is that of an external expression
    pub(crate) fn external_at(&self, idx: ExprIdx) -> Option<&Handle<B>> {
        (!idx.is_internal()).then(|| &self.external[usize::from(idx)])
    }

    // SAFETY: op must be part of the data of self
    pub(crate) unsafe fn op<'a>(&'a self, op: &'a OpData) -> OpRef<'a> {
        // SAFETY: op_impls was resolved from op_types
        unsafe { OpRef::new(&self.op_impls, &self.data.op_types, op) }
    }

    // SAFETY: info must be part of the data of self
    pub(crate) unsafe fn info<'a>(&'a self, info: &'a VarInfoData) -> VarInfoRef<'a> {
        // SAFETY: info_impls was resolved from info_types
        unsafe { VarInfoRef::new(&self.info_impls, &self.data.info_types, info) }
    }
}

#[rustfmt::skip]
impl<B: ExprSource> Hash for Root<B> {
    fn hash<H: Hasher>(&self, state: &mut H) { self.hash.hash(state); }
}
#[rustfmt::skip]
impl<B: ExprSource> PartialEq for Root<B> {
    fn eq(&self, other: &Self) -> bool { self.hash == other.hash }
}
impl<B: ExprSource> Eq for Root<B> {}
#[rustfmt::skip]
impl<B: ExprSource> PartialOrd for Root<B> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
#[rustfmt::skip]
impl<B: ExprSource> Ord for Root<B> {
    fn cmp(&self, other: &Self) -> Ordering { self.hash.cmp(&other.hash) }
}

impl<B: ExprSource> fmt::Debug for Root<B>
where
    for<'a> B::View<'a>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Root")
            .field("hash", &format_args!("{}", self.hash))
            .field("expr", &self.view())
            .finish()
    }
}

/// A handle to an expression of a root: a combination of an index into
/// the expr table + an arc to the underlying root
pub struct Handle<B: ExprSource> {
    root: Arc<Root<B>>,
    // Always an internal expression of root
    idx: ExprIdx,
}

impl<B: ExprSource> Handle<B> {
    pub fn view(&self) -> B::View<'_> {
        self.root.expr_at(self.idx)
    }
    pub fn root(&self) -> &Arc<Root<B>> {
        &self.root
    }
}

impl<B: ExprSource> Handle<B> {
    fn key(&self) -> (Blake3Hash, ExprIdx) {
        (self.root.hash, self.idx)
    }
}

#[rustfmt::skip]
impl<B: ExprSource> Clone for Handle<B> {
    fn clone(&self) -> Self { Self { root: self.root.clone(), idx: self.idx } }
}
#[rustfmt::skip]
impl<B: ExprSource> Hash for Handle<B> {
    fn hash<H: Hasher>(&self, state: &mut H) { self.key().hash(state); }
}
#[rustfmt::skip]
impl<B: ExprSource> PartialEq for Handle<B> {
    fn eq(&self, other: &Self) -> bool { self.key() == other.key() }
}
impl<B: ExprSource> Eq for Handle<B> {}
#[rustfmt::skip]
impl<B: ExprSource> PartialOrd for Handle<B> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
#[rustfmt::skip]
impl<B: ExprSource> Ord for Handle<B> {
    fn cmp(&self, other: &Self) -> Ordering { self.key().cmp(&other.key()) }
}

impl<B: ExprSource> fmt::Debug for Handle<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Handle({}, {:?})", self.root.hash, self.idx)
    }
}

// ---------------------------------------------------------------------------
// References between expressions
// ---------------------------------------------------------------------------

/// A reference to an expression: one that is being built, and will be
/// stored in the same root, or one in a root that is already built.
pub enum Ref<B: ExprSource> {
    Internal(Rc<B>),
    External(Handle<B>),
}

impl<B: ExprSource> Clone for Ref<B> {
    fn clone(&self) -> Self {
        match self {
            Self::Internal(expr) => Self::Internal(expr.clone()),
            Self::External(handle) => Self::External(handle.clone()),
        }
    }
}

impl<B: ExprSource + fmt::Debug> fmt::Debug for Ref<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Internal(expr) => expr.fmt(f),
            Self::External(handle) => handle.fmt(f),
        }
    }
}

impl<B: ExprSource> From<B> for Ref<B> {
    fn from(expr: B) -> Self {
        Self::Internal(Rc::new(expr))
    }
}

impl<B: ExprSource> From<Handle<B>> for Ref<B> {
    fn from(handle: Handle<B>) -> Self {
        Self::External(handle)
    }
}

// The archived form of a Ref: an index into the internal or
// the external expressions of the root.
#[derive(
    Clone, Copy, Archive, Serialize, Portable, CheckBytes, Hash, PartialEq, Eq, PartialOrd, Ord,
)]
#[rkyv(as = Self)]
#[bytecheck(crate = bytecheck)]
#[repr(transparent)]
pub struct ExprIdx(i32_le);

impl ExprIdx {
    fn internal(slot: usize) -> Self {
        let slot = i32::try_from(slot).expect("internal expression index exceeds i32::MAX");
        Self(slot.into())
    }

    // External slots 0, 1, ... are encoded as -1, -2, ... .
    fn external(slot: usize) -> Self {
        let slot = i32::try_from(slot).expect("external expression index exceeds i32::MAX");
        Self((!slot).into())
    }

    pub fn is_internal(&self) -> bool {
        self.0 >= 0
    }
}
impl From<ExprIdx> for usize {
    fn from(idx: ExprIdx) -> usize {
        let raw = idx.0.to_native();
        (if raw < 0 { !raw } else { raw }) as usize
    }
}

impl fmt::Debug for ExprIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_internal() {
            "internal"
        } else {
            "external"
        };
        write!(f, "{kind}:{}", usize::from(*self))
    }
}

/// Implemented by serializers that collect the expressions of a root.
pub trait ExprRefContext<B: ExprSource> {
    /// Returns the index of `expr`, adding it if it is new.
    fn intern_expr(&mut self, expr: &Ref<B>) -> ExprIdx;
    /// The internal expression at `slot`, if one has been interned.
    fn internal_expr(&self, slot: usize) -> Option<Rc<B>>;
    /// The external expressions interned so far, in slot order.
    fn external_exprs(&self) -> &[Handle<B>];
}

impl<B: ExprSource> Archive for Ref<B> {
    type Archived = ExprIdx;
    type Resolver = ExprIdx;

    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        munge!(let ExprIdx(index) = out);
        index.write(resolver.0);
    }
}

impl<B: ExprSource, S: Fallible + ExprRefContext<B> + ?Sized> Serialize<S> for Ref<B> {
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        Ok(serializer.intern_expr(self))
    }
}

pub struct ExprRefCtx<B: ExprSource> {
    // Internal expressions are identified by address. Each is kept
    // alive by `internal`, so an address cannot be reused.
    internal_ids: BTreeMap<*const B, ExprIdx>,
    internal: Vec<Rc<B>>,
    external_ids: BTreeMap<Handle<B>, ExprIdx>,
    external: Vec<Handle<B>>,
}

impl<B: ExprSource> Default for ExprRefCtx<B> {
    fn default() -> Self {
        Self {
            internal_ids: BTreeMap::new(),
            internal: Vec::new(),
            external_ids: BTreeMap::new(),
            external: Vec::new(),
        }
    }
}

impl<B: ExprSource> ExprRefContext<B> for ExprRefCtx<B> {
    fn intern_expr(&mut self, expr: &Ref<B>) -> ExprIdx {
        match expr {
            Ref::External(handle) => {
                if let Some(&idx) = self.external_ids.get(handle) {
                    return idx;
                }
                let idx = ExprIdx::external(self.external.len());
                self.external.push(handle.clone());
                self.external_ids.insert(handle.clone(), idx);
                idx
            }
            Ref::Internal(handle) => {
                if let Some(&idx) = self.internal_ids.get(&Rc::as_ptr(handle)) {
                    return idx;
                }
                let idx = ExprIdx::internal(self.internal.len());
                self.internal.push(handle.clone());
                self.internal_ids.insert(Rc::as_ptr(handle), idx);
                idx
            }
        }
    }
    fn internal_expr(&self, slot: usize) -> Option<Rc<B>> {
        self.internal.get(slot).cloned()
    }
    fn external_exprs(&self) -> &[Handle<B>] {
        &self.external
    }
}

// ---------------------------------------------------------------------------
// The archived data
// ---------------------------------------------------------------------------

// The unarchived form of a root.
#[derive(Archive, Serialize)]
#[rkyv(archived = RootData)]
pub struct RootSource<B: ExprSource> {
    // Interned first, so the root is always internal expression 0.
    root: Ref<B>,
    exprs: ExprsTable<B>,
    // The type tables are written from what has been interned
    // so far, so they must stay after exprs.
    op_types: OpTypeTable,
    info_types: InfoTypeTable,
}

// An external expression: an expression of another root.
#[derive(Archive, Serialize, Portable, Clone, Copy, Debug)]
#[rkyv(as = Self)]
#[repr(C)]
pub struct ExtExprData {
    pub root_hash: Blake3Hash,
    // The index of the expression within its own root
    pub idx: ExprIdx,
}

// The expressions of a root. Internal expressions are stored by data,
// external expressions by hash reference.
//
// The unarchived version is a zero-sized placeholder: serializing it
// writes out the expressions the serializer has interned, as well as
// every expression those refer to.
pub struct ExprsTable<B>(PhantomData<fn() -> B>);

#[derive(Portable)]
#[repr(C)]
pub struct ExprsTableData<B: ExprSource> {
    internal: ArchivedVec<B::Data>,
    external: ArchivedVec<ExtExprData>,
}

pub struct ExprsTableResolver {
    internal: (usize, VecResolver),
    external: (usize, VecResolver),
}

impl<B: ExprSource> Archive for ExprsTable<B> {
    type Archived = ExprsTableData<B>;
    type Resolver = ExprsTableResolver;

    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        munge!(let ExprsTableData { internal, external } = out);
        let (len, vec) = resolver.internal;
        ArchivedVec::resolve_from_len(len, vec, internal);
        let (len, vec) = resolver.external;
        ArchivedVec::resolve_from_len(len, vec, external);
    }
}

impl<B, S> Serialize<S> for ExprsTable<B>
where
    B: ExprSource + Archive<Archived = B::Data> + Serialize<S>,
    S: Fallible + Allocator + Writer + ExprRefContext<B> + ?Sized,
    S::Error: Source,
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        // Serializing an expression interns the expressions it refers to,
        // so keep going until no new ones turn up.
        let mut exprs = Vec::new();
        while let Some(expr) = serializer.internal_expr(exprs.len()) {
            let resolver = expr.serialize(serializer)?;
            exprs.push((expr, resolver));
        }

        // Everything the expressions point to is written: now write the
        // expressions themselves, next to each other.
        let len = exprs.len();
        let pos = serializer.align_for::<B::Archived>()?;
        for (expr, resolver) in exprs {
            // SAFETY: resolver is the result of serializing expr. The
            // serializer was aligned for a B::Archived above, and writing
            // one leaves it aligned for the next.
            unsafe { serializer.resolve_aligned(&*expr, resolver)? };
        }
        let internal = (len, VecResolver::from_pos(pos));

        let external: Vec<ExtExprData> = (serializer.external_exprs().iter())
            .map(|handle| ExtExprData {
                root_hash: handle.root.hash,
                idx: handle.idx,
            })
            .collect();
        let resolver = ArchivedVec::serialize_from_slice(&external, serializer)?;
        Ok(ExprsTableResolver {
            internal,
            external: (external.len(), resolver),
        })
    }
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

// The serializer state for building a root.
pub struct BuildCtx<B: ExprSource> {
    exprs: ExprRefCtx<B>,
    ops: OpTypeCtx,
    infos: InfoTypeCtx,
}

pub type BuildSerializer<'a, B> =
    Strategy<Serializer<AlignedVec, ArenaHandle<'a>, BuildCtx<B>>, rancor::Error>;

impl<B: ExprSource, W, A, E> ExprRefContext<B> for Strategy<Serializer<W, A, BuildCtx<B>>, E> {
    fn intern_expr(&mut self, expr: &Ref<B>) -> ExprIdx {
        self.sharing.exprs.intern_expr(expr)
    }
    fn internal_expr(&self, slot: usize) -> Option<Rc<B>> {
        self.sharing.exprs.internal_expr(slot)
    }
    fn external_exprs(&self) -> &[Handle<B>] {
        self.sharing.exprs.external_exprs()
    }
}

impl<B: ExprSource, W, A, E> TypeTableContext<dyn Operation>
    for Strategy<Serializer<W, A, BuildCtx<B>>, E>
{
    fn intern_type(&mut self, name: TypeName) -> TypeIdx {
        self.sharing.ops.intern_type(name)
    }
    fn interned_types(&self) -> &[TypeName] {
        self.sharing.ops.interned_types()
    }
    fn written(&self, value: &OwnedDyn<dyn Operation>) -> Option<usize> {
        self.sharing.ops.written(value)
    }
    fn set_written(&mut self, value: &OwnedDyn<dyn Operation>, pos: usize) {
        self.sharing.ops.set_written(value, pos)
    }
}

impl<B: ExprSource, W, A, E> TypeTableContext<dyn Info>
    for Strategy<Serializer<W, A, BuildCtx<B>>, E>
{
    fn intern_type(&mut self, name: TypeName) -> TypeIdx {
        self.sharing.infos.intern_type(name)
    }
    fn interned_types(&self) -> &[TypeName] {
        self.sharing.infos.interned_types()
    }
    fn written(&self, value: &OwnedDyn<dyn Info>) -> Option<usize> {
        self.sharing.infos.written(value)
    }
    fn set_written(&mut self, value: &OwnedDyn<dyn Info>, pos: usize) {
        self.sharing.infos.set_written(value, pos)
    }
}

// Checks expr and everything it refers to, and builds a root of it.
pub(crate) fn build<B>(expr: B) -> Result<Arc<Root<B>>, BuildError>
where
    B: ExprSource,
    RootSource<B>: for<'a> Serialize<BuildSerializer<'a, B>>,
{
    let root = Rc::new(expr);
    B::check(&root, &mut HashSet::new())?;

    let source = RootSource {
        root: Ref::Internal(root),
        exprs: ExprsTable(PhantomData),
        op_types: OpTypeTable::default(),
        info_types: InfoTypeTable::default(),
    };
    let (bytes, pos, external) = with_arena(|arena| {
        let ctx = BuildCtx {
            exprs: ExprRefCtx::default(),
            ops: OpTypeCtx::default(),
            infos: InfoTypeCtx::default(),
        };
        let mut serializer = Serializer::new(AlignedVec::new(), arena.acquire(), ctx);
        let pos = rkyv::api::serialize_using::<_, rancor::Error>(&source, &mut serializer)?;
        Ok((serializer.writer, pos, serializer.sharing.exprs.external))
    })
    .map_err(BuildError::Serialize)?;

    let hash = Blake3Hash::of(&bytes);
    // SAFETY: bytes was just written by serializing a RootSource,
    // which archives as a RootData at pos.
    let data = unsafe { OwnedArchive::<RootData<B>>::new_unchecked(bytes, pos) };
    let op_impls = resolve_types(&data.op_types)?;
    let info_impls = resolve_types(&data.info_types)?;

    Ok(Arc::new(Root {
        data,
        hash,
        op_impls,
        info_impls,
        external,
    }))
}

// Resolves a type table, requiring every entry to have an implementation.
fn resolve_types<T>(table: &ArchivedTypeTable<T>) -> Result<TypeImpls<T>, BuildError>
where
    T: DynTyped + ?Sized,
{
    let impls = table.resolve();
    let missing = impls.missing().next();
    match missing {
        None => Ok(impls),
        Some(idx) => {
            let name = &table.names()[idx];
            Err(BuildError::UnknownType {
                dialect: name.dialect().into(),
                name: name.name().into(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Storing and loading
// ---------------------------------------------------------------------------

impl<B: ExprSource> Root<B> {
    /// The archived data, whose hash identifies this root. Storing these
    /// bytes is enough to load the root again, given the roots it refers to.
    pub fn bytes(&self) -> &[u8] {
        self.data.bytes()
    }

    /// The other roots this root refers to expressions of.
    pub fn external_roots(&self) -> impl Iterator<Item = &Arc<Root<B>>> {
        self.external.iter().map(|handle| &handle.root)
    }
}

/// The stored data of a root, before the roots it refers to are known.
pub struct UnresolvedRoot<B: ExprSource> {
    data: OwnedArchive<RootData<B>>,
    hash: Blake3Hash,
}

impl<B: ExprSource> UnresolvedRoot<B> {
    /// # Safety
    /// `bytes` must be the bytes of a root of this kind of expression (see
    /// [`Root::bytes`]), written by a compatible build, and `hash` their
    /// hash. Nothing is validated.
    ///
    /// # Panics
    /// If the bytes are not aligned (see `OwnedArchive::from_shared`).
    pub unsafe fn new(bytes: SharedBytes, hash: Blake3Hash) -> Self {
        let len = (*bytes).as_ref().len();
        // The root data is what a root's serializer writes last.
        let pos = rkyv::api::root_position::<RootData<B>>(len);
        // SAFETY: the caller certifies that bytes holds a root.
        let data = unsafe { OwnedArchive::from_shared(bytes, pos) };
        Self { data, hash }
    }

    pub fn hash(&self) -> Blake3Hash {
        self.hash
    }

    /// The hashes of the roots this root refers to, which
    /// [`UnresolvedRoot::resolve`] needs.
    pub fn external_hashes(&self) -> impl Iterator<Item = Blake3Hash> + '_ {
        self.data.exprs.external.iter().map(|ext| ext.root_hash)
    }

    /// Completes the root, looking up the roots it refers to by hash and
    /// the implementations of its operations and infos in the dialects.
    pub fn resolve(
        self,
        mut external: impl FnMut(Blake3Hash) -> Option<Arc<Root<B>>>,
    ) -> Result<Arc<Root<B>>, BuildError> {
        let external = (self.data.exprs.external.iter())
            .map(|ext| {
                let root = external(ext.root_hash).ok_or(BuildError::UnknownRoot(ext.root_hash))?;
                Ok(Handle { root, idx: ext.idx })
            })
            .collect::<Result<_, BuildError>>()?;
        let op_impls = resolve_types(&self.data.op_types)?;
        let info_impls = resolve_types(&self.data.info_types)?;
        Ok(Arc::new(Root {
            data: self.data,
            hash: self.hash,
            op_impls,
            info_impls,
            external,
        }))
    }
}
