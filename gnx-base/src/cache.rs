//! A content-addressed cache of archived values over an object [`Store`].
//!
//! A blob is stored under the hash of its bytes, so a value is stored once
//! however often it is put. Values are read back without being copied or
//! deserialized: a [`Blob`] gives access to the archived form in place. When
//! the store hands out local files (a local store, or one behind
//! [`Store::mirrored`]) the bytes are memory-mapped.
//!
//! Something that is not known by its content, such as "the result of this
//! pipeline on that expression", is found through a binding: see
//! [`Cache::bind`] and [`Cache::resolve`].
//!
//! Nothing read from the store is validated. A store must only hold what
//! compatible builds of this library wrote to it.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::{Arc, Mutex, Weak};

use anyhash::{Hasher as _, HasherLe, HasherWrite};
use gnx_expr::{
    Blake3Hash, BuildError, ExprSource, OwnedArchive, Root, SharedBytes, UnresolvedRoot,
};
use object_store::{GetResultPayload, ObjectStoreExt, PutMode, PutPayload};
use rkyv::api::high::HighSerializer;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Portable, Serialize, rancor};

use crate::store::{Path, Store};

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("cache store: {0}")]
    Store(#[from] object_store::Error),
    #[error("cache I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("blob {0} is not in the cache")]
    Missing(Blake3Hash),
    #[error("a cache key is already bound to {existing}, not {new}")]
    Conflict {
        existing: Blake3Hash,
        new: Blake3Hash,
    },
    #[error("malformed cache binding")]
    Binding,
    #[error("cached expression: {0}")]
    Root(#[from] BuildError),
    #[error("serializing a cache entry: {0}")]
    Serialize(rancor::Error),
}

/// Identifies a stored `T` by the hash of its bytes.
pub struct Key<T: ?Sized> {
    hash: Blake3Hash,
    _type: PhantomData<fn() -> T>,
}

impl<T: ?Sized> Key<T> {
    /// # Safety
    /// The blob with this hash, if the cache holds one, must have been
    /// stored as a `T`.
    pub unsafe fn from_hash(hash: Blake3Hash) -> Self {
        Self {
            hash,
            _type: PhantomData,
        }
    }

    pub fn hash(&self) -> Blake3Hash {
        self.hash
    }
}

impl<T: ?Sized> Clone for Key<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ?Sized> Copy for Key<T> {}
impl<T: ?Sized> PartialEq for Key<T> {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}
impl<T: ?Sized> Eq for Key<T> {}
impl<T: ?Sized> fmt::Debug for Key<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Key<{}>({})", type_name::<T>(), self.hash)
    }
}

/// A stored `T`, accessed in its archived form without copying.
pub struct Blob<T: Archive>
where
    T::Archived: Portable,
{
    key: Key<T>,
    data: OwnedArchive<T::Archived>,
}

impl<T: Archive> Blob<T>
where
    T::Archived: Portable,
{
    pub fn key(&self) -> Key<T> {
        self.key
    }
}

impl<T: Archive> Deref for Blob<T>
where
    T::Archived: Portable,
{
    type Target = T::Archived;

    fn deref(&self) -> &T::Archived {
        self.data.get()
    }
}

/// A shared handle to a cache. Clones refer to the same cache.
#[derive(Clone)]
pub struct Cache(Arc<Inner>);

struct Inner {
    store: Store,
    // The roots that are loaded, so that a root is only loaded once.
    roots: Mutex<HashMap<(TypeId, Blake3Hash), Weak<dyn Any + Send + Sync>>>,
}

impl fmt::Debug for Cache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Cache").field(&self.0.store).finish()
    }
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

impl Cache {
    pub fn new(store: Store) -> Self {
        Self(Arc::new(Inner {
            store,
            roots: Mutex::default(),
        }))
    }

    /// A cache that lives in this process and is lost with it.
    pub fn in_memory() -> Self {
        Self::new(Store::in_memory())
    }

    pub fn store(&self) -> &Store {
        &self.0.store
    }

    /// Stores `value`. The result gives access to its archived form.
    pub async fn put<T>(&self, value: &T) -> Result<Blob<T>, CacheError>
    where
        T: Archive + for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
        T::Archived: Portable,
    {
        let bytes = rkyv::to_bytes::<rancor::Error>(value).map_err(CacheError::Serialize)?;
        let hash = Blake3Hash::of(&bytes);
        self.put_bytes(hash, &bytes).await?;
        // SAFETY: bytes was just written by serializing a T.
        Ok(unsafe { blob(Arc::new(bytes), hash) })
    }

    /// Reads a stored value.
    pub async fn get<T>(&self, key: Key<T>) -> Result<Blob<T>, CacheError>
    where
        T: Archive,
        T::Archived: Portable,
    {
        let bytes = self.get_bytes(key.hash).await?;
        // SAFETY: a key certifies that the blob was stored as a T.
        Ok(unsafe { blob(bytes, key.hash) })
    }

    /// Records that `key` stands for the stored value `to`. Binding a key
    /// again to the same value does nothing.
    ///
    /// # Errors
    /// [`CacheError::Conflict`] if the key is bound to another value.
    pub async fn bind<T: ?Sized>(
        &self,
        key: &impl anyhash::Hash,
        to: Key<T>,
    ) -> Result<(), CacheError> {
        let path = binding_path::<T>(key);
        let payload = PutPayload::from(to.hash.as_bytes().to_vec());
        match (self.0.store.inner())
            .put_opts(&path, payload, PutMode::Create.into())
            .await
        {
            Ok(_) => Ok(()),
            Err(object_store::Error::AlreadyExists { .. }) => {
                let existing = self.binding(&path).await?.ok_or(CacheError::Binding)?;
                if existing == to.hash {
                    Ok(())
                } else {
                    Err(CacheError::Conflict {
                        existing,
                        new: to.hash,
                    })
                }
            }
            Err(error) => Err(error.into()),
        }
    }

    /// The stored value `key` was bound to, if any.
    pub async fn resolve<T: ?Sized>(
        &self,
        key: &impl anyhash::Hash,
    ) -> Result<Option<Key<T>>, CacheError> {
        let hash = self.binding(&binding_path::<T>(key)).await?;
        // SAFETY: the binding is specific to T, and binding it took a Key<T>.
        Ok(hash.map(|hash| unsafe { Key::from_hash(hash) }))
    }

    /// Stores a root expression and every root it refers to.
    pub fn put_root<'a, B>(
        &'a self,
        root: &'a Arc<Root<B>>,
    ) -> BoxFuture<'a, Result<Key<Root<B>>, CacheError>>
    where
        B: ExprSource + 'static,
        Root<B>: Send + Sync,
    {
        Box::pin(async move {
            for external in root.external_roots() {
                self.put_root(external).await?;
            }
            self.put_bytes(root.hash(), root.bytes()).await?;
            let erased: Arc<dyn Any + Send + Sync> = root.clone();
            (self.0.roots.lock().unwrap())
                .insert((TypeId::of::<B>(), root.hash()), Arc::downgrade(&erased));
            // SAFETY: the blob with that hash holds this root.
            Ok(unsafe { Key::from_hash(root.hash()) })
        })
    }

    /// Loads a root expression, and the roots it refers to. A root that is
    /// still loaded is shared rather than loaded again.
    pub fn get_root<'a, B>(
        &'a self,
        key: Key<Root<B>>,
    ) -> BoxFuture<'a, Result<Arc<Root<B>>, CacheError>>
    where
        B: ExprSource + 'static,
        Root<B>: Send + Sync,
    {
        Box::pin(async move {
            let slot = (TypeId::of::<B>(), key.hash);
            let loaded = |cache: &Self| {
                let roots = cache.0.roots.lock().unwrap();
                let root = roots.get(&slot)?.upgrade()?;
                Some(root.downcast::<Root<B>>().expect("root of another kind"))
            };
            if let Some(root) = loaded(self) {
                return Ok(root);
            }
            let bytes = self.get_bytes(key.hash).await?;
            // SAFETY: a key certifies that the blob holds a root of this kind.
            let unresolved = || unsafe { UnresolvedRoot::<B>::new(bytes.clone(), key.hash) };
            // Not held while loading: the data of a root need not be Send.
            let hashes: Vec<_> = unresolved().external_hashes().collect();
            let mut externals = HashMap::new();
            for hash in hashes {
                if !externals.contains_key(&hash) {
                    // SAFETY: a root only refers to roots of its own kind.
                    let external = unsafe { Key::from_hash(hash) };
                    externals.insert(hash, self.get_root(external).await?);
                }
            }
            let root = unresolved().resolve(|hash| externals.get(&hash).cloned())?;
            let mut roots = self.0.roots.lock().unwrap();
            roots.retain(|_, root| root.strong_count() != 0);
            // Another task may have loaded the root in the meantime.
            if let Some(existing) = roots.get(&slot).and_then(Weak::upgrade) {
                return Ok(existing.downcast().expect("root of another kind"));
            }
            let erased: Arc<dyn Any + Send + Sync> = root.clone();
            roots.insert(slot, Arc::downgrade(&erased));
            Ok(root)
        })
    }

    async fn put_bytes(&self, hash: Blake3Hash, bytes: &[u8]) -> Result<(), CacheError> {
        let payload = PutPayload::from(bytes.to_vec());
        match (self.0.store.inner())
            .put_opts(&blob_path(hash), payload, PutMode::Create.into())
            .await
        {
            // The same bytes are stored already.
            Ok(_) | Err(object_store::Error::AlreadyExists { .. }) => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    // The bytes of a blob, aligned for an archive: mapped if the store
    // keeps them in a local file.
    async fn get_bytes(&self, hash: Blake3Hash) -> Result<SharedBytes, CacheError> {
        let result = match self.0.store.inner().get(&blob_path(hash)).await {
            Ok(result) => result,
            Err(object_store::Error::NotFound { .. }) => return Err(CacheError::Missing(hash)),
            Err(error) => return Err(error.into()),
        };
        match result.payload {
            GetResultPayload::File(file, _) => {
                // SAFETY: blobs are immutable: one is written in full under
                // another name before it appears under its hash, and never
                // changed afterwards.
                let map = unsafe { memmap2::Mmap::map(&file)? };
                Ok(Arc::new(map))
            }
            GetResultPayload::Stream(_) => {
                let bytes = result.bytes().await?;
                let mut aligned = AlignedVec::<16>::with_capacity(bytes.len());
                aligned.extend_from_slice(&bytes);
                Ok(Arc::new(aligned))
            }
        }
    }

    async fn binding(&self, path: &Path) -> Result<Option<Blake3Hash>, CacheError> {
        let result = match self.0.store.inner().get(path).await {
            Ok(result) => result,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let bytes = result.bytes().await?;
        let hash: [u8; 32] = bytes[..].try_into().map_err(|_| CacheError::Binding)?;
        Ok(Some(Blake3Hash::from_bytes(hash)))
    }
}

// SAFETY: bytes must hold an archived T, and hash be their hash.
unsafe fn blob<T>(bytes: SharedBytes, hash: Blake3Hash) -> Blob<T>
where
    T: Archive,
    T::Archived: Portable,
{
    let pos = rkyv::api::root_position::<T::Archived>((*bytes).as_ref().len());
    Blob {
        // SAFETY: as certified by the caller.
        key: unsafe { Key::from_hash(hash) },
        data: unsafe { OwnedArchive::from_shared(bytes, pos) },
    }
}

fn blob_path(hash: Blake3Hash) -> Path {
    Path::from(format!("blobs/{hash}"))
}

// Where the binding of `key` to a T is stored.
fn binding_path<T: ?Sized>(key: &impl anyhash::Hash) -> Path {
    let mut hasher = StableHasher::new();
    hasher.write_str(type_name::<T>());
    key.hash(&mut hasher);
    Path::from(format!("keys/{}", hasher.finish()))
}

/// A hasher for [`anyhash::Hash`] whose result is the same wherever and
/// whenever it is computed, given the same writes.
pub struct StableHasher(HasherLe<Blake3Hash, Blake3Writer>);

impl StableHasher {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(HasherLe::new(Blake3Writer(blake3::Hasher::new())))
    }

    pub fn finish(&self) -> Blake3Hash {
        self.0.finish()
    }
}

impl HasherWrite for StableHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
    fn write_u8(&mut self, i: u8) {
        self.0.write_u8(i);
    }
    fn write_u16(&mut self, i: u16) {
        self.0.write_u16(i);
    }
    fn write_u32(&mut self, i: u32) {
        self.0.write_u32(i);
    }
    fn write_u64(&mut self, i: u64) {
        self.0.write_u64(i);
    }
    fn write_u128(&mut self, i: u128) {
        self.0.write_u128(i);
    }
    // A length is written the same on every platform.
    fn write_usize(&mut self, i: usize) {
        self.0.write_u64(i as u64);
    }
    fn write_isize(&mut self, i: isize) {
        self.0.write_u64(i as u64);
    }
}

struct Blake3Writer(blake3::Hasher);

impl HasherWrite for Blake3Writer {
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}

impl anyhash::Hasher<Blake3Hash> for Blake3Writer {
    fn finish(&self) -> Blake3Hash {
        Blake3Hash::from_bytes(*self.0.finalize().as_bytes())
    }
}
