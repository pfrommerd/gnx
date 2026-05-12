use std::path::Path as StdPath;
use std::sync::Arc;

pub use object_store::path::Path;
pub use object_store::{
    DynObjectStore, GetOptions, GetResult, ListResult, ObjectMeta, ObjectStore, ObjectStoreExt,
    PutOptions, PutPayload, PutPayloadMut, PutResult, Result,
};

/// Shared object-store handle used by gnx-io.
#[derive(Clone, Debug)]
pub struct Store {
    inner: Arc<dyn ObjectStore>,
}

impl Store {
    pub fn new(inner: Arc<dyn ObjectStore>) -> Self {
        Self { inner }
    }

    pub fn in_memory() -> Self {
        Self::new(Arc::new(object_store::memory::InMemory::new()))
    }

    pub fn local<P: AsRef<StdPath>>(root: P) -> Result<Self> {
        object_store::local::LocalFileSystem::new_with_prefix(root)
            .map(|store| Self::new(Arc::new(store)))
    }

    pub fn inner(&self) -> &Arc<dyn ObjectStore> {
        &self.inner
    }

    pub fn object<P: Into<Path>>(&self, path: P) -> Object {
        Object {
            store: self.clone(),
            path: path.into(),
        }
    }

    pub async fn put<P, B>(&self, path: P, payload: B) -> Result<PutResult>
    where
        P: Into<Path>,
        B: Into<PutPayload>,
    {
        let path = path.into();
        self.inner.put(&path, payload.into()).await
    }

    pub async fn get<P: Into<Path>>(&self, path: P) -> Result<GetResult> {
        let path = path.into();
        self.inner.get(&path).await
    }

    pub async fn get_bytes<P: Into<Path>>(&self, path: P) -> Result<Vec<u8>> {
        let path = path.into();
        self.inner
            .get(&path)
            .await?
            .bytes()
            .await
            .map(|bytes| bytes.to_vec())
    }

    pub async fn head<P: Into<Path>>(&self, path: P) -> Result<ObjectMeta> {
        let path = path.into();
        self.inner.head(&path).await
    }

    pub async fn delete<P: Into<Path>>(&self, path: P) -> Result<()> {
        let path = path.into();
        self.inner.delete(&path).await
    }
}

/// A concrete object location within a [`Store`].
#[derive(Clone, Debug)]
pub struct Object {
    store: Store,
    path: Path,
}

impl Object {
    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn relative<P: AsRef<str>>(&self, path: P) -> Self {
        Self {
            store: self.store.clone(),
            path: self
                .path
                .clone()
                .join(path.as_ref())
                .expect("valid object path"),
        }
    }

    pub async fn put<B: Into<PutPayload>>(&self, payload: B) -> Result<PutResult> {
        self.store.inner.put(&self.path, payload.into()).await
    }

    pub async fn get(&self) -> Result<GetResult> {
        self.store.inner.get(&self.path).await
    }

    pub async fn get_bytes(&self) -> Result<Vec<u8>> {
        self.store
            .inner
            .get(&self.path)
            .await?
            .bytes()
            .await
            .map(|bytes| bytes.to_vec())
    }

    pub async fn head(&self) -> Result<ObjectMeta> {
        self.store.inner.head(&self.path).await
    }

    pub async fn delete(&self) -> Result<()> {
        self.store.inner.delete(&self.path).await
    }
}
