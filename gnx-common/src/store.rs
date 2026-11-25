use std::path::Path as StdPath;
use std::sync::Arc;

pub use object_store::Error;
pub use object_store::path::Path;
pub use object_store::{
    DynObjectStore, GetOptions, GetResult, GetResultPayload, ListResult, ObjectMeta, ObjectStore,
    ObjectStoreExt, PutMode, PutOptions, PutPayload, PutPayloadMut, PutResult, Result,
};

/// Shared handle to an object store.
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
            path: self.path.clone().join(path.as_ref()),
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

impl Store {
    /// `remote`, with every object that is read or written kept in the
    /// local directory `root` as well. Reading an object then gives a local
    /// file, which can be memory-mapped, and reads it from `remote` at most
    /// once.
    ///
    /// Objects are assumed never to change once written: a local copy is
    /// not checked against `remote` again.
    pub fn mirrored<P: AsRef<StdPath>>(remote: Store, root: P) -> Result<Self> {
        let local = object_store::local::LocalFileSystem::new_with_prefix(root)?;
        Ok(Self::new(Arc::new(Mirrored {
            remote: remote.inner,
            local: Arc::new(local),
        })))
    }
}

/// See [`Store::mirrored`].
#[derive(Debug)]
struct Mirrored {
    remote: Arc<dyn ObjectStore>,
    local: Arc<object_store::local::LocalFileSystem>,
}

impl std::fmt::Display for Mirrored {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mirrored({}, {})", self.remote, self.local)
    }
}

// Whether a read asks for a whole object, whatever its state.
fn is_plain(options: &GetOptions) -> bool {
    let GetOptions {
        if_match,
        if_none_match,
        if_modified_since,
        if_unmodified_since,
        range,
        version,
        head,
        extensions: _,
    } = options;
    if_match.is_none()
        && if_none_match.is_none()
        && if_modified_since.is_none()
        && if_unmodified_since.is_none()
        && range.is_none()
        && version.is_none()
        && !head
}

#[async_trait::async_trait]
impl ObjectStore for Mirrored {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        let result = self
            .remote
            .put_opts(location, payload.clone(), opts)
            .await?;
        self.local.put(location, payload).await?;
        Ok(result)
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: object_store::PutMultipartOptions,
    ) -> Result<Box<dyn object_store::MultipartUpload>> {
        self.remote.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        if !is_plain(&options) {
            return self.remote.get_opts(location, options).await;
        }
        match self.local.get(location).await {
            Err(Error::NotFound { .. }) => {}
            result => return result,
        }
        let bytes = self.remote.get(location).await?.bytes().await?;
        self.local.put(location, bytes.into()).await?;
        self.local.get(location).await
    }

    fn delete_stream(
        &self,
        locations: futures_util::stream::BoxStream<'static, Result<Path>>,
    ) -> futures_util::stream::BoxStream<'static, Result<Path>> {
        use futures_util::StreamExt;
        let local = self.local.clone();
        (self.remote.delete_stream(locations))
            .then(move |deleted| {
                let local = local.clone();
                async move {
                    let location = deleted?;
                    match local.delete(&location).await {
                        Ok(()) | Err(Error::NotFound { .. }) => Ok(location),
                        Err(error) => Err(error),
                    }
                }
            })
            .boxed()
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> futures_util::stream::BoxStream<'static, Result<ObjectMeta>> {
        self.remote.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.remote.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: object_store::CopyOptions,
    ) -> Result<()> {
        self.remote.copy_opts(from, to, options).await?;
        // Whatever was mirrored at the destination is out of date.
        match self.local.delete(to).await {
            Ok(()) | Err(Error::NotFound { .. }) => Ok(()),
            Err(error) => Err(error),
        }
    }
}
