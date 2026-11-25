use std::sync::Arc;

use futures_executor::block_on;
use gnx_common::cache::{Cache, CacheError, Key};
use gnx_common::pass::Builder;
use gnx_common::store::{GetResultPayload, ObjectStoreExt, Path, Store};
use gnx_expr::{Op, Operation, RootExpr, SubExprBuilder, VarInfo, dialect};
use rkyv::{Archive, Portable, Serialize};

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(C)]
struct Neg;
impl Operation for Neg {}

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(C)]
struct Call;
impl Operation for Call {}

dialect! {
    dyn Operation, "cache_test" => {
        "neg" => Neg,
        "call" => Call,
    }
}

#[derive(Archive, Serialize)]
struct Artifact {
    name: String,
    bytes: Vec<u8>,
}

fn artifact() -> Artifact {
    Artifact {
        name: "module".to_owned(),
        bytes: (0..=255).collect(),
    }
}

fn info() -> VarInfo {
    VarInfo::new(0usize)
}

/// A root whose only equation calls another root.
fn nested() -> (Arc<RootExpr>, Arc<RootExpr>) {
    let mut inner = Builder::new();
    let x = inner.input(info());
    let y = inner.emit1(Op::new(Neg), &[x], info());
    let inner = inner.finish(vec![y]).build().unwrap();

    let mut outer = Builder::new();
    let x = outer.input(info());
    let called = outer.emit(
        Op::new(Call),
        &[x],
        [info()],
        vec![SubExprBuilder::new(inner.clone().handle())],
    );
    let outer = outer.finish(called).build().unwrap();
    (outer, inner)
}

fn caches() -> Vec<(tempfile::TempDir, Box<dyn Fn() -> Cache>)> {
    // The directory is unused by the in-memory store.
    let memory = Store::in_memory();
    let dir = tempfile::tempdir().unwrap();
    let local = Store::local(dir.path()).unwrap();
    vec![
        (
            tempfile::tempdir().unwrap(),
            Box::new(move || Cache::new(memory.clone())),
        ),
        (dir, Box::new(move || Cache::new(local.clone()))),
    ]
}

#[test]
fn values_round_trip_by_content() {
    for (_dir, cache) in caches() {
        let stored = block_on(cache().put(&artifact())).unwrap();
        assert_eq!(stored.name, "module");
        // The same value is the same blob.
        let again = block_on(cache().put(&artifact())).unwrap();
        assert_eq!(stored.key(), again.key());

        // A cache that has never seen the value reads it from the store.
        let loaded = block_on(cache().get(stored.key())).unwrap();
        assert_eq!(loaded.name, "module");
        assert_eq!(loaded.bytes.len(), 256);
        assert_eq!(loaded.bytes[255], 255);
    }
}

#[test]
fn missing_blobs_are_errors() {
    let stored = block_on(Cache::in_memory().put(&artifact())).unwrap();
    let missing = block_on(Cache::in_memory().get(stored.key()));
    assert!(matches!(missing, Err(CacheError::Missing(hash)) if hash == stored.key().hash()));
}

#[test]
fn bindings_resolve_and_do_not_change() {
    for (_dir, cache) in caches() {
        let cache = cache();
        let first = block_on(cache.put(&artifact())).unwrap().key();
        let second = block_on(cache.put(&"other".to_owned())).unwrap().key();

        let unbound: Option<Key<Artifact>> = block_on(cache.resolve(&("request", 1u32))).unwrap();
        assert!(unbound.is_none());

        block_on(cache.bind(&("request", 1u32), first)).unwrap();
        block_on(cache.bind(&("request", 1u32), first)).unwrap();
        assert_eq!(
            block_on(cache.resolve::<Artifact>(&("request", 1u32))).unwrap(),
            Some(first)
        );
        // Another key, and the same key for another type, are unbound.
        assert!(
            block_on(cache.resolve::<Artifact>(&("request", 2u32)))
                .unwrap()
                .is_none()
        );
        assert!(
            block_on(cache.resolve::<String>(&("request", 1u32)))
                .unwrap()
                .is_none()
        );

        // SAFETY: the key is only bound, never read.
        let other = unsafe { Key::<Artifact>::from_hash(second.hash()) };
        let conflict = block_on(cache.bind(&("request", 1u32), other));
        assert!(matches!(
            conflict,
            Err(CacheError::Conflict { existing, new })
                if existing == first.hash() && new == second.hash()
        ));
    }
}

#[test]
fn roots_round_trip_with_the_roots_they_refer_to() {
    for (_dir, cache) in caches() {
        let (outer, inner) = nested();
        let key = block_on(cache().put_root(&outer)).unwrap();
        assert_eq!(key.hash(), outer.hash());
        let shown = outer.view().to_string();
        let inner_hash = inner.hash();
        drop((outer, inner));

        let fresh = cache();
        let loaded = block_on(fresh.get_root(key)).unwrap();
        assert_eq!(loaded.hash(), key.hash());
        assert_eq!(loaded.view().to_string(), shown);
        assert_eq!(loaded.view().check().map_err(|e| format!("{e:?}")), Ok(()));
        let externals: Vec<_> = loaded.external_roots().map(|root| root.hash()).collect();
        assert_eq!(externals, [inner_hash]);

        // A root that is still loaded is not loaded again.
        let again = block_on(fresh.get_root(key)).unwrap();
        assert!(Arc::ptr_eq(&loaded, &again));
    }
}

#[test]
fn a_mirror_serves_local_files() {
    let remote = Store::in_memory();
    let dir = tempfile::tempdir().unwrap();
    let mirrored = Store::mirrored(remote.clone(), dir.path()).unwrap();
    let path = Path::from("blobs/a");

    // Written before the mirror existed: fetched on first read.
    block_on(remote.put(path.clone(), b"payload".to_vec())).unwrap();
    for _ in 0..2 {
        let result = block_on(mirrored.inner().get(&path)).unwrap();
        assert!(matches!(result.payload, GetResultPayload::File(..)));
        assert_eq!(&block_on(result.bytes()).unwrap()[..], b"payload");
    }

    // Written through the mirror: stored remotely and locally.
    let path = Path::from("blobs/b");
    block_on(mirrored.put(path.clone(), b"second".to_vec())).unwrap();
    assert_eq!(block_on(remote.get_bytes(path.clone())).unwrap(), b"second");
    let result = block_on(mirrored.inner().get(&path)).unwrap();
    assert!(matches!(result.payload, GetResultPayload::File(..)));

    // A cache over a mirror reads through it.
    let stored = block_on(Cache::new(remote).put(&artifact())).unwrap();
    let loaded = block_on(Cache::new(mirrored).get(stored.key())).unwrap();
    assert_eq!(loaded.name, "module");
}
