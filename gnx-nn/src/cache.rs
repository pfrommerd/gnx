//! Session-owned decoder state. Keys and slot layout are static graph structure;
//! the cursor and slot contents are mutable, mappable graph leaves.
use gnx_graph::Graph;
use gnx_math::{Array, ArrayRef, DType, lax};
use std::{borrow::Borrow, collections::HashMap};

/// Deterministic binary path. Splitting does not allocate a tracer or depend on
/// construction order. Use each child for one slot or split it further.
#[derive(Clone, Graph, Debug, Eq, PartialEq, Hash)]
pub struct CacheKey {
    path: String,
}
impl CacheKey {
    pub fn root() -> Self {
        Self {
            path: "root".into(),
        }
    }
    pub fn split(&self) -> (Self, Self) {
        (
            Self {
                path: format!("{}0", self.path),
            },
            Self {
                path: format!("{}1", self.path),
            },
        )
    }
}
impl Borrow<str> for CacheKey {
    fn borrow(&self) -> &str {
        &self.path
    }
}

/// Mutable state for one decoding session. A clone shares its cells; create a
/// new cache to start another session. Under `map` each slot and the cursor
/// acquire a batch axis, while keys and the maximum time extent stay static.
#[derive(Clone, Graph)]
pub struct KVCache {
    capacity: usize,
    cursor: ArrayRef,
    slots: HashMap<CacheKey, ArrayRef>,
}

/// A snapshot of the two time-major cache slots after a chunk write. Only
/// the first `len` rows exist logically. Unlike an input-shape witness,
/// `len` may be computed inside a device loop and may differ by mapped lane.
#[derive(Clone, Graph)]
pub struct KvPrefix {
    pub(crate) key: Array,
    pub(crate) value: Array,
    pub(crate) len: Array,
}

impl KVCache {
    /// Allocate a cache with a logical time extent of one. A fixed storage bound
    /// permits data-dependent positions and growth inside compiled loops.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0 && i32::try_from(capacity).is_ok());
        Self {
            capacity,
            cursor: ArrayRef::new(Array::scalar(0_i32)),
            slots: HashMap::new(),
        }
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn position(&self) -> Array {
        self.cursor.get()
    }
    /// Advance after all layers have written this token. `position == capacity`
    /// is the exhausted sentinel; callers must check for room before writing.
    pub fn step(&self) {
        self.advance(1);
    }
    /// Advance once after writing a whole chunk. The caller must ensure the
    /// chunk fits in the remaining capacity before running the forward pass.
    pub fn advance(&self, tokens: usize) {
        assert!(tokens <= self.capacity);
        self.cursor.set(self.position() + tokens as i32);
    }
    pub fn init_time(&mut self, key: CacheKey, features: &[usize], dtype: DType) {
        let mut shape = vec![self.capacity];
        shape.extend_from_slice(features);
        self.init_state(key, &shape, dtype);
    }
    /// Fixed-size state, e.g. recurrent or convolution history.
    pub fn init_state(&mut self, key: CacheKey, shape: &[usize], dtype: DType) {
        assert!(
            !self.slots.contains_key(&key),
            "duplicate cache key {key:?}"
        );
        self.slots
            .insert(key, ArrayRef::new(lax::zeros(shape, dtype)));
    }
    pub fn slot(&self, key: &CacheKey) -> &ArrayRef {
        &self.slots[key]
    }
    /// Write two `[time, ...]` chunks and return their shared logical prefix,
    /// without advancing the cursor. Both chunks must fit at the current position.
    pub fn write_kv(&self, keys: &(CacheKey, CacheKey), key: &Array, value: &Array) -> KvPrefix {
        let t = key.fixed_dims().expect("fixed chunk dimensions")[0];
        assert_eq!(value.fixed_dims().expect("fixed chunk dimensions")[0], t);
        assert_ne!(keys.0, keys.1, "K and V need distinct slots");
        KvPrefix {
            key: self.write_chunk(&keys.0, key),
            value: self.write_chunk(&keys.1, value),
            len: self.position() + t as i32,
        }
    }
    /// Write `[time, ...]` at the current position without advancing. The
    /// chunk must fit entirely in the remaining capacity.
    fn write_chunk(&self, key: &CacheKey, chunk: &Array) -> Array {
        let slot = self.slot(key);
        let storage = slot.get();
        let shape = storage.fixed_dims().expect("fixed cache storage");
        assert_eq!(shape.first(), Some(&self.capacity), "expected a time slot");
        let dims = chunk.fixed_dims().expect("fixed chunk dimensions");
        assert!(dims.first().is_some_and(|&t| t > 0 && t <= self.capacity));
        assert_eq!(shape[1..], dims[1..]);
        let zero = Array::scalar(0_i32);
        let pos = self.position();
        let mut offsets = vec![&zero; shape.len()];
        offsets[0] = &pos;
        let updated = lax::dynamic_update_slice(&storage, chunk, &offsets);
        slot.set(updated.clone());
        updated
    }
}
