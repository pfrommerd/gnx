use gnx_math::{Array, ArrayRef, DType, random};
use std::{collections::BTreeMap, sync::Mutex};

/// Named streams with independent traced counters. Names are UTF-8 bytes;
/// derive each stream by folding both halves of its stable FNV-1a-64 tag.
pub struct Rngs {
    root: Array,
    zero: ArrayRef,
    counters: Mutex<BTreeMap<String, ArrayRef>>,
}

/// Default streams use seed zero and the ThreeFry algorithm.
impl Default for Rngs {
    fn default() -> Self {
        Self::key(0, random::Algorithm::ThreeFry2x32)
    }
}

impl Rngs {
    /// Create named streams from a seed, equivalent to `Rngs::new(random::key(seed, algorithm))`.
    pub fn key(seed: u64, algorithm: random::Algorithm) -> Self {
        Self::new(random::key(seed, algorithm))
    }

    pub fn new(root: Array) -> Self {
        assert!(
            root.dtype().is_key() && root.shape().dims().is_empty(),
            "random draw requires one scalar key; map over batched keys"
        );
        Self {
            root,
            zero: ArrayRef::new(Array::scalar(0_u32)),
            counters: Default::default(),
        }
    }

    /// Return the next key from `name`. Each stream's U32 counter starts at
    /// zero and wraps after 2^32 draws. Writes follow ordinary ArrayRef rules.
    pub fn next(&self, name: &str) -> Array {
        let tag = name.bytes().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        let hi = Array::scalar((tag >> 32) as u32);
        let lo = Array::scalar(tag as u32);
        let base = random::fold_in(&random::fold_in(&self.root, &hi), &lo);
        let mut counters = self.counters.lock().unwrap();
        let counter = counters
            .entry(name.into())
            .or_insert_with(|| self.zero.independent());
        let n = counter.get();
        let next = n.astype(DType::I64) + 1;
        counter.set(next.astype(DType::U32));
        random::fold_in(&base, &n)
    }
}

/// Conventional neural-network stream names on the generic random stream collection.
pub trait RngsExt {
    fn params(&self) -> Array;
    fn dropout(&self) -> Array;
}
impl RngsExt for Rngs {
    fn params(&self) -> Array {
        self.next("params")
    }
    fn dropout(&self) -> Array {
        self.next("dropout")
    }
}
