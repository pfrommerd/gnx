use super::{GraphId, Error};
use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};

// A GraphContext stores already-constructed graph nodes
// in a generic type-erased way
// The proper way of interacting with it is through the ctx_build! macro,
// which releases the &mut borrow on the context while building child nodes
#[derive(Default)]
pub struct GraphContext {
    // The boxes contain HashMap<GraphId, T>
    maps: HashMap<TypeId, Box<dyn Any>>,
    // All the GraphIds we've seen so far
    seen: HashSet<GraphId>,
}

impl GraphContext {
    pub fn new() -> Self {
        Self::default()
    }

    // For use by the ctx_build_shared macro
    // through which you should interact with the GraphContext
    // The macro released the &mut borrow while building child nodes
    pub fn _reserve<T: Clone + 'static, E: Error>(&mut self, id: GraphId) -> Result<Option<T>, E> {
        if self.seen.contains(&id) {
            return Err(E::invalid_id(id));
        }
        self.seen.insert(id);
        let map = self
            .maps
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(HashMap::<GraphId, T>::new()));
        let map = map
            .downcast_mut::<HashMap<GraphId, T>>()
            .ok_or(E::custom("Internal context error"))?;
        if let Some(s) = map.get(&id) {
            Ok(Some(s.clone()))
        } else {
            Ok(None)
        }
    }
    pub fn _finish<T: Clone + 'static, E: Error>(&mut self, id: GraphId, value: T) -> Result<(), E> {
        let map = self
            .maps
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(HashMap::<GraphId, T>::new()));
        // self.seen.insert(id);
        let map = map
            .downcast_mut::<HashMap<GraphId, T>>()
            .ok_or(E::custom("Internal context error"))?;
        map.insert(id, value);
        Ok(())
    }

    pub fn create<T: Clone + 'static, E: Error, F: FnOnce(&mut Self) -> Result<T, E>>(
        &mut self,
        id: GraphId,
        builder: F,
    ) -> Result<T, E> {
        let value = self._reserve::<T, E>(id)?;
        match value {
            Some(value) => Ok(value),
            None => {
                let value = builder(self)?;
                self._finish::<T, E>(id, value.clone())?;
                Ok(value)
            }
        }
    }
}
