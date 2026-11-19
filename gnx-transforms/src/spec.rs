//! Resolve transformation specs (`in_axes`, `out_axes`, `wrt`) written as
//! [`Dag`] graphs against the tracer leaves of an argument or result graph.
//!
//! Matching follows key paths (tuple and list indices, struct field names;
//! transparent wrapper keys are skipped). A spec leaf covers every value leaf
//! under it, as a JAX pytree prefix does: `dag!(0, *)` maps every leaf of the
//! first argument along axis 0 and none of the second.

use gnx_graph::Key;
use gnx_graph::util::{Dag, DagChild, DagNode};

use crate::{TransformError, TransformResult};

/// The spec entry for one leaf. `exact` is whether the spec leaf sits on the
/// value leaf itself rather than on one of its ancestors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Resolved<T> {
    pub value: Option<T>,
    pub exact: bool,
}

fn show(path: &[Key]) -> String {
    let parts: Vec<_> = path
        .iter()
        .filter_map(|key| match key {
            Key::Named(name) => Some(format!(".{name}")),
            Key::Index(index) => Some(format!("[{index}]")),
            Key::Transparent => None,
        })
        .collect();
    if parts.is_empty() {
        "the root".into()
    } else {
        parts.concat()
    }
}

fn lookup<'a, T>(mut dag: &'a Dag<T>, path: &[Key]) -> Result<(Option<&'a T>, bool), String> {
    let mut keys = path
        .iter()
        .filter(|key| !matches!(key, Key::Transparent))
        .enumerate()
        .peekable();
    loop {
        match dag {
            Dag::Leaf(value) => return Ok((value.as_ref(), keys.peek().is_none())),
            Dag::Node(node) => {
                let Some((depth, key)) = keys.next() else {
                    return Err(format!(
                        "the spec is more detailed than the value at {}",
                        show(path)
                    ));
                };
                let child = match (node, key) {
                    (DagNode::List(children), Key::Index(index)) => usize::try_from(*index)
                        .ok()
                        .and_then(|index| children.get(index)),
                    (DagNode::List(_), _) => None,
                    (DagNode::Map(children), key) => children.get(key),
                };
                let Some(child) = child else {
                    let prefix: Vec<_> = path
                        .iter()
                        .filter(|key| !matches!(key, Key::Transparent))
                        .take(depth + 1)
                        .cloned()
                        .collect();
                    return Err(format!("the spec has no entry for {}", show(&prefix)));
                };
                dag = match child {
                    DagChild::Owned(dag) => dag,
                    DagChild::Shared(dag) => dag.as_ref(),
                };
            }
        }
    }
}

/// One resolved entry per leaf path.
pub(crate) fn resolve<T: Clone>(
    spec: &Dag<T>,
    paths: &[Vec<Key>],
    what: &str,
) -> TransformResult<Vec<Resolved<T>>> {
    paths
        .iter()
        .map(|path| {
            lookup(spec, path)
                .map(|(value, exact)| Resolved {
                    value: value.cloned(),
                    exact,
                })
                .map_err(|error| TransformError::invalid(format!("{what}: {error}")))
        })
        .collect()
}
