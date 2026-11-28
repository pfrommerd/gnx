//! Project matches on the typed graph onto tracer leaves. Axes use annotations;
//! differentiation uses match locations and preserves exact-versus-prefix semantics.

use gnx_graph::{
    Filter, Graph, Key, Leaf,
    util::{Dag, DagChild, DagNode},
};

use crate::filters::AxisType;
use crate::{TransformError, TransformResult};

/// The spec entry for one leaf. `exact` is whether the spec leaf sits on the
/// value leaf itself rather than on one of its ancestors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved<T> {
    pub value: Option<T>,
    pub exact: bool,
}

pub(crate) fn axes<B: AxisType>(
    filter: &B::Filter,
    graph: &impl Graph,
    paths: &[Vec<Key>],
    name: &str,
) -> TransformResult<Vec<Resolved<usize>>> {
    B::validate(filter, graph, paths, name)?;
    Ok(project(crate::axis_matches::axes(graph, filter), paths))
}
pub(crate) fn wrt<L: Leaf, F: Filter<Leaf = L>>(
    filter: &F,
    graph: &impl Graph,
    paths: &[Vec<Key>],
) -> Vec<Resolved<()>> {
    project(crate::axis_matches::locations(graph, filter), paths)
}
fn project<T: Copy>(matches: Vec<(Vec<Key>, T)>, paths: &[Vec<Key>]) -> Vec<Resolved<T>> {
    // Only remove transparent keys when comparing a graph path to the canonical tracer paths.
    let matches: std::collections::HashMap<Vec<_>, _> = matches
        .into_iter()
        .map(|(mut p, v)| {
            p.retain(|k| !matches!(k, Key::Transparent));
            (p, v)
        })
        .collect();
    paths
        .iter()
        .map(|path| {
            let path: Vec<_> = path
                .iter()
                .filter(|k| !matches!(k, Key::Transparent))
                .cloned()
                .collect();
            for n in (0..=path.len()).rev() {
                if let Some(value) = matches.get(&path[..n]) {
                    return Resolved {
                        value: Some(*value),
                        exact: n == path.len(),
                    };
                }
            }
            Resolved {
                value: None,
                exact: false,
            }
        })
        .collect()
}

pub(crate) fn traced_output_axes<B: AxisType>(
    filter: &B::Filter,
    graph: &impl Graph,
) -> TransformResult<Vec<usize>> {
    let paths =
        crate::axis_matches::locations(graph, &gnx_graph::Of::<gnx_expr::ObjectTracer>::filter())
            .into_iter()
            .map(|(p, _)| p)
            .collect::<Vec<_>>();
    output_axes::<B>(filter, graph, &paths)
}

pub(crate) fn output_axes<B: AxisType>(
    filter: &B::Filter,
    graph: &impl Graph,
    paths: &[Vec<Key>],
) -> TransformResult<Vec<usize>> {
    axes::<B>(filter, graph, paths, "out_axes")?
        .into_iter()
        .zip(paths)
        .map(|(axis, path)| {
            axis.value.ok_or_else(|| {
                TransformError::invalid(format!(
                    "out_axes must give an axis for every result (missing {})",
                    show(path)
                ))
            })
        })
        .collect()
}

pub(crate) fn at_path(name: &str, path: &[Key], error: TransformError) -> TransformError {
    TransformError::invalid(format!("{name} at {}: {error}", show(path)))
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

fn validate_path<T>(mut dag: &Dag<T>, path: &[Key]) -> Result<(), String> {
    let mut keys = path
        .iter()
        .filter(|key| !matches!(key, Key::Transparent))
        .enumerate();
    loop {
        match dag {
            Dag::Leaf(_) => return Ok(()),
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

pub(crate) fn validate_dag<T>(
    spec: &Dag<T>,
    paths: &[Vec<Key>],
    name: &str,
) -> TransformResult<()> {
    paths.iter().try_for_each(|p| {
        validate_path(spec, p).map_err(|error| TransformError::invalid(format!("{name}: {error}")))
    })
}
