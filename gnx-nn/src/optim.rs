//! Composable additive gradient updates on graphs. Pass a filtered view of the
//! loss arguments to [`Optimizer`] to select the same cells as autodiff.

use gnx_expr::{CellKey, ObjectTracer};
use gnx_graph::{
    Graph, GraphContext, Key, Leaf, Of, OwnedGraph,
    util::{GraphExt, IterSource},
};
use gnx_math::{Array, ArrayRef, DType, lax};
use std::{collections::HashMap, fmt};

#[derive(Debug)]
pub struct OptimError(pub String);
impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for OptimError {}
pub type OptimResult<T> = std::result::Result<T, OptimError>;

/// A pure transformation. Return additive updates and the next state, without
/// modifying the supplied cells. All stages receive the same pre-step params.
pub trait GradientTransform {
    type State<V: OwnedGraph>: OwnedGraph;
    fn init<P: Graph<Owned: OwnedGraph>>(&self, params: &P) -> OptimResult<Self::State<P::Owned>>;
    fn update<V: OwnedGraph, P: Graph>(
        &self,
        updates: V,
        state: &Self::State<V>,
        params: &P,
    ) -> OptimResult<(V, Self::State<V>)>;
}

// Keep path and alias topology separate from handle identity: gradient cells
// are fresh, but must agree with the parameter cells about which paths tie.
struct Arrays {
    paths: Vec<(Vec<Key>, usize)>,
    values: Vec<Array>,
    cells: Vec<Option<ArrayRef>>,
}
impl Arrays {
    fn of(g: &impl Graph) -> OptimResult<Self> {
        let mut paths = Vec::new();
        let mut cells = Vec::new();
        let mut values = Vec::new();
        let mut aliases = HashMap::new();
        for (path, object) in g.collect_leaves_with_paths(Of::<ObjectTracer>::filter()) {
            let (value, cell) = match &object {
                ObjectTracer::Ref(_) => (
                    Array::try_from_value(object)
                        .map_err(|_| OptimError("expected an array value".into()))?,
                    None,
                ),
                ObjectTracer::Cell(c) => {
                    let key = CellKey::from(c);
                    if let Some(&i) = aliases.get(&key) {
                        paths.push((path, i));
                        continue;
                    }
                    aliases.insert(key, values.len());
                    let cell = ArrayRef::try_from_value(object)
                        .map_err(|_| OptimError("expected an array cell".into()))?;
                    (cell.get(), Some(cell))
                }
            };
            if !matches!(
                value.dtype(),
                DType::BF16 | DType::F16 | DType::F32 | DType::F64
            ) {
                return Err(OptimError(format!(
                    "expected a real floating-point parameter at {path:?}"
                )));
            }
            paths.push((path, values.len()));
            cells.push(cell);
            values.push(value);
        }
        Ok(Self {
            paths,
            values,
            cells,
        })
    }
    fn rebuild<V: OwnedGraph>(&self, graph: V, values: Vec<Array>) -> OptimResult<V> {
        if values.len() != self.cells.len() {
            return Err(OptimError("wrong number of updates".into()));
        }
        let objects: Vec<ObjectTracer> = values
            .into_iter()
            .zip(&self.cells)
            .map(|(a, c)| {
                if c.is_some() {
                    ArrayRef::new(a).into_tracer().into()
                } else {
                    a.into_tracer().into()
                }
            })
            .collect();
        graph
            .replace(
                Of::<ObjectTracer>::filter(),
                IterSource::new(self.paths.iter().map(|(_, i)| objects[*i].clone())),
                &mut GraphContext::new(),
            )
            .map_err(|e| OptimError(e.to_string()))
    }
    fn check_cells(&self) -> OptimResult<()> {
        if self.cells.iter().any(Option::is_none) {
            return Err(OptimError(
                "Optimizer needs mutable array cells; filter the parameter graph".into(),
            ));
        }
        Ok(())
    }
    /// Parameter slot for each distinct update. Ties must agree on the supplied paths.
    fn slots_for(&self, updates: &Self) -> OptimResult<Vec<usize>> {
        let paths: HashMap<_, _> = self.paths.iter().map(|(path, i)| (path, *i)).collect();
        let mut slots = vec![None; updates.values.len()];
        let mut owners = vec![None; self.values.len()];
        for (path, j) in &updates.paths {
            let j = *j;
            let &i = paths
                .get(path)
                .ok_or_else(|| OptimError(format!("no parameter at update path {path:?}")))?;
            if *slots[j].get_or_insert(i) != i || *owners[i].get_or_insert(j) != j {
                return Err(OptimError(format!(
                    "inconsistent parameter and update ties at {path:?}"
                )));
            }
            let (p, u) = (&self.values[i], &updates.values[j]);
            if p.shape() != u.shape()
                || p.dtype() != u.dtype()
                || self.cells[i].is_some() != updates.cells[j].is_some()
            {
                return Err(OptimError(format!("incompatible update at {path:?}")));
            }
        }
        Ok(slots
            .into_iter()
            .map(|i| i.expect("every slot has a path"))
            .collect())
    }
}

/// Owns the transform state; writes once per distinct selected cell. Parameter
/// and update graphs may have different Rust types. Updates may omit parameter
/// paths, but every supplied path must match a compatible parameter.
pub struct Optimizer<V: OwnedGraph, T: GradientTransform> {
    tx: T,
    state: T::State<V>,
}
impl<V: OwnedGraph, T: GradientTransform> Optimizer<V, T> {
    pub fn new<P: Graph<Owned = V>>(params: &P, tx: T) -> OptimResult<Self> {
        Arrays::of(params)?.check_cells()?;
        Ok(Self {
            state: tx.init(params)?,
            tx,
        })
    }
    pub fn state(&self) -> &T::State<V> {
        &self.state
    }
    pub fn update<P: Graph>(&mut self, params: &P, grads: V) -> OptimResult<()> {
        let current = Arrays::of(params)?;
        current.check_cells()?;
        current.slots_for(&Arrays::of(&grads)?)?;
        let (updates, next) = self.tx.update(grads, &self.state, params)?;
        let updates = Arrays::of(&updates)?;
        let slots = current.slots_for(&updates)?;
        for (i, u) in slots.into_iter().zip(updates.values) {
            current.cells[i]
                .as_ref()
                .expect("validated cell")
                .set(&current.values[i] + u);
        }
        self.state = next;
        Ok(())
    }
}

pub struct Chain<A, B>(A, B);
pub fn chain<A, B>(first: A, second: B) -> Chain<A, B> {
    Chain(first, second)
}
impl<A: GradientTransform, B: GradientTransform> GradientTransform for Chain<A, B> {
    type State<V: OwnedGraph> = (A::State<V>, B::State<V>);
    fn init<P: Graph<Owned: OwnedGraph>>(&self, p: &P) -> OptimResult<Self::State<P::Owned>> {
        Ok((self.0.init(p)?, self.1.init(p)?))
    }
    fn update<V: OwnedGraph, P: Graph>(
        &self,
        u: V,
        s: &Self::State<V>,
        p: &P,
    ) -> OptimResult<(V, Self::State<V>)> {
        let (u, a) = self.0.update(u, &s.0, p)?;
        let (u, b) = self.1.update(u, &s.1, p)?;
        Ok((u, (a, b)))
    }
}

pub struct Scale(f64);
pub fn scale(factor: f64) -> Scale {
    Scale(factor)
}
pub fn sgd(learning_rate: f64) -> Scale {
    scale(-learning_rate)
}
fn times(x: &Array, factor: f64) -> Array {
    x * Array::scalar(factor).astype(x.dtype())
}
impl GradientTransform for Scale {
    type State<V: OwnedGraph> = ();
    fn init<P: Graph<Owned: OwnedGraph>>(&self, _: &P) -> OptimResult<()> {
        Ok(())
    }
    fn update<V: OwnedGraph, P: Graph>(&self, u: V, _: &(), _: &P) -> OptimResult<(V, ())> {
        let a = Arrays::of(&u)?;
        let v = a.values.iter().map(|x| times(x, self.0)).collect();
        Ok((a.rebuild(u, v)?, ()))
    }
}

pub struct AddDecayedWeights(f64);
pub fn add_decayed_weights(decay: f64) -> AddDecayedWeights {
    AddDecayedWeights(decay)
}
impl GradientTransform for AddDecayedWeights {
    type State<V: OwnedGraph> = ();
    fn init<P: Graph<Owned: OwnedGraph>>(&self, _: &P) -> OptimResult<()> {
        Ok(())
    }
    fn update<V: OwnedGraph, P: Graph>(&self, u: V, _: &(), p: &P) -> OptimResult<(V, ())> {
        let a = Arrays::of(&u)?;
        let p = Arrays::of(p)?;
        let slots = p.slots_for(&a)?;
        let v = a
            .values
            .iter()
            .zip(slots)
            .map(|(x, i)| x + times(&p.values[i], self.0))
            .collect();
        Ok((a.rebuild(u, v)?, ()))
    }
}

// Accumulate reduced-precision gradients in F32, preserving F64 inputs.
fn zeros(x: &Array) -> Array {
    lax::zeros(x.shape(), crate::tensor::work(x.dtype()))
}
pub struct Trace(f64);
pub fn trace(decay: f64) -> Trace {
    Trace(decay)
}
impl GradientTransform for Trace {
    type State<V: OwnedGraph> = Vec<Array>;
    fn init<P: Graph<Owned: OwnedGraph>>(&self, p: &P) -> OptimResult<Self::State<P::Owned>> {
        Ok(Arrays::of(p)?.values.iter().map(zeros).collect())
    }
    fn update<V: OwnedGraph, P: Graph>(
        &self,
        u: V,
        s: &Self::State<V>,
        p: &P,
    ) -> OptimResult<(V, Self::State<V>)> {
        let a = Arrays::of(&u)?;
        let params = Arrays::of(p)?;
        check_moments(s, &params)?;
        let mut next = s.clone();
        let mut out = Vec::with_capacity(a.values.len());
        for (g, i) in a.values.iter().zip(params.slots_for(&a)?) {
            next[i] = g.astype(s[i].dtype()) + times(&s[i], self.0);
            out.push(next[i].astype(g.dtype()));
        }
        Ok((a.rebuild(u, out)?, next))
    }
}
fn check_moments(m: &[Array], a: &Arrays) -> OptimResult<()> {
    if m.len() != a.values.len()
        || m.iter()
            .zip(&a.values)
            .any(|(m, x)| m.shape() != x.shape() || m.dtype() != crate::tensor::work(x.dtype()))
    {
        Err(OptimError(
            "moment state does not match gradient layout".into(),
        ))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct AdamConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}
impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}
#[derive(Clone, Graph)]
pub struct AdamState {
    pub step: Array,
    pub first: Vec<Array>,
    pub second: Vec<Array>,
}
pub struct ScaleByAdam(AdamConfig);
pub fn scale_by_adam(config: AdamConfig) -> ScaleByAdam {
    ScaleByAdam(config)
}
impl GradientTransform for ScaleByAdam {
    type State<V: OwnedGraph> = AdamState;
    fn init<P: Graph<Owned: OwnedGraph>>(&self, p: &P) -> OptimResult<Self::State<P::Owned>> {
        let c = self.0;
        if !(0.0..1.0).contains(&c.beta1)
            || !(0.0..1.0).contains(&c.beta2)
            || !c.epsilon.is_finite()
            || c.epsilon <= 0.0
        {
            return Err(OptimError(
                "Adam requires betas in [0, 1) and positive finite epsilon".into(),
            ));
        }
        let z: Vec<_> = Arrays::of(p)?.values.iter().map(zeros).collect();
        Ok(AdamState {
            step: Array::scalar(0u32),
            first: z.clone(),
            second: z,
        })
    }
    fn update<V: OwnedGraph, P: Graph>(
        &self,
        u: V,
        s: &AdamState,
        p: &P,
    ) -> OptimResult<(V, AdamState)> {
        let a = Arrays::of(&u)?;
        let params = Arrays::of(p)?;
        check_moments(&s.first, &params)?;
        check_moments(&s.second, &params)?;
        let step = &s.step + Array::scalar(1u32);
        let correction = |beta: f64| {
            if beta == 0.0 {
                Array::scalar(1.0)
            } else {
                Array::scalar(1.0) - lax::exp(&times(&step.astype(DType::F64), beta.ln()))
            }
        };
        let c = self.0;
        let (b1, b2) = (correction(c.beta1), correction(c.beta2));
        let (mut first, mut second, mut out) = (
            s.first.clone(),
            s.second.clone(),
            Vec::with_capacity(a.values.len()),
        );
        for (g, i) in a.values.iter().zip(params.slots_for(&a)?) {
            let (m, v) = (&s.first[i], &s.second[i]);
            let t = m.dtype();
            let x = g.astype(t);
            let m = times(m, c.beta1) + times(&x, 1.0 - c.beta1);
            let v = times(v, c.beta2) + times(&(&x * &x), 1.0 - c.beta2);
            let root = lax::sqrt(&(&v / b2.astype(t)));
            out.push(
                ((&m / b1.astype(t)) / (root + Array::scalar(c.epsilon).astype(t)))
                    .astype(g.dtype()),
            );
            first[i] = m;
            second[i] = v;
        }
        Ok((
            a.rebuild(u, out)?,
            AdamState {
                step,
                first,
                second,
            },
        ))
    }
}
pub fn adam(lr: f64) -> Chain<ScaleByAdam, Scale> {
    chain(scale_by_adam(AdamConfig::default()), sgd(lr))
}
pub fn adamw(lr: f64, weight_decay: f64) -> Chain<ScaleByAdam, Chain<AddDecayedWeights, Scale>> {
    chain(
        scale_by_adam(AdamConfig::default()),
        chain(add_decayed_weights(weight_decay), sgd(lr)),
    )
}
