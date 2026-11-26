//! Bias-free feed-forward blocks and token-wise sparse expert routing.
use crate::{Linear, Param, Rngs, RngsExt, activations, tensor::*};
use gnx_graph::Graph;
use gnx_math::{Array, ArrayRef, DType, lax, random};
#[derive(Clone, Graph)]
pub struct SwiGlu {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}
impl SwiGlu {
    pub fn init(hidden: usize, width: usize, t: DType, r: &Rngs) -> Self {
        Self {
            gate: Linear::init_with(hidden, width, false, t, r),
            up: Linear::init_with(hidden, width, false, t, r),
            down: Linear::init_with(width, hidden, false, t, r),
        }
    }
    pub fn forward(&self, x: Array) -> Array {
        self.down
            .forward(activations::silu(&self.gate.forward(x.clone())) * self.up.forward(x))
    }
}

#[derive(Clone, Graph)]
pub struct RoutedExperts {
    pub router: Linear,
    /// Per-expert correction used only for sigmoid top-k selection, not combination weights.
    pub correction: ArrayRef,
    pub gate: Param,
    pub up: Param,
    pub down: Param,
    pub shared: Vec<SwiGlu>,
    pub top_k: usize,
    pub sigmoid: bool,
    pub normalize: bool,
    pub scale: f64,
}
impl RoutedExperts {
    pub fn init(
        hidden: usize,
        width: usize,
        experts: usize,
        top_k: usize,
        shared: usize,
        sigmoid: bool,
        t: DType,
        r: &Rngs,
    ) -> Self {
        assert!(top_k > 0 && top_k <= experts && hidden > 0 && width > 0);
        let stack = |i: usize, o: usize| {
            let f = work(t);
            // Random draws use a flattened iota. GNX shape metadata stores axis
            // extents in 32 bits; a K2 expert bank exceeds this as a single axis.
            let chunk = u32::MAX as usize / i.checked_mul(o).expect("expert size overflow");
            assert!(chunk > 0, "one expert matrix exceeds the random draw limit");
            let draw = |n| random::uniform(&r.params(), [n, o, i], f);
            let z = if experts <= chunk {
                draw(experts)
            } else {
                let parts: Vec<_> = (0..experts)
                    .step_by(chunk)
                    .map(|start| draw(chunk.min(experts - start)))
                    .collect();
                lax::concatenate(&parts.iter().collect::<Vec<_>>(), 0)
            };
            Param::new(((z - 0.5) * (2. / (i as f64).sqrt())).astype(t))
        };
        Self {
            router: Linear::init_with(hidden, experts, false, t, r),
            correction: ArrayRef::new(lax::zeros([experts], work(t))),
            gate: stack(hidden, width),
            up: stack(hidden, width),
            down: stack(width, hidden),
            shared: (shared > 0)
                .then(|| SwiGlu::init(hidden, shared * width, t, r))
                .into_iter()
                .collect(),
            top_k,
            sigmoid,
            normalize: true,
            scale: 1.,
        }
    }
    /// Returns `[tokens,k]` expert indices and combination weights.
    pub fn route(&self, x: &Array) -> (Array, Array) {
        let d = dims(x);
        assert_eq!(d.len(), 2);
        let f = work(x.dtype());
        let logits = project(&x.astype(f), &self.router.weight.value().astype(f));
        let logits = match self.router.bias() {
            Some(bias) => logits + bias.value().astype(f),
            None => logits,
        };
        let scores = if self.sigmoid {
            activations::sigmoid(&logits)
        } else {
            activations::softmax(&logits)
        };
        let choice = if self.sigmoid {
            &scores + self.correction.get()
        } else {
            scores.clone()
        };
        let (_, idx) = lax::top_k(&choice, self.top_k);
        let n = dims(&scores)[1];
        let row = lax::iota([d[0], 1], 0, DType::I32)
            * i32::try_from(n).expect("expert count exceeds I32");
        let weights = scores.reshape([d[0] * n]).at(&idx + row);
        let weights = if self.normalize {
            &weights / sum_keep(&weights, &[1])
        } else {
            weights
        };
        (idx, weights * self.scale)
    }
    pub fn forward(&self, x: Array) -> Array {
        let d = dims(&x);
        let hidden = *d.last().unwrap();
        let tokens = d[..d.len() - 1].iter().product();
        let flat = x.reshape([tokens, hidden]);
        let (ids, weights) = self.route(&flat);
        let k = self.top_k;
        let input = lax::broadcast_in_dim(&flat, [tokens, k, hidden], &[0, 2]);
        let apply = |z: &Array, w: &Param| {
            let w = w.value().at(&ids);
            lax::dot_general(
                z,
                &w,
                lax::DotDimensions {
                    lhs_batch: vec![0, 1],
                    rhs_batch: vec![0, 1],
                    lhs_contracting: vec![2],
                    rhs_contracting: vec![3],
                },
            )
        };
        let a = activations::silu(&apply(&input, &self.gate)) * apply(&input, &self.up);
        let y = apply(&a, &self.down) * weights.reshape([tokens, k, 1]).astype(x.dtype());
        let mut y = lax::reduce_sum(&y, &[1]).reshape(x.shape());
        for expert in &self.shared {
            y = y + expert.forward(x.clone());
        }
        y
    }
}
