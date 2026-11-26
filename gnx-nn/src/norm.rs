//! Channels-last normalization. Half-precision inputs accumulate statistics in F32.
use crate::{Param, tensor::*};
use gnx_graph::Graph;
use gnx_math::{Array, ArrayRef, DType, lax};
fn mean(x: &Array, axes: &[usize]) -> Array {
    let d = dims(x);
    let n = axes.iter().map(|&a| d[a]).product::<usize>();
    assert!(n > 0);
    sum_keep(x, axes) * (1. / n as f64)
}
fn normalize(x: &Array, axes: &[usize], eps: f64, center: bool) -> Array {
    let z = if center { x - mean(x, axes) } else { x.clone() };
    let var = mean(&(&z * &z), axes);
    z * lax::rsqrt(&(var + eps))
}
fn weight(n: usize, t: DType) -> Param {
    assert!(n > 0);
    Param::new(constant(1., [n], t))
}
fn input(x: &Array, p: &Param) -> Array {
    assert_eq!(dims(x).last(), dims(&p.value()).first());
    x.astype(work(x.dtype()))
}
#[derive(Clone, Graph)]
pub struct RmsNorm {
    pub scale: Param,
    pub eps: f64,
}
impl RmsNorm {
    pub fn init(features: usize, eps: f64, t: DType) -> Self {
        Self {
            scale: weight(features, t),
            eps,
        }
    }
    pub fn forward(&self, x: Array) -> Array {
        let z = input(&x, &self.scale);
        let z = normalize(&z, &[dims(&z).len() - 1], self.eps, false);
        z.astype(x.dtype()) * self.scale.value()
    }
}
#[derive(Clone, Graph)]
pub struct LayerNorm {
    pub scale: Param,
    pub bias: Param,
    pub eps: f64,
}
impl LayerNorm {
    pub fn init(features: usize, eps: f64, t: DType) -> Self {
        Self {
            scale: weight(features, t),
            bias: Param::new(lax::zeros([features], t)),
            eps,
        }
    }
    pub fn forward(&self, x: Array) -> Array {
        let z = input(&x, &self.scale);
        let z = normalize(&z, &[dims(&z).len() - 1], self.eps, true);
        z.astype(x.dtype()) * self.scale.value() + self.bias.value()
    }
}
#[derive(Clone, Graph)]
pub struct GroupNorm {
    pub scale: Param,
    pub bias: Param,
    pub groups: usize,
    pub eps: f64,
}
impl GroupNorm {
    pub fn init(features: usize, groups: usize, eps: f64, t: DType) -> Self {
        assert!(groups > 0 && features % groups == 0);
        Self {
            scale: weight(features, t),
            bias: Param::new(lax::zeros([features], t)),
            groups,
            eps,
        }
    }
    pub fn forward(&self, x: Array) -> Array {
        let z = input(&x, &self.scale);
        let mut d = dims(&z);
        assert!(d.len() >= 2);
        let c = d.pop().unwrap();
        let mut axes = (1..d.len()).collect::<Vec<_>>();
        d.extend([self.groups, c / self.groups]);
        axes.push(d.len() - 1);
        let z = normalize(&z.reshape(d), &axes, self.eps, true).reshape(x.shape());
        z.astype(x.dtype()) * self.scale.value() + self.bias.value()
    }
}
#[derive(Clone, Graph)]
pub struct BatchNorm {
    pub scale: Param,
    pub bias: Param,
    pub running_mean: ArrayRef,
    pub running_var: ArrayRef,
    pub momentum: f64,
    pub eps: f64,
}
impl BatchNorm {
    /// `momentum` is the fraction of the previous running statistic retained.
    pub fn init(features: usize, momentum: f64, eps: f64, t: DType) -> Self {
        assert!((0.0..=1.).contains(&momentum));
        Self {
            scale: weight(features, t),
            bias: Param::new(lax::zeros([features], t)),
            running_mean: ArrayRef::new(lax::zeros([features], work(t))),
            running_var: ArrayRef::new(constant(1., [features], work(t))),
            momentum,
            eps,
        }
    }
    pub fn forward(&self, x: Array, training: bool) -> Array {
        let z = input(&x, &self.scale);
        let (m, v) = if training {
            let d = dims(&z);
            assert!(d.len() >= 2);
            let axes = (0..d.len() - 1).collect::<Vec<_>>();
            let m = mean(&z, &axes);
            let delta = &z - &m;
            let v = mean(&(&delta * &delta), &axes);
            let update = |cell: &ArrayRef, new: &Array| {
                let old = cell.get();
                cell.set(&old * self.momentum + new.reshape(old.shape()) * (1. - self.momentum))
            };
            update(&self.running_mean, &m);
            update(&self.running_var, &v);
            (m, v)
        } else {
            (self.running_mean.get(), self.running_var.get())
        };
        let y = (&z - m) * lax::rsqrt(&(v + self.eps));
        y.astype(x.dtype()) * self.scale.value() + self.bias.value()
    }
}
