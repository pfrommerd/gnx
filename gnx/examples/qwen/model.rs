use crate::tensor::*;
use gnx::{
    Array, Shape,
    array::{ArrayRef, DType},
    lax, np,
    ops::bounded::BoundedAxis,
};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(
    Deserialize,
    Clone,
    gnx_graph::Leaf,
    gnx_graph::GraphHash,
    gnx_graph::GraphPartialEq,
    gnx_graph::GraphEq,
)]
pub struct Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rope_parameters: Rope,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
}
#[derive(
    Deserialize,
    Clone,
    gnx_graph::Leaf,
    gnx_graph::GraphHash,
    gnx_graph::GraphPartialEq,
    gnx_graph::GraphEq,
)]
pub struct Rope {
    rope_theta: f64,
    partial_rotary_factor: f64,
}
pub type Weights = HashMap<String, Array>;
const ROOT: &str = "model.language_model";

// Each JIT receives weights as explicit array arguments: equal-shaped layers share traces.
#[gnx::jit]
fn mlp(x: Array, gate: Array, up: Array, down: Array) -> Array {
    linear(&(silu(&linear(&x, &gate)) * linear(&x, &up)), &down)
}

#[gnx::jit]
fn delta_rule(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    decay: Array,
    state: Array,
) -> (Array, Array) {
    let s = dims(&state);
    let (h, d, f) = (s[0], s[1], s[2]);
    let state = state * decay.reshape([h, 1, 1]);
    let delta = (v - dot(&k, &state, 1, 1, 1)) * beta.reshape([h, 1]);
    let next = state + k.reshape([h, d, 1]) * delta.reshape([h, 1, f]);
    (dot(&q, &next, 1, 1, 1), next)
}

impl Config {
    pub fn validate(&self, w: &Weights) -> Result<(), String> {
        let h = self.hidden_size;
        let i = self.intermediate_size;
        let d = self.head_dim;
        let q = self.num_attention_heads;
        let k = self.num_key_value_heads;
        let kh = self.linear_num_key_heads;
        let vh = self.linear_num_value_heads;
        let kd = kh * self.linear_key_head_dim;
        let vd = vh * self.linear_value_head_dim;
        let rot = (d as f64 * self.rope_parameters.partial_rotary_factor) as usize;
        if self.model_type != "qwen3_5_text"
            || self.layer_types.len() != self.num_hidden_layers
            || k == 0
            || q == 0
            || kh == 0
            || vh == 0
            || q % k != 0
            || vh % kh != 0
            || self.linear_conv_kernel_dim < 2
            || rot == 0
            || rot > d
            || rot % 2 != 0
        {
            return Err("unsupported text configuration".into());
        }
        let check = |name: String, s: &[usize]| {
            let a = w
                .get(&name)
                .ok_or_else(|| format!("missing weight {name}"))?;
            if dims(a) != s {
                return Err(format!("{name}: expected {s:?}, got {:?}", dims(a)));
            }
            Ok(())
        };
        check(format!("{ROOT}.embed_tokens.weight"), &[self.vocab_size, h])?;
        check(format!("{ROOT}.norm.weight"), &[h])?;
        if !self.tie_word_embeddings {
            check("lm_head.weight".into(), &[self.vocab_size, h])?;
        }
        for (n, ty) in self.layer_types.iter().enumerate() {
            let wt = |key: &str, s: &[usize]| check(format!("{ROOT}.layers.{n}.{key}"), s);
            wt("input_layernorm.weight", &[h])?;
            wt("post_attention_layernorm.weight", &[h])?;
            wt("mlp.gate_proj.weight", &[i, h])?;
            wt("mlp.up_proj.weight", &[i, h])?;
            wt("mlp.down_proj.weight", &[h, i])?;
            match ty.as_str() {
                "full_attention" => {
                    for (name, out) in [("q", 2 * q * d), ("k", k * d), ("v", k * d)] {
                        wt(&format!("self_attn.{name}_proj.weight"), &[out, h])?;
                    }
                    wt("self_attn.o_proj.weight", &[h, q * d])?;
                    for key in ["q", "k"] {
                        wt(&format!("self_attn.{key}_norm.weight"), &[d])?;
                    }
                }
                "linear_attention" => {
                    for (name, out) in [("qkv", 2 * kd + vd), ("z", vd), ("a", vh), ("b", vh)] {
                        wt(&format!("linear_attn.in_proj_{name}.weight"), &[out, h])?;
                    }
                    wt("linear_attn.out_proj.weight", &[h, vd])?;
                    wt("linear_attn.norm.weight", &[self.linear_value_head_dim])?;
                    wt(
                        "linear_attn.conv1d.weight",
                        &[2 * kd + vd, 1, self.linear_conv_kernel_dim],
                    )?;
                    for key in ["A_log", "dt_bias"] {
                        wt(&format!("linear_attn.{key}"), &[vh])?;
                    }
                }
                _ => return Err(format!("unsupported layer type: {ty}")),
            }
        }
        Ok(())
    }
    fn rope(&self, x: &Array, pos: &Array) -> Array {
        let d = self.head_dim;
        let r = (d as f64 * self.rope_parameters.partial_rotary_factor) as usize;
        let half = r / 2;
        let freq = lax::iota(&[half], 0, DType::F32);
        let log_scale = -self.rope_parameters.rope_theta.ln() / half as f64;
        let angles = pos.astype(DType::F32) * np::exp(&(freq * log_scale));
        let (cos, sin) = (np::cos(&angles), np::sin(&angles));
        let a = x.at((.., ..half));
        let b = x.at((.., half..r));
        np::concatenate(
            &[
                &(&a * &cos - &b * &sin),
                &(a * sin + b * cos),
                &x.at((.., r..d)),
            ],
            1,
        )
    }
    fn attention(
        &self,
        x: &Array,
        w: impl Fn(&str) -> Array,
        cache: &(ArrayRef, ArrayRef),
        pos: &Array,
        time: &BoundedAxis,
    ) -> Array {
        let (h, k, d) = (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        );
        let qg = linear(x, &w("q_proj.weight")).reshape([h, 2 * d]);
        let gate = lax::logistic(&qg.at((.., d..)));
        let q = self.rope(
            &rms(&qg.at((.., ..d)), &w("q_norm.weight"), self.rms_norm_eps),
            pos,
        );
        let key = self.rope(
            &rms(
                &linear(x, &w("k_proj.weight")).reshape([k, d]),
                &w("k_norm.weight"),
                self.rms_norm_eps,
            ),
            pos,
        );
        let append = |slot: &ArrayRef, v: Array| {
            let v = v.reshape([k, 1, d]);
            let zero = Array::scalar(0_i32);
            let updated = lax::dynamic_update_slice(&slot.get(), &v, &[&zero, pos, &zero]);
            slot.set(updated.clone());
            updated
        };
        let key = append(&cache.0, key);
        let value = append(&cache.1, linear(x, &w("v_proj.weight")));
        let value = time.prefix(&value, 1).padded(0.);
        let scores = dot(&q.reshape([k, h / k, d]), &key, 2, 2, 1) * (d as f64).sqrt().recip();
        let scores = time.prefix(&scores, 2).padded(f64::NEG_INFINITY);
        let probs = np::softmax(&scores);
        let y = dot(&probs, &value, 2, 1, 1).reshape([h, d]) * gate;
        linear(&y.reshape([h * d]), &w("o_proj.weight"))
    }
    fn recurrent(
        &self,
        x: &Array,
        w: impl Fn(&str) -> Array,
        prev: &[Array],
    ) -> (Array, [Array; 2]) {
        let (kh, h, d, f, t) = (
            self.linear_num_key_heads,
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_conv_kernel_dim,
        );
        let kd = kh * d;
        let vd = h * f;
        let c = 2 * kd + vd;
        let proj = |name: &str| linear(x, &w(&format!("in_proj_{name}.weight")));
        let window = np::concatenate(&[&prev[0], &proj("qkv").reshape([c, 1])], 1);
        let qkv = silu(&np::sum(
            &(&window * w("conv1d.weight").astype(DType::F32).reshape([c, t])),
            1,
        ));
        let heads = |start| {
            let x = norm(&qkv.at(start..start + kd).reshape([kh, d]), 1., 1e-6);
            lax::broadcast_in_dim(&x, Shape::fixed([kh, h / kh, d]), &[0, 2]).reshape([h, d])
        };
        let q = heads(0) * (d as f64).sqrt().recip();
        let k = heads(kd);
        let v = qkv.at(2 * kd..c).reshape([h, f]);
        let a = proj("a") + w("dt_bias").astype(DType::F32);
        let softplus = lax::max(&a, &np::zeros(dims(&a), a.dtype()))
            + np::log1p(&np::exp(&np::negative(&np::abs(&a))));
        let rate = np::exp(&w("A_log").astype(DType::F32));
        let decay = np::exp(&np::negative(&(rate * softplus)));
        let state = prev[1].clone();
        let (out, state) = delta_rule(q, k, v, lax::logistic(&proj("b")), decay, state);
        let gated = norm(&out, f as f64, self.rms_norm_eps)
            * w("norm.weight").astype(DType::F32)
            * silu(&proj("z").reshape([h, f]));
        (
            linear(&gated.reshape([vd]), &w("out_proj.weight")),
            [window.at((.., 1..t)), state],
        )
    }
    pub fn state_shapes(&self, capacity: usize) -> impl Iterator<Item = Vec<usize>> + '_ {
        self.layer_types.iter().flat_map(move |ty| {
            if ty == "full_attention" {
                let kv = vec![self.num_key_value_heads, capacity, self.head_dim];
                [kv.clone(), kv]
            } else {
                let (h, d, f) = (
                    self.linear_num_value_heads,
                    self.linear_key_head_dim,
                    self.linear_value_head_dim,
                );
                [
                    vec![
                        2 * self.linear_num_key_heads * d + h * f,
                        self.linear_conv_kernel_dim - 1,
                    ],
                    vec![h, d, f],
                ]
            }
        })
    }
    fn logits(&self, w: &Weights, id: &Array, (cursor, capacity, layers): &DecodeState) -> Array {
        let pos = cursor.get();
        let length = &pos + 1;
        let time = BoundedAxis::new(*capacity, &length);
        let embed = &w[&format!("{ROOT}.embed_tokens.weight")];
        let row = lax::dynamic_slice(embed, &[id, &Array::scalar(0_i32)], &[1, self.hidden_size]);
        let mut x = row.astype(DType::F32).reshape([self.hidden_size]);
        for (n, ty) in self.layer_types.iter().enumerate() {
            let wt = |key: &str| w[&format!("{ROOT}.layers.{n}.{key}")].clone();
            let y = rms(&x, &wt("input_layernorm.weight"), self.rms_norm_eps);
            let slots = &layers[n];
            let y = if ty == "full_attention" {
                self.attention(
                    &y,
                    |key| wt(&format!("self_attn.{key}")),
                    slots,
                    &pos,
                    &time,
                )
            } else {
                let (y, [history, carry]) = self.recurrent(
                    &y,
                    |key| wt(&format!("linear_attn.{key}")),
                    &[slots.0.get(), slots.1.get()],
                );
                slots.0.set(history);
                slots.1.set(carry);
                y
            };
            x = x + y;
            let y = rms(
                &x,
                &wt("post_attention_layernorm.weight"),
                self.rms_norm_eps,
            );
            x = x + mlp(
                y,
                wt("mlp.gate_proj.weight"),
                wt("mlp.up_proj.weight"),
                wt("mlp.down_proj.weight"),
            );
        }
        let logits = {
            let y = rms(&x, &w[&format!("{ROOT}.norm.weight")], self.rms_norm_eps);
            linear(
                &y,
                if self.tie_word_embeddings {
                    embed
                } else {
                    &w["lm_head.weight"]
                },
            )
        };
        cursor.set(length);
        logits
    }
}

/// Cursor, time bound, and two persistent slots per model layer.
pub type DecodeState = (ArrayRef, usize, Vec<(ArrayRef, ArrayRef)>);
pub type Model = (Config, Vec<(String, Array)>);

/// One traced computation for prompt and decode tokens, including state writes.
#[gnx::jit]
pub fn step(model: &Model, id: &Array, state: &DecodeState) -> Array {
    let (config, params) = model;
    let weights = params.iter().cloned().collect();
    lax::top_k(&config.logits(&weights, id, state), 1).1
}

#[cfg(test)]
#[gnx::jit]
pub fn step_logits(model: &Model, id: &Array, state: &DecodeState) -> Array {
    model
        .0
        .logits(&model.1.iter().cloned().collect(), id, state)
}
/// Prefill supplies the first prediction. Feed each non-delimiter token back
/// through the same step and collect it in a capacity-sized device buffer.
#[gnx::jit]
pub fn decode(
    model: &Model,
    first: &Array,
    state: &DecodeState,
    end: &Array,
) -> (Array, Array, Array) {
    let (cursor, capacity, _) = state;
    let capacity = *capacity;
    let limit = lax::scalar(capacity, DType::I32);
    let count = ArrayRef::new(Array::scalar(0_i32));
    let tokens = ArrayRef::new(np::zeros([capacity], DType::I32));
    let last = gnx::transforms::cf::while_array(
        &first.reshape([]),
        |id| {
            let pending = lax::ne(id, end);
            let room = lax::lt(&cursor.get(), &limit);
            lax::bitwise_and(&pending, &room)
        },
        |id| {
            let n = count.get();
            tokens.set(lax::dynamic_update_slice(
                &tokens.get(),
                &id.reshape([1]),
                &[&n],
            ));
            count.set(n + 1);
            step(model, id, state).reshape([])
        },
    )
    .expect("decoder while loop has a scalar token carry");
    (tokens.get(), count.get(), last)
}
