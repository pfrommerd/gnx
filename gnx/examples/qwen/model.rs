use crate::tensor::*;
use gnx::{
    Array,
    array::DType,
    ops::{self, elementwise as e},
};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
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
    pub eos_token_id: u32,
    pub max_position_embeddings: usize,
}
#[derive(Deserialize)]
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
    fn rope(&self, x: &Array, pos: usize) -> Array {
        let d = self.head_dim;
        let r = (d as f64 * self.rope_parameters.partial_rotary_factor) as usize;
        let half = r / 2;
        let freq = apply(ops::creation::iota(0), &[], &[half], DType::F32);
        let log_scale = scalar(-self.rope_parameters.rope_theta.ln() / half as f64);
        let angles = scalar(pos as f64) * unary(e::exp(), &(freq * log_scale));
        let (cos, sin) = (unary(e::cos(), &angles), unary(e::sin(), &angles));
        let a = view(x, 1, 0, half);
        let b = view(x, 1, half, r);
        cat(
            &cat(&(&a * &cos - &b * &sin), &(a * sin + b * cos), 1),
            &view(x, 1, r, d),
            1,
        )
    }
    fn attention(
        &self,
        x: &Array,
        w: impl Fn(&str) -> Array,
        prev: &[Array],
        pos: usize,
    ) -> (Array, [Array; 2]) {
        let (h, k, d) = (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        );
        let qg = linear(x, &w("q_proj.weight")).reshape([h, 2 * d]);
        let gate = sigmoid(&view(&qg, 1, d, 2 * d));
        let q = self.rope(
            &rms(&view(&qg, 1, 0, d), &w("q_norm.weight"), self.rms_norm_eps),
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
        let append = |v: Array, index: usize| {
            let v = v.reshape([k, 1, d]);
            if prev.is_empty() {
                v
            } else {
                cat(&prev[index], &v, 1)
            }
        };
        let key = append(key, 0);
        let value = append(linear(x, &w("v_proj.weight")), 1);
        let scores = dot(&q.reshape([k, h / k, d]), &key, 2, 2, 1) * scalar(1. / (d as f64).sqrt());
        let exp = unary(e::exp(), &(&scores - top1(&scores).0));
        let probs = &exp / keep_sum(&exp);
        let y = dot(&probs, &value, 2, 1, 1).reshape([h, d]) * gate;
        (
            linear(&y.reshape([h * d]), &w("o_proj.weight")),
            [key, value],
        )
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
        let history = prev.first().cloned().unwrap_or_else(|| zeros(&[c, t - 1]));
        let window = cat(&history, &proj("qkv").reshape([c, 1]), 1);
        let qkv = silu(&sum(
            &(&window * cast(&w("conv1d.weight"), DType::F32).reshape([c, t])),
            1,
        ));
        let heads = |start| {
            let x = norm(&view(&qkv, 0, start, start + kd).reshape([kh, d]), 1., 1e-6);
            broadcast(&x, &[kh, h / kh, d], &[0, 2]).reshape([h, d])
        };
        let q = heads(0) * scalar(1. / (d as f64).sqrt());
        let k = heads(kd);
        let v = view(&qkv, 0, 2 * kd, c).reshape([h, f]);
        let a = proj("a") + cast(&w("dt_bias"), DType::F32);
        let softplus = binary_max_zero(&a)
            + unary(
                e::log1p(),
                &unary(e::exp(), &unary(e::neg(), &unary(e::abs(), &a))),
            );
        let decay = unary(
            e::exp(),
            &unary(
                e::neg(),
                &(unary(e::exp(), &cast(&w("A_log"), DType::F32)) * softplus),
            ),
        );
        let state = prev.get(1).cloned().unwrap_or_else(|| zeros(&[h, d, f]));
        let (out, state) = delta_rule(q, k, v, sigmoid(&proj("b")), decay, state);
        let gated = norm(&out, f as f64, self.rms_norm_eps)
            * cast(&w("norm.weight"), DType::F32)
            * silu(&proj("z").reshape([h, f]));
        (
            linear(&gated.reshape([vd]), &w("out_proj.weight")),
            [view(&window, 1, 1, t), state],
        )
    }
    /// Both prompt prefill and generation consume one token and advance the same layer caches.
    pub fn step(
        &self,
        w: &Weights,
        id: u32,
        pos: usize,
        prev: &[Array],
        predict: bool,
    ) -> (Vec<Array>, Option<Array>) {
        let embed = &w[&format!("{ROOT}.embed_tokens.weight")];
        let mut x = cast(&view(embed, 0, id as usize, id as usize + 1), DType::F32)
            .reshape([self.hidden_size]);
        let mut cache = Vec::with_capacity(2 * self.num_hidden_layers);
        for (n, ty) in self.layer_types.iter().enumerate() {
            let wt = |key: &str| w[&format!("{ROOT}.layers.{n}.{key}")].clone();
            let y = rms(&x, &wt("input_layernorm.weight"), self.rms_norm_eps);
            let state = if prev.is_empty() {
                &[]
            } else {
                &prev[2 * n..2 * n + 2]
            };
            let (y, next) = if ty == "full_attention" {
                self.attention(&y, |key| wt(&format!("self_attn.{key}")), state, pos)
            } else {
                self.recurrent(&y, |key| wt(&format!("linear_attn.{key}")), state)
            };
            cache.extend(next);
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
        let token = predict.then(|| {
            let y = rms(&x, &w[&format!("{ROOT}.norm.weight")], self.rms_norm_eps);
            top1(&linear(
                &y,
                if self.tie_word_embeddings {
                    embed
                } else {
                    &w["lm_head.weight"]
                },
            ))
            .1
        });
        (cache, token)
    }
}
fn binary_max_zero(x: &Array) -> Array {
    apply(e::max(), &[x, &zeros(&dims(x))], &dims(x), DType::F32)
}
