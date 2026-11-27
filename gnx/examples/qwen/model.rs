use crate::tensor::*;
use gnx::{
    Array, Graph, Shape,
    array::{ArrayRef, DType},
    lax, nn,
    nn::activations::{sigmoid, silu},
    np,
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

#[derive(Clone, Graph)]
pub struct Model {
    pub config: Config,
    embedding: nn::Embedding,
    blocks: Vec<Block>,
    norm: nn::RmsNorm,
    head: nn::Linear,
}

#[derive(Clone, Graph)]
struct Block {
    input_norm: nn::RmsNorm,
    // Graph derives support structs; exactly one attention branch is populated.
    attention: Option<FullAttention>,
    recurrent: Option<DeltaAttention>,
    post_norm: nn::RmsNorm,
    mlp: nn::SwiGlu,
}

#[derive(Clone, Graph)]
struct FullAttention {
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    out: nn::Linear,
    q_norm: nn::RmsNorm,
    k_norm: nn::RmsNorm,
    rotary: nn::Rotary,
    rotary_dim: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

#[derive(Clone, Graph)]
struct DeltaAttention {
    qkv: nn::Linear,
    z: nn::Linear,
    a: nn::Linear,
    b: nn::Linear,
    out: nn::Linear,
    conv: nn::Conv,
    norm: nn::RmsNorm,
    a_log: nn::Param,
    dt_bias: nn::Param,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel_size: usize,
}

// Project at checkpoint precision, then compute the gate and residual in F32.
fn project(layer: &nn::Linear, x: &Array) -> Array {
    layer
        .forward(x.astype(layer.weight.value().dtype()))
        .astype(DType::F32)
}

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
}

impl FullAttention {
    fn rope(&self, x: &Array, pos: &Array) -> Array {
        let d = self.head_dim;
        let r = self.rotary_dim;
        let heads = dims(x)[0];
        let prefix = x.at((.., ..r)).reshape([1, 1, heads, r]);
        let rotated = self
            .rotary
            .apply_at(&prefix, pos, false)
            .reshape([heads, r]);
        if r == d {
            rotated
        } else {
            np::concatenate(&[&rotated, &x.at((.., r..))], 1)
        }
    }
    fn forward(
        &self,
        x: &Array,
        cache: &(ArrayRef, ArrayRef),
        pos: &Array,
        time: &BoundedAxis,
    ) -> Array {
        let (h, k, d) = (self.heads, self.kv_heads, self.head_dim);
        let qg = project(&self.q, x).reshape([h, 2 * d]);
        let gate = sigmoid(&qg.at((.., d..)));
        let q = self.rope(&self.q_norm.forward(qg.at((.., ..d))), pos);
        let key = self.rope(
            &self.k_norm.forward(project(&self.k, x).reshape([k, d])),
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
        let value = append(&cache.1, project(&self.v, x));
        let value = time.prefix(&value, 1).padded(0.);
        let capacity = dims(&key)[1];
        let kv = |a: &Array| lax::transpose(a, &[1, 0, 2]).reshape([1, capacity, k, d]);
        let keep = lax::broadcast_in_dim(&Array::scalar(true), [1, capacity], &[]);
        let mask = time.prefix(&keep, 1).padded(0.);
        let attended = nn::scaled_dot_product_attention(
            &q.reshape([1, 1, h, d]),
            &kv(&key),
            &kv(&value),
            (d as f64).sqrt().recip(),
            false,
            Some(&mask),
        );
        let y = attended.reshape([h, d]) * gate;
        project(&self.out, &y.reshape([h * d]))
    }
}

impl DeltaAttention {
    #[gnx::jit]
    fn forward(&self, x: &Array, slots: &(ArrayRef, ArrayRef)) -> Array {
        let (kh, h, d, f, t) = (
            self.key_heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
            self.kernel_size,
        );
        let kd = kh * d;
        let vd = h * f;
        let c = 2 * kd + vd;
        let window = np::concatenate(&[&slots.0.get(), &project(&self.qkv, x).reshape([c, 1])], 1);
        let input = lax::transpose(&window, &[1, 0]).reshape([1, t, c]);
        let qkv = silu(&self.conv.forward(input).reshape([c]));
        let heads = |start| {
            let x = qkv.at(start..start + kd).reshape([kh, d]);
            // Delta-rule keys and queries use L2 normalization, not RMSNorm.
            let x = &x * lax::rsqrt(&(np::sum(&(&x * &x), -1).reshape([kh, 1]) + 1e-6));
            lax::broadcast_in_dim(&x, Shape::fixed([kh, h / kh, d]), &[0, 2]).reshape([h, d])
        };
        let q = heads(0) * (d as f64).sqrt().recip();
        let k = heads(kd);
        let v = qkv.at(2 * kd..c).reshape([h, f]);
        let a = project(&self.a, x) + self.dt_bias.value();
        let softplus = lax::max(&a, &np::zeros(dims(&a), a.dtype()))
            + np::log1p(&np::exp(&np::negative(&np::abs(&a))));
        let rate = np::exp(&self.a_log.value());
        let decay = np::exp(&np::negative(&(rate * softplus)));
        let (out, state) = delta_rule(q, k, v, sigmoid(&project(&self.b, x)), decay, slots.1.get());
        slots.0.set(window.at((.., 1..t)));
        slots.1.set(state);
        let gated = self.norm.forward(out) * silu(&project(&self.z, x).reshape([h, f]));
        project(&self.out, &gated.reshape([vd]))
    }
}

impl Config {
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
}

impl Model {
    pub fn from_weights(config: Config, weights: Weights) -> Result<Self, String> {
        config.validate(&weights)?;
        let embedding = nn::Embedding {
            weight: nn::Param::new(weights[&format!("{ROOT}.embed_tokens.weight")].clone()),
        };
        let head = if config.tie_word_embeddings {
            embedding.tied_head()
        } else {
            nn::Linear::new(weights["lm_head.weight"].clone(), None)
        };
        // The checkpoint stores ordinary RMSNorm scales as offsets from one.
        let norm = |w: Array| nn::RmsNorm {
            scale: nn::Param::new(w.astype(DType::F32) + 1.),
            eps: config.rms_norm_eps,
        };
        let final_norm = norm(weights[&format!("{ROOT}.norm.weight")].clone());
        let blocks = config
            .layer_types
            .iter()
            .enumerate()
            .map(|(i, kind)| {
                let weight = |key: &str| weights[&format!("{ROOT}.layers.{i}.{key}")].clone();
                let linear = |key: &str| nn::Linear::new(weight(&format!("{key}.weight")), None);
                let (attention, recurrent) = if kind == "full_attention" {
                    (
                        Some(FullAttention {
                            q: linear("self_attn.q_proj"),
                            k: linear("self_attn.k_proj"),
                            v: linear("self_attn.v_proj"),
                            out: linear("self_attn.o_proj"),
                            q_norm: norm(weight("self_attn.q_norm.weight")),
                            k_norm: norm(weight("self_attn.k_norm.weight")),
                            rotary: nn::Rotary::new(config.rope_parameters.rope_theta),
                            rotary_dim: (config.head_dim as f64
                                * config.rope_parameters.partial_rotary_factor)
                                as usize,
                            heads: config.num_attention_heads,
                            kv_heads: config.num_key_value_heads,
                            head_dim: config.head_dim,
                        }),
                        None,
                    )
                } else {
                    let kernel = weight("linear_attn.conv1d.weight").astype(DType::F32);
                    let channels = dims(&kernel)[0];
                    (
                        None,
                        Some(DeltaAttention {
                            qkv: linear("linear_attn.in_proj_qkv"),
                            z: linear("linear_attn.in_proj_z"),
                            a: linear("linear_attn.in_proj_a"),
                            b: linear("linear_attn.in_proj_b"),
                            out: linear("linear_attn.out_proj"),
                            conv: nn::Conv::new(
                                lax::transpose(&kernel, &[2, 1, 0]),
                                None,
                                channels,
                            ),
                            // The gated delta-rule norm stores its scale directly.
                            norm: nn::RmsNorm {
                                scale: nn::Param::new(
                                    weight("linear_attn.norm.weight").astype(DType::F32),
                                ),
                                eps: config.rms_norm_eps,
                            },
                            a_log: nn::Param::new(weight("linear_attn.A_log").astype(DType::F32)),
                            dt_bias: nn::Param::new(
                                weight("linear_attn.dt_bias").astype(DType::F32),
                            ),
                            key_heads: config.linear_num_key_heads,
                            value_heads: config.linear_num_value_heads,
                            key_dim: config.linear_key_head_dim,
                            value_dim: config.linear_value_head_dim,
                            kernel_size: config.linear_conv_kernel_dim,
                        }),
                    )
                };
                Block {
                    input_norm: norm(weight("input_layernorm.weight")),
                    post_norm: norm(weight("post_attention_layernorm.weight")),
                    attention,
                    recurrent,
                    mlp: nn::SwiGlu {
                        gate: linear("mlp.gate_proj"),
                        up: linear("mlp.up_proj"),
                        down: linear("mlp.down_proj"),
                    },
                }
            })
            .collect();
        Ok(Self {
            config,
            embedding,
            blocks,
            norm: final_norm,
            head,
        })
    }
    #[gnx::jit]
    pub fn logits(&self, id: &Array, state: &DecodeState) -> Array {
        let (cursor, capacity, layers) = state;
        let pos = cursor.get();
        let length = &pos + 1;
        let time = BoundedAxis::new(*capacity, &length);
        let mut x = self.embedding.forward(id).astype(DType::F32);
        assert_eq!(self.blocks.len(), layers.len());
        for (block, slots) in self.blocks.iter().zip(layers) {
            let y = block.input_norm.forward(x.clone());
            let y = match (&block.attention, &block.recurrent) {
                (Some(a), None) => a.forward(&y, slots, &pos, &time),
                (None, Some(a)) => a.forward(&y, slots),
                _ => unreachable!("block has exactly one attention layer"),
            };
            x = x + y;
            let y = block
                .mlp
                .forward_with(&block.post_norm.forward(x.clone()), project);
            x = x + y;
        }
        let logits = project(&self.head, &self.norm.forward(x));
        cursor.set(length);
        logits
    }
    /// One traced computation for prompt and decode tokens, including state writes.
    #[gnx::jit]
    pub fn step(&self, id: &Array, state: &DecodeState) -> Array {
        lax::top_k(&self.logits(id, state), 1).1
    }

    /// Prefill supplies the first prediction. Feed each non-delimiter token back
    /// through the same step and collect it in a capacity-sized device buffer.
    #[gnx::jit]
    pub fn decode(&self, first: &Array, state: &DecodeState, end: &Array) -> (Array, Array, Array) {
        let (cursor, capacity, _) = state;
        let capacity = *capacity;
        let limit = lax::scalar(capacity, DType::I32);
        let count = ArrayRef::new(Array::scalar(0_i32));
        let tokens = ArrayRef::new(np::zeros([capacity], DType::I32));
        let last = gnx::transforms::cf::while_loop(
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
                self.step(id, state).reshape([])
            },
        )
        .expect("decoder while loop has a scalar token carry");
        (tokens.get(), count.get(), last)
    }
}

/// Cursor, time bound, and two persistent slots per model layer.
pub type DecodeState = (ArrayRef, usize, Vec<(ArrayRef, ArrayRef)>);
