use super::*;

const EMBEDDING: [[f32; 2]; 3] = [[1., 0.], [0., 2.], [-2., 1.]];
fn fixture() -> (Config, Weights) {
    let cfg=serde_json::from_value(serde_json::json!({"model_type":"qwen3_5_text","hidden_size":2,"intermediate_size":2,"vocab_size":3,"num_hidden_layers":1,"layer_types":["full_attention"],"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"linear_num_key_heads":1,"linear_num_value_heads":1,"linear_key_head_dim":2,"linear_value_head_dim":2,"linear_conv_kernel_dim":2,"rope_parameters":{"rope_theta":10000.,"partial_rotary_factor":1.},"rms_norm_eps":0.00001,"tie_word_embeddings":true,"eos_token_id":3,"max_position_embeddings":4})).unwrap();
    let mut w = Weights::new();
    let mut add = |key: &str, shape: &[usize], v: &[f32]| {
        let host = HostBuffer::copy_from_slice(
            ArrayInfo::new(Shape::fixed(shape.iter().copied()), DType::F32),
            &v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap();
        w.insert(
            format!("model.language_model.{key}"),
            Array::from_data(DataHandle::new(host)),
        );
    };
    add("embed_tokens.weight", &[3, 2], &EMBEDDING.concat());
    for k in [
        "norm",
        "layers.0.input_layernorm",
        "layers.0.post_attention_layernorm",
        "layers.0.self_attn.k_norm",
        "layers.0.self_attn.q_norm",
    ] {
        add(&format!("{k}.weight"), &[2], &[0.; 2]);
    }
    for k in ["k", "v", "o"] {
        add(
            &format!("layers.0.self_attn.{k}_proj.weight"),
            &[2, 2],
            &[1., 0., 0., 1.],
        );
    }
    add(
        "layers.0.self_attn.q_proj.weight",
        &[4, 2],
        &[1., 0., 0., 1., 0., 0., 0., 0.],
    );
    for k in ["gate", "up", "down"] {
        add(&format!("layers.0.mlp.{k}_proj.weight"), &[2, 2], &[0.; 4]);
    }
    (cfg, w)
}

/// Independent unpadded reference for the one-head fixture: identity Q/K/V
/// projections, gate sigmoid(0), zero MLP, tied output weights.
fn reference(ids: &[usize]) -> Vec<f32> {
    fn norm(x: [f32; 2]) -> [f32; 2] {
        let scale = (0.5 * (x[0] * x[0] + x[1] * x[1]) + 1e-5).sqrt();
        x.map(|v| v / scale)
    }
    let values: Vec<_> = ids.iter().map(|&id| norm(EMBEDDING[id])).collect();
    let keys: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let [a, b] = norm(x);
            let (s, c) = (i as f32).sin_cos();
            [a * c - b * s, b * c + a * s]
        })
        .collect();
    let dot = |a: [f32; 2], b: [f32; 2]| a[0] * b[0] + a[1] * b[1];
    let q = *keys.last().unwrap();
    let weights: Vec<_> = keys
        .iter()
        .map(|&k| (dot(q, k) / 2f32.sqrt()).exp())
        .collect();
    let total = weights.iter().sum::<f32>();
    let mut result = EMBEDDING[*ids.last().unwrap()];
    for (weight, value) in weights.iter().zip(values) {
        for j in 0..2 {
            result[j] += 0.5 * weight / total * value[j];
        }
    }
    EMBEDDING.iter().map(|&e| dot(e, norm(result))).collect()
}
fn check_logits<D: QwenDevice>(device: D) -> Result<()> {
    let (model, w) = fixture();
    let capacity = 4;
    let model = load_model(model, w, &device)?;
    let mut first_program = None;
    for ids in [[0, 2, 1, 0], [1, 0, 2, 1]] {
        let init = HostBuffer::new(
            ArrayInfo::new(Shape::fixed([1, capacity, 2]), DType::F32),
            f32::NAN.to_le_bytes().repeat(capacity * 2),
        )?;
        let backing = device.upload(&DataHandle::new(init))?;
        let slot = || ArrayRef::new(Array::from_data(backing.clone()));
        let state = (ArrayRef::new(index(0)?), capacity, vec![(slot(), slot())]);
        for (pos, &id) in ids.iter().enumerate() {
            let logits = model.logits(&index(id as i32)?, &state);
            let hash = gnx::expr(&(&logits, &state))?
                .prepare()?
                .into_parts()
                .0
                .hash();
            assert_eq!(
                *first_program.get_or_insert(hash),
                hash,
                "position and fresh state must not specialize the executable"
            );
            block_on(device.eval(
                &(&logits, &state),
                ExecOpts {
                    transfers: TransferPolicy::AllowCopy,
                },
            ))?;
            let logits = device.readback(logits.data().unwrap())?;
            for (actual, expected) in logits.bytes().chunks_exact(4).zip(reference(&ids[..=pos])) {
                let actual = f32::from_le_bytes(actual.try_into()?);
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "position {pos}: {actual} vs {expected}"
                );
            }
        }
    }
    Ok(())
}
// A tiny model with an untied output head and deterministic transition
// 2 -> 0 -> 1 -> 2. Exercise the actual prefill, compiled loop and readback.
fn check_reply<D: QwenDevice>(device: D) -> Result<()> {
    let (mut cfg, mut weights) = fixture();
    cfg.tie_word_embeddings = false;
    let head = [EMBEDDING[2], EMBEDDING[0], EMBEDDING[1]].concat();
    weights.insert(
        "lm_head.weight".into(),
        Array::from_data(DataHandle::new(HostBuffer::copy_from_slice(
            ArrayInfo::new(Shape::fixed([3, 2]), DType::F32),
            &head
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?)),
    );
    let mut decoder = Decoder {
        capacity: 4,
        timings: false,
        model: load_model(cfg, weights, &device)?,
        device,
        tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
        end: 2,
    };
    assert_eq!(decoder.generate(&[1])?, Vec::<u32>::new()); // first prediction is the delimiter
    assert_eq!(decoder.generate(&[1, 2])?, [0, 1]); // delimiter predicted at the last cache position
    decoder.capacity = 2;
    assert!(decoder.generate(&[2]).unwrap_err().is::<ContextExhausted>());
    decoder.capacity = 1;
    assert!(decoder.generate(&[1])?.is_empty()); // delimiter wins even with no remaining space
    Ok(())
}
// Check the depthwise kernel layout and both recurrent slots against a scalar
// reference. Q/K/V/Z and output projections are identity; beta and decay are 1/2.
#[test]
#[cfg(feature = "qwen-torch")]
fn torch_delta_logits() -> Result<()> {
    let device = gnx_torch::TorchDevice::cpu(gnx::cache::Cache::in_memory())?;
    let (mut config, mut w) = fixture();
    config.layer_types[0] = "linear_attention".into();
    let mut add = |key: &str, shape: &[usize], values: &[f32]| -> Result<()> {
        let host = HostBuffer::copy_from_slice(
            ArrayInfo::new(Shape::fixed(shape.iter().copied()), DType::F32),
            &values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        w.insert(
            format!("model.language_model.layers.0.linear_attn.{key}"),
            Array::from_data(DataHandle::new(host)),
        );
        Ok(())
    };
    let identity = [1., 0., 0., 1.];
    add("in_proj_qkv.weight", &[6, 2], &identity.repeat(3))?;
    for key in ["in_proj_z.weight", "out_proj.weight"] {
        add(key, &[2, 2], &identity)?;
    }
    for key in ["in_proj_a.weight", "in_proj_b.weight"] {
        add(key, &[1, 2], &[0.; 2])?;
    }
    add("conv1d.weight", &[6, 1, 2], &[0.5, 1.].repeat(6))?;
    add("norm.weight", &[2], &[1.; 2])?;
    for key in ["A_log", "dt_bias"] {
        add(key, &[1], &[0.])?;
    }
    let model = load_model(config, w, &device)?;
    let slots = new_state(&model.config, 2, &device)?;
    let mut prev = [0.; 2];
    let mut state = [[0.; 2]; 2];
    let silu = |v: f32| v / (1. + (-v).exp());
    let rms = |v: [f32; 2]| v.map(|a| a / ((v[0] * v[0] + v[1] * v[1]) * 0.5 + 1e-5).sqrt());
    for id in [0, 1] {
        let x = rms(EMBEDDING[id]);
        let v: [f32; 2] = std::array::from_fn(|i| silu(prev[i] * 0.5 + x[i]));
        prev = x;
        let k = v.map(|a| a / (v[0] * v[0] + v[1] * v[1] + 1e-6).sqrt());
        let mut out = [0.; 2];
        for j in 0..2 {
            let delta = 0.5 * (v[j] - 0.5 * (k[0] * state[0][j] + k[1] * state[1][j]));
            for i in 0..2 {
                state[i][j] = 0.5 * state[i][j] + k[i] * delta;
                out[j] += k[i] * state[i][j] / 2f32.sqrt();
            }
        }
        let out = rms(out);
        let hidden = rms(std::array::from_fn(|i| {
            EMBEDDING[id][i] + out[i] * silu(x[i])
        }));
        let logits = model.logits(&index(id as i32)?, &slots);
        block_on(device.eval(
            &(&logits, &slots),
            ExecOpts {
                transfers: TransferPolicy::AllowCopy,
            },
        ))?;
        let actual = device.readback(logits.data().unwrap())?;
        for (bytes, e) in actual.bytes().chunks_exact(4).zip(EMBEDDING) {
            let expected = e[0] * hidden[0] + e[1] * hidden[1];
            assert!((f32::from_le_bytes(bytes.try_into()?) - expected).abs() < 2e-5);
        }
    }
    Ok(())
}

#[cfg(feature = "qwen-example")]
#[test]
fn xla_reply_loop() -> Result<()> {
    check_reply(gnx_xla::XlaDevice::cpu(gnx::cache::Cache::in_memory())?)
}
#[cfg(feature = "qwen-torch")]
#[test]
fn torch_reply_loop() -> Result<()> {
    check_reply(gnx_torch::TorchDevice::cpu(gnx::cache::Cache::in_memory())?)
}

#[test]
#[cfg(feature = "qwen-example")]
fn xla_decoder_logits() -> Result<()> {
    check_logits(gnx_xla::XlaDevice::cpu(gnx::cache::Cache::in_memory())?)
}
#[test]
#[cfg(feature = "qwen-torch")]
fn torch_decoder_logits() -> Result<()> {
    check_logits(gnx_torch::TorchDevice::cpu(gnx::cache::Cache::in_memory())?)
}

#[cfg(feature = "qwen-example")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_decoder_logits() -> Result<()> {
    check_logits(gnx_xla::XlaDevice::cuda(0, gnx::cache::Cache::in_memory())?)?;
    check_logits(gnx_torch::TorchDevice::cuda(
        0,
        gnx::cache::Cache::in_memory(),
    )?)
}
