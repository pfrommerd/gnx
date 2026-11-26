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
    let model = pack_model(model, w);
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
            let logits = model::step_logits(&model, &index(id as i32)?, &state);
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
        model: pack_model(cfg, weights),
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
