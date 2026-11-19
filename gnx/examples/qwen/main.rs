//! Greedy text completion with Qwen3.5-0.8B or Qwen3.8-27B on GNX/XLA.
mod model;
mod tensor;

use clap::Parser;
use futures_executor::block_on;
use gnx::{
    Array, Eval, ExecOpts, Shape,
    array::{ArrayInfo, DType},
    buffer::{DataHandle, HostBuffer, TransferPolicy},
};
use gnx_xla::XlaDevice;
use model::{Config, Weights};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};
use tokenizers::Tokenizer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[derive(Deserialize)]
struct Checkpoint {
    text_config: Config,
}

fn weights(dir: &Path) -> Result<Weights> {
    let mut weights = Weights::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_none_or(|s| s != "safetensors") {
            continue;
        }
        let bytes = fs::read(&path)?;
        let shard = SafeTensors::deserialize(&bytes)?;
        for (name, v) in shard.tensors() {
            if !name.starts_with("model.language_model.") && name != "lm_head.weight" {
                continue;
            }
            let dtype = match v.dtype() {
                safetensors::Dtype::BF16 => DType::BF16,
                safetensors::Dtype::F32 => DType::F32,
                other => return Err(format!("{name}: unsupported weight dtype {other:?}").into()),
            };
            let info = ArrayInfo::new(Shape::fixed(v.shape().iter().copied()), dtype);
            let array = Array::from_data(DataHandle::new(HostBuffer::copy_from_slice(
                info,
                v.data(),
            )?));
            if weights.insert(name.clone(), array).is_some() {
                return Err(format!("duplicate weight {name} in {}", path.display()).into());
            }
        }
    }
    Ok(weights)
}
/// Generate text or chat with a local Qwen checkpoint.
#[derive(Parser)]
#[command(name = "qwen")]
struct Args {
    /// Run on a CUDA-visible GPU (default ordinal: 0). Omit to use the CPU.
    #[arg(long, num_args = 0..=1, default_missing_value = "0", value_name = "ORDINAL")]
    cuda: Option<usize>,
    /// Directory containing the checkpoint, config.json and tokenizer.json.
    #[arg(long, value_name = "DIR", default_value = "Qwen3.5-0.8B")]
    model_dir: PathBuf,
    /// Text to complete in one-shot mode.
    #[arg(long, value_name = "TEXT", default_value = "The capital of France is")]
    prompt: String,
    /// Start an interactive conversation.
    #[arg(long, conflicts_with = "prompt")]
    chat: bool,
    /// Maximum tokens to generate per reply.
    #[arg(long, value_name = "N", default_value_t = 16)]
    max_new_tokens: usize,
}

fn main() -> Result<()> {
    let Args {
        model_dir: dir,
        prompt,
        chat,
        max_new_tokens: limit,
        cuda,
    } = Args::parse();
    let device = match cuda {
        Some(ordinal) => XlaDevice::cuda(ordinal)?,
        None => XlaDevice::cpu()?,
    };
    let config_path = dir.join("config.json");
    let config = fs::read(&config_path).map_err(|e| {
        format!(
            "cannot read {}: {e}. Check --model-dir; it must point to a downloaded Qwen checkpoint",
            config_path.display()
        )
    })?;
    let Checkpoint { text_config: model } = serde_json::from_slice(&config)?;
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("cannot load {}: {e}", tokenizer_path.display()))?;
    let mut w = weights(&dir)?;
    model.validate(&w)?;
    if cuda.is_some() {
        // Upload each weight once; subsequent token steps reuse its device buffer.
        for array in w.values_mut() {
            *array =
                Array::from_data(device.import(array.data().unwrap(), TransferPolicy::AllowCopy)?);
        }
    }
    let session = Decoder {
        model,
        w,
        device,
        tokenizer,
    };
    if chat {
        return session.chat(limit);
    }
    let encoding = session.tokenizer.encode(prompt.as_str(), false)?;
    let generated = session.generate(encoding.get_ids(), limit, |_| Ok(()))?;
    println!("{prompt}{}", session.tokenizer.decode(&generated, true)?);
    Ok(())
}

struct Decoder {
    model: Config,
    w: Weights,
    device: XlaDevice,
    tokenizer: Tokenizer,
}
impl Decoder {
    fn chat(&self, limit: usize) -> Result<()> {
        let mut history = String::new();
        eprintln!("Enter a message. /reset clears the conversation; /quit or Ctrl-D exits.");
        loop {
            print!("You: ");
            io::stdout().flush()?;
            let mut line = String::new();
            if io::stdin().read_line(&mut line)? == 0 {
                break;
            }
            let message = line.trim();
            match message {
                "/quit" => break,
                "/reset" => {
                    history.clear();
                    continue;
                }
                "" => continue,
                _ => (),
            }
            // Text-only Qwen chat template with thinking disabled. Past assistant
            // messages contain the answer without the empty thinking prefix.
            let turn = format!("<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n");
            let prompt = format!("{history}{turn}<think>\n\n</think>\n\n");
            let encoded = self.tokenizer.encode(prompt.as_str(), false)?;
            if !self.fits(encoded.len(), limit) {
                eprintln!("Context is full. Use /reset or enter a shorter message.");
                continue;
            }
            print!("Assistant: ");
            io::stdout().flush()?;
            let mut stream = self.tokenizer.decode_stream(true);
            let ids = self.generate(encoded.get_ids(), limit, |id| {
                if let Some(text) = stream.step(id)? {
                    print!("{text}");
                    io::stdout().flush()?;
                }
                Ok(())
            })?;
            println!();
            history.push_str(&turn);
            history.push_str(self.tokenizer.decode(&ids, true)?.trim());
            history.push_str("<|im_end|>\n");
        }
        Ok(())
    }

    fn fits(&self, prompt_len: usize, limit: usize) -> bool {
        prompt_len.saturating_add(limit.saturating_sub(1)) <= self.model.max_position_embeddings
    }
    fn generate(
        &self,
        ids: &[u32],
        limit: usize,
        mut on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<Vec<u32>> {
        let Self {
            model,
            w,
            device,
            tokenizer,
        } = self;
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= model.vocab_size) {
            return Err("expected a nonempty text prompt with valid token IDs".into());
        }
        if !self.fits(ids.len(), limit) {
            return Err("requested sequence exceeds model context length".into());
        }
        if limit == 0 {
            return Ok(vec![]);
        }
        let mut cache = vec![];
        let mut generated = vec![];
        let mut id = ids[0];
        for pos in 0..ids.len() + limit - 1 {
            let predict = pos + 1 >= ids.len();
            let (next, token) = model.step(w, id, pos, &cache, predict);
            // Materialize all cache entries together, keeping the allocations on the device.
            let mut roots = next.clone();
            roots.extend(token.iter().cloned());
            block_on(device.eval(&roots, ExecOpts::default()))?;
            cache = next;
            if let Some(token) = token {
                id =
                    u32::from_le_bytes(device.download(token.data().unwrap())?.bytes().try_into()?);
                if id == model.eos_token_id || Some(id) == tokenizer.token_to_id("<|im_end|>") {
                    break;
                }
                on_token(id)?;
                generated.push(id);
            } else {
                id = ids[pos + 1];
            }
        }
        Ok(generated)
    }
}
