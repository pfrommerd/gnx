//! Greedy text completion with Qwen3.5-0.8B or Qwen3.8-27B on GNX.
#[path = "../common/logging.rs"]
mod logging;
mod model;

use clap::Parser;
use futures_executor::block_on;
use gnx::{
    Array, Device, DeviceHandle, Eval, ExecOpts, Shape,
    array::{ArrayInfo, DType},
    buffer::{DataHandle, HostBuffer, TransferPolicy},
    nn::{CacheKey, KVCache},
};
use model::{Config, Model, Weights};
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
    /// Device: default (XLA CUDA 0, then CPU), xla/cpu, torch/cpu, or <backend>/cuda:N.
    #[arg(long, default_value = "default")]
    device: String,
    /// Directory containing the checkpoint, config.json and tokenizer.json.
    #[arg(long, value_name = "DIR", default_value = "Qwen3.5-0.8B")]
    model_dir: PathBuf,
    /// Raw prompt for one-shot generation, which ends at <|im_end|>.
    #[arg(
        long,
        value_name = "TEXT",
        default_value = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )]
    prompt: String,
    /// Start an interactive conversation.
    #[arg(long, conflicts_with = "prompt")]
    chat: bool,
    /// Bound for the KV time axis, shared across all turns (capped by the checkpoint).
    #[arg(long, default_value_t = 2048)]
    context_length: usize,
    /// Report time to first token and subsequent decode tokens/second.
    #[arg(long)]
    timings: bool,
}

fn main() -> Result<()> {
    let Args {
        device,
        model_dir: dir,
        prompt,
        chat,
        context_length,
        timings,
    } = Args::parse();
    let _logging = logging::init(&device, "error")?;
    let device: DeviceHandle = device.parse()?;
    let config_path = dir.join("config.json");
    let config = fs::read(&config_path).map_err(|e| {
        format!(
            "cannot read {}: {e}. Check --model-dir; it must point to a downloaded Qwen checkpoint",
            config_path.display()
        )
    })?;
    let Checkpoint { text_config: model } = serde_json::from_slice(&config)?;
    let capacity = context_length.min(model.max_position_embeddings);
    if capacity == 0 || capacity > i32::MAX as usize || model.vocab_size > i32::MAX as usize {
        return Err("context and vocabulary must fit the decoder's i32 indices".into());
    }
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("cannot load {}: {e}", tokenizer_path.display()))?;
    let end = tokenizer
        .token_to_id("<|im_end|>")
        .filter(|&id| (id as usize) < model.vocab_size)
        .ok_or("tokenizer must define a valid <|im_end|> token")?;
    let mut w = weights(&dir)?;
    // Upload once (XLA CPU can borrow the aligned host storage).
    for array in w.values_mut() {
        *array = Array::from_data(block_on(
            device.upload(array.data().unwrap(), TransferPolicy::AllowCopy),
        )?);
    }
    let session = Decoder {
        capacity,
        timings,
        model: load_model(model, w, &device)?,
        device,
        tokenizer,
        end,
    };
    if chat {
        return session.chat();
    }
    let encoding = session.tokenizer.encode(prompt.as_str(), false)?;
    let generated = session.generate(encoding.get_ids())?;
    println!("{prompt}{}", session.tokenizer.decode(&generated, true)?);
    Ok(())
}

fn load_model(config: Config, weights: Weights, device: &impl Device) -> Result<Model> {
    let model = Model::from_weights(config, weights, CacheKey::root())?;
    // Materialize checkpoint layout and scale conversions before tracing the first token.
    block_on(device.eval(
        &model,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))?;
    Ok(model)
}

fn index(i: i32) -> io::Result<Array> {
    Ok(Array::from_data(DataHandle::new(
        HostBuffer::copy_from_slice(
            ArrayInfo::new(Shape::fixed([]), DType::I32),
            &i.to_le_bytes(),
        )?,
    )))
}
fn start_session<D: Device>(model: &Model, capacity: usize, device: &D) -> Result<KVCache> {
    let mut cache = KVCache::new(capacity);
    model.init_cache(&mut cache);
    block_on(device.eval(
        &cache,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))?;
    Ok(cache)
}

#[cfg(test)]
mod tests;

struct Decoder<D: Device> {
    capacity: usize,
    timings: bool,
    model: Model,
    device: D,
    tokenizer: Tokenizer,
    end: u32,
}

#[derive(Debug, thiserror::Error)]
#[error("context exhausted ({0} tokens) before <|im_end|>")]
struct ContextExhausted(usize);

impl<D: Device> Decoder<D> {
    fn chat(&self) -> Result<()> {
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
            if encoded.len() > self.capacity {
                eprintln!("Context is full. Use /reset or enter a shorter message.");
                continue;
            }
            print!("Assistant: ");
            io::stdout().flush()?;
            let ids = match self.generate(encoded.get_ids()) {
                Ok(ids) => ids,
                Err(e) if e.is::<ContextExhausted>() => {
                    println!("{e}. Use /reset or enter a shorter message.");
                    continue;
                }
                Err(e) => return Err(e),
            };
            let reply = self.tokenizer.decode(&ids, true)?;
            println!("{reply}");
            history.push_str(&turn);
            history.push_str(reply.trim());
            history.push_str("<|im_end|>\n");
        }
        Ok(())
    }

    fn generate(&self, ids: &[u32]) -> Result<Vec<u32>> {
        let Self {
            model, device, end, ..
        } = self;
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= model.config.vocab_size) {
            return Err("expected a nonempty text prompt with valid token IDs".into());
        }
        if ids.len() > self.capacity {
            return Err(format!(
                "prompt exceeds the configured context bound ({} tokens)",
                self.capacity
            )
            .into());
        }
        let opts = ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        };
        let started = std::time::Instant::now();
        let cache = start_session(model, self.capacity, device)?;
        let mut first = None;
        for &id in ids {
            let prediction = model.step(&index(id as i32)?, &cache);
            block_on(device.eval(&(&prediction, &cache), opts))?;
            first = Some(prediction);
        }
        let first = first.expect("nonempty prompt");
        if self.timings {
            block_on(device.download(first.data().expect("prefill evaluated")))?;
            eprintln!("First token: {:.3} s", started.elapsed().as_secs_f64());
        }
        let decode_started = std::time::Instant::now();
        let result = model.decode(&first, &index(*end as i32)?, &cache);
        drop(cache);
        block_on(device.eval(&result, opts))?;
        let read = |x: &Array| block_on(device.download(x.data().expect("decode evaluated")));
        let (tokens, count, last) = result;
        if u32::from_le_bytes(read(&last)?.bytes().try_into()?) != *end {
            return Err(ContextExhausted(self.capacity).into());
        }
        let count = usize::try_from(i32::from_le_bytes(read(&count)?.bytes().try_into()?))?;
        if count > self.capacity - ids.len() {
            return Err("invalid decode length".into());
        }
        let buffer = read(&tokens)?;
        let generated = buffer
            .bytes()
            .chunks_exact(4)
            .take(count)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if self.timings && count > 0 {
            let elapsed = decode_started.elapsed().as_secs_f64();
            eprintln!(
                "Decode: {count} tokens in {elapsed:.3} s ({:.1} tokens/s, includes first-use compilation)",
                count as f64 / elapsed
            );
        }
        Ok(generated)
    }
}
