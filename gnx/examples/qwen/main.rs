//! Greedy text completion with Qwen3.5-0.8B or Qwen3.8-27B on GNX.
mod model;
mod tensor;

use clap::{Parser, ValueEnum};
use futures_executor::block_on;
use gnx::{
    Array, Eval, ExecOpts, Shape,
    array::{ArrayInfo, ArrayRef, DType},
    buffer::{DataHandle, HostBuffer, TransferPolicy},
};
use model::{Config, DecodeState, Model, Weights};
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
#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Xla,
    Torch,
}

trait QwenDevice: Eval {
    fn upload(&self, data: &DataHandle) -> io::Result<DataHandle>;
    fn readback(&self, data: &DataHandle) -> io::Result<HostBuffer>;
}
macro_rules! qwen_device {
    ($device:ty) => {
        impl QwenDevice for $device {
            fn upload(&self, data: &DataHandle) -> io::Result<DataHandle> {
                self.import(data, TransferPolicy::AllowCopy)
            }
            fn readback(&self, data: &DataHandle) -> io::Result<HostBuffer> {
                self.download(data)
            }
        }
    };
}
#[cfg(feature = "qwen-example")]
qwen_device!(gnx_xla::XlaDevice);
#[cfg(feature = "qwen-torch")]
qwen_device!(gnx_torch::TorchDevice);

/// Generate text or chat with a local Qwen checkpoint.
#[derive(Parser)]
#[command(name = "qwen")]
struct Args {
    /// Execution backend (defaults to XLA if built, otherwise Torch).
    #[arg(long, value_enum, default_value = if cfg!(feature="qwen-example") {"xla"} else {"torch"})]
    backend: Backend,
    /// Run on a CUDA-visible GPU (default ordinal: 0). Omit to use the CPU.
    #[arg(long, num_args = 0..=1, default_missing_value = "0", value_name = "ORDINAL")]
    cuda: Option<usize>,
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
    let args = Args::parse();
    match args.backend {
        #[cfg(feature = "qwen-example")]
        Backend::Xla => {
            let device = match args.cuda {
                Some(n) => gnx_xla::XlaDevice::cuda(n, gnx::cache::Cache::in_memory())?,
                None => gnx_xla::XlaDevice::cpu(gnx::cache::Cache::in_memory())?,
            };
            let upload = args.cuda.is_some();
            run(args, device, upload)
        }
        #[cfg(not(feature = "qwen-example"))]
        Backend::Xla => Err("XLA backend not built; enable --features qwen-example".into()),
        #[cfg(feature = "qwen-torch")]
        Backend::Torch => {
            let device = match args.cuda {
                Some(n) => gnx_torch::TorchDevice::cuda(n, gnx::cache::Cache::in_memory())?,
                None => gnx_torch::TorchDevice::cpu(gnx::cache::Cache::in_memory())?,
            };
            run(args, device, true)
        }
        #[cfg(not(feature = "qwen-torch"))]
        Backend::Torch => Err("Torch backend not built; enable --features qwen-torch".into()),
    }
}

fn run<D: QwenDevice>(args: Args, device: D, import_weights: bool) -> Result<()> {
    let Args {
        model_dir: dir,
        prompt,
        chat,
        ..
    } = args;
    let config_path = dir.join("config.json");
    let config = fs::read(&config_path).map_err(|e| {
        format!(
            "cannot read {}: {e}. Check --model-dir; it must point to a downloaded Qwen checkpoint",
            config_path.display()
        )
    })?;
    let Checkpoint { text_config: model } = serde_json::from_slice(&config)?;
    let capacity = args.context_length.min(model.max_position_embeddings);
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
    if import_weights {
        // Upload each weight once; subsequent token steps reuse its device buffer.
        for array in w.values_mut() {
            *array = Array::from_data(device.upload(array.data().unwrap())?);
        }
    }
    let session = Decoder {
        capacity,
        timings: args.timings,
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

fn load_model(config: Config, weights: Weights, device: &impl QwenDevice) -> Result<Model> {
    let model = Model::from_weights(config, weights)?;
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
fn new_state<D: QwenDevice>(model: &Config, capacity: usize, device: &D) -> Result<DecodeState> {
    let mut slots = model.state_shapes(capacity).map(|s| {
        let host = HostBuffer::try_fill(ArrayInfo::new(Shape::fixed(s), DType::F32), |_| {
            Ok::<_, io::Error>(())
        })?;
        Ok::<_, io::Error>(ArrayRef::new(Array::from_data(
            device.upload(&DataHandle::new(host))?,
        )))
    });
    let mut layers = Vec::new();
    while let Some(a) = slots.next() {
        layers.push((a?, slots.next().expect("two slots per layer")?));
    }
    Ok((ArrayRef::new(index(0)?), capacity, layers))
}

#[cfg(test)]
mod tests;

struct Decoder<D: QwenDevice> {
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

impl<D: QwenDevice> Decoder<D> {
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
        let state = new_state(&model.config, self.capacity, device)?;
        let mut first = None;
        for &id in ids {
            let prediction = model.step(&index(id as i32)?, &state);
            block_on(device.eval(&(&prediction, &state), opts))?;
            first = Some(prediction);
        }
        let first = first.expect("nonempty prompt");
        if self.timings {
            device.readback(first.data().expect("prefill evaluated"))?;
            eprintln!("First token: {:.3} s", started.elapsed().as_secs_f64());
        }
        let decode_started = std::time::Instant::now();
        let result = model.decode(&first, &state, &index(*end as i32)?);
        drop(state);
        block_on(device.eval(&result, opts))?;
        let read = |x: &Array| device.readback(x.data().expect("decode evaluated"));
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
