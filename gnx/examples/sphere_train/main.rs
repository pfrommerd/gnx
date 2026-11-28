//! Train a small sphere autoencoder on a fixed batch of synthetic images.
#[path = "../common/logging.rs"]
mod logging;
use std::{error::Error, num::NonZeroU64};

use clap::Parser;
use futures_executor::block_on;
use gnx::{
    Array, Device, DeviceHandle, Eval, ExecOpts,
    array::{ArrayInfo, DType},
    buffer::{DataHandle, HostBuffer, TransferPolicy},
    graph::{Of, View},
    nn::{InitializeWith, Param, Rngs, RngsExt, optim},
    np, random, value_and_grad,
};
use gnx_models::sphere::{
    BackboneConfig, Decoder, DecoderConfig, Encoder, EncoderConfig, Manifold,
};
use indicatif::{ProgressBar, ProgressStyle};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Model = (Encoder, Decoder);

#[derive(Parser)]
#[command(about = "Fit a tiny sphere autoencoder to stripes and checkerboards")]
struct Args {
    /// Device: default (XLA CUDA 0, then CPU), xla/cpu, torch/cpu, or <backend>/cuda:N.
    #[arg(long, default_value = "default")]
    device: String,
    #[arg(long, default_value = "50")]
    steps: NonZeroU64,
    #[arg(long, default_value_t = 0.003)]
    learning_rate: f64,
}

fn backbone() -> BackboneConfig {
    BackboneConfig {
        image_size: 16,
        patch_size: 4,
        width: 32,
        layers: 1,
        heads: 2,
        token_channels: 8,
        manifold: Some(Manifold::Sphere),
        ..Default::default()
    }
}

fn images(size: usize) -> Result<Array> {
    let mut bytes = Vec::with_capacity(3 * 3 * size * size * 4);
    for pattern in 0..3 {
        for channel in 0..3 {
            for y in 0..size {
                for x in 0..size {
                    let tile = match pattern {
                        0 => x / 4,
                        1 => y / 4,
                        _ => x / 4 + y / 4,
                    };
                    let pixel: f32 = if (tile + channel) % 2 == 0 { 1.0 } else { -1.0 };
                    bytes.extend(pixel.to_ne_bytes());
                }
            }
        }
    }
    let info = ArrayInfo::new([3, 3, size, size], DType::F32);
    Ok(Array::from_data(DataHandle::new(
        HostBuffer::copy_from_slice(info, &bytes)?,
    )))
}

fn mse(model: &Model, images: Array) -> Array {
    let latent = model.0.forward(images.clone(), None);
    let error = model.1.forward(latent, None) - &images;
    np::sum(&(&error * &error), np::ALL)
        / (images.fixed_dims().unwrap().iter().product::<usize>() as f64)
}

fn scalar<D: Device>(device: &D, value: &Array) -> Result<f32> {
    Ok(f32::from_ne_bytes(
        block_on(device.download(value.data().unwrap()))?
            .bytes()
            .try_into()?,
    ))
}

fn format_mse(mse: f32) -> String {
    if mse > 0.0 && mse < 1e-6 {
        format!("{mse:.6e}")
    } else {
        format!("{mse:.6}")
    }
}

fn log_mse(updates: u64, mse: f32) {
    if updates == 0 {
        log::info!("Initial MSE: {}", format_mse(mse));
    } else if updates % 100 == 0 {
        log::info!("Iteration {updates}  MSE: {}", format_mse(mse));
    }
}

// Log scaling keeps improvements visible across several orders of magnitude.
fn loss_sparkline(losses: &[f32]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let logs = losses.iter().map(|&x| x.max(f32::MIN_POSITIVE).log10());
    let low = logs.clone().fold(f32::INFINITY, f32::min);
    let high = logs.clone().fold(f32::NEG_INFINITY, f32::max);
    logs.map(|x| {
        let level = if high > low {
            (x - low) / (high - low)
        } else {
            0.0
        };
        BARS[(level * 7.0).round() as usize]
    })
    .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.learning_rate.is_finite() || args.learning_rate <= 0.0 {
        return Err("learning rate must be positive and finite".into());
    }
    let (progress, _log_guard) = logging::init(&args.device, "error,sphere_train=info")?;
    let device: DeviceHandle = args.device.parse()?;
    let opts = ExecOpts {
        transfers: TransferPolicy::AllowCopy,
    };
    let cfg = backbone();
    let images = images(cfg.image_size)?;
    let rngs = Rngs::default();
    let encoder = Encoder::init(
        EncoderConfig {
            backbone: cfg.clone(),
        },
        (),
        &rngs,
    );
    // Start each patch with a nonzero direction before projecting onto the sphere.
    encoder
        .head
        .weight
        .set(random::normal(&rngs.params(), [cfg.token_channels, cfg.width], DType::F32) * 0.02);
    let decoder = Decoder::init(
        DecoderConfig {
            backbone: cfg,
            ..Default::default()
        },
        (),
        &rngs,
    );
    let model = (encoder, decoder);
    block_on(device.eval(&model, opts))?;
    let initial_head = model.0.head.weight.value();
    let loss_and_grad =
        value_and_grad!(|m: &Model, x: Array| mse(m, x), wrt = Of::<Param>::filter());
    let inputs = (&model, images.clone());
    let params = View::new(&inputs, Of::<Param>::filter());
    let mut optimizer = optim::Optimizer::new(&params, optim::adam(args.learning_rate))?;
    let bar = progress.add(ProgressBar::new(args.steps.get()));
    bar.set_style(
        ProgressStyle::with_template(
            "Training  {bar:24.red/bright.black} {percent:>3}%  {pos}/{len}  {msg}  ETA {eta}",
        )?
        .progress_chars("━╸─"),
    );
    let mut samples = Vec::with_capacity(17);
    let sample_every = args.steps.get().div_ceil(16);
    for step in 0..args.steps.get() {
        let (loss, grads) = loss_and_grad(&model, images.clone());
        optimizer.update(&params, grads)?;
        block_on(device.eval(&(&loss, &model, optimizer.state()), opts))?;
        let loss = scalar(&device, &loss)?;
        if step % sample_every == 0 {
            samples.push(loss);
        }
        // The forward pass used the weights after `step` completed updates.
        log_mse(step, loss);
        bar.set_message(format!("MSE: {}", format_mse(loss)));
        bar.inc(1);
    }
    let loss = mse(&model, images);
    let delta = model.0.head.weight.value() - initial_head;
    let movement = np::sum(&(&delta * &delta), np::ALL);
    block_on(device.eval(&(&loss, &movement), opts))?;
    let final_mse = scalar(&device, &loss)?;
    samples.push(final_mse);
    log_mse(args.steps.get(), final_mse);
    log::info!(
        "Encoder head update (L2): {:.6}",
        scalar(&device, &movement)?.sqrt()
    );
    let summary = format!("MSE: {}", format_mse(final_mse));
    let sparkline = loss_sparkline(&samples);
    if progress.is_hidden() {
        println!("Trained  {0}/{0}  {summary}\nLoss {sparkline}", args.steps);
    } else {
        bar.set_prefix(sparkline);
        bar.set_style(
            ProgressStyle::with_template(
                "Trained   {bar:24.green} 100%  {pos}/{len}  {msg}\nLoss {prefix}",
            )?
            .progress_chars("━╸─"),
        );
        bar.finish_with_message(summary);
    }
    Ok(())
}
