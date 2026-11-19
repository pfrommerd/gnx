use std::io;

use gnx_math::HostId;
use gnx_xla::{Worker, XlaDevice};
use uuid::Uuid;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let address = args
        .next()
        .unwrap_or_else(|| "127.0.0.1:7440".into())
        .parse()?;
    let host = args
        .next()
        .map(|id| Uuid::parse_str(&id).map(HostId))
        .transpose()?;
    if args.next().is_some() {
        return Err(io::Error::other("usage: gnx-xla-worker [listen-address] [host-uuid]").into());
    }
    let device = match host {
        Some(host) => XlaDevice::cpu_on(host)?,
        None => XlaDevice::cpu()?,
    };
    let mut worker = Worker::bind(address, device).await?;
    println!("{}", worker.local_addr());
    Ok(worker.wait().await?)
}
