use gnx::DeviceHandle;
use std::io::ErrorKind;

#[test]
fn rejects_invalid_selectors() {
    for spec in [
        "",
        "cpu",
        "cuda:0",
        "gpu/cpu",
        "xla/cpu:0",
        "torch/cuda",
        "xla/cuda:-1",
        "torch/cuda:0/1",
    ] {
        assert_eq!(
            spec.parse::<DeviceHandle>().err().unwrap().kind(),
            ErrorKind::InvalidInput,
            "{spec}"
        );
    }
}

#[cfg(not(feature = "xla"))]
#[test]
fn default_requires_xla() {
    assert_eq!(
        "default".parse::<DeviceHandle>().err().unwrap().kind(),
        ErrorKind::Unsupported
    );
}

#[cfg(feature = "xla")]
#[test]
fn default_selects_available_xla_target() {
    let expected = "xla/cuda:0".parse::<DeviceHandle>().is_ok();
    match "default".parse().unwrap() {
        DeviceHandle::Xla(device) => {
            assert_eq!(device.platform(), if expected { "cuda" } else { "cpu" })
        }
        #[cfg(feature = "torch")]
        DeviceHandle::Torch(_) => panic!("expected XLA"),
    }
}

#[cfg(all(feature = "xla", feature = "torch"))]
#[test]
fn cpu_handles_share_the_selected_device_on_clone() {
    for spec in ["xla/cpu", "torch/cpu"] {
        let handle: DeviceHandle = spec.parse().unwrap();
        match (&handle, handle.clone()) {
            (DeviceHandle::Xla(a), DeviceHandle::Xla(b)) if spec == "xla/cpu" => {
                assert!(std::sync::Arc::ptr_eq(a, &b))
            }
            (DeviceHandle::Torch(a), DeviceHandle::Torch(b)) if spec == "torch/cpu" => {
                assert!(std::sync::Arc::ptr_eq(a, &b))
            }
            _ => panic!("wrong backend for {spec}"),
        }
    }
}
