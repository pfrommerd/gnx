//! The same program and the same boundary semantics on both CPU runtimes.
#![cfg(feature = "qwen-torch")]
use futures_executor::block_on;
use gnx::{
    Array, ExecOpts, Shape,
    array::{ArrayInfo, DType, Dim},
    buffer::{DataHandle, HostBuffer},
    cache::Cache,
    device::BoundedBuffer,
    expr::{Symbol, Value},
    jit,
    ops::bounded::{axis_length, pad_to_bound},
};

const COPY: ExecOpts = ExecOpts {
    transfers: gnx::buffer::TransferPolicy::AllowCopy,
};

fn input(n: usize, values: &[i32]) -> DataHandle {
    DataHandle::new(
        HostBuffer::copy_from_slice(
            ArrayInfo::new(Shape::fixed([2, n]), DType::I32),
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    )
}
fn signature(s: u32) -> ArrayInfo {
    ArrayInfo::new(
        [
            Dim::Fixed(2),
            Dim::Bounded {
                symbol: Symbol::new(s),
                max: 3,
            },
        ]
        .into_iter()
        .collect(),
        DType::I32,
    )
}
#[gnx::jit]
fn add_prefixes(x: &gnx::Array, y: &gnx::Array) -> gnx::Array {
    x + y
}

macro_rules! exercise {
    ($d:expr) => {{
        let d = $d;
        let traces=std::cell::Cell::new(0);
        let f = jit(|a:Array,b:Array,c:Array| {
            traces.set(traces.get()+1);
            let sum = add_prefixes(&a, &b);
            vec![
                axis_length(&a, 1),
                sum.clone(),
                pad_to_bound(&sum),
                axis_length(&c, 1),
            ]
        });
        let arg=|id|Array::placeholder(signature(id));
        let (p,_) = f.lower((arg(7),arg(7),arg(8))).unwrap();
        let exec = block_on(d.compile(&p)).unwrap();
        let read = |v: &Value| {
            d.download(v.downcast::<DataHandle>().unwrap())
                .unwrap()
                .bytes()
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        for n in [0, 1, 3, 2] {
            let other = 3 - n;
            // Nonzero padding must never escape as a logical value, even when the
            // caller supplies device storage instead of a compact host tensor.
            let backing = d
                .import(&input(3, &[1, 2, 3, 4, 5, 6]), COPY.transfers)
                .unwrap();
            let a = Value::new(BoundedBuffer::new(backing, Shape::fixed([2, n])).unwrap());
            let b = Value::new(input(n, &vec![10; 2 * n]));
            let c = Value::new(input(other, &vec![0; 2 * other]));
            let result = exec.run(&[a, b, c], COPY).unwrap();
            assert_eq!(read(&result[0]), [n as i32]);
            assert_eq!(read(&result[3]), [other as i32]);
            let expected = (0..6)
                .filter(|i| i % 3 < n)
                .map(|i| i as i32 + 11)
                .collect::<Vec<_>>();
            assert_eq!(
                result[1]
                    .downcast::<DataHandle>()
                    .unwrap()
                    .array_info()
                    .shape(),
                &Shape::fixed([2, n])
            );
            assert_eq!(read(&result[1]), expected);
            assert_eq!(
                read(&result[2]),
                (0..6)
                    .map(|i| if i % 3 < n { i as i32 + 11 } else { 0 })
                    .collect::<Vec<_>>()
            );
        }
        for (a, b) in [(1, 2), (4, 4)] {
            let arg = |n| Value::new(input(n, &vec![0; 2 * n]));
            assert!(exec.run(&[arg(a), arg(b), arg(0)], COPY).is_err());
        }
        let zero=ArrayInfo::new(Shape::fixed([2,0]),DType::I32).with_bounded_axis(1,Symbol::new(0),0).unwrap();
        let (empty,_)=jit(|a:Array|vec![axis_length(&a,1),pad_to_bound(&a)]).lower((Array::placeholder(zero),)).unwrap();
        let out=block_on(d.compile(&empty)).unwrap().run(&[Value::new(input(0,&[]))],COPY).unwrap();
        assert_eq!(read(&out[0]),[0]); assert!(read(&out[1]).is_empty());

        // A capacity-preserving reshape could move padding into a valid lane.
        // Reject it unless the caller explicitly pads first.
        let (bad,_)=jit(|a:Array|a.reshape([6])).lower((arg(0),)).unwrap();
        assert!(matches!(block_on(d.compile(&bad)),Err(e) if e.kind()==std::io::ErrorKind::Unsupported));
        let (p2,_) = f.lower((arg(17),arg(17),arg(18))).unwrap();
        assert_eq!(traces.get(),1,"equivalent symbolic signatures share the JIT trace");
        let again = block_on(d.compile(&p2)).unwrap();
        (exec, again)
    }};
}
#[cfg(feature = "qwen-example")]
#[test]
fn xla_bounded() {
    exercise!(gnx_xla::XlaDevice::cpu(Cache::in_memory()).unwrap());
}
#[test]
fn torch_bounded() {
    let (a, b) = exercise!(gnx_torch::TorchDevice::cpu(Cache::in_memory()).unwrap());
    assert!(
        a.modules()
            .zip(b.modules())
            .all(|(x, y)| std::ptr::eq(x, y))
    );
}
