#![cfg(feature = "mesh")]

use std::io::ErrorKind;
use std::net::SocketAddr;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

use gnx::Eval;
use gnx_expr::{Eqn, Expr, Value, ValueInfo, VarScope};
use gnx_math::{
    ArrayInfo, DType, DataHandle, Device, ExecOpts, HostBuffer, HostId, MemoryLocation, Shape,
    TransferPolicy,
    ops::add_op,
    shard::{Sharding, constraint_op, replicate_op},
};
use gnx_xla::Mesh;

const COPY: ExecOpts = ExecOpts {
    transfers: TransferPolicy::AllowCopy,
};

fn program(length: usize, tiled: bool) -> (ArrayInfo, Expr) {
    let info = ArrayInfo::new(Shape::fixed([length]), DType::F32);
    let v = ValueInfo::new(info.clone());
    let mut vars = VarScope::new();
    let [a, b, constrained, sum] = std::array::from_fn(|_| vars.create_var());
    let op = if tiled {
        constraint_op(Sharding::new(vec![vec!["data".into()]]).unwrap())
    } else {
        replicate_op()
    };
    let expr = Expr::new(
        vec![],
        vec![(a, v.clone()), (b, v.clone())],
        vec![
            Eqn::new(op, vec![a], vec![(constrained, v.clone())]),
            Eqn::new(add_op(), vec![constrained, b], vec![(sum, v)]),
        ],
        vec![sum, sum],
    );
    (info, expr)
}

fn host(info: &ArrayInfo, numbers: &[f32]) -> Value {
    let bytes = numbers.iter().flat_map(|n| n.to_le_bytes()).collect();
    Value::new(DataHandle::new(
        HostBuffer::new(info.clone(), bytes).unwrap(),
    ))
}

async fn verify_value(mesh: &Mesh, data: &DataHandle, expected: &[f32]) {
    let packed = mesh.download(data).await.unwrap();
    let actual: Vec<_> = packed
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(actual, expected);
}

async fn start_worker(host: HostId) -> (Child, SocketAddr) {
    let mut worker = Command::new(env!("CARGO_BIN_EXE_gnx-xla-worker"))
        .args(["127.0.0.1:0", &host.0.to_string()])
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut stdout = BufReader::new(worker.stdout.take().unwrap()).lines();
    let address = tokio::time::timeout(Duration::from_secs(20), stdout.next_line())
        .await
        .expect("worker startup timeout")
        .unwrap()
        .expect("worker closed stdout")
        .parse()
        .unwrap();
    (worker, address)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shardy_partitions_one_logical_device_over_tarpc_workers() {
    let hosts = [HostId::fresh(), HostId::fresh()];
    let (_left, left) = start_worker(hosts[0]).await;
    let (_right, right) = start_worker(hosts[1]).await;
    let mesh = Mesh::connect("data", [left, right]).await.unwrap();
    let workers: Vec<_> = mesh.workers().collect();
    assert_eq!(workers[0].host, hosts[0]);
    assert_eq!(workers[1].host, hosts[1]);
    assert_ne!(workers[0].allocation_owner, workers[1].allocation_owner);

    let (info, tiled) = program(4, true);
    let kernel = mesh.compile(&tiled).await.unwrap();
    let a = host(&info, &[1., 2., 3., 4.]);
    let b = host(&info, &[10., 20., 30., 40.]);
    let args = [a, b];
    assert_eq!(
        kernel
            .run(&args, ExecOpts::default())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    let results = kernel.run(&args, COPY).await.unwrap();
    let data = results[0].downcast::<DataHandle>().unwrap();
    assert!(data.to_host().is_err());
    for (i, shard) in data.shards().iter().enumerate() {
        assert_eq!(
            shard.bounds,
            std::iter::once(2 * i..2 * (i + 1)).collect::<Vec<_>>()
        );
        assert_eq!(shard.location, MemoryLocation::host(hosts[i]));
        assert_eq!(
            data.shard_data(i)
                .unwrap()
                .single_allocation()
                .unwrap()
                .id()
                .owner,
            workers[i].allocation_owner
        );
    }
    let repeated = results[1].downcast::<DataHandle>().unwrap();
    assert_eq!(
        data.shard_data(0)
            .unwrap()
            .single_allocation()
            .unwrap()
            .id(),
        repeated
            .shard_data(0)
            .unwrap()
            .single_allocation()
            .unwrap()
            .id()
    );
    verify_value(&mesh, data, &[11., 22., 33., 44.]).await;

    let next = mesh
        .execute(tiled, results, ExecOpts::default())
        .await
        .unwrap();
    verify_value(
        &mesh,
        next[0].downcast::<DataHandle>().unwrap(),
        &[22., 44., 66., 88.],
    )
    .await;
    assert_eq!(
        mesh.compile(&program(3, true).1)
            .await
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Unsupported
    );

    let (_, replicated) = program(4, false);
    let replicas = mesh.execute(replicated.clone(), args, COPY).await.unwrap();
    let data = replicas[0].downcast::<DataHandle>().unwrap();
    assert_eq!(data.shards()[0].bounds, data.shards()[1].bounds);
    assert_ne!(data.shards()[0].replica, data.shards()[1].replica);
    verify_value(&mesh, data, &[11., 22., 33., 44.]).await;

    // A change from tiled to fully replicated would move data.
    assert_eq!(
        mesh.execute(replicated.clone(), next.clone(), ExecOpts::default())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    let copied = mesh.execute(replicated, next, COPY).await.unwrap();
    verify_value(
        &mesh,
        copied[0].downcast::<DataHandle>().unwrap(),
        &[44., 88., 132., 176.],
    )
    .await;

    // The same Mesh works through the high-level graph evaluation boundary.
    let array = gnx::Array::from_data(copied[0].downcast::<DataHandle>().unwrap().clone());
    let tiled =
        gnx::shard::constrain(&array, Sharding::new(vec![vec!["data".into()]]).unwrap()).unwrap();
    mesh.eval(&tiled, COPY).await.unwrap();
    let data = tiled.data().unwrap();
    assert_eq!(data.shards()[0].bounds[0], 0..2);
    verify_value(&mesh, data, &[44., 88., 132., 176.]).await;
}
