use gnx_graph::{GraphEq, GraphHash, OwnedGraph};
use gnx_transforms::Jitable;
use gnx_transforms::jit;
use gnx_transforms::jit_fn;
use gnx_transforms::jittable;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

static ATTRIBUTE_TRACES: AtomicUsize = AtomicUsize::new(0);

#[jit]
fn cached_attribute(x: &u32) -> u32 {
    ATTRIBUTE_TRACES.fetch_add(1, Ordering::Relaxed);
    *x + 1
}

#[test]
fn jit_attribute_reuses_its_static_wrapper() {
    let start = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                start.wait();
                assert_eq!(cached_attribute(&10), 11);
            });
        }
    });
    let mut value = 10;
    assert_eq!(cached_attribute(&value), 11);
    value = 11;
    assert_eq!(cached_attribute(&value), 12);
    value = 10;
    assert_eq!(cached_attribute(&value), 11);
    assert_eq!(ATTRIBUTE_TRACES.load(Ordering::Relaxed), 2);
}

#[test]
fn jit_methods_key_traces_on_the_receiver() {
    #[derive(Clone, gnx_derive::Graph)]
    struct Offset {
        value: u32,
    }
    static TRACES: AtomicUsize = AtomicUsize::new(0);
    impl Offset {
        #[jit]
        fn add(&self, x: &u32) -> u32 {
            TRACES.fetch_add(1, Ordering::Relaxed);
            self.value + *x
        }
        #[jit]
        fn into_value(self) -> u32 {
            self.value
        }
    }
    for value in [2, 3, 2] {
        let offset = Offset { value };
        assert_eq!(offset.add(&5), value + 5);
        assert_eq!(offset.into_value(), value);
    }
    assert_eq!(TRACES.load(Ordering::Relaxed), 2);
}

#[test]
fn different_signatures_can_trace_concurrently() {
    let (started, ready) = (std::sync::Mutex::new(0), std::sync::Condvar::new());
    let function = jit_fn!(|x: u32| {
        let mut count = started.lock().unwrap();
        *count += 1;
        ready.notify_all();
        let (_guard, timeout) = ready
            .wait_timeout_while(count, std::time::Duration::from_secs(2), |count| *count < 2)
            .unwrap();
        assert!(
            !timeout.timed_out(),
            "unrelated trace signatures were serialized"
        );
        x + 1
    });
    std::thread::scope(|scope| {
        let left = scope.spawn(|| function(1));
        let right = scope.spawn(|| function(2));
        assert_eq!(left.join().unwrap(), 2);
        assert_eq!(right.join().unwrap(), 3);
    });
}

#[test]
fn nested_call_does_not_wait_for_its_own_trace_slot() {
    #[jit]
    fn recursive(value: u32) -> u32 {
        thread_local! {
            static ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        if ACTIVE.replace(true) {
            value
        } else {
            let result = recursive(value) + 1;
            ACTIVE.set(false);
            result
        }
    }

    let (send, receive) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || send.send(recursive(7)).unwrap());
    assert_eq!(
        receive.recv_timeout(std::time::Duration::from_secs(5)),
        Ok(8),
    );
    worker.join().unwrap();
    assert_eq!(recursive(7), 8);
}

#[test]
fn one_callable_can_have_separate_typed_jits_for_different_signatures() {
    use gnx_transforms::{Function, TryCallable, TryJitable};
    use std::convert::Infallible;

    struct Overloaded;

    impl Function<(&u32,)> for Overloaded {
        type Output = u32;
    }

    impl TryCallable<(&u32,)> for Overloaded {
        type Error = Infallible;

        fn try_invoke(&self, (value,): (&u32,)) -> Result<u32, Infallible> {
            Ok(value + 1)
        }
    }

    impl TryJitable<(&u32,)> for Overloaded {
        fn try_invoke_owned(&self, (value,): (Arc<u32>,)) -> Result<u32, Infallible> {
            self.try_invoke((value.as_ref(),))
        }
    }

    impl Function<(Arc<u32>,)> for Overloaded {
        type Output = u32;
    }

    impl TryCallable<(Arc<u32>,)> for Overloaded {
        type Error = Infallible;

        fn try_invoke(&self, (value,): (Arc<u32>,)) -> Result<u32, Infallible> {
            Ok(*value + 2)
        }
    }

    impl TryJitable<(Arc<u32>,)> for Overloaded {
        fn try_invoke_owned(&self, input: (Arc<u32>,)) -> Result<u32, Infallible> {
            self.try_invoke(input)
        }
    }

    let borrowed = jit::<(&u32,), _>(Overloaded);
    let owned = jit::<(Arc<u32>,), _>(Overloaded);
    assert_eq!(borrowed.try_invoke((&4,)), Ok(5));
    assert_eq!(owned.try_invoke((Arc::new(4),)), Ok(6));
    assert_eq!(borrowed.try_invoke((&4,)), Ok(5));
}

#[jit]
fn f(x: u32) -> u32 {
    x + 1
}

fn g(x: u32) -> u32 {
    let jitted = jit_fn!(|y| f(y + x));
    let z = 2 * x;
    jitted(z)
}

#[jit]
fn f_ref(x: &u32) -> u32 {
    *x + 1
}

fn g_ref<T: OwnedGraph + GraphEq + GraphHash>(x: T) -> T
where
    T::Builder<gnx_expr::ObjectTracer>: Send + Sync,
{
    let jitted = jit_fn!(|_: &_| x.duplicate());
    jitted(&x)
}

#[test]
fn test_jit() {
    // Check that the type of f is unaltered
    // by the jit transform.
    let _: fn(u32) -> u32 = f;
    assert_eq!(f(1), 2);
    assert_eq!(g(1), 4);
}

#[test]
fn test_jit_refs() {
    let _: fn(&u32) -> u32 = f_ref;
    assert_eq!(f_ref(&1), 2);
    assert_eq!(g_ref(2), 2);
}

#[test]
fn test_jit_fn_wraps_function_items() {
    let jitted = jit_fn!(f);
    assert_eq!(jitted(3), 4);
}

#[test]
fn test_jittable_invoke_owned() {
    let callable = jittable!(fn(&u32) -> u32, |x: &u32| *x + 1);
    assert_eq!(Jitable::invoke_owned(&callable, (Arc::new(1),)), 2);
}
