//! Op interfaces: open, link-time registries of trait implementations for
//! operations.
//!
//! An *interface* is any object-safe trait, used as `dyn Trait`. Crates that
//! define either the trait or the operation can implement it on the operation.
//! Other crates can supply a stateless provider. Both forms register the pair
//! with [`register_op_interface!`](crate::register_op_interface). Consumers
//! then query an operation at run time:
//!
//! ```ignore
//! let rule = eqn.op().interface::<dyn XlaLowering>()?;
//! rule.lower_xla(ctx, eqn, &operands)?;
//! ```
//!
//! The registry stores a type-erased cast `&dyn Operation -> &dyn Trait` per
//! `(interface, operation)` pair, so it is independent of the trait's methods.
//! Operations are the archived types that implement `Operation`.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::sync::LazyLock;

use crate::util::TypedIn;
use crate::{Operation, TypeName};

/// Typed cast for one `(interface, operation)` pair.
pub struct Caster<I: ?Sized + 'static>(pub for<'a> fn(&'a dyn Operation) -> Option<&'a I>);

/// A registered implementation. Construct with
/// [`register_op_interface!`](crate::register_op_interface).
#[doc(hidden)]
pub struct InterfaceImpl {
    interface: fn() -> TypeId,
    operation: TypeName,
    /// Returns a `&'static Caster<I>` for the registered interface `I`.
    caster: fn() -> &'static (dyn Any + Send + Sync),
}

impl InterfaceImpl {
    /// `caster` must return a `Caster<I>`; the registration macro guarantees this.
    pub const fn new<I: ?Sized + 'static, O: TypedIn<dyn Operation>>(
        caster: fn() -> &'static (dyn Any + Send + Sync),
    ) -> Self {
        Self {
            interface: TypeId::of::<I>,
            operation: O::NAME,
            caster,
        }
    }
}

inventory::collect!(InterfaceImpl);

/// Register implementations of an interface for one or more operation types.
///
/// Use `OperationType => Provider` to register a unit struct as a stateless
/// provider. Its methods can read operation parameters from the equation.
///
/// ```ignore
/// impl XlaLowering for AddOperation { ... }
/// gnx_expr::register_op_interface!(dyn XlaLowering: AddOperation);
/// gnx_expr::register_op_interface!(dyn BatchRule: CondOperation, SwitchOperation);
/// ```
#[macro_export]
macro_rules! register_op_interface {
    ($iface:ty : $op:ty => $provider:path $(,)?) => {
        $crate::__private::inventory::submit! {
            $crate::InterfaceImpl::new::<$iface, $op>(|| {
                static PROVIDER: $provider = $provider;
                static CASTER: $crate::Caster<$iface> = $crate::Caster(|_| Some(&PROVIDER));
                &CASTER
            })
        }
    };
    ($iface:ty : $($op:ty),+ $(,)?) => {$(
        $crate::__private::inventory::submit! {
            $crate::InterfaceImpl::new::<$iface, $op>(|| {
                static CASTER: $crate::Caster<$iface> = $crate::Caster(|op| {
                    op.downcast_ref::<$op>().map(|op| op as &$iface)
                });
                &CASTER
            })
        }
    )+};
}

#[derive(Clone, Copy)]
enum Entry {
    Unique(&'static InterfaceImpl),
    Duplicate,
}

static TABLE: LazyLock<HashMap<(TypeId, TypeName), Entry>> = LazyLock::new(|| {
    let mut table = HashMap::new();
    for registration in inventory::iter::<InterfaceImpl> {
        table
            .entry(((registration.interface)(), registration.operation))
            .and_modify(|entry| *entry = Entry::Duplicate)
            .or_insert(Entry::Unique(registration));
    }
    table
});

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InterfaceError {
    #[error("{dialect}.{op} does not implement {interface}")]
    Missing {
        dialect: &'static str,
        op: &'static str,
        interface: &'static str,
    },
    #[error("{dialect}.{op} has multiple {interface} registrations")]
    Duplicate {
        dialect: &'static str,
        op: &'static str,
        interface: &'static str,
    },
}

fn lookup(op: &dyn Operation, interface: TypeId) -> Option<Entry> {
    TABLE.get(&(interface, op.type_name())).copied()
}

impl dyn Operation {
    /// The registered implementation of interface `I` (a `dyn Trait`) for this operation.
    pub fn interface<I: ?Sized + 'static>(&self) -> Result<&I, InterfaceError> {
        let error = |duplicate| {
            let (dialect, op, interface) = (self.dialect(), self.name(), type_name::<I>());
            if duplicate {
                InterfaceError::Duplicate {
                    dialect,
                    op,
                    interface,
                }
            } else {
                InterfaceError::Missing {
                    dialect,
                    op,
                    interface,
                }
            }
        };
        match lookup(self, TypeId::of::<I>()) {
            Some(Entry::Unique(registration)) => {
                let caster = (registration.caster)()
                    .downcast_ref::<Caster<I>>()
                    .expect("registry keyed by interface TypeId");
                Ok((caster.0)(self).expect("registry keyed by operation name"))
            }
            Some(Entry::Duplicate) => Err(error(true)),
            None => Err(error(false)),
        }
    }

    /// Whether exactly one implementation of `I` is registered for this operation.
    pub fn implements<I: ?Sized + 'static>(&self) -> bool {
        matches!(lookup(self, TypeId::of::<I>()), Some(Entry::Unique(_)))
    }
}
