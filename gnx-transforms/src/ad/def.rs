//! Declarative differentiation rules, in the style of JAX's `ad.defjvp`,
//! `ad.deflinear` and `ad.defbilinear`.
//!
//! Rules are closures over traced [`Val`]s. The framework re-emits the primal,
//! skips symbolic-zero tangents and cotangents, sums per-operand
//! contributions, checks arities and result metadata, and registers the
//! generated [`JvpRule`] / [`TransposeRule`] implementations:
//!
//! ```ignore
//! // Product rule; reverse mode follows from the rules of `mul`.
//! defbilinear!(MulOperation, |_, ct, [_, y]| Ok(ct * y.val()?), |_, ct, [x, _]| Ok(x.val()? * ct));
//!
//! // Linear unary op: the tangent applies the op again; only the transpose is given.
//! deflinear!(TransposeOperation, |cx, ct, _| ct.transpose(&inverse(cx.operation::<TransposeOperation>()?.perm())));
//!
//! // General form: one tangent contribution per operand, `None` for operands
//! // that are not differentiable.
//! defjvp!(SinOperation, |_, t, [x]| Ok(t * x.cos()));
//! ```
//!
//! Every closure receives the rule context `cx` first:
//!
//! - **JVP closures** (`defjvp!`) take `(cx, t, primals)`: the operand's
//!   tangent and all primal operands, and return that operand's contribution
//!   to the result tangent. [`Ctx::ans`] is the primal result.
//! - **Transpose closures** (`deftranspose!`, `deflinear!`, `defbilinear!`)
//!   take `(cx, ct, operands)`: the result cotangent and each operand as an
//!   [`Arg`] (a residual value, or a linear operand known only by its
//!   metadata), and return the operand's cotangent. They run only for linear
//!   operands and only when the cotangent is nonzero.
//!
//! These forms cover single-result operations without regions. Operations
//! that own regions (control flow) implement [`JvpRule`] and
//! [`TransposeRule`] directly.

use std::cell::{Cell, RefCell};
use std::ops;

use gnx_expr::pass::Builder;
use gnx_expr::util::TypedIn;
use gnx_expr::{Eqn, Op, OpRef, Operation, Var, VarInfo};

use super::jvp::{Dual, Tangent, accumulate, instantiate};
use super::transpose::Operand;
use crate::util::{self, copy_eqn};
use crate::{TransformError, TransformResult};

#[doc(inline)]
pub use crate::{defbilinear, defjvp, deflinear, deftranspose};

/// The context of one rule application: the equation being differentiated and
/// the builder that rule closures emit into.
pub struct Ctx<'a> {
    b: RefCell<Builder>,
    eqn: Eqn<'a>,
    ans: Cell<Option<Var>>,
}

impl<'a> Ctx<'a> {
    pub fn eqn(&self) -> Eqn<'a> {
        self.eqn
    }

    pub fn op(&self) -> OpRef<'a> {
        self.eqn.op()
    }

    /// Metadata of the (single) result.
    pub fn out(&self) -> VarInfo {
        let output = self.eqn.outputs().next().expect("a single result");
        output.info_ref().to_owned()
    }

    /// The primal result. Available in JVP closures only.
    pub fn ans(&'a self) -> TransformResult<Val<'a>> {
        self.ans.get().map(|var| self.val(var)).ok_or_else(|| {
            TransformError::invalid("the primal result is only available in jvp rules")
        })
    }

    /// The typed operation being differentiated.
    pub fn operation<T: TypedIn<dyn Operation>>(&self) -> TransformResult<&'a T> {
        self.op()
            .downcast::<T>()
            .ok_or_else(|| TransformError::invalid("unexpected operation in differentiation rule"))
    }

    pub fn zeros(&'a self, info: &VarInfo) -> Val<'a> {
        let var = util::zeros(&mut self.b.borrow_mut(), info);
        self.val(var)
    }

    /// Elementwise `predicate ? on_true : on_false`.
    pub fn select(&'a self, predicate: Val<'a>, on_true: Val<'a>, on_false: Val<'a>) -> Val<'a> {
        let var = util::select(
            &mut self.b.borrow_mut(),
            predicate.var,
            on_true.var,
            on_false.var,
        );
        self.val(var)
    }

    /// Emit an arbitrary single-result operation.
    pub fn emit(&'a self, op: Op, inputs: &[Val<'a>], info: VarInfo) -> Val<'a> {
        let inputs: Vec<_> = inputs.iter().map(|value| value.var).collect();
        let var = self.b.borrow_mut().emit1(op, &inputs, info);
        self.val(var)
    }

    /// Re-emit this rule's operation (with its attributes) on new operands.
    pub fn apply(&'a self, inputs: &[Val<'a>]) -> Val<'a> {
        self.emit(self.op().to_owned(), inputs, self.out())
    }

    fn val(&'a self, var: Var) -> Val<'a> {
        Val { cx: self, var }
    }

    fn info(&self, var: Var) -> VarInfo {
        self.b.borrow().info(var).clone()
    }
}

/// Run `f` with a rule context that temporarily owns the builder.
fn with_ctx<R>(b: &mut Builder, eqn: Eqn<'_>, f: impl FnOnce(&Ctx<'_>) -> R) -> R {
    let cx = Ctx {
        b: RefCell::new(std::mem::take(b)),
        eqn,
        ans: Cell::new(None),
    };
    let result = f(&cx);
    *b = cx.b.into_inner();
    result
}

/// A traced value inside a rule closure.
#[derive(Clone, Copy)]
pub struct Val<'a> {
    cx: &'a Ctx<'a>,
    var: Var,
}

impl<'a> Val<'a> {
    pub fn var(self) -> Var {
        self.var
    }

    pub fn info(self) -> VarInfo {
        self.cx.info(self.var)
    }

    pub fn dims(self) -> TransformResult<Vec<usize>> {
        util::dims(&self.info())
    }

    pub fn zeros_like(self) -> Val<'a> {
        self.cx.zeros(&self.info())
    }

    fn map(
        self,
        f: impl FnOnce(&mut Builder, Var) -> TransformResult<Var>,
    ) -> TransformResult<Self> {
        let var = f(&mut self.cx.b.borrow_mut(), self.var)?;
        Ok(self.cx.val(var))
    }

    /// Operand axis `i` becomes result axis `mapping[i]` of `shape`.
    pub fn broadcast_in_dim(self, shape: &[usize], mapping: &[usize]) -> TransformResult<Self> {
        self.map(|b, x| util::broadcast_in_dim(b, x, shape, mapping))
    }

    pub fn reduce_sum(self, axes: &[usize]) -> TransformResult<Self> {
        self.map(|b, x| util::reduce_sum(b, x, axes))
    }

    /// Result axis `i` is operand axis `perm[i]`.
    pub fn transpose(self, perm: &[usize]) -> TransformResult<Self> {
        self.map(|b, x| util::transpose(b, x, perm))
    }

    pub fn rev(self, axes: &[usize]) -> TransformResult<Self> {
        self.map(|b, x| Ok(util::rev(b, x, axes)))
    }

    pub fn reshape(self, info: VarInfo) -> Self {
        self.cx.emit(gnx_common::ops::reshape_op(), &[self], info)
    }
}

impl<'a> ops::Add for Val<'a> {
    type Output = Val<'a>;
    fn add(self, rhs: Self) -> Self {
        let var = util::add(&mut self.cx.b.borrow_mut(), self.var, rhs.var);
        self.cx.val(var)
    }
}

impl<'a> ops::Mul for Val<'a> {
    type Output = Val<'a>;
    fn mul(self, rhs: Self) -> Self {
        let var = util::mul(&mut self.cx.b.borrow_mut(), self.var, rhs.var);
        self.cx.val(var)
    }
}

/// An operand seen by a transpose closure.
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    /// A value computed by the primal program.
    Residual(Val<'a>),
    /// An operand the equation is linear in; only its metadata is known.
    Linear(&'a VarInfo),
}

impl<'a> Arg<'a> {
    pub fn is_linear(self) -> bool {
        matches!(self, Arg::Linear(_))
    }

    /// The residual value, or an error if the operand is linear.
    pub fn val(self) -> TransformResult<Val<'a>> {
        match self {
            Arg::Residual(value) => Ok(value),
            Arg::Linear(_) => Err(TransformError::invalid(
                "the equation is not linear: a transpose rule needs this operand as a constant",
            )),
        }
    }

    pub fn info(self) -> VarInfo {
        match self {
            Arg::Residual(value) => value.info(),
            Arg::Linear(info) => info.clone(),
        }
    }

    pub fn dims(self) -> TransformResult<Vec<usize>> {
        util::dims(&self.info())
    }
}

/// A JVP closure for an operation with `N` operands.
pub type JvpFn<'r, const N: usize> =
    dyn for<'a> Fn(&'a Ctx<'a>, Val<'a>, [Val<'a>; N]) -> TransformResult<Val<'a>> + 'r;

/// A transpose closure for an operation with `N` operands.
pub type TransposeFn<'r, const N: usize> =
    dyn for<'a> Fn(&'a Ctx<'a>, Val<'a>, [Arg<'a>; N]) -> TransformResult<Val<'a>> + 'r;

/// Fix a closure's signature (used by the macros for type inference).
#[doc(hidden)]
pub fn jvp_rule<'r, const N: usize, F>(f: &'r F) -> &'r JvpFn<'r, N>
where
    F: for<'a> Fn(&'a Ctx<'a>, Val<'a>, [Val<'a>; N]) -> TransformResult<Val<'a>> + 'r,
{
    f
}

#[doc(hidden)]
pub fn transpose_rule<'r, const N: usize, F>(f: &'r F) -> &'r TransposeFn<'r, N>
where
    F: for<'a> Fn(&'a Ctx<'a>, Val<'a>, [Arg<'a>; N]) -> TransformResult<Val<'a>> + 'r,
{
    f
}

fn operation(eqn: Eqn<'_>) -> String {
    format!("{}.{}", eqn.op().dialect(), eqn.op().name())
}

fn single_result(eqn: Eqn<'_>) -> TransformResult<VarInfo> {
    let mut outputs = eqn.outputs();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output.info_ref().to_owned()),
        _ => Err(TransformError::invalid(format!(
            "{} must have exactly one result to use a declarative rule",
            operation(eqn)
        ))),
    }
}

fn operands<'s, T, const N: usize>(eqn: Eqn<'_>, values: &'s [T]) -> TransformResult<&'s [T; N]> {
    values.try_into().map_err(|_| {
        TransformError::invalid(format!(
            "{} has {} operands but its rule expects {N}",
            operation(eqn),
            values.len()
        ))
    })
}

fn check(eqn: Eqn<'_>, what: &str, value: Val<'_>, expected: &VarInfo) -> TransformResult<Var> {
    let info = value.info();
    if &info != expected {
        return Err(TransformError::invalid(format!(
            "{what} rule of {} returned {info}, expected {expected}",
            operation(eqn)
        )));
    }
    Ok(value.var)
}

/// Emit the primal result and record it as [`Ctx::ans`].
fn primal<'a, const N: usize>(
    cx: &'a Ctx<'a>,
    inputs: &[Dual; N],
    captures: &[Vec<Dual>],
) -> [Val<'a>; N] {
    let primals = inputs.map(|dual| dual.primal);
    let captures: Vec<Vec<_>> = captures
        .iter()
        .map(|duals| duals.iter().map(|dual| dual.primal).collect())
        .collect();
    let ans = copy_eqn(&mut cx.b.borrow_mut(), cx.eqn, &primals, &captures)[0];
    cx.ans.set(Some(ans));
    primals.map(|var| cx.val(var))
}

fn sum<'a>(cx: &'a Ctx<'a>, total: Tangent, term: Var) -> Tangent {
    accumulate(&mut cx.b.borrow_mut(), total, Tangent::Var(term))
}

/// [`JvpRule`] body for `defjvp!`: sum one contribution per nonzero tangent.
pub fn jvp<const N: usize>(
    b: &mut Builder,
    eqn: Eqn<'_>,
    inputs: &[Dual],
    captures: &[Vec<Dual>],
    rules: [Option<&JvpFn<'_, N>>; N],
) -> TransformResult<Vec<Dual>> {
    let inputs = operands::<_, N>(eqn, inputs)?;
    let out = single_result(eqn)?;
    with_ctx(b, eqn, |cx| {
        let primals = primal(cx, inputs, captures);
        let mut tangent = Tangent::Zero;
        for (dual, rule) in inputs.iter().zip(rules) {
            let (Tangent::Var(t), Some(rule)) = (dual.tangent, rule) else {
                continue;
            };
            let term = check(eqn, "jvp", rule(cx, cx.val(t), primals)?, &out)?;
            tangent = sum(cx, tangent, term);
        }
        Ok(vec![Dual {
            primal: cx.ans.get().expect("primal result"),
            tangent,
        }])
    })
}

/// [`JvpRule`] body for `deflinear!`: the tangent applies the operation to the
/// operand tangents, materializing zeros only when some tangent is nonzero.
pub fn linear_jvp(
    b: &mut Builder,
    eqn: Eqn<'_>,
    inputs: &[Dual],
    captures: &[Vec<Dual>],
) -> TransformResult<Vec<Dual>> {
    single_result(eqn)?;
    let primals: Vec<_> = inputs.iter().map(|dual| dual.primal).collect();
    let captures: Vec<Vec<_>> = captures
        .iter()
        .map(|duals| duals.iter().map(|dual| dual.primal).collect())
        .collect();
    let primal = copy_eqn(b, eqn, &primals, &captures)[0];
    let tangent = if inputs.iter().all(|dual| dual.tangent.is_zero()) {
        Tangent::Zero
    } else {
        let tangents: Vec<_> = inputs
            .iter()
            .map(|dual| {
                let info = b.info(dual.primal).clone();
                instantiate(b, dual.tangent, &info)
            })
            .collect();
        Tangent::Var(copy_eqn(b, eqn, &tangents, &captures)[0])
    };
    Ok(vec![Dual { primal, tangent }])
}

/// [`JvpRule`] body for `defbilinear!`: `op(tx, y) + op(x, ty)`.
pub fn bilinear_jvp(
    b: &mut Builder,
    eqn: Eqn<'_>,
    inputs: &[Dual],
    captures: &[Vec<Dual>],
) -> TransformResult<Vec<Dual>> {
    let [x, y] = operands::<_, 2>(eqn, inputs)?;
    single_result(eqn)?;
    with_ctx(b, eqn, |cx| {
        let [xp, yp] = primal(cx, &[*x, *y], captures);
        let mut tangent = Tangent::Zero;
        if let Tangent::Var(tx) = x.tangent {
            tangent = sum(cx, tangent, cx.apply(&[cx.val(tx), yp]).var);
        }
        if let Tangent::Var(ty) = y.tangent {
            tangent = sum(cx, tangent, cx.apply(&[xp, cx.val(ty)]).var);
        }
        Ok(vec![Dual {
            primal: cx.ans.get().expect("primal result"),
            tangent,
        }])
    })
}

/// [`TransposeRule`] body: one cotangent per linear operand.
pub fn transpose<const N: usize>(
    b: &mut Builder,
    eqn: Eqn<'_>,
    operands: &[Operand],
    cotangents: &[Tangent],
    rules: [Option<&TransposeFn<'_, N>>; N],
) -> TransformResult<Vec<Tangent>> {
    let operands = self::operands::<_, N>(eqn, operands)?;
    single_result(eqn)?;
    let [ct] = cotangents else {
        return Err(TransformError::invalid("expected one cotangent"));
    };
    let Tangent::Var(ct) = *ct else {
        return Ok(vec![Tangent::Zero; N]);
    };
    with_ctx(b, eqn, |cx| {
        let args: [Arg<'_>; N] = std::array::from_fn(|i| match &operands[i] {
            Operand::Residual(var) => Arg::Residual(cx.val(*var)),
            Operand::Linear(info) => Arg::Linear(info),
        });
        let ct = cx.val(ct);
        operands
            .iter()
            .zip(rules)
            .enumerate()
            .map(|(index, (operand, rule))| {
                let Operand::Linear(info) = operand else {
                    return Ok(Tangent::Zero);
                };
                let rule = rule.ok_or_else(|| {
                    TransformError::invalid(format!(
                        "{} is not linear in operand {index}",
                        operation(eqn)
                    ))
                })?;
                Ok(Tangent::Var(check(
                    eqn,
                    "transpose",
                    rule(cx, ct, args)?,
                    info,
                )?))
            })
            .collect()
    })
}

// ----- macros -------------------------------------------------------------------

/// Define forward-mode differentiation for a single-result operation with one
/// closure per operand (`None` for non-differentiable operands). Each closure
/// `|cx, t, [x, y, ..]|` returns the operand's contribution to the result
/// tangent; contributions of nonzero tangents are summed.
///
/// For reverse mode the contributions must be linear in `t`, built from
/// operations that themselves have transpose rules. For an operation owned by
/// another crate, `defjvp!(extern OpType, ...)` registers a local rule provider.
///
/// ```ignore
/// defjvp!(AddOperation, |_, t, _| Ok(t), |_, t, _| Ok(t));
/// defjvp!(SelectOperation, None,
///     |cx, t, [p, _, _]| Ok(cx.select(p, t, t.zeros_like())),
///     |cx, t, [p, _, _]| Ok(cx.select(p, t.zeros_like(), t)));
/// ```
#[macro_export]
macro_rules! defjvp {
    // Register a local rule provider when the operation is owned by another crate.
    (extern $op:ty, $($rules:tt)+) => {
        $crate::__ad_def!(@norm external_jvp $op [] $($rules)+);
    };
    ($op:ty, $($rules:tt)+) => {
        $crate::__ad_def!(@norm jvp $op [] $($rules)+);
    };
}

/// Define the transposition of an operation that is linear in some operands,
/// with one closure per operand (`None` for operands it is never linear in).
/// Each closure `|cx, ct, [x, y, ..]|` receives [`Arg`]s and returns the
/// operand's cotangent.
///
/// ```ignore
/// deftranspose!(AddOperation, |_, ct, _| Ok(ct), |_, ct, _| Ok(ct));
/// ```
#[macro_export]
macro_rules! deftranspose {
    (extern $op:ty, $($rules:tt)+) => {
        $crate::__ad_def!(@norm external_transpose $op [] $($rules)+);
    };
    ($op:ty, $($rules:tt)+) => {
        $crate::__ad_def!(@norm transpose $op [] $($rules)+);
    };
}

/// Define both rules for an operation that is linear in all its operands: the
/// tangent is the operation applied to the tangents, and the closures (as in
/// [`deftranspose!`]) give each operand's cotangent.
///
/// ```ignore
/// deflinear!(RevOperation, |cx, ct, _| ct.rev(&cx.indices("axes")?));
/// ```
#[macro_export]
macro_rules! deflinear {
    ($op:ty, $($rules:tt)+) => {
        $crate::__ad_def!(@norm linear $op [] $($rules)+);
    };
}

/// Define both rules for a binary operation that is linear in each operand
/// separately (a product): the tangent is `op(tx, y) + op(x, ty)`, and the two
/// closures (as in [`deftranspose!`]) give each operand's cotangent.
///
/// ```ignore
/// defbilinear!(MulOperation, |_, ct, [_, y]| Ok(ct * y.val()?), |_, ct, [x, _]| Ok(x.val()? * ct));
/// ```
#[macro_export]
macro_rules! defbilinear {
    ($op:ty, $($rules:tt)+) => {
        $crate::__ad_def!(@norm bilinear $op [] $($rules)+);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __ad_def {
    // Normalize each rule into a group `(None)` or `(Some <closure>)`.
    (@norm $kind:ident $op:ty [$($g:tt)*] $(,)?) => {
        $crate::__ad_def!(@emit $kind $op [$($g)*]);
    };
    (@norm $kind:ident $op:ty [$($g:tt)*] None $(, $($rest:tt)*)?) => {
        $crate::__ad_def!(@norm $kind $op [$($g)* (None)] $($($rest)*)?);
    };
    (@norm $kind:ident $op:ty [$($g:tt)*] $rule:expr $(, $($rest:tt)*)?) => {
        $crate::__ad_def!(@norm $kind $op [$($g)* (Some $rule)] $($($rest)*)?);
    };

    (@emit jvp $op:ty [$($g:tt)*]) => {
        $crate::__ad_def!(@impl_jvp $op [$($g)*]);
        $crate::__private::gnx_expr::register_op_interface!(dyn $crate::ad::JvpRule: $op);
    };
    (@emit external_jvp $op:ty [$($g:tt)*]) => {
        const _: () = {
            struct Rule;
            $crate::__ad_def!(@impl_jvp Rule [$($g)*]);
            $crate::__private::gnx_expr::register_op_interface!(dyn $crate::ad::JvpRule: $op => Rule);
        };
    };
    (@impl_jvp $op:ty [$($g:tt)*]) => {
        impl $crate::ad::JvpRule for $op {
            fn jvp(
                &self,
                b: &mut $crate::__private::gnx_expr::pass::Builder,
                eqn: $crate::__private::gnx_expr::Eqn<'_>,
                inputs: &[$crate::ad::Dual],
                captures: &[::std::vec::Vec<$crate::ad::Dual>],
            ) -> $crate::TransformResult<::std::vec::Vec<$crate::ad::Dual>> {
                const N: usize = $crate::__ad_def!(@count $($g)*);
                $crate::ad::def::jvp::<N>(
                    b, eqn, inputs, captures,
                    [$($crate::__ad_def!(@rule jvp_rule N $g)),*],
                )
            }
        }
    };
    (@emit transpose $op:ty [$($g:tt)*]) => {
        $crate::__ad_def!(@impl_transpose $op [$($g)*]);
        $crate::__private::gnx_expr::register_op_interface!(dyn $crate::ad::TransposeRule: $op);
    };
    (@emit external_transpose $op:ty [$($g:tt)*]) => {
        const _: () = {
            struct Rule;
            $crate::__ad_def!(@impl_transpose Rule [$($g)*]);
            $crate::__private::gnx_expr::register_op_interface!(dyn $crate::ad::TransposeRule: $op => Rule);
        };
    };
    (@impl_transpose $op:ty [$($g:tt)*]) => {
        impl $crate::ad::TransposeRule for $op {
            fn transpose(
                &self,
                b: &mut $crate::__private::gnx_expr::pass::Builder,
                eqn: $crate::__private::gnx_expr::Eqn<'_>,
                operands: &[$crate::ad::Operand],
                _captures: &[::std::vec::Vec<$crate::__private::gnx_expr::Var>],
                cotangents: &[$crate::ad::Tangent],
            ) -> $crate::TransformResult<::std::vec::Vec<$crate::ad::Tangent>> {
                const N: usize = $crate::__ad_def!(@count $($g)*);
                $crate::ad::def::transpose::<N>(
                    b, eqn, operands, cotangents,
                    [$($crate::__ad_def!(@rule transpose_rule N $g)),*],
                )
            }
        }
    };
    (@emit linear $op:ty [$($g:tt)*]) => {
        $crate::__ad_def!(@derived linear_jvp $op);
        $crate::__ad_def!(@emit transpose $op [$($g)*]);
    };
    (@emit bilinear $op:ty [$($g:tt)*]) => {
        $crate::__ad_def!(@derived bilinear_jvp $op);
        $crate::__ad_def!(@emit transpose $op [$($g)*]);
    };

    (@derived $jvp:ident $op:ty) => {
        impl $crate::ad::JvpRule for $op {
            fn jvp(
                &self,
                b: &mut $crate::__private::gnx_expr::pass::Builder,
                eqn: $crate::__private::gnx_expr::Eqn<'_>,
                inputs: &[$crate::ad::Dual],
                captures: &[::std::vec::Vec<$crate::ad::Dual>],
            ) -> $crate::TransformResult<::std::vec::Vec<$crate::ad::Dual>> {
                $crate::ad::def::$jvp(b, eqn, inputs, captures)
            }
        }
        $crate::__private::gnx_expr::register_op_interface!(dyn $crate::ad::JvpRule: $op);
    };

    (@count $($g:tt)*) => { <[()]>::len(&[$($crate::__ad_def!(@unit $g)),*]) };
    (@unit $g:tt) => { () };
    (@rule $helper:ident $n:ident (None)) => { ::core::option::Option::None };
    (@rule $helper:ident $n:ident (Some $rule:expr)) => {
        ::core::option::Option::Some($crate::ad::def::$helper::<$n, _>(&$rule))
    };
}
