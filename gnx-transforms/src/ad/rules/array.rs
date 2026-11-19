//! Forward-mode and transposition rules for the `array` dialect.

use gnx_math_base::ops::{
    AddOperation, BroadcastInDimOperation, MulOperation, ReshapeOperation, RevOperation,
    SelectOperation, TransposeOperation,
};

use crate::ad::def::{defbilinear, defjvp, deflinear, deftranspose};
use crate::util::inverse;

// Addition is linear, but its tangent sums only the nonzero operand tangents
// instead of materializing zeros, so it uses the general forms.
defjvp!(AddOperation, |_, t, _| Ok(t), |_, t, _| Ok(t));
deftranspose!(AddOperation, |_, ct, _| Ok(ct), |_, ct, _| Ok(ct));

defbilinear!(
    MulOperation,
    |_, ct, [_, y]| Ok(ct * y.val()?),
    |_, ct, [x, _]| Ok(x.val()? * ct)
);

defjvp!(
    SelectOperation,
    None,
    |cx, t, [p, _, _]| Ok(cx.select(p, t, t.zeros_like())),
    |cx, t, [p, _, _]| Ok(cx.select(p, t.zeros_like(), t))
);
deftranspose!(
    SelectOperation,
    None,
    |cx, ct, [p, _, _]| Ok(cx.select(p.val()?, ct, ct.zeros_like())),
    |cx, ct, [p, _, _]| Ok(cx.select(p.val()?, ct.zeros_like(), ct))
);

deflinear!(BroadcastInDimOperation, |cx, ct, [x]| {
    let mapping = cx.operation::<BroadcastInDimOperation>()?.dims();
    let (input, output) = (x.dims()?, ct.dims()?);
    let axes: Vec<_> = (0..output.len())
        .filter(|&axis| {
            mapping
                .iter()
                .position(|&destination| destination == axis)
                .is_none_or(|source| input[source] != output[axis])
        })
        .collect();
    Ok(ct.reduce_sum(&axes)?.reshape(x.info()))
});

deflinear!(TransposeOperation, |cx, ct, _| ct
    .transpose(&inverse(cx.operation::<TransposeOperation>()?.perm())));

deflinear!(RevOperation, |cx, ct, _| ct
    .rev(cx.operation::<RevOperation>()?.axes()));

deflinear!(ReshapeOperation, |_, ct, [x]| Ok(ct.reshape(x.info())));
