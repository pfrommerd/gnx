//! Forward-mode and transposition rules for the `array` dialect.

use gnx_math::ops::{
    AddOperation, BroadcastInDimOperation, MulOperation, ReduceSumOperation, ReshapeOperation,
    RevOperation, SelectOperation, TransposeOperation,
};

use crate::TransformError;
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
    let mapping = cx.indices("dims")?;
    let (input, output) = (x.dims()?, ct.dims()?);
    if input
        .iter()
        .zip(&mapping)
        .any(|(&size, &axis)| output[axis] != size)
    {
        return Err(TransformError::invalid(
            "transposing a broadcast that expands size-1 axes is not supported",
        ));
    }
    let axes: Vec<_> = (0..output.len())
        .filter(|axis| !mapping.contains(axis))
        .collect();
    ct.reduce_sum(&axes)
});

deflinear!(ReduceSumOperation, |cx, ct, [x]| {
    let axes = cx.indices("axes")?;
    let input = x.dims()?;
    let mapping: Vec<_> = (0..input.len())
        .filter(|axis| !axes.contains(axis))
        .collect();
    ct.broadcast_in_dim(&input, &mapping)
});

deflinear!(TransposeOperation, |cx, ct, _| ct
    .transpose(&inverse(&cx.indices("perm")?)));

deflinear!(RevOperation, |cx, ct, _| ct.rev(&cx.indices("axes")?));

deflinear!(ReshapeOperation, |_, ct, [x]| Ok(ct.reshape(x.info())));
