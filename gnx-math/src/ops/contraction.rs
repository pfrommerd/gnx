use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt, signature};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DotDimensions {
    pub lhs_contracting: Vec<usize>,
    pub rhs_contracting: Vec<usize>,
    pub lhs_batch: Vec<usize>,
    pub rhs_batch: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DotGeneralOperation {
    dimensions: DotDimensions,
}
impl DotGeneralOperation {
    pub fn dimensions(&self) -> &DotDimensions {
        &self.dimensions
    }
}
gnx_common::array_operation!(
    DotGeneralOperation,
    "dot_general",
    |op, eqn| {
        arity(eqn, 2)?;
        let d = &op.dimensions;
        if d.lhs_contracting.len() != d.rhs_contracting.len()
            || d.lhs_batch.len() != d.rhs_batch.len()
        {
            return Err(signature("dot_general dimension counts must match"));
        }
        Ok(())
    },
    dimensions
);
pub fn dot_general(dimensions: DotDimensions) -> Op {
    Op::new(DotGeneralOperation { dimensions })
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConvAxes {
    pub batch: usize,
    pub feature: usize,
    pub spatial: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConvKernelAxes {
    pub input_feature: usize,
    pub output_feature: usize,
    pub spatial: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConvParameters {
    pub input: ConvAxes,
    pub kernel: ConvKernelAxes,
    pub output: ConvAxes,
    pub strides: Vec<usize>,
    pub padding: Vec<(i64, i64)>,
    pub lhs_dilation: Vec<usize>,
    pub rhs_dilation: Vec<usize>,
    pub feature_groups: usize,
    pub batch_groups: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ConvGeneralDilatedOperation {
    parameters: ConvParameters,
}
impl ConvGeneralDilatedOperation {
    pub fn parameters(&self) -> &ConvParameters {
        &self.parameters
    }
}
gnx_common::array_operation!(
    ConvGeneralDilatedOperation,
    "conv_general_dilated",
    |op, eqn| {
        arity(eqn, 2)?;
        let p = &op.parameters;
        let rank = p.input.spatial.len();
        if [
            p.kernel.spatial.len(),
            p.output.spatial.len(),
            p.strides.len(),
            p.padding.len(),
            p.lhs_dilation.len(),
            p.rhs_dilation.len(),
        ]
        .iter()
        .any(|&n| n != rank)
            || [
                p.strides.as_slice(),
                p.lhs_dilation.as_slice(),
                p.rhs_dilation.as_slice(),
            ]
            .iter()
            .any(|v| v.contains(&0))
            || p.feature_groups == 0
            || p.batch_groups == 0
        {
            return Err(signature("invalid convolution parameters"));
        }
        Ok(())
    },
    parameters
);
pub fn conv_general_dilated(parameters: ConvParameters) -> Op {
    Op::new(ConvGeneralDilatedOperation { parameters })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ScaledDotOperation {
    dimensions: DotDimensions,
    lhs_scaled: bool,
    rhs_scaled: bool,
}
impl ScaledDotOperation {
    pub fn dimensions(&self) -> &DotDimensions {
        &self.dimensions
    }
    pub fn lhs_scaled(&self) -> bool {
        self.lhs_scaled
    }
    pub fn rhs_scaled(&self) -> bool {
        self.rhs_scaled
    }
}
gnx_common::array_operation!(
    ScaledDotOperation,
    "scaled_dot",
    |op, eqn| {
        arity(
            eqn,
            2 + usize::from(op.lhs_scaled) + usize::from(op.rhs_scaled),
        )
    },
    dimensions
);
pub fn scaled_dot(dimensions: DotDimensions, lhs_scaled: bool, rhs_scaled: bool) -> Op {
    Op::new(ScaledDotOperation {
        dimensions,
        lhs_scaled,
        rhs_scaled,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedDotDimensions {
    pub dot: DotDimensions,
    pub lhs_ragged: usize,
    pub rhs_group: Option<usize>,
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RaggedDotGeneralOperation {
    dimensions: RaggedDotDimensions,
}
impl RaggedDotGeneralOperation {
    pub fn dimensions(&self) -> &RaggedDotDimensions {
        &self.dimensions
    }
}
gnx_common::array_operation!(
    RaggedDotGeneralOperation,
    "ragged_dot_general",
    |_op, eqn| arity(eqn, 3),
    dimensions
);
pub fn ragged_dot_general(dimensions: RaggedDotDimensions) -> Op {
    Op::new(RaggedDotGeneralOperation { dimensions })
}
