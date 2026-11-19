use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt, result_rank, signature};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CumulativeKind {
    Sum,
    Product,
    Min,
    Max,
    LogSumExp,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CumulativeOperation {
    kind: CumulativeKind,
    axis: usize,
    reverse: bool,
}
impl CumulativeOperation {
    pub fn kind(&self) -> CumulativeKind {
        self.kind
    }
    pub fn axis(&self) -> usize {
        self.axis
    }
    pub fn reverse(&self) -> bool {
        self.reverse
    }
}
gnx_array::array_operation!(
    CumulativeOperation,
    "cumulative",
    |op, eqn| {
        arity(eqn, 1)?;
        if op.axis >= result_rank(eqn)? {
            return Err(signature("cumulative axis is out of range"));
        }
        Ok(())
    },
    kind
);
macro_rules! cumulative {
    ($($name:ident => $kind:ident),*) => {$(
        pub fn $name(axis: usize, reverse: bool) -> Op {
            Op::new(CumulativeOperation { kind: CumulativeKind::$kind, axis, reverse })
        }
    )*};
}
cumulative!(cumsum => Sum, cumprod => Product, cummin => Min, cummax => Max, cumlogsumexp => LogSumExp);

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SortOperation {
    axis: usize,
    keys: usize,
    stable: bool,
}
impl SortOperation {
    pub fn axis(&self) -> usize {
        self.axis
    }
    pub fn keys(&self) -> usize {
        self.keys
    }
    pub fn stable(&self) -> bool {
        self.stable
    }
}
gnx_array::array_operation!(
    SortOperation,
    "sort",
    |op, eqn| {
        if op.keys == 0
            || op.keys > eqn.inputs().len()
            || eqn.inputs().len() != eqn.outputs().len()
            || !eqn.subexprs().is_empty()
        {
            return Err(signature("invalid sort arity or key count"));
        }
        Ok(())
    },
    axis
);
pub fn sort(axis: usize, keys: usize, stable: bool) -> Op {
    Op::new(SortOperation { axis, keys, stable })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TopKOperation {
    k: usize,
}
impl TopKOperation {
    pub fn k(&self) -> usize {
        self.k
    }
}
gnx_array::array_operation!(
    TopKOperation,
    "top_k",
    |_op, eqn| {
        if eqn.inputs().len() != 1 || eqn.outputs().len() != 2 || !eqn.subexprs().is_empty() {
            return Err(signature("top_k needs one input and two results"));
        }
        Ok(())
    },
    k
);
pub fn top_k(k: usize) -> Op {
    Op::new(TopKOperation { k })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ApproxTopKOperation {
    k: usize,
    axis: usize,
    recall_bits: u32,
    largest: bool,
    aggregate: bool,
    original_size: Option<usize>,
}
impl ApproxTopKOperation {
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn axis(&self) -> usize {
        self.axis
    }
    pub fn recall(&self) -> f32 {
        f32::from_bits(self.recall_bits)
    }
    pub fn largest(&self) -> bool {
        self.largest
    }
    pub fn aggregate(&self) -> bool {
        self.aggregate
    }
    pub fn original_size(&self) -> Option<usize> {
        self.original_size
    }
}
gnx_array::array_operation!(
    ApproxTopKOperation,
    "approx_top_k",
    |op, eqn| {
        if eqn.inputs().len() != 1
            || eqn.outputs().len() != 2
            || !eqn.subexprs().is_empty()
            || !(op.recall() > 0. && op.recall() <= 1.)
        {
            return Err(signature("invalid approx_top_k signature or recall"));
        }
        Ok(())
    },
    k
);
pub fn approx_top_k(k: usize, axis: usize, recall: f32, largest: bool) -> Op {
    Op::new(ApproxTopKOperation {
        k,
        axis,
        recall_bits: recall.to_bits(),
        largest,
        aggregate: true,
        original_size: None,
    })
}

pub fn approx_top_k_with_options(
    k: usize,
    axis: usize,
    recall: f32,
    largest: bool,
    aggregate: bool,
    original_size: Option<usize>,
) -> Op {
    Op::new(ApproxTopKOperation {
        k,
        axis,
        recall_bits: recall.to_bits(),
        largest,
        aggregate,
        original_size,
    })
}
