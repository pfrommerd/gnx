use super::{Op, arity, count, result_rank, signature};

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Portable,
)]
#[rkyv(as = Self)]
#[repr(u8)]
pub enum CumulativeKind {
    Sum,
    Product,
    Min,
    Max,
    LogSumExp,
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = CumulativeOperation)]
pub struct Cumulative {
    kind: CumulativeKind,
    axis: u64,
    reverse: bool,
}
impl CumulativeOperation {
    pub fn kind(&self) -> CumulativeKind {
        self.kind
    }
    pub fn axis(&self) -> usize {
        count(self.axis)
    }
    pub fn reverse(&self) -> bool {
        self.reverse
    }
}
gnx_common::array_operation!(
    CumulativeOperation,
    "cumulative",
    |op, eqn| {
        arity(eqn, 1)?;
        if op.axis() >= result_rank(eqn)? {
            return Err(signature("cumulative axis is out of range"));
        }
        Ok(())
    },
    kind
);
macro_rules! cumulative {
    ($($name:ident => $kind:ident),*) => {$(
        pub fn $name(axis: usize, reverse: bool) -> Op {
            Op::new(Cumulative { kind: CumulativeKind::$kind, axis: axis as u64, reverse })
        }
    )*};
}
cumulative!(cumsum => Sum, cumprod => Product, cummin => Min, cummax => Max, cumlogsumexp => LogSumExp);

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = SortOperation)]
pub struct Sort {
    axis: u64,
    keys: u64,
    stable: bool,
}
impl SortOperation {
    pub fn axis(&self) -> usize {
        count(self.axis)
    }
    pub fn keys(&self) -> usize {
        count(self.keys)
    }
    pub fn stable(&self) -> bool {
        self.stable
    }
}
gnx_common::array_operation!(
    SortOperation,
    "sort",
    |op, eqn| {
        if op.keys() == 0
            || op.keys() > eqn.inputs().len()
            || eqn.inputs().len() != eqn.outputs().len()
            || eqn.subexprs().len() != 0
        {
            return Err(signature("invalid sort arity or key count"));
        }
        Ok(())
    },
    axis
);
pub fn sort(axis: usize, keys: usize, stable: bool) -> Op {
    Op::new(Sort {
        axis: axis as u64,
        keys: keys as u64,
        stable,
    })
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = TopKOperation)]
pub struct TopK {
    k: u64,
}
impl TopKOperation {
    pub fn k(&self) -> usize {
        count(self.k)
    }
}
gnx_common::array_operation!(
    TopKOperation,
    "top_k",
    |_op, eqn| {
        if eqn.inputs().len() != 1 || eqn.outputs().len() != 2 || eqn.subexprs().len() != 0 {
            return Err(signature("top_k needs one input and two results"));
        }
        Ok(())
    },
    k
);
pub fn top_k(k: usize) -> Op {
    Op::new(TopK { k: k as u64 })
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = ApproxTopKOperation)]
pub struct ApproxTopK {
    k: u64,
    axis: u64,
    recall_bits: u32,
    largest: bool,
    aggregate: bool,
    original_size: Option<u64>,
}
impl ApproxTopKOperation {
    pub fn k(&self) -> usize {
        count(self.k)
    }
    pub fn axis(&self) -> usize {
        count(self.axis)
    }
    pub fn recall(&self) -> f32 {
        f32::from_bits(self.recall_bits.to_native())
    }
    pub fn largest(&self) -> bool {
        self.largest
    }
    pub fn aggregate(&self) -> bool {
        self.aggregate
    }
    pub fn original_size(&self) -> Option<usize> {
        self.original_size.as_ref().map(|&size| count(size))
    }
}
gnx_common::array_operation!(
    ApproxTopKOperation,
    "approx_top_k",
    |op, eqn| {
        if eqn.inputs().len() != 1
            || eqn.outputs().len() != 2
            || eqn.subexprs().len() != 0
            || !(op.recall() > 0. && op.recall() <= 1.)
        {
            return Err(signature("invalid approx_top_k signature or recall"));
        }
        Ok(())
    },
    k
);
pub fn approx_top_k(k: usize, axis: usize, recall: f32, largest: bool) -> Op {
    approx_top_k_with_options(k, axis, recall, largest, true, None)
}

pub fn approx_top_k_with_options(
    k: usize,
    axis: usize,
    recall: f32,
    largest: bool,
    aggregate: bool,
    original_size: Option<usize>,
) -> Op {
    Op::new(ApproxTopK {
        k: k as u64,
        axis: axis as u64,
        recall_bits: recall.to_bits(),
        largest,
        aggregate,
        original_size: original_size.map(|size| size as u64),
    })
}
