//! Fixed-shape array indexing. Omitted trailing axes are full slices.
use crate::{
    Array, ArrayInfo, ArrayRef, DType, invoke_array,
    ops::{creation::constant, indexing::*, signature},
};
use gnx_expr::LoweringError;
use std::ops::{self, Bound, RangeBounds};
type Result<T> = std::result::Result<T, LoweringError>;
/// A selector for one axis. Tensor indices must be in bounds at evaluation time.
pub enum AxisIndex {
    Item(usize),
    Range(Bound<usize>, Bound<usize>),
    Tensor(Array),
}
/// A single selector or tuple of selectors for [`Array::at`].
pub trait IntoIndex {
    fn into_index(self) -> Vec<AxisIndex>;
}
impl From<usize> for AxisIndex {
    fn from(i: usize) -> Self {
        Self::Item(i)
    }
}
impl From<&Array> for AxisIndex {
    fn from(a: &Array) -> Self {
        Self::Tensor(a.clone())
    }
}
impl From<Array> for AxisIndex {
    fn from(a: Array) -> Self {
        Self::Tensor(a)
    }
}
macro_rules! ranges {
    ($($ty:ty),+) => {$(
        impl From<$ty> for AxisIndex {
            fn from(range: $ty) -> Self {
                Self::Range(range.start_bound().cloned(), range.end_bound().cloned())
            }
        }
    )+};
}
ranges!(
    ops::Range<usize>,
    ops::RangeFrom<usize>,
    ops::RangeTo<usize>,
    ops::RangeInclusive<usize>,
    ops::RangeToInclusive<usize>
);
impl From<ops::RangeFull> for AxisIndex {
    fn from(_: ops::RangeFull) -> Self {
        Self::Range(Bound::Unbounded, Bound::Unbounded)
    }
}
impl IntoIndex for () {
    fn into_index(self) -> Vec<AxisIndex> {
        vec![]
    }
}
impl IntoIndex for Vec<AxisIndex> {
    fn into_index(self) -> Vec<AxisIndex> {
        self
    }
}
impl<T: Into<AxisIndex>> IntoIndex for T {
    fn into_index(self) -> Vec<AxisIndex> {
        vec![self.into()]
    }
}
macro_rules! tuple {
    ($($ty:ident:$i:tt),+) => {
        impl<$($ty: Into<AxisIndex>),+> IntoIndex for ($($ty,)+) {
            fn into_index(self) -> Vec<AxisIndex> { vec![$(self.$i.into()),+] }
        }
    };
}
tuple!(A:0);
tuple!(A:0,B:1);
tuple!(A:0,B:1,C:2);
tuple!(A:0,B:1,C:2,D:3);
tuple!(A:0,B:1,C:2,D:3,E:4);
tuple!(A:0,B:1,C:2,D:3,E:4,F:5);
tuple!(A:0,B:1,C:2,D:3,E:4,F:5,G:6);
tuple!(A:0,B:1,C:2,D:3,E:4,F:5,G:6,H:7);
struct Window {
    ranges: Vec<(usize, usize, usize)>,
    shape: Vec<usize>,
    tensor: Option<(usize, Array)>,
    sliced: bool,
}
impl Window {
    fn parse(x: &Array, idx: impl IntoIndex) -> Result<Self> {
        let d = x.fixed_dims()?;
        let selectors = idx.into_index();
        if selectors.len() > d.len() {
            return Err(signature("too many indices"));
        }
        let mut iter = selectors.into_iter();
        let mut w = Self {
            ranges: vec![],
            shape: vec![],
            tensor: None,
            sliced: false,
        };
        for n in d {
            let mut keep = true;
            let (a, b) = match iter.next().unwrap_or_else(|| (..).into()) {
                AxisIndex::Tensor(idx) => {
                    if w.tensor.is_some() {
                        return Err(signature("only one tensor index is supported"));
                    }
                    w.tensor = Some((w.shape.len(), idx));
                    (0, n)
                }
                AxisIndex::Item(i) => {
                    if i >= n {
                        return Err(signature("index out of bounds"));
                    }
                    keep = false;
                    (i, i + 1)
                }
                AxisIndex::Range(start, end) => {
                    let after =
                        |i: usize| i.checked_add(1).ok_or_else(|| signature("index overflow"));
                    let a = match start {
                        Bound::Unbounded => 0,
                        Bound::Included(i) => i,
                        Bound::Excluded(i) => after(i)?,
                    };
                    let b = match end {
                        Bound::Unbounded => n,
                        Bound::Included(i) => after(i)?,
                        Bound::Excluded(i) => i,
                    };
                    if a > b || b > n {
                        return Err(signature("slice out of bounds"));
                    }
                    (a, b)
                }
            };
            w.ranges.push((a, b, 1));
            w.sliced |= a != 0 || b != n;
            if keep {
                w.shape.push(b - a);
            }
        }
        Ok(w)
    }
}
impl Array {
    /// Index a fixed-shape array. Integers remove an axis, ranges preserve it,
    /// and an I32/I64 index array replaces one axis with its own shape.
    pub fn try_at(&self, index: impl IntoIndex) -> Result<Array> {
        let w = Window::parse(self, index)?;
        let mut a = if w.sliced {
            let shape = w.ranges.iter().map(|(a, b, _)| b - a).collect::<Vec<_>>();
            invoke_array(
                slice(&w.ranges),
                &[self],
                &ArrayInfo::new(shape, self.dtype()),
            )
        } else {
            self.clone()
        };
        if w.shape.len() != w.ranges.len() {
            a = a.reshape(w.shape);
        }
        match w.tensor {
            Some((axis, idx)) => a.try_take(&idx, axis),
            None => Ok(a),
        }
    }
    #[track_caller]
    pub fn at(&self, index: impl IntoIndex) -> Array {
        self.try_at(index).expect("Array::at")
    }
    /// Gather along an explicit axis. Runtime indices must be nonnegative and in bounds.
    pub fn try_take(&self, indices: &Array, axis: usize) -> Result<Array> {
        let mut sizes = self.fixed_dims()?;
        let mut ix = indices.fixed_dims()?;
        if axis >= sizes.len() || !matches!(indices.dtype(), DType::I32 | DType::I64) {
            return Err(signature("take requires a valid axis and I32/I64 indices"));
        }
        let mut out = sizes.clone();
        out.splice(axis..axis + 1, ix.iter().copied());
        if ix.contains(&0) {
            return Ok(invoke_array(
                crate::ops::zeros_op(),
                &[],
                &ArrayInfo::new(out, self.dtype()),
            ));
        }
        if sizes[axis] == 0 {
            return Err(signature("cannot take from empty axis"));
        }
        sizes[axis] = 1;
        let offset = (0..axis).chain(axis + ix.len()..out.len()).collect();
        ix.push(1);
        let p = GatherParameters {
            dimensions: GatherDimensions {
                offset,
                collapsed: vec![axis],
                start_index_map: vec![axis],
                operand_batch: vec![],
                index_batch: vec![],
            },
            slice_sizes: sizes,
            mode: BoundsMode::PromiseInBounds,
            sorted: false,
            unique: false,
        };
        Ok(invoke_array(
            gather(p),
            &[self, &indices.reshape(ix)],
            &ArrayInfo::new(out, self.dtype()),
        ))
    }
    #[track_caller]
    pub fn take(&self, indices: &Array, axis: usize) -> Array {
        self.try_take(indices, axis).expect("Array::take")
    }
    /// Functional update. Integer/range selectors only; value must exactly match the selected shape and dtype.
    pub fn try_with_at(&self, index: impl IntoIndex, value: &Array) -> Result<Array> {
        let w = Window::parse(self, index)?;
        if w.tensor.is_some() {
            return Err(signature("tensor-indexed writes are not supported"));
        }
        if value.dtype() != self.dtype() || value.fixed_dims()? != w.shape {
            return Err(signature("indexed update shape or dtype mismatch"));
        }
        let sizes = w.ranges.iter().map(|(a, b, _)| b - a).collect::<Vec<_>>();
        if sizes.contains(&0) {
            return Ok(self.clone());
        }
        let update = value.reshape(sizes);
        if !w.sliced {
            return Ok(update);
        }
        let starts = w
            .ranges
            .iter()
            .map(|&(a, _, _)| {
                invoke_array(
                    constant(i64::try_from(a).expect("slice offset exceeds I64")),
                    &[],
                    &ArrayInfo::new((), DType::I64),
                )
            })
            .collect::<Vec<_>>();
        let mut args = vec![self, &update];
        args.extend(starts.iter());
        Ok(invoke_array(
            dynamic_update_slice(),
            &args,
            &ArrayInfo::new(self.shape(), self.dtype()),
        ))
    }
    #[track_caller]
    pub fn with_at(&self, index: impl IntoIndex, value: &Array) -> Array {
        self.try_with_at(index, value).expect("Array::with_at")
    }
}
impl ArrayRef {
    /// Trace a read and return an immutable array, not a view of this cell.
    pub fn try_at(&self, index: impl IntoIndex) -> Result<Array> {
        self.get().try_at(index)
    }
    #[track_caller]
    pub fn at(&self, index: impl IntoIndex) -> Array {
        self.get().at(index)
    }
    pub fn try_take(&self, indices: &Array, axis: usize) -> Result<Array> {
        self.get().try_take(indices, axis)
    }
    #[track_caller]
    pub fn take(&self, indices: &Array, axis: usize) -> Array {
        self.get().take(indices, axis)
    }
    /// Read the cell, replace the selected window, then write the resulting array back.
    pub fn try_set_at(&self, index: impl IntoIndex, value: &Array) -> Result<()> {
        let updated = self.get().try_with_at(index, value)?;
        self.set(updated);
        Ok(())
    }
    #[track_caller]
    pub fn set_at(&self, index: impl IntoIndex, value: &Array) {
        self.try_set_at(index, value).expect("ArrayRef::set_at")
    }
}
