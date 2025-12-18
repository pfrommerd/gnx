use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::{Deref, DerefMut};


// An array type contains uninitialized data
pub unsafe trait Array {
    type Item;
    // A maybe-uninitialized version of the buffer type
    type MaybeUninit;

    // Methods for the accessing MaybeUninit type
    // Note that these methods do *not* do any bounds checking
    // but the implementer must guarantee that if idx < size they point
    // to a valid, consistent memory location.
    fn uninit_index(array: &Self::MaybeUninit, idx: usize) -> *const Self::Item;
    fn uninit_index_mut(array: &mut Self::MaybeUninit, idx: usize) -> *mut Self::Item;
    fn uninit_size(array: &Self::MaybeUninit) -> usize;

    fn uninit_slice(array: &Self::MaybeUninit, idx: usize, len: usize) -> &MaybeUninit<[Self::Item]>;
    fn uninit_slice_mut(array: &mut Self::MaybeUninit, idx: usize, len: usize) -> &MaybeUninit<[Self::Item]>;
}

unsafe impl<T, const N: usize> Array for [T; N] {
    type Item = T;
    type MaybeUninit = MaybeUninit<Self>;
    fn uninit_index(buf: &MaybeUninit<Self>, idx: usize) -> *const T { unsafe { (buf.as_ptr() as *const T).add(idx) } }
    fn uninit_index_mut(buf: &mut MaybeUninit<Self>, idx: usize) -> *mut T { unsafe { (buf.as_mut_ptr() as *mut T).add(idx) } }
    fn uninit_size(buf: &MaybeUninit<Self>) -> usize { N }

    fn uninit_slice(array: &Self::MaybeUninit, idx: usize, len: usize) -> &MaybeUninit<[Self::Item]> {
       unsafe { std::slice::from_raw_parts(array.as_ptr().add(idx), len) }
    }
}

// Vector provides a vec-like interface
pub trait Vector where Self: IntoIterator + DerefMut<Target = [Self::Item]>,
              for<'a> &'a Self: IntoIterator<Item = &'a Self::Item>,
          for<'a> &'a mut Self: IntoIterator<Item = &'a mut Self::Item> {
    fn new() -> Self;

    fn len(&self) -> usize;
    fn capacity(&self) -> usize;

    fn push(&mut self, item: Self::Item);
    fn pop(&mut self) -> Option<Self::Item>;
}

impl<T> Vector for Vec<T> {
    fn new() -> Self { Vec::new() }

    fn len(&self) -> usize { Vec::len(self) }
    fn capacity(&self) -> usize { Vec::capacity(self) }

    fn push(&mut self, item: T) { Vec::push(self, item) }
    fn pop(&mut self) -> Option<T> { Vec::pop(self) }
}

pub struct ArrayVec<A: Array> {
    // SAFETY: len <= buffer.size()
    len: usize,
    buffer: A::MaybeUninit
}

impl<A: Array> Drop for ArrayVec<A> {
    fn drop(&mut self) {
        unsafe { 
            for i in 0..self.len { std::mem::drop(unsafe {
                A::uninit_index(self.buffer, i).read()
            }) }
        }
    }
}

pub struct ArrayVecIter<A: Array> {
    idx: usize,
    vec: ManuallyDrop<ArrayVec<A>>
}

impl<A: Array> Drop for ArrayVecIter<A> {
    fn drop(&mut self) {
        unsafe { 
            for i in self.idx..self.vec.len {
                std::mem::drop(unsafe {
                    A::uninit_index(self.vec.buffer, i).read() 
                })
            }
        }
    }
}

impl<A: Array> Iterator for ArrayVecIter<A> {
    type Item = A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.vec.len {
            Some(unsafe {
                let v = A::uninit_index(self.vec.buffer, self.idx).read();
                self.idx = self.idx + 1;
            })
        } else {
            None
        }
    }
}

impl<A: Array> IntoIterator for ArrayVec<A> {
    type Item = A::Item;
    type IntoIter = ArrayVecIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        ArrayVecIter { idx: 0, vec: ManuallyDrop::new(self) }
    }
}

impl<A: Array> Deref for ArrayVec<A> {
    type Target = [A::Item];
    fn deref(&self) -> &Self::Target {
        unsafe {
            A::uninit_slice(self.buffer, self.len)
            std::slice::from_raw_parts(self.buffer.as_ptr() as *const T, self.len)
        }
    }
}
impl<A: Array> DerefMut for ArrayVec<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut T, self.len)
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a ArrayVec<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut ArrayVec<T, N> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}


impl<T, const N: usize> VecLike<T> for ArrayVec<T, N> {
    fn new() -> Self {
        ArrayVec { len: 0, buffer: MaybeUninit::uninit() }
    }
    fn len(&self) -> usize { self.len }
    fn capacity(&self) -> usize { N }

    fn push(&mut self, item: T) { todo!() }
    fn pop(&mut self) -> Option<T> { todo!() }
}




// inlineable vec/map implementations
enum InlineVec<I: Storage> {
    Fixed(ArrayVec<T, N>),
    Variable(Vec<T>),
}

pub struct Vector<T, const N: usize> {
    data: _InlineVec<T, N>,
}


impl<T, const N: usize, const S: usize> From<[T; S]> for InlineVec<T, N> {
    fn from(value: [T; S]) -> Self {
        if S <= N {
            let mut buffer: MaybeUninit<[T; N]> = MaybeUninit::uninit();
            unsafe {
                // SAFETY: S < T so we can fill the buffer
                let buf: *mut T = buffer.as_mut_ptr() as *mut T;
                // If there is a panic here, value is effectively leaked!
                // This is safe because leaking is not Undefined Behavior.
                buf.copy_from_nonoverlapping(value.as_ptr(), S);
            }
            InlineVec { data: _InlineVec::Fixed { len: S, buffer } }
        } else {
            InlineVec { data: _InlineVec::Variable(value.into()) }
        }
    }
}

impl<T, const N: usize> From<Vec<T>> for InlineVec<T, N> {
    fn from(value: Vec<T>) -> Self {
        InlineVec { data: _InlineVec::Variable(value) }
    }
}