use crate::graph::GraphViewer;

#[derive(Clone, Copy)]
pub struct Of<T>(std::marker::PhantomData<T>);

// impl<T> GraphViewer<T> for Of<T> {
//     type Ref<'r> = &'r T;
// }
