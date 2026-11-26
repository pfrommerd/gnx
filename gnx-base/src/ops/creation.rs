use super::{Op, arity, signature};
use crate::{ArrayInfo, IntoScalar, Scalar};

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = ConstantOperation)]
pub struct Constant {
    value: Scalar,
}
impl ConstantOperation {
    pub fn value(&self) -> Scalar {
        rkyv::deserialize::<Scalar, rkyv::rancor::Panic>(&self.value).unwrap()
    }
}
crate::array_operation!(
    ConstantOperation,
    "constant",
    |op, eqn| {
        arity(eqn, 0)?;
        let info = ArrayInfo::from_info(&eqn.outputs().next().unwrap().info_ref().to_owned())
            .ok_or_else(|| signature("constant requires array result"))?;
        if !info.shape().dims().is_empty() {
            return Err(signature("constant must produce a scalar"));
        }
        op.value().exact(info.dtype())?;
        Ok(())
    },
    value
);
pub fn constant(value: impl IntoScalar) -> Op {
    scalar_op(value.into_scalar())
}
pub fn scalar_op(value: Scalar) -> Op {
    Op::new(Constant { value })
}
