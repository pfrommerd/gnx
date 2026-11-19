use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt};

simple_operations! {
    RealOperation, real_op, real, 1;
    ImagOperation, imag_op, imag, 1;
    ConvertElementTypeOperation, convert_element_type_op, convert_element_type, 1;
    ConjOperation, conj_op, conj, 1;
    BitcastConvertTypeOperation, bitcast_convert_type_op, bitcast_convert_type, 1;
    ComplexOperation, complex_op, complex, 2;
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ReducePrecisionOperation {
    exponent_bits: u8,
    mantissa_bits: u8,
}
impl ReducePrecisionOperation {
    pub fn exponent_bits(&self) -> u8 {
        self.exponent_bits
    }
    pub fn mantissa_bits(&self) -> u8 {
        self.mantissa_bits
    }
}
gnx_math_base::array_operation!(ReducePrecisionOperation, "reduce_precision", |eqn| arity(
    eqn, 1
));
pub fn reduce_precision(exponent_bits: u8, mantissa_bits: u8) -> Op {
    Op::new(ReducePrecisionOperation {
        exponent_bits,
        mantissa_bits,
    })
}
