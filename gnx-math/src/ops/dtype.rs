use super::{Op, arity};

simple_operations! {
    RealOperation, real_op, real, "real", 1;
    ImagOperation, imag_op, imag, "imag", 1;
    ConvertElementTypeOperation, convert_element_type_op, convert_element_type, "convert_element_type", 1;
    ConjOperation, conj_op, conj, "conj", 1;
    BitcastConvertTypeOperation, bitcast_convert_type_op, bitcast_convert_type, "bitcast_convert_type", 1;
    ComplexOperation, complex_op, complex, "complex", 2;
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = ReducePrecisionOperation)]
pub struct ReducePrecision {
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
gnx_base::array_operation!(ReducePrecisionOperation, "reduce_precision", |eqn| arity(
    eqn, 1
));
pub fn reduce_precision(exponent_bits: u8, mantissa_bits: u8) -> Op {
    Op::new(ReducePrecision {
        exponent_bits,
        mantissa_bits,
    })
}
