use super::{Op, arity};

simple_operations! {
    AndOperation, and_op, and, "and", 2;
    OrOperation, or_op, or, "or", 2;
    XorOperation, xor_op, xor, "xor", 2;
    NotOperation, not_op, not, "not", 1;
    ClzOperation, clz_op, clz, "clz", 1;
    PopulationCountOperation, population_count_op, population_count, "population_count", 1;
    ShiftLeftOperation, shift_left_op, shift_left, "shift_left", 2;
    ShiftRightArithmeticOperation, shift_right_arithmetic_op, shift_right_arithmetic, "shift_right_arithmetic", 2;
    ShiftRightLogicalOperation, shift_right_logical_op, shift_right_logical, "shift_right_logical", 2;
}
