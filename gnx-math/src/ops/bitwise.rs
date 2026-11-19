use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt};

simple_operations! {
    AndOperation, and_op, and, 2;
    OrOperation, or_op, or, 2;
    XorOperation, xor_op, xor, 2;
    NotOperation, not_op, not, 1;
    ClzOperation, clz_op, clz, 1;
    PopulationCountOperation, population_count_op, population_count, 1;
    ShiftLeftOperation, shift_left_op, shift_left, 2;
    ShiftRightArithmeticOperation, shift_right_arithmetic_op, shift_right_arithmetic, 2;
    ShiftRightLogicalOperation, shift_right_logical_op, shift_right_logical, 2;
}
