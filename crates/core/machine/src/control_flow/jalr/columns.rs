use crate::adapter::{register::i_type::ITypeReader, state::CPUState};
use sp1_derive::AlignedBorrow;
use sp1_hypercube::Word;
use std::mem::size_of;

use crate::operations::{
    bitwise::BitwiseOperation, u16_operation::U16toU8Operation, AddOperation,
};
pub const NUM_JALR_COLS: usize = size_of::<JalrColumns<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct JalrColumns<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ITypeReader<T>,

    /// The value of the first operand.
    pub op_a_value: Word<T>,

    /// Whether or not the current row is a real row.
    pub is_real: T,

    /// Instance of `AddOperation` to handle addition logic in `JumpChip`.
    pub add_operation: AddOperation<T>,

    /// Computation of `pc + 4` if `op_a != X0`.
    pub op_a_operation: AddOperation<T>,

    /// Instance of `BitwiseOperation` to handle bitwise logic.
    pub bitwise_operation: BitwiseOperation<T>,

    /// Instance of `U16toU8Operation` to handle u16 to u8 conversion.
    pub u16_to_u8_operation: U16toU8Operation<T>,
}
