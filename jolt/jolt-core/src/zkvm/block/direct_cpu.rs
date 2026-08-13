//! Block-native CPU/R1CS relation for direct Jolt-Nova.
//!
//! Every active row is materialized with Jolt's canonical `R1CSCycleInputs`
//! and checked against the complete `R1CS_CONSTRAINTS` table inside the same
//! Nova step as lookup, registers, and authenticated RAM. The last row uses an
//! explicit next-block lookahead witness, so block boundaries cannot drop PC or
//! inline-sequence constraints. The running CPU transcript is carried in Nova;
//! D6 remains responsible for aggregation of deferred native PCS obligations.

use std::collections::BTreeMap;

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::Zero;
use nova_snark::{
    frontend::{
        gadgets::{boolean::AllocatedBit, num::AllocatedNum},
        ConstraintSystem, LinearCombination, SynthesisError,
    },
    traits::circuit::StepCircuit,
};
use tracer::{instruction::Cycle, TraceBlock};

use crate::{
    field::JoltField,
    transcripts::{PoseidonTranscript, Transcript},
    zkvm::{
        instruction::{
            CircuitFlags, Flags, InstructionFlags, InstructionLookup, NUM_CIRCUIT_FLAGS,
            NUM_INSTRUCTION_FLAGS,
        },
        lookup_table::LookupTables,
        r1cs::{
            constraints::R1CS_CONSTRAINTS,
            inputs::{JoltR1CSInputs, R1CSCycleInputs, ALL_R1CS_INPUTS},
        },
    },
};

use super::{
    direct_lookup::{
        alloc_u64_bits, alloc_witness_num, allocated_poseidon_initial, bit_as_num,
        enforce_num_equal, field_from_digest, nova_from_fr, nova_to_storage, poseidon_absorb,
        NO_LOOKUP_TABLE_ID,
    },
    direct_ram::{
        direct_ram_initial_z, prepare_direct_ram_stage, ram_registry_root, DirectRamStepCircuit,
        DIRECT_RAM_Z_ARITY, RAM_ROOT_SLOT,
    },
    direct_register::{validate_combined_sequence, REGISTER_STATE_OFFSET},
    recursive_relations::alloc_nova_constant,
    DirectChunkedError, DirectChunkedPreprocessing, DirectLookupSubclaim, DirectRamSubclaim,
    DirectRegisterSubclaim, DirectRelation, DirectRelationState, NovaPrimaryEngine,
    NovaPrimarySpartanSnark, NovaScalar, NovaSecondaryEngine, NovaSecondarySpartanSnark,
};

const DIRECT_CPU_QUERY_DOMAIN: &[u8] = b"direct-cpu-r1cs-v1";
const DIRECT_BYTECODE_LEAF_DOMAIN: &[u8] = b"direct-bytecode-leaf-v1";
const DIRECT_BYTECODE_NODE_DOMAIN: &[u8] = b"direct-bytecode-node-v1";
static TERMINAL_CPU_LOOKAHEAD: Cycle = Cycle::NoOp;
const R1CS_INPUT_COUNT: usize = ALL_R1CS_INPUTS.len();
const CPU_BYTECODE_ROOT_SLOT: usize = DIRECT_RAM_Z_ARITY;
const CPU_EXPECTED_PC_SLOT: usize = CPU_BYTECODE_ROOT_SLOT + 1;
const CPU_EXPECTED_UNEXPANDED_PC_SLOT: usize = CPU_EXPECTED_PC_SLOT + 1;
const CPU_EXPECTED_VIRTUAL_SLOT: usize = CPU_EXPECTED_UNEXPANDED_PC_SLOT + 1;
const CPU_EXPECTED_FIRST_SLOT: usize = CPU_EXPECTED_VIRTUAL_SLOT + 1;
const CPU_EXPECTED_NOOP_SLOT: usize = CPU_EXPECTED_FIRST_SLOT + 1;
const CPU_CLAIM_SLOT: usize = CPU_EXPECTED_NOOP_SLOT + 1;
const CPU_TRANSCRIPT_STATE_SLOT: usize = CPU_CLAIM_SLOT + 1;
const CPU_TRANSCRIPT_ROUND_SLOT: usize = CPU_CLAIM_SLOT + 2;
pub(super) const DIRECT_CPU_Z_ARITY: usize = CPU_TRANSCRIPT_ROUND_SLOT + 1;

type DirectCpuNovaSnark =
    nova_snark::nova::RecursiveSNARK<NovaPrimaryEngine, NovaSecondaryEngine, DirectCpuStepCircuit>;
type DirectCpuCompressedSnark = nova_snark::nova::CompressedSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    DirectCpuStepCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;
type DirectCpuPublicParams =
    nova_snark::nova::PublicParams<NovaPrimaryEngine, NovaSecondaryEngine, DirectCpuStepCircuit>;

#[derive(Clone, Debug, PartialEq)]
pub struct DirectCpuCycleWitness {
    pub active: bool,
    pub inputs: [i128; R1CS_INPUT_COUNT],
    pub instruction_tag: u16,
    pub rs1_enabled: bool,
    pub rs1_address: u8,
    pub rs2_enabled: bool,
    pub rs2_address: u8,
    pub rd_enabled: bool,
    pub rd_address: u8,
    pub table_id: u8,
    pub instruction_flags: [bool; NUM_INSTRUCTION_FLAGS],
    pub next_is_noop: bool,
    pub bytecode_path: Vec<Fr>,
}

impl Default for DirectCpuCycleWitness {
    fn default() -> Self {
        Self {
            active: false,
            inputs: [0; R1CS_INPUT_COUNT],
            instruction_tag: 0,
            rs1_enabled: false,
            rs1_address: 0,
            rs2_enabled: false,
            rs2_address: 0,
            rd_enabled: false,
            rd_address: 0,
            table_id: NO_LOOKUP_TABLE_ID,
            instruction_flags: [false; NUM_INSTRUCTION_FLAGS],
            next_is_noop: false,
            bytecode_path: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectCpuBlockWitness {
    pub block_index: usize,
    pub global_cycle_start: usize,
    pub active_cycles: usize,
    pub cycle_capacity: usize,
    pub terminated: bool,
    pub bytecode_root: Fr,
    pub bytecode_depth: usize,
    pub query_root: Fr,
    pub block_claim: Fr,
    pub cycles: Vec<DirectCpuCycleWitness>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectCpuSubclaim {
    pub block: DirectCpuBlockWitness,
    pub transcript_state_before: [u8; 32],
    pub transcript_round_before: u32,
    pub transcript_state_after: [u8; 32],
    pub transcript_round_after: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectCpuStageStatement {
    pub program_digest: [u8; 32],
    pub lookup_table_commitment: [u8; 32],
    pub bytecode_root: Fr,
    pub block_count: usize,
    pub total_cycles: usize,
    pub initial_registers: [u64; common::constants::REGISTER_COUNT as usize],
    pub final_registers: [u64; common::constants::REGISTER_COUNT as usize],
    pub ram_addresses: Vec<u64>,
    pub ram_registry_root: Fr,
    pub initial_ram_root: Fr,
    pub final_ram_root: Fr,
    pub folded_cpu_claim: Fr,
    pub terminated: bool,
    pub relations: BTreeMap<DirectRelation, DirectRelationState>,
}

#[derive(Clone)]
pub struct DirectCpuStageProof {
    pub statement: DirectCpuStageStatement,
    pub lookup_subclaims: Vec<DirectLookupSubclaim>,
    pub register_subclaims: Vec<DirectRegisterSubclaim>,
    pub ram_subclaims: Vec<DirectRamSubclaim>,
    pub cpu_subclaims: Vec<DirectCpuSubclaim>,
    pub nova_recursive_snark: Vec<u8>,
    pub spartan_proof: Vec<u8>,
    pub initial_z: Vec<[u8; 32]>,
    pub final_z: Vec<[u8; 32]>,
}

fn bytecode_row_values(pc: usize, row: &jolt_riscv::JoltInstructionRow) -> Vec<Fr> {
    let flags = row.circuit_flags();
    let instruction_flags = row.instruction_flags();
    let table_id = <jolt_riscv::JoltInstructionRow as InstructionLookup<
        { common::constants::XLEN },
    >>::lookup_table(row)
    .as_ref()
    .map(|table| LookupTables::enum_index(table) as u8)
    .unwrap_or(NO_LOOKUP_TABLE_ID);
    let mut values = vec![
        Fr::from(pc as u64),
        Fr::from(row.instruction_kind.tag().0 as u64),
        Fr::from(row.address as u64),
        Fr::from_i128(row.operands.imm),
        Fr::from(u64::from(row.operands.rs1.is_some())),
        Fr::from(row.operands.rs1.unwrap_or_default() as u64),
        Fr::from(u64::from(row.operands.rs2.is_some())),
        Fr::from(row.operands.rs2.unwrap_or_default() as u64),
        Fr::from(u64::from(row.operands.rd.is_some())),
        Fr::from(row.operands.rd.unwrap_or_default() as u64),
        Fr::from(table_id as u64),
    ];
    values.extend(flags.into_iter().map(|flag| Fr::from(u64::from(flag))));
    values.extend(
        instruction_flags
            .into_iter()
            .map(|flag| Fr::from(u64::from(flag))),
    );
    values
}

fn cpu_hash(domain: &'static [u8], values: &[Fr]) -> Fr {
    let mut transcript = PoseidonTranscript::new(domain);
    for value in values {
        transcript.append_scalar(b"cpu_hash_word", value);
    }
    Fr::from_le_bytes_mod_order(&transcript.state)
}

fn bytecode_leaf(pc: usize, row: &jolt_riscv::JoltInstructionRow) -> Fr {
    cpu_hash(DIRECT_BYTECODE_LEAF_DOMAIN, &bytecode_row_values(pc, row))
}

fn bytecode_node(left: Fr, right: Fr) -> Fr {
    cpu_hash(DIRECT_BYTECODE_NODE_DOMAIN, &[left, right])
}

fn bytecode_tree(preprocessing: &DirectChunkedPreprocessing) -> Vec<Vec<Fr>> {
    let mut levels = vec![preprocessing
        .bytecode
        .bytecode
        .iter()
        .enumerate()
        .map(|(pc, row)| bytecode_leaf(pc, row))
        .collect::<Vec<_>>()];
    debug_assert!(!levels[0].is_empty() && levels[0].len().is_power_of_two());
    while levels.last().unwrap().len() > 1 {
        levels.push(
            levels
                .last()
                .unwrap()
                .chunks_exact(2)
                .map(|pair| bytecode_node(pair[0], pair[1]))
                .collect(),
        );
    }
    levels
}

fn bytecode_path(tree: &[Vec<Fr>], mut pc: usize) -> Vec<Fr> {
    let mut path = Vec::with_capacity(tree.len() - 1);
    for level in &tree[..tree.len() - 1] {
        path.push(level[pc ^ 1]);
        pc >>= 1;
    }
    path
}

fn bytecode_root_from_path(mut leaf: Fr, mut pc: usize, path: &[Fr]) -> Fr {
    for sibling in path {
        leaf = if pc & 1 == 0 {
            bytecode_node(leaf, *sibling)
        } else {
            bytecode_node(*sibling, leaf)
        };
        pc >>= 1;
    }
    leaf
}

fn cycle_static_witness(
    preprocessing: &DirectChunkedPreprocessing,
    cycle: &Cycle,
) -> Result<
    (
        usize,
        jolt_riscv::JoltInstructionRow,
        u8,
        [bool; NUM_INSTRUCTION_FLAGS],
    ),
    DirectChunkedError,
> {
    let instruction = cycle
        .instruction()
        .try_jolt_instruction_row()
        .map_err(|kind| {
            DirectChunkedError::InvalidConfiguration(format!(
                "CPU cycle is not a final Jolt row: {kind:?}"
            ))
        })?;
    let pc = preprocessing.bytecode.get_pc(&instruction).ok_or_else(|| {
        DirectChunkedError::InvalidConfiguration(format!(
            "CPU bytecode map is missing instruction at {:#x}",
            instruction.address
        ))
    })?;
    let table_id = <jolt_riscv::JoltInstructionRow as InstructionLookup<
        { common::constants::XLEN },
    >>::lookup_table(&instruction)
    .as_ref()
    .map(|table| LookupTables::enum_index(table) as u8)
    .unwrap_or(NO_LOOKUP_TABLE_ID);
    Ok((pc, instruction, table_id, instruction.instruction_flags()))
}

fn row_values(row: &R1CSCycleInputs) -> [i128; R1CS_INPUT_COUNT] {
    ALL_R1CS_INPUTS.map(|input| row.get_input_value(input))
}

fn eval_lc_clear(lc: &crate::zkvm::r1cs::constraints::LC, inputs: &[i128; R1CS_INPUT_COUNT]) -> Fr {
    let mut value = lc.const_term().map(Fr::from_i128).unwrap_or_else(Fr::zero);
    for i in 0..lc.num_terms() {
        let term = lc.term(i).expect("term index is in range");
        value += Fr::from_i128(inputs[term.input_index]) * Fr::from_i128(term.coeff);
    }
    value
}

fn validate_clear_row(
    block_index: usize,
    row_index: usize,
    inputs: &[i128; R1CS_INPUT_COUNT],
) -> Result<(), DirectChunkedError> {
    for named in R1CS_CONSTRAINTS.iter() {
        let a = eval_lc_clear(&named.cons.a, inputs);
        let b = eval_lc_clear(&named.cons.b, inputs);
        if !a.is_zero() && !b.is_zero() {
            return Err(DirectChunkedError::InvalidBlock {
                block_index,
                reason: format!("CPU/R1CS row {row_index} violates {:?}", named.label),
            });
        }
    }
    Ok(())
}

fn append_cpu_cycle(transcript: &mut PoseidonTranscript, cycle: &DirectCpuCycleWitness) {
    transcript.append_scalar(b"cpu_active", &Fr::from(u64::from(cycle.active)));
    for value in cycle.inputs {
        transcript.append_scalar(b"cpu_input", &Fr::from_i128(value));
    }
    for value in [
        Fr::from(cycle.instruction_tag as u64),
        Fr::from(u64::from(cycle.rs1_enabled)),
        Fr::from(cycle.rs1_address as u64),
        Fr::from(u64::from(cycle.rs2_enabled)),
        Fr::from(cycle.rs2_address as u64),
        Fr::from(u64::from(cycle.rd_enabled)),
        Fr::from(cycle.rd_address as u64),
        Fr::from(cycle.table_id as u64),
        Fr::from(u64::from(cycle.next_is_noop)),
    ] {
        transcript.append_scalar(b"cpu_static", &value);
    }
    for flag in cycle.instruction_flags {
        transcript.append_scalar(b"cpu_instruction_flag", &Fr::from(u64::from(flag)));
    }
    for sibling in &cycle.bytecode_path {
        transcript.append_scalar(b"cpu_bytecode_sibling", sibling);
    }
}

fn cpu_query_root(block: &DirectCpuBlockWitness) -> Fr {
    let mut transcript = PoseidonTranscript::new(DIRECT_CPU_QUERY_DOMAIN);
    transcript.append_scalar(b"block_index", &Fr::from(block.block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(block.global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(block.active_cycles as u64));
    transcript.append_scalar(b"terminated", &Fr::from(u64::from(block.terminated)));
    transcript.append_scalar(b"bytecode_root", &block.bytecode_root);
    transcript.append_scalar(b"bytecode_depth", &Fr::from(block.bytecode_depth as u64));
    for cycle in &block.cycles {
        append_cpu_cycle(&mut transcript, cycle);
    }
    Fr::from_le_bytes_mod_order(&transcript.state)
}

fn derive_cpu_subclaim(
    preprocessing: &DirectChunkedPreprocessing,
    block: &TraceBlock,
    capacity: usize,
    lookahead: Option<&Cycle>,
    transcript: &mut PoseidonTranscript,
) -> Result<DirectCpuSubclaim, DirectChunkedError> {
    super::direct::validate_block(block, capacity)?;
    let before = transcript.state;
    let round_before = transcript.n_rounds;
    let tree = bytecode_tree(preprocessing);
    let bytecode_root = tree.last().unwrap()[0];
    let bytecode_depth = tree.len() - 1;
    let mut cycles = Vec::with_capacity(capacity);
    for (row_index, cycle) in block.cycles.iter().enumerate() {
        let next = block.cycles.get(row_index + 1).or(lookahead);
        let row = R1CSCycleInputs::from_cycle_with_next::<Fr>(&preprocessing.bytecode, cycle, next);
        let inputs = row_values(&row);
        validate_clear_row(block.block_index, row_index, &inputs)?;
        let (pc, instruction, table_id, instruction_flags) =
            cycle_static_witness(preprocessing, cycle)?;
        cycles.push(DirectCpuCycleWitness {
            active: true,
            inputs,
            instruction_tag: instruction.instruction_kind.tag().0,
            rs1_enabled: instruction.operands.rs1.is_some(),
            rs1_address: instruction.operands.rs1.unwrap_or_default(),
            rs2_enabled: instruction.operands.rs2.is_some(),
            rs2_address: instruction.operands.rs2.unwrap_or_default(),
            rd_enabled: instruction.operands.rd.is_some(),
            rd_address: instruction.operands.rd.unwrap_or_default(),
            table_id,
            instruction_flags,
            next_is_noop: next.is_some_and(|next| matches!(next, Cycle::NoOp)),
            bytecode_path: bytecode_path(&tree, pc),
        });
    }
    let noop = preprocessing.bytecode.bytecode[0];
    cycles.resize_with(capacity, || DirectCpuCycleWitness {
        instruction_tag: noop.instruction_kind.tag().0,
        rs1_enabled: noop.operands.rs1.is_some(),
        rs1_address: noop.operands.rs1.unwrap_or_default(),
        rs2_enabled: noop.operands.rs2.is_some(),
        rs2_address: noop.operands.rs2.unwrap_or_default(),
        rd_enabled: noop.operands.rd.is_some(),
        rd_address: noop.operands.rd.unwrap_or_default(),
        table_id: NO_LOOKUP_TABLE_ID,
        instruction_flags: noop.instruction_flags(),
        bytecode_path: vec![Fr::zero(); bytecode_depth],
        ..Default::default()
    });
    let mut block_witness = DirectCpuBlockWitness {
        block_index: block.block_index,
        global_cycle_start: block.global_cycle_start,
        active_cycles: block.active_cycles,
        cycle_capacity: capacity,
        terminated: block.end_state.terminated,
        bytecode_root,
        bytecode_depth,
        query_root: Fr::zero(),
        block_claim: Fr::zero(),
        cycles,
    };
    block_witness.query_root = cpu_query_root(&block_witness);
    transcript.append_scalar(b"block_index", &Fr::from(block.block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(block.global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(block.active_cycles as u64));
    transcript.append_scalar(
        b"terminated",
        &Fr::from(u64::from(block.end_state.terminated)),
    );
    transcript.append_scalar(b"bytecode_root", &bytecode_root);
    transcript.append_scalar(b"bytecode_depth", &Fr::from(bytecode_depth as u64));
    transcript.append_scalar(b"cpu_query_root", &block_witness.query_root);
    let challenge: Fr = transcript.challenge_scalar();
    let mut power = Fr::from(1u64);
    for cycle in &block_witness.cycles {
        for value in cycle.inputs {
            block_witness.block_claim += power * Fr::from_i128(value);
            power *= challenge;
        }
    }
    transcript.append_scalar(b"cpu_block_claim", &block_witness.block_claim);
    Ok(DirectCpuSubclaim {
        block: block_witness,
        transcript_state_before: before,
        transcript_round_before: round_before,
        transcript_state_after: transcript.state,
        transcript_round_after: transcript.n_rounds,
    })
}

fn verify_cpu_subclaim(
    subclaim: &DirectCpuSubclaim,
    transcript: &mut PoseidonTranscript,
) -> Result<(), DirectChunkedError> {
    let block = &subclaim.block;
    if subclaim.transcript_state_before != transcript.state
        || subclaim.transcript_round_before != transcript.n_rounds
        || block.cycles.len() != block.cycle_capacity
        || block.active_cycles == 0
        || block.active_cycles > block.cycle_capacity
        || block.bytecode_depth != block.cycles[0].bytecode_path.len()
        || block.query_root != cpu_query_root(block)
    {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: block.block_index,
            reason: "CPU subclaim shape or commitment is invalid".to_string(),
        });
    }
    for (row, cycle) in block.cycles.iter().enumerate() {
        if cycle.active != (row < block.active_cycles) {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: format!("CPU active prefix fails at row {row}"),
            });
        }
        if cycle.active {
            validate_clear_row(block.block_index, row, &cycle.inputs)?;
        } else if cycle.inputs != [0; R1CS_INPUT_COUNT]
            || cycle.instruction_tag != 0
            || cycle.rs1_enabled
            || cycle.rs1_address != 0
            || cycle.rs2_enabled
            || cycle.rs2_address != 0
            || cycle.rd_enabled
            || cycle.rd_address != 0
            || cycle.table_id != NO_LOOKUP_TABLE_ID
            || cycle.instruction_flags
                != std::array::from_fn(|flag| flag == InstructionFlags::IsNoop as usize)
            || cycle.next_is_noop
            || cycle.bytecode_path.iter().any(|sibling| !sibling.is_zero())
        {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: format!("inactive CPU row {row} is not zero"),
            });
        }
        let leaf_values = vec![
            Fr::from_i128(cycle.inputs[JoltR1CSInputs::PC.to_index()]),
            Fr::from(cycle.instruction_tag as u64),
            Fr::from_i128(cycle.inputs[JoltR1CSInputs::UnexpandedPC.to_index()]),
            Fr::from_i128(cycle.inputs[JoltR1CSInputs::Imm.to_index()]),
            Fr::from(u64::from(cycle.rs1_enabled)),
            Fr::from(cycle.rs1_address as u64),
            Fr::from(u64::from(cycle.rs2_enabled)),
            Fr::from(cycle.rs2_address as u64),
            Fr::from(u64::from(cycle.rd_enabled)),
            Fr::from(cycle.rd_address as u64),
            Fr::from(cycle.table_id as u64),
        ]
        .into_iter()
        .chain((0..NUM_CIRCUIT_FLAGS).map(|flag| Fr::from_i128(cycle.inputs[21 + flag])))
        .chain(
            cycle
                .instruction_flags
                .into_iter()
                .map(|flag| Fr::from(u64::from(flag))),
        )
        .collect::<Vec<_>>();
        let pc = usize::try_from(cycle.inputs[JoltR1CSInputs::PC.to_index()]).map_err(|_| {
            DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: format!("CPU PC is negative or too wide at row {row}"),
            }
        })?;
        if cycle.bytecode_path.len() != block.bytecode_depth
            || (cycle.active
                && bytecode_root_from_path(
                    cpu_hash(DIRECT_BYTECODE_LEAF_DOMAIN, &leaf_values),
                    pc,
                    &cycle.bytecode_path,
                ) != block.bytecode_root)
        {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: format!("CPU bytecode authentication fails at row {row}"),
            });
        }
    }
    transcript.append_scalar(b"block_index", &Fr::from(block.block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(block.global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(block.active_cycles as u64));
    transcript.append_scalar(b"terminated", &Fr::from(u64::from(block.terminated)));
    transcript.append_scalar(b"bytecode_root", &block.bytecode_root);
    transcript.append_scalar(b"bytecode_depth", &Fr::from(block.bytecode_depth as u64));
    transcript.append_scalar(b"cpu_query_root", &block.query_root);
    let challenge: Fr = transcript.challenge_scalar();
    let mut claim = Fr::zero();
    let mut power = Fr::from(1u64);
    for cycle in &block.cycles {
        for value in cycle.inputs {
            claim += power * Fr::from_i128(value);
            power *= challenge;
        }
    }
    if claim != block.block_claim {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: block.block_index,
            reason: "CPU folded row claim is invalid".to_string(),
        });
    }
    transcript.append_scalar(b"cpu_block_claim", &claim);
    if subclaim.transcript_state_after != transcript.state
        || subclaim.transcript_round_after != transcript.n_rounds
    {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: block.block_index,
            reason: "CPU transcript binding is invalid".to_string(),
        });
    }
    Ok(())
}

fn alloc_i128<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
    value: i128,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    alloc_witness_num(cs, nova_from_fr(&Fr::from_i128(value))?)
}

fn eval_lc_allocated<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    lc: &crate::zkvm::r1cs::constraints::LC,
    inputs: &[AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let constant = lc.const_term().unwrap_or(0);
    let value = lc.const_term().map(Fr::from_i128).unwrap_or_else(Fr::zero)
        + (0..lc.num_terms())
            .map(|i| {
                let term = lc.term(i).unwrap();
                Fr::from_le_bytes_mod_order(&nova_to_storage(
                    inputs[term.input_index].get_value().unwrap_or_default(),
                )) * Fr::from_i128(term.coeff)
            })
            .fold(Fr::zero(), |a, b| a + b);
    let output = alloc_witness_num(cs.namespace(|| "LC output"), nova_from_fr(&value)?)?;
    let mut expression =
        LinearCombination::zero() + (nova_from_fr(&Fr::from_i128(constant))?, CS::one());
    for i in 0..lc.num_terms() {
        let term = lc.term(i).unwrap();
        expression = expression
            + (
                nova_from_fr(&Fr::from_i128(term.coeff))?,
                inputs[term.input_index].get_variable(),
            );
    }
    cs.enforce(
        || "CPU linear combination",
        |_| expression - output.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(output)
}

fn enforce_equal_when_active<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    active: &AllocatedBit,
    left: &AllocatedNum<NovaScalar>,
    right: &AllocatedNum<NovaScalar>,
    label: &'static str,
) {
    cs.enforce(
        || label,
        |lc| lc + active.get_variable(),
        |lc| lc + left.get_variable() - right.get_variable(),
        |lc| lc,
    );
}

fn synthesize_cpu_hash<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    domain: &'static [u8],
    values: &[&AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let mut state = allocated_poseidon_initial(cs.namespace(|| "CPU hash initial"), domain)?;
    for (index, value) in values.iter().enumerate() {
        state = poseidon_absorb(
            cs.namespace(|| format!("CPU hash word {index}")),
            &state,
            value,
            "cpu_hash_word",
        )?;
    }
    Ok(state.state)
}

fn select_cpu_num<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    condition: &AllocatedBit,
    when_true: &AllocatedNum<NovaScalar>,
    when_false: &AllocatedNum<NovaScalar>,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let delta = super::direct_lookup::sub_nums(
        cs.namespace(|| "CPU selection delta"),
        when_true,
        when_false,
        label,
    )?;
    let selected = super::direct_lookup::mul_nums(
        cs.namespace(|| "CPU selected delta"),
        &bit_as_num(condition),
        &delta,
        label,
    )?;
    super::direct_lookup::add_nums(
        cs.namespace(|| "CPU selection output"),
        when_false,
        &selected,
        label,
    )
}

fn synthesize_bytecode_root<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    leaf: &AllocatedNum<NovaScalar>,
    pc_bits: &[AllocatedBit],
    path: &[AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    if path.len() > pc_bits.len() {
        return Err(SynthesisError::Unsatisfiable(
            "bytecode path exceeds PC width".to_string(),
        ));
    }
    let mut node = leaf.clone();
    for (level, (bit, sibling)) in pc_bits.iter().zip(path).enumerate() {
        let left = select_cpu_num(
            cs.namespace(|| format!("bytecode left {level}")),
            bit,
            sibling,
            &node,
            "bytecode left selection",
        )?;
        let right = select_cpu_num(
            cs.namespace(|| format!("bytecode right {level}")),
            bit,
            &node,
            sibling,
            "bytecode right selection",
        )?;
        node = synthesize_cpu_hash(
            cs.namespace(|| format!("bytecode node {level}")),
            DIRECT_BYTECODE_NODE_DOMAIN,
            &[&left, &right],
        )?;
    }
    Ok(node)
}

struct AllocatedCpuStatic {
    pc_bits: Vec<AllocatedBit>,
    instruction_tag: AllocatedNum<NovaScalar>,
    rs1_enabled: AllocatedBit,
    rs1_address: AllocatedNum<NovaScalar>,
    rs2_enabled: AllocatedBit,
    rs2_address: AllocatedNum<NovaScalar>,
    rd_enabled: AllocatedBit,
    rd_address: AllocatedNum<NovaScalar>,
    table_id: AllocatedNum<NovaScalar>,
    instruction_flags: Vec<AllocatedBit>,
    next_is_noop: AllocatedBit,
    bytecode_path: Vec<AllocatedNum<NovaScalar>>,
}

struct AllocatedCpuRow {
    active: AllocatedBit,
    inputs: Vec<AllocatedNum<NovaScalar>>,
    static_data: AllocatedCpuStatic,
}

fn cpu_input(
    inputs: &[AllocatedNum<NovaScalar>],
    input: JoltR1CSInputs,
) -> &AllocatedNum<NovaScalar> {
    &inputs[input.to_index()]
}

fn enforce_bits_equal_when<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    condition: LinearCombination<NovaScalar>,
    left: &AllocatedBit,
    right: &AllocatedBit,
    label: &'static str,
) {
    cs.enforce(
        || label,
        |_| condition,
        |lc| lc + left.get_variable() - right.get_variable(),
        |lc| lc,
    );
}

fn signed_u128_from_bits<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    unsigned: &AllocatedNum<NovaScalar>,
    bits: &[AllocatedBit],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    if bits.len() != 128 {
        return Err(SynthesisError::Unsatisfiable(
            "signed u128 requires 128 bits".to_string(),
        ));
    }
    let mut two_128 = NovaScalar::one();
    for _ in 0..128 {
        two_128 = two_128.double();
    }
    let correction = super::direct_lookup::scale_num(
        cs.namespace(|| "signed u128 correction"),
        &bit_as_num(&bits[127]),
        two_128,
        "signed u128 correction",
    )?;
    super::direct_lookup::sub_nums(
        cs.namespace(|| "signed u128 result"),
        unsigned,
        &correction,
        "signed u128 result",
    )
}

fn select_last_num<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    selectors: &[AllocatedNum<NovaScalar>],
    values: &[&AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    if selectors.len() != values.len() {
        return Err(SynthesisError::Unsatisfiable(
            "last-value selection length mismatch".to_string(),
        ));
    }
    let mut result = alloc_nova_constant(cs.namespace(|| "last value zero"), NovaScalar::zero())?;
    for (row, (selector, value)) in selectors.iter().zip(values).enumerate() {
        let term = super::direct_lookup::mul_nums(
            cs.namespace(|| format!("last value term {row}")),
            selector,
            value,
            "last value term",
        )?;
        result = super::direct_lookup::add_nums(
            cs.namespace(|| format!("last value sum {row}")),
            &result,
            &term,
            "last value sum",
        )?;
    }
    Ok(result)
}

fn allocate_cpu_static<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    witness: &DirectCpuCycleWitness,
    pc: &AllocatedNum<NovaScalar>,
) -> Result<AllocatedCpuStatic, SynthesisError> {
    let (pc_copy, pc_bits) = alloc_u64_bits(
        cs.namespace(|| "bytecode PC"),
        witness.inputs[JoltR1CSInputs::PC.to_index()] as u64,
        "bytecode PC",
    )?;
    enforce_num_equal(
        cs.namespace(|| "bytecode PC equals CPU PC"),
        &pc_copy,
        pc,
        "bytecode PC equals CPU PC",
    );
    let instruction_tag = alloc_witness_num(
        cs.namespace(|| "instruction tag"),
        NovaScalar::from(witness.instruction_tag as u64),
    )?;
    let rs1_enabled =
        AllocatedBit::alloc(cs.namespace(|| "rs1 enabled"), Some(witness.rs1_enabled))?;
    let (rs1_address, _) = alloc_u64_bits(
        cs.namespace(|| "rs1 address"),
        witness.rs1_address as u64,
        "CPU rs1 address",
    )?;
    let rs2_enabled =
        AllocatedBit::alloc(cs.namespace(|| "rs2 enabled"), Some(witness.rs2_enabled))?;
    let (rs2_address, _) = alloc_u64_bits(
        cs.namespace(|| "rs2 address"),
        witness.rs2_address as u64,
        "CPU rs2 address",
    )?;
    let rd_enabled = AllocatedBit::alloc(cs.namespace(|| "rd enabled"), Some(witness.rd_enabled))?;
    let (rd_address, _) = alloc_u64_bits(
        cs.namespace(|| "rd address"),
        witness.rd_address as u64,
        "CPU rd address",
    )?;
    let table_id = alloc_witness_num(
        cs.namespace(|| "lookup table id"),
        NovaScalar::from(witness.table_id as u64),
    )?;
    let instruction_flags = witness
        .instruction_flags
        .iter()
        .enumerate()
        .map(|(index, flag)| {
            AllocatedBit::alloc(
                cs.namespace(|| format!("instruction flag {index}")),
                Some(*flag),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let next_is_noop =
        AllocatedBit::alloc(cs.namespace(|| "next is noop"), Some(witness.next_is_noop))?;
    let bytecode_path = witness
        .bytecode_path
        .iter()
        .enumerate()
        .map(|(level, sibling)| {
            alloc_witness_num(
                cs.namespace(|| format!("bytecode sibling {level}")),
                nova_from_fr(sibling)?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AllocatedCpuStatic {
        pc_bits,
        instruction_tag,
        rs1_enabled,
        rs1_address,
        rs2_enabled,
        rs2_address,
        rd_enabled,
        rd_address,
        table_id,
        instruction_flags,
        next_is_noop,
        bytecode_path,
    })
}

fn bind_cpu_row<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    row: usize,
    cpu: &AllocatedCpuRow,
    lookup: &super::direct_lookup::AllocatedDirectLookupCycle,
    register: &super::direct_register::AllocatedRegisterCycle,
    ram: &super::direct_ram::AllocatedRamCycle,
    bytecode_root: &AllocatedNum<NovaScalar>,
) -> Result<(), SynthesisError> {
    for (label, other) in [
        ("CPU-lookup active", &lookup.active),
        ("CPU-register active", &register.active),
        ("CPU-RAM active", &ram.active),
    ] {
        cs.enforce(
            || format!("{label} {row}"),
            |lc| lc + cpu.active.get_variable() - other.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
    }

    let left_lookup = cpu_input(&cpu.inputs, JoltR1CSInputs::LeftLookupOperand);
    let right_lookup = cpu_input(&cpu.inputs, JoltR1CSInputs::RightLookupOperand);
    let signed_right = signed_u128_from_bits(
        cs.namespace(|| format!("signed lookup operand {row}")),
        &lookup.right_operand,
        &lookup.right_operand_bits,
    )?;
    for (label, left, right) in [
        ("left lookup operand", left_lookup, &lookup.left_operand),
        ("right lookup operand", right_lookup, &signed_right),
        (
            "lookup output",
            cpu_input(&cpu.inputs, JoltR1CSInputs::LookupOutput),
            &lookup.output,
        ),
        (
            "rs1 value",
            cpu_input(&cpu.inputs, JoltR1CSInputs::Rs1Value),
            &register.rs1.value,
        ),
        (
            "rs2 value",
            cpu_input(&cpu.inputs, JoltR1CSInputs::Rs2Value),
            &register.rs2.value,
        ),
        (
            "rd write value",
            cpu_input(&cpu.inputs, JoltR1CSInputs::RdWriteValue),
            &register.rd.post_value,
        ),
        (
            "RAM address",
            cpu_input(&cpu.inputs, JoltR1CSInputs::RamAddress),
            &ram.address,
        ),
        (
            "RAM read value",
            cpu_input(&cpu.inputs, JoltR1CSInputs::RamReadValue),
            &ram.read_value,
        ),
        (
            "RAM write value",
            cpu_input(&cpu.inputs, JoltR1CSInputs::RamWriteValue),
            &ram.write_value,
        ),
    ] {
        enforce_equal_when_active(
            cs.namespace(|| format!("bind {label} {row}")),
            &cpu.active,
            left,
            right,
            label,
        );
    }

    for (label, left, right) in [
        (
            "rs1 enabled",
            bit_as_num(&cpu.static_data.rs1_enabled),
            bit_as_num(&register.rs1.enabled),
        ),
        (
            "rs2 enabled",
            bit_as_num(&cpu.static_data.rs2_enabled),
            bit_as_num(&register.rs2.enabled),
        ),
        (
            "rd enabled",
            bit_as_num(&cpu.static_data.rd_enabled),
            bit_as_num(&register.rd.enabled),
        ),
    ] {
        enforce_equal_when_active(
            cs.namespace(|| format!("bind {label} {row}")),
            &cpu.active,
            &left,
            &right,
            label,
        );
    }
    for (label, left, right) in [
        (
            "rs1 address",
            &cpu.static_data.rs1_address,
            &register.rs1.address,
        ),
        (
            "rs2 address",
            &cpu.static_data.rs2_address,
            &register.rs2.address,
        ),
        (
            "rd address",
            &cpu.static_data.rd_address,
            &register.rd.address,
        ),
        ("lookup table", &cpu.static_data.table_id, &lookup.table_id),
    ] {
        enforce_equal_when_active(
            cs.namespace(|| format!("bind {label} {row}")),
            &cpu.active,
            left,
            right,
            label,
        );
    }

    let load = cpu_input(&cpu.inputs, JoltR1CSInputs::OpFlags(CircuitFlags::Load));
    let store = cpu_input(&cpu.inputs, JoltR1CSInputs::OpFlags(CircuitFlags::Store));
    enforce_equal_when_active(
        cs.namespace(|| format!("bind RAM read kind {row}")),
        &cpu.active,
        load,
        &bit_as_num(&ram.is_read),
        "RAM read kind",
    );
    enforce_equal_when_active(
        cs.namespace(|| format!("bind RAM write kind {row}")),
        &cpu.active,
        store,
        &bit_as_num(&ram.is_write),
        "RAM write kind",
    );

    let raf = lookup.raf_identity_path.get_variable();
    let raf_expected = [
        CircuitFlags::AddOperands,
        CircuitFlags::SubtractOperands,
        CircuitFlags::MultiplyOperands,
        CircuitFlags::Advice,
    ]
    .into_iter()
    .fold(LinearCombination::<NovaScalar>::zero(), |lc, flag| {
        lc + cpu_input(&cpu.inputs, JoltR1CSInputs::OpFlags(flag)).get_variable()
    });
    cs.enforce(
        || format!("RAF path follows bytecode flags {row}"),
        |_| raf_expected - raf,
        |lc| lc + CS::one(),
        |lc| lc,
    );
    for bit in 0..128 {
        enforce_bits_equal_when(
            cs.namespace(|| format!("identity lookup index {row} {bit}")),
            LinearCombination::zero() + raf,
            &lookup.lookup_index_bits[bit],
            &lookup.right_operand_bits[bit],
            "identity lookup index",
        );
    }
    for bit in 0..64 {
        enforce_bits_equal_when(
            cs.namespace(|| format!("interleaved right index {row} {bit}")),
            LinearCombination::zero() + CS::one() - raf,
            &lookup.lookup_index_bits[2 * bit],
            &lookup.right_operand_bits[bit],
            "interleaved right index",
        );
        enforce_bits_equal_when(
            cs.namespace(|| format!("interleaved left index {row} {bit}")),
            LinearCombination::zero() + CS::one() - raf,
            &lookup.lookup_index_bits[2 * bit + 1],
            &lookup.left_operand_bits[bit],
            "interleaved left index",
        );
    }

    let flags = (0..NUM_CIRCUIT_FLAGS)
        .map(|flag| &cpu.inputs[21 + flag])
        .collect::<Vec<_>>();
    let mut leaf_values = vec![
        cpu_input(&cpu.inputs, JoltR1CSInputs::PC),
        &cpu.static_data.instruction_tag,
        cpu_input(&cpu.inputs, JoltR1CSInputs::UnexpandedPC),
        cpu_input(&cpu.inputs, JoltR1CSInputs::Imm),
    ];
    let rs1_enabled = bit_as_num(&cpu.static_data.rs1_enabled);
    let rs2_enabled = bit_as_num(&cpu.static_data.rs2_enabled);
    let rd_enabled = bit_as_num(&cpu.static_data.rd_enabled);
    leaf_values.extend([
        &rs1_enabled,
        &cpu.static_data.rs1_address,
        &rs2_enabled,
        &cpu.static_data.rs2_address,
        &rd_enabled,
        &cpu.static_data.rd_address,
        &cpu.static_data.table_id,
    ]);
    leaf_values.extend(flags.iter().copied());
    let instruction_flag_nums = cpu
        .static_data
        .instruction_flags
        .iter()
        .map(bit_as_num)
        .collect::<Vec<_>>();
    leaf_values.extend(instruction_flag_nums.iter());
    let leaf = synthesize_cpu_hash(
        cs.namespace(|| format!("bytecode leaf {row}")),
        DIRECT_BYTECODE_LEAF_DOMAIN,
        &leaf_values,
    )?;
    let computed_root = synthesize_bytecode_root(
        cs.namespace(|| format!("bytecode path {row}")),
        &leaf,
        &cpu.static_data.pc_bits,
        &cpu.static_data.bytecode_path,
    )?;
    enforce_equal_when_active(
        cs.namespace(|| format!("bytecode root binding {row}")),
        &cpu.active,
        &computed_root,
        bytecode_root,
        "bytecode root binding",
    );
    for (label, value) in [
        ("inactive instruction tag", &cpu.static_data.instruction_tag),
        ("inactive rs1 address", &cpu.static_data.rs1_address),
        ("inactive rs2 address", &cpu.static_data.rs2_address),
        ("inactive rd address", &cpu.static_data.rd_address),
    ] {
        cs.enforce(
            || format!("{label} {row}"),
            |lc| lc + CS::one() - cpu.active.get_variable(),
            |lc| lc + value.get_variable(),
            |lc| lc,
        );
    }
    for (label, bit) in [
        ("inactive rs1 enabled", &cpu.static_data.rs1_enabled),
        ("inactive rs2 enabled", &cpu.static_data.rs2_enabled),
        ("inactive rd enabled", &cpu.static_data.rd_enabled),
        ("inactive next noop", &cpu.static_data.next_is_noop),
    ] {
        cs.enforce(
            || format!("{label} {row}"),
            |lc| lc + CS::one() - cpu.active.get_variable(),
            |lc| lc + bit.get_variable(),
            |lc| lc,
        );
    }
    let no_table = NovaScalar::from(NO_LOOKUP_TABLE_ID as u64);
    cs.enforce(
        || format!("inactive lookup table {row}"),
        |lc| lc + CS::one() - cpu.active.get_variable(),
        |lc| lc + cpu.static_data.table_id.get_variable() - (no_table, CS::one()),
        |lc| lc,
    );
    for (flag, bit) in cpu.static_data.instruction_flags.iter().enumerate() {
        let expected = if flag == InstructionFlags::IsNoop as usize {
            NovaScalar::one()
        } else {
            NovaScalar::zero()
        };
        cs.enforce(
            || format!("inactive instruction flag {row} {flag}"),
            |lc| lc + CS::one() - cpu.active.get_variable(),
            |lc| lc + bit.get_variable() - (expected, CS::one()),
            |lc| lc,
        );
    }
    for (level, sibling) in cpu.static_data.bytecode_path.iter().enumerate() {
        cs.enforce(
            || format!("inactive bytecode path {row} {level}"),
            |lc| lc + CS::one() - cpu.active.get_variable(),
            |lc| lc + sibling.get_variable(),
            |lc| lc,
        );
    }

    let left_rs1 = super::direct_lookup::mul_nums(
        cs.namespace(|| format!("left rs1 term {row}")),
        &bit_as_num(
            &cpu.static_data.instruction_flags[InstructionFlags::LeftOperandIsRs1Value as usize],
        ),
        cpu_input(&cpu.inputs, JoltR1CSInputs::Rs1Value),
        "left rs1 term",
    )?;
    let left_pc = super::direct_lookup::mul_nums(
        cs.namespace(|| format!("left PC term {row}")),
        &bit_as_num(&cpu.static_data.instruction_flags[InstructionFlags::LeftOperandIsPC as usize]),
        cpu_input(&cpu.inputs, JoltR1CSInputs::UnexpandedPC),
        "left PC term",
    )?;
    let expected_left = super::direct_lookup::add_nums(
        cs.namespace(|| format!("left instruction input {row}")),
        &left_rs1,
        &left_pc,
        "left instruction input",
    )?;
    enforce_num_equal(
        cs.namespace(|| format!("left instruction binding {row}")),
        &expected_left,
        cpu_input(&cpu.inputs, JoltR1CSInputs::LeftInstructionInput),
        "left instruction binding",
    );

    let right_rs2 = super::direct_lookup::mul_nums(
        cs.namespace(|| format!("right rs2 term {row}")),
        &bit_as_num(
            &cpu.static_data.instruction_flags[InstructionFlags::RightOperandIsRs2Value as usize],
        ),
        cpu_input(&cpu.inputs, JoltR1CSInputs::Rs2Value),
        "right rs2 term",
    )?;
    let right_imm = super::direct_lookup::mul_nums(
        cs.namespace(|| format!("right immediate term {row}")),
        &bit_as_num(
            &cpu.static_data.instruction_flags[InstructionFlags::RightOperandIsImm as usize],
        ),
        cpu_input(&cpu.inputs, JoltR1CSInputs::Imm),
        "right immediate term",
    )?;
    let expected_right = super::direct_lookup::add_nums(
        cs.namespace(|| format!("right instruction input {row}")),
        &right_rs2,
        &right_imm,
        "right instruction input",
    )?;
    enforce_num_equal(
        cs.namespace(|| format!("right instruction binding {row}")),
        &expected_right,
        cpu_input(&cpu.inputs, JoltR1CSInputs::RightInstructionInput),
        "right instruction binding",
    );

    for (label, left, right, output) in [
        (
            "instruction product",
            cpu_input(&cpu.inputs, JoltR1CSInputs::LeftInstructionInput),
            cpu_input(&cpu.inputs, JoltR1CSInputs::RightInstructionInput),
            cpu_input(&cpu.inputs, JoltR1CSInputs::Product),
        ),
        (
            "branch product",
            cpu_input(&cpu.inputs, JoltR1CSInputs::LookupOutput),
            &bit_as_num(&cpu.static_data.instruction_flags[InstructionFlags::Branch as usize]),
            cpu_input(&cpu.inputs, JoltR1CSInputs::ShouldBranch),
        ),
    ] {
        cs.enforce(
            || format!("{label} {row}"),
            |lc| lc + left.get_variable(),
            |lc| lc + right.get_variable(),
            |lc| lc + output.get_variable(),
        );
    }
    cs.enforce(
        || format!("jump product {row}"),
        |lc| {
            lc + cpu_input(&cpu.inputs, JoltR1CSInputs::OpFlags(CircuitFlags::Jump)).get_variable()
        },
        |lc| lc + CS::one() - cpu.static_data.next_is_noop.get_variable(),
        |lc| lc + cpu_input(&cpu.inputs, JoltR1CSInputs::ShouldJump).get_variable(),
    );
    Ok(())
}

#[derive(Clone, Default)]
struct DirectCpuStepCircuit {
    lookup: Option<DirectLookupSubclaim>,
    register: Option<DirectRegisterSubclaim>,
    ram: Option<DirectRamSubclaim>,
    cpu: Option<DirectCpuSubclaim>,
    final_step: bool,
}

impl DirectCpuStepCircuit {
    fn for_subclaims(
        lookup: DirectLookupSubclaim,
        register: DirectRegisterSubclaim,
        ram: DirectRamSubclaim,
        cpu: DirectCpuSubclaim,
        final_step: bool,
    ) -> Self {
        Self {
            lookup: Some(lookup),
            register: Some(register),
            ram: Some(ram),
            cpu: Some(cpu),
            final_step,
        }
    }
}

impl StepCircuit<NovaScalar> for DirectCpuStepCircuit {
    fn arity(&self) -> usize {
        DIRECT_CPU_Z_ARITY
    }

    fn synthesize<CS: ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<NovaScalar>],
    ) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
        if z.len() != DIRECT_CPU_Z_ARITY {
            return Err(SynthesisError::Unsatisfiable(
                "direct CPU state arity".to_string(),
            ));
        }
        let lookup = self
            .lookup
            .as_ref()
            .ok_or(SynthesisError::AssignmentMissing)?;
        let register = self
            .register
            .as_ref()
            .ok_or(SynthesisError::AssignmentMissing)?;
        let ram = self.ram.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
        let cpu = self.cpu.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
        let block = &cpu.block;
        if lookup.block.block_index != block.block_index
            || register.block.block_index != block.block_index
            || ram.block.block_index != block.block_index
            || lookup.block.global_cycle_start != block.global_cycle_start
            || register.block.global_cycle_start != block.global_cycle_start
            || ram.block.global_cycle_start != block.global_cycle_start
            || lookup.block.active_cycles != block.active_cycles
            || register.block.active_cycles != block.active_cycles
            || ram.block.active_cycles != block.active_cycles
            || lookup.block.cycle_capacity != block.cycle_capacity
            || register.block.cycle_capacity != block.cycle_capacity
            || ram.block.cycle_capacity != block.cycle_capacity
            || lookup.block.terminated != block.terminated
            || register.block.terminated != block.terminated
            || ram.block.terminated != block.terminated
        {
            return Err(SynthesisError::Unsatisfiable(
                "D2-D5 metadata mismatch".to_string(),
            ));
        }
        let ram_circuit = DirectRamStepCircuit::for_subclaims(
            lookup.clone(),
            register.clone(),
            ram.clone(),
            self.final_step,
        );
        let (ram_output, lookup_cycles, register_cycles, ram_cycles) = {
            let mut namespace = cs.namespace(|| "D2-D4 relation");
            ram_circuit.synthesize_with_observations(&mut namespace, &z[..DIRECT_RAM_Z_ARITY])?
        };
        let block_index = alloc_witness_num(
            cs.namespace(|| "CPU block index"),
            NovaScalar::from(block.block_index as u64),
        )?;
        let cycle_start = alloc_witness_num(
            cs.namespace(|| "CPU cycle start"),
            NovaScalar::from(block.global_cycle_start as u64),
        )?;
        let active_cycles = alloc_witness_num(
            cs.namespace(|| "CPU active cycles"),
            NovaScalar::from(block.active_cycles as u64),
        )?;
        enforce_num_equal(
            cs.namespace(|| "CPU block index state"),
            &block_index,
            &z[0],
            "CPU block index state",
        );
        enforce_num_equal(
            cs.namespace(|| "CPU cycle start state"),
            &cycle_start,
            &z[2],
            "CPU cycle start state",
        );
        let bytecode_root = alloc_witness_num(
            cs.namespace(|| "CPU bytecode root"),
            nova_from_fr(&block.bytecode_root)?,
        )?;
        let bytecode_depth = alloc_witness_num(
            cs.namespace(|| "CPU bytecode depth"),
            NovaScalar::from(block.bytecode_depth as u64),
        )?;
        let terminated =
            AllocatedBit::alloc(cs.namespace(|| "CPU terminated"), Some(block.terminated))?;
        let terminated_num = bit_as_num(&terminated);
        enforce_num_equal(
            cs.namespace(|| "CPU bytecode root public state"),
            &bytecode_root,
            &z[CPU_BYTECODE_ROOT_SLOT],
            "CPU bytecode root public state",
        );
        let mut allocated_rows = Vec::with_capacity(block.cycle_capacity);
        for (row_index, row) in block.cycles.iter().enumerate() {
            let active = AllocatedBit::alloc(
                cs.namespace(|| format!("CPU active {row_index}")),
                Some(row.active),
            )?;
            let inputs = row
                .inputs
                .iter()
                .enumerate()
                .map(|(input, value)| {
                    alloc_i128(
                        cs.namespace(|| format!("CPU row {row_index} input {input}")),
                        *value,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let active_num = bit_as_num(&active);
            for (constraint_index, named) in R1CS_CONSTRAINTS.iter().enumerate() {
                let a = eval_lc_allocated(
                    cs.namespace(|| format!("CPU row {row_index} A {constraint_index}")),
                    &named.cons.a,
                    &inputs,
                )?;
                let b = eval_lc_allocated(
                    cs.namespace(|| format!("CPU row {row_index} B {constraint_index}")),
                    &named.cons.b,
                    &inputs,
                )?;
                let gated_a = super::direct_lookup::mul_nums(
                    cs.namespace(|| format!("CPU row {row_index} active A {constraint_index}")),
                    &active_num,
                    &a,
                    "active CPU R1CS A",
                )?;
                cs.enforce(
                    || format!("CPU row {row_index} {:?}", named.label),
                    |lc| lc + gated_a.get_variable(),
                    |lc| lc + b.get_variable(),
                    |lc| lc,
                );
            }
            for (input_index, input) in inputs.iter().enumerate() {
                cs.enforce(
                    || format!("inactive CPU input zero {row_index} {input_index}"),
                    |lc| lc + CS::one() - active.get_variable(),
                    |lc| lc + input.get_variable(),
                    |lc| lc,
                );
            }
            let static_data = allocate_cpu_static(
                cs.namespace(|| format!("CPU static row {row_index}")),
                row,
                cpu_input(&inputs, JoltR1CSInputs::PC),
            )?;
            allocated_rows.push(AllocatedCpuRow {
                active,
                inputs,
                static_data,
            });
        }
        if allocated_rows.is_empty()
            || allocated_rows.len() != lookup_cycles.len()
            || allocated_rows.len() != register_cycles.len()
            || allocated_rows.len() != ram_cycles.len()
            || allocated_rows
                .iter()
                .any(|row| row.static_data.bytecode_path.len() != block.bytecode_depth)
        {
            return Err(SynthesisError::Unsatisfiable(
                "D5 row or bytecode path shape mismatch".to_string(),
            ));
        }
        cs.enforce(
            || "first CPU row active",
            |lc| lc + allocated_rows[0].active.get_variable() - CS::one(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        for row in 1..allocated_rows.len() {
            cs.enforce(
                || format!("CPU active prefix {row}"),
                |lc| lc + allocated_rows[row].active.get_variable(),
                |lc| lc + CS::one() - allocated_rows[row - 1].active.get_variable(),
                |lc| lc,
            );
        }
        let active_sum = allocated_rows
            .iter()
            .fold(LinearCombination::<NovaScalar>::zero(), |lc, row| {
                lc + row.active.get_variable()
            });
        cs.enforce(
            || "CPU active row count",
            |_| active_sum - active_cycles.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        for row in 0..allocated_rows.len() {
            bind_cpu_row(
                cs.namespace(|| format!("D5 cross-relation row {row}")),
                row,
                &allocated_rows[row],
                &lookup_cycles[row],
                &register_cycles[row],
                &ram_cycles[row],
                &bytecode_root,
            )?;
        }
        for (label, current, expected) in [
            (
                "expected PC",
                cpu_input(&allocated_rows[0].inputs, JoltR1CSInputs::PC),
                &z[CPU_EXPECTED_PC_SLOT],
            ),
            (
                "expected unexpanded PC",
                cpu_input(&allocated_rows[0].inputs, JoltR1CSInputs::UnexpandedPC),
                &z[CPU_EXPECTED_UNEXPANDED_PC_SLOT],
            ),
            (
                "expected virtual flag",
                cpu_input(
                    &allocated_rows[0].inputs,
                    JoltR1CSInputs::OpFlags(CircuitFlags::VirtualInstruction),
                ),
                &z[CPU_EXPECTED_VIRTUAL_SLOT],
            ),
            (
                "expected first-in-sequence flag",
                cpu_input(
                    &allocated_rows[0].inputs,
                    JoltR1CSInputs::OpFlags(CircuitFlags::IsFirstInSequence),
                ),
                &z[CPU_EXPECTED_FIRST_SLOT],
            ),
            (
                "expected noop flag",
                &bit_as_num(
                    &allocated_rows[0].static_data.instruction_flags
                        [InstructionFlags::IsNoop as usize],
                ),
                &z[CPU_EXPECTED_NOOP_SLOT],
            ),
        ] {
            enforce_num_equal(cs.namespace(|| label), current, expected, label);
        }
        let mut query = allocated_poseidon_initial(
            cs.namespace(|| "CPU query initial"),
            DIRECT_CPU_QUERY_DOMAIN,
        )?;
        for (label, value) in [
            ("block_index", &block_index),
            ("cycle_start", &cycle_start),
            ("active_cycles", &active_cycles),
            ("terminated", &terminated_num),
            ("bytecode_root", &bytecode_root),
            ("bytecode_depth", &bytecode_depth),
        ] {
            query = poseidon_absorb(
                cs.namespace(|| format!("CPU query {label}")),
                &query,
                value,
                label,
            )?;
        }
        for (row_index, row) in allocated_rows.iter().enumerate() {
            let active = bit_as_num(&row.active);
            query = poseidon_absorb(
                cs.namespace(|| format!("CPU query active absorb {row_index}")),
                &query,
                &active,
                "cpu_active",
            )?;
            for (input_index, input) in row.inputs.iter().enumerate() {
                query = poseidon_absorb(
                    cs.namespace(|| format!("CPU query {row_index} {input_index}")),
                    &query,
                    input,
                    "cpu_input",
                )?;
            }
            let static_values = vec![
                row.static_data.instruction_tag.clone(),
                bit_as_num(&row.static_data.rs1_enabled),
                row.static_data.rs1_address.clone(),
                bit_as_num(&row.static_data.rs2_enabled),
                row.static_data.rs2_address.clone(),
                bit_as_num(&row.static_data.rd_enabled),
                row.static_data.rd_address.clone(),
                row.static_data.table_id.clone(),
                bit_as_num(&row.static_data.next_is_noop),
            ];
            for (static_index, value) in static_values.iter().enumerate() {
                query = poseidon_absorb(
                    cs.namespace(|| format!("CPU static query {row_index} {static_index}")),
                    &query,
                    value,
                    "cpu_static",
                )?;
            }
            for (flag, value) in row.static_data.instruction_flags.iter().enumerate() {
                query = poseidon_absorb(
                    cs.namespace(|| format!("CPU instruction flag query {row_index} {flag}")),
                    &query,
                    &bit_as_num(value),
                    "cpu_instruction_flag",
                )?;
            }
            for (level, sibling) in row.static_data.bytecode_path.iter().enumerate() {
                query = poseidon_absorb(
                    cs.namespace(|| format!("CPU bytecode path query {row_index} {level}")),
                    &query,
                    sibling,
                    "cpu_bytecode_sibling",
                )?;
            }
        }
        let claimed_query = alloc_witness_num(
            cs.namespace(|| "claimed CPU query"),
            nova_from_fr(&block.query_root)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "CPU query trace binding"),
            &query.state,
            &claimed_query,
            "CPU query trace binding",
        );
        let mut transcript =
            super::recursive_relations::AllocatedRecursivePoseidonTranscriptState {
                state: z[CPU_TRANSCRIPT_STATE_SLOT].clone(),
                n_rounds: z[CPU_TRANSCRIPT_ROUND_SLOT].clone(),
            };
        for (label, value) in [
            ("block_index", &block_index),
            ("cycle_start", &cycle_start),
            ("active_cycles", &active_cycles),
            ("terminated", &terminated_num),
            ("bytecode_root", &bytecode_root),
            ("bytecode_depth", &bytecode_depth),
            ("cpu_query_root", &claimed_query),
        ] {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("CPU transcript {label}")),
                &transcript,
                value,
                label,
            )?;
        }
        transcript = super::direct_lookup::poseidon_challenge(
            cs.namespace(|| "CPU row folding challenge"),
            &transcript,
        )?;
        let challenge = transcript.state.clone();
        let mut claim = alloc_nova_constant(cs.namespace(|| "CPU claim zero"), NovaScalar::zero())?;
        let mut power =
            alloc_nova_constant(cs.namespace(|| "CPU claim power one"), NovaScalar::one())?;
        for (row_index, row) in allocated_rows.iter().enumerate() {
            for (input_index, input) in row.inputs.iter().enumerate() {
                let term = super::direct_lookup::mul_nums(
                    cs.namespace(|| format!("CPU claim term {row_index} {input_index}")),
                    &power,
                    input,
                    "CPU claim term",
                )?;
                claim = super::direct_lookup::add_nums(
                    cs.namespace(|| format!("CPU claim sum {row_index} {input_index}")),
                    &claim,
                    &term,
                    "CPU claim sum",
                )?;
                power = super::direct_lookup::mul_nums(
                    cs.namespace(|| format!("CPU claim power {row_index} {input_index}")),
                    &power,
                    &challenge,
                    "CPU claim power",
                )?;
            }
        }
        let claimed_block = alloc_witness_num(
            cs.namespace(|| "claimed CPU block claim"),
            nova_from_fr(&block.block_claim)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "CPU block claim binding"),
            &claim,
            &claimed_block,
            "CPU block claim binding",
        );
        transcript = poseidon_absorb(
            cs.namespace(|| "CPU block claim transcript"),
            &transcript,
            &claim,
            "cpu_block_claim",
        )?;
        let running_claim = super::direct_lookup::add_nums(
            cs.namespace(|| "CPU running claim"),
            &z[CPU_CLAIM_SLOT],
            &claim,
            "CPU running claim",
        )?;
        let claimed_state = alloc_witness_num(
            cs.namespace(|| "claimed CPU transcript state"),
            nova_from_fr(&field_from_digest(&cpu.transcript_state_after))?,
        )?;
        let claimed_round = alloc_witness_num(
            cs.namespace(|| "claimed CPU transcript round"),
            NovaScalar::from(cpu.transcript_round_after as u64),
        )?;
        enforce_num_equal(
            cs.namespace(|| "CPU transcript state binding"),
            &transcript.state,
            &claimed_state,
            "CPU transcript state binding",
        );
        enforce_num_equal(
            cs.namespace(|| "CPU transcript round binding"),
            &transcript.n_rounds,
            &claimed_round,
            "CPU transcript round binding",
        );
        let mut last_selectors = Vec::with_capacity(allocated_rows.len());
        for row in 0..allocated_rows.len() {
            let current = bit_as_num(&allocated_rows[row].active);
            let next = if row + 1 < allocated_rows.len() {
                bit_as_num(&allocated_rows[row + 1].active)
            } else {
                alloc_nova_constant(cs.namespace(|| "CPU inactive sentinel"), NovaScalar::zero())?
            };
            let one = alloc_nova_constant(
                cs.namespace(|| format!("CPU selector one {row}")),
                NovaScalar::one(),
            )?;
            let one_minus_next = super::direct_lookup::sub_nums(
                cs.namespace(|| format!("CPU one minus next active {row}")),
                &one,
                &next,
                "CPU one minus next active",
            )?;
            last_selectors.push(super::direct_lookup::mul_nums(
                cs.namespace(|| format!("CPU last selector {row}")),
                &current,
                &one_minus_next,
                "CPU last selector",
            )?);
        }
        let next_pc = select_last_num(
            cs.namespace(|| "next block PC"),
            &last_selectors,
            &allocated_rows
                .iter()
                .map(|row| cpu_input(&row.inputs, JoltR1CSInputs::NextPC))
                .collect::<Vec<_>>(),
        )?;
        let next_unexpanded_pc = select_last_num(
            cs.namespace(|| "next block unexpanded PC"),
            &last_selectors,
            &allocated_rows
                .iter()
                .map(|row| cpu_input(&row.inputs, JoltR1CSInputs::NextUnexpandedPC))
                .collect::<Vec<_>>(),
        )?;
        let next_virtual = select_last_num(
            cs.namespace(|| "next block virtual flag"),
            &last_selectors,
            &allocated_rows
                .iter()
                .map(|row| cpu_input(&row.inputs, JoltR1CSInputs::NextIsVirtual))
                .collect::<Vec<_>>(),
        )?;
        let next_first = select_last_num(
            cs.namespace(|| "next block first flag"),
            &last_selectors,
            &allocated_rows
                .iter()
                .map(|row| cpu_input(&row.inputs, JoltR1CSInputs::NextIsFirstInSequence))
                .collect::<Vec<_>>(),
        )?;
        let next_noop_values = allocated_rows
            .iter()
            .map(|row| bit_as_num(&row.static_data.next_is_noop))
            .collect::<Vec<_>>();
        let next_noop = select_last_num(
            cs.namespace(|| "next block noop flag"),
            &last_selectors,
            &next_noop_values.iter().collect::<Vec<_>>(),
        )?;
        let final_step =
            AllocatedBit::alloc(cs.namespace(|| "CPU final step"), Some(self.final_step))?;
        let final_step_num = bit_as_num(&final_step);
        enforce_num_equal(
            cs.namespace(|| "CPU closure equals trace termination"),
            &final_step_num,
            &terminated_num,
            "CPU closure equals trace termination",
        );
        for (label, value) in [
            ("terminal next PC", &next_pc),
            ("terminal next unexpanded PC", &next_unexpanded_pc),
            ("terminal next virtual", &next_virtual),
            ("terminal next first", &next_first),
        ] {
            cs.enforce(
                || label,
                |lc| lc + final_step.get_variable(),
                |lc| lc + value.get_variable(),
                |lc| lc,
            );
        }
        cs.enforce(
            || "terminal next row is noop",
            |lc| lc + final_step.get_variable(),
            |lc| lc + next_noop.get_variable() - CS::one(),
            |lc| lc,
        );
        let mut output = ram_output;
        output.push(bytecode_root);
        output.push(next_pc);
        output.push(next_unexpanded_pc);
        output.push(next_virtual);
        output.push(next_first);
        output.push(next_noop);
        output.push(running_claim);
        output.push(transcript.state);
        output.push(transcript.n_rounds);
        Ok(output)
    }
}

fn direct_cpu_initial_z(
    preprocessing: &DirectChunkedPreprocessing,
    registers: &[u64; common::constants::REGISTER_COUNT as usize],
    ram_registry_root: Fr,
    ram_root: Fr,
    bytecode_root: Fr,
) -> Vec<NovaScalar> {
    let mut z = direct_ram_initial_z(preprocessing, registers, ram_registry_root, ram_root);
    z.push(nova_from_fr(&bytecode_root).expect("bytecode root canonical"));
    let entry_pc = crate::zkvm::bytecode::entry_bytecode_index(&preprocessing.bytecode) as i128;
    for value in [
        entry_pc,
        preprocessing.bytecode.entry_address as i128,
        0,
        0,
        0,
    ] {
        z.push(nova_from_fr(&Fr::from_i128(value)).expect("initial CPU row canonical"));
    }
    z.push(NovaScalar::zero());
    let transcript = PoseidonTranscript::new(DIRECT_CPU_QUERY_DOMAIN);
    z.push(nova_from_fr(&field_from_digest(&transcript.state)).expect("CPU transcript canonical"));
    z.push(NovaScalar::zero());
    z
}

fn setup_cpu_pp(
    circuit: &DirectCpuStepCircuit,
) -> Result<DirectCpuPublicParams, DirectChunkedError> {
    DirectCpuPublicParams::setup(
        circuit,
        &*nova_snark::traits::snark::default_ck_hint::<NovaPrimaryEngine>(),
        &*nova_snark::traits::snark::default_ck_hint::<NovaSecondaryEngine>(),
    )
    .map_err(|e| DirectChunkedError::InvalidProofShape(format!("CPU Nova setup failed: {e:?}")))
}

fn cpu_relations() -> BTreeMap<DirectRelation, DirectRelationState> {
    DirectRelation::ALL
        .into_iter()
        .map(|relation| {
            (
                relation,
                match relation {
                    DirectRelation::Pcs => DirectRelationState::Deferred,
                    _ => DirectRelationState::Proven,
                },
            )
        })
        .collect()
}

pub fn prove_direct_cpu_stage<I>(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    blocks: I,
) -> Result<DirectCpuStageProof, DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    let blocks = blocks.into_iter().collect::<Vec<_>>();
    if blocks.is_empty() {
        return Err(DirectChunkedError::EmptyTrace);
    }
    let d4 = prepare_direct_ram_stage(preprocessing, capacity, &blocks)?;
    let mut cpu_transcript = PoseidonTranscript::new(DIRECT_CPU_QUERY_DOMAIN);
    let mut cpus = Vec::with_capacity(blocks.len());
    for position in 0..blocks.len() {
        let lookahead = blocks
            .get(position + 1)
            .and_then(|next| next.cycles.first())
            .or_else(|| {
                blocks[position]
                    .end_state
                    .terminated
                    .then_some(&TERMINAL_CPU_LOOKAHEAD)
            });
        cpus.push(derive_cpu_subclaim(
            preprocessing,
            &blocks[position],
            capacity,
            lookahead,
            &mut cpu_transcript,
        )?);
    }
    let mut verify_transcript = PoseidonTranscript::new(DIRECT_CPU_QUERY_DOMAIN);
    for cpu in &cpus {
        verify_cpu_subclaim(cpu, &mut verify_transcript)?;
    }
    let z0 = direct_cpu_initial_z(
        preprocessing,
        &d4.initial_registers,
        d4.registry_root,
        d4.initial_root,
        cpus[0].block.bytecode_root,
    );
    let first = DirectCpuStepCircuit::for_subclaims(
        d4.lookup[0].clone(),
        d4.registers[0].clone(),
        d4.rams[0].clone(),
        cpus[0].clone(),
        blocks.len() == 1,
    );
    let pp = setup_cpu_pp(&first)?;
    let mut recursive = DirectCpuNovaSnark::new(&pp, &first, &z0).map_err(|e| {
        DirectChunkedError::InvalidProofShape(format!("CPU Nova init failed: {e:?}"))
    })?;
    recursive.prove_step(&pp, &first).map_err(|e| {
        DirectChunkedError::InvalidProofShape(format!("CPU Nova first step failed: {e:?}"))
    })?;
    for position in 1..blocks.len() {
        let circuit = DirectCpuStepCircuit::for_subclaims(
            d4.lookup[position].clone(),
            d4.registers[position].clone(),
            d4.rams[position].clone(),
            cpus[position].clone(),
            position + 1 == blocks.len(),
        );
        recursive.prove_step(&pp, &circuit).map_err(|e| {
            DirectChunkedError::InvalidProofShape(format!("CPU Nova step failed: {e:?}"))
        })?;
    }
    let output = recursive.verify(&pp, blocks.len(), &z0).map_err(|e| {
        DirectChunkedError::InvalidProofShape(format!("CPU Nova self verify failed: {e:?}"))
    })?;
    let (pk, vk) = DirectCpuCompressedSnark::setup(&pp).map_err(|e| {
        DirectChunkedError::InvalidProofShape(format!("CPU Spartan setup failed: {e:?}"))
    })?;
    let compressed = DirectCpuCompressedSnark::prove(&pp, &pk, &recursive).map_err(|e| {
        DirectChunkedError::InvalidProofShape(format!("CPU Spartan prove failed: {e:?}"))
    })?;
    if compressed.verify(&vk, blocks.len(), &z0).map_err(|e| {
        DirectChunkedError::InvalidProofShape(format!("CPU Spartan verify failed: {e:?}"))
    })? != output
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "CPU Nova/Spartan output mismatch".to_string(),
        ));
    }
    let folded_cpu_claim = cpus.iter().map(|c| c.block.block_claim).sum();
    if output[CPU_CLAIM_SLOT]
        != nova_from_fr(&folded_cpu_claim).map_err(|e| {
            DirectChunkedError::InvalidProofShape(format!("CPU claim conversion: {e:?}"))
        })?
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "CPU folded claim mismatch".to_string(),
        ));
    }
    let proof = DirectCpuStageProof {
        statement: DirectCpuStageStatement {
            program_digest: preprocessing.program_digest,
            lookup_table_commitment: preprocessing.lookup_table_commitment,
            bytecode_root: cpus[0].block.bytecode_root,
            block_count: blocks.len(),
            total_cycles: d4.total_cycles,
            initial_registers: d4.initial_registers,
            final_registers: d4.final_registers,
            ram_addresses: d4.addresses,
            ram_registry_root: d4.registry_root,
            initial_ram_root: d4.initial_root,
            final_ram_root: d4.final_root,
            folded_cpu_claim,
            terminated: true,
            relations: cpu_relations(),
        },
        lookup_subclaims: d4.lookup,
        register_subclaims: d4.registers,
        ram_subclaims: d4.rams,
        cpu_subclaims: cpus,
        nova_recursive_snark: postcard::to_stdvec(&recursive).map_err(|e| {
            DirectChunkedError::InvalidProofShape(format!("CPU Nova serialization: {e:?}"))
        })?,
        spartan_proof: postcard::to_stdvec(&compressed).map_err(|e| {
            DirectChunkedError::InvalidProofShape(format!("CPU Spartan serialization: {e:?}"))
        })?,
        initial_z: z0.iter().copied().map(nova_to_storage).collect(),
        final_z: output.iter().copied().map(nova_to_storage).collect(),
    };
    verify_direct_cpu_stage(preprocessing, capacity, &proof)?;
    Ok(proof)
}

pub fn verify_direct_cpu_stage(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    proof: &DirectCpuStageProof,
) -> Result<(), DirectChunkedError> {
    let n = proof.statement.block_count;
    if n == 0
        || proof.statement.program_digest != preprocessing.program_digest
        || proof.statement.lookup_table_commitment != preprocessing.lookup_table_commitment
        || proof.statement.bytecode_root != bytecode_tree(preprocessing).last().unwrap()[0]
        || proof.statement.relations != cpu_relations()
        || !proof.statement.terminated
        || proof
            .statement
            .ram_addresses
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || proof.statement.ram_registry_root != ram_registry_root(&proof.statement.ram_addresses)
        || proof.lookup_subclaims.len() != n
        || proof.register_subclaims.len() != n
        || proof.ram_subclaims.len() != n
        || proof.cpu_subclaims.len() != n
        || proof
            .cpu_subclaims
            .iter()
            .any(|subclaim| subclaim.block.cycle_capacity != capacity)
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D5 envelope mismatch".to_string(),
        ));
    }
    let (block_count, total_cycles) = validate_combined_sequence(
        preprocessing,
        capacity,
        &proof.lookup_subclaims,
        &proof.register_subclaims,
    )?;
    if block_count != n
        || total_cycles != proof.statement.total_cycles
        || proof.register_subclaims[0].block.start_registers != proof.statement.initial_registers
        || proof.register_subclaims[n - 1].block.end_registers != proof.statement.final_registers
        || proof.ram_subclaims[0].block.start_root != proof.statement.initial_ram_root
        || proof.ram_subclaims[n - 1].block.end_root != proof.statement.final_ram_root
        || proof.statement.ram_registry_root != proof.ram_subclaims[0].block.registry_root
        || proof
            .cpu_subclaims
            .iter()
            .any(|subclaim| subclaim.block.bytecode_root != proof.statement.bytecode_root)
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D5 public relation boundaries mismatch".to_string(),
        ));
    }
    let mut lookup_t =
        PoseidonTranscript::new(super::direct_lookup::DIRECT_LOOKUP_TRANSCRIPT_DOMAIN);
    for s in &proof.lookup_subclaims {
        super::direct_lookup::verify_native_subclaim(s, &mut lookup_t)?;
    }
    let mut reg_t =
        PoseidonTranscript::new(super::direct_register::DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
    for s in &proof.register_subclaims {
        super::direct_register::verify_native_register_subclaim(s, &mut reg_t)?;
    }
    let mut ram_t = PoseidonTranscript::new(super::direct_ram::DIRECT_RAM_QUERY_DOMAIN);
    for (position, s) in proof.ram_subclaims.iter().enumerate() {
        super::direct_ram::verify_ram_subclaim(s, &mut ram_t)?;
        if s.block.registry_root != proof.statement.ram_registry_root
            || s.block.address_count != proof.statement.ram_addresses.len()
            || (position > 0
                && proof.ram_subclaims[position - 1].block.end_root != s.block.start_root)
        {
            return Err(DirectChunkedError::InvalidProofShape(
                "D5 RAM continuity mismatch".to_string(),
            ));
        }
    }
    let mut cpu_t = PoseidonTranscript::new(DIRECT_CPU_QUERY_DOMAIN);
    for (position, cpu) in proof.cpu_subclaims.iter().enumerate() {
        verify_cpu_subclaim(cpu, &mut cpu_t)?;
        if cpu.block.block_index != position
            || cpu.block.block_index != proof.lookup_subclaims[position].block.block_index
        {
            return Err(DirectChunkedError::InvalidProofShape(
                "D5 block continuity mismatch".to_string(),
            ));
        }
    }
    let folded: Fr = proof
        .cpu_subclaims
        .iter()
        .map(|c| c.block.block_claim)
        .sum();
    if folded != proof.statement.folded_cpu_claim {
        return Err(DirectChunkedError::InvalidProofShape(
            "D5 public CPU claim mismatch".to_string(),
        ));
    }
    let z0 = direct_cpu_initial_z(
        preprocessing,
        &proof.statement.initial_registers,
        proof.statement.ram_registry_root,
        proof.statement.initial_ram_root,
        proof.statement.bytecode_root,
    );
    let setup = DirectCpuStepCircuit::for_subclaims(
        proof.lookup_subclaims[0].clone(),
        proof.register_subclaims[0].clone(),
        proof.ram_subclaims[0].clone(),
        proof.cpu_subclaims[0].clone(),
        n == 1,
    );
    let pp = setup_cpu_pp(&setup)?;
    let recursive = postcard::from_bytes::<DirectCpuNovaSnark>(&proof.nova_recursive_snark)
        .map_err(|e| {
            DirectChunkedError::InvalidProofShape(format!("CPU Nova deserialize: {e:?}"))
        })?;
    let output = recursive
        .verify(&pp, n, &z0)
        .map_err(|e| DirectChunkedError::InvalidProofShape(format!("CPU Nova verify: {e:?}")))?;
    let (_, vk) = DirectCpuCompressedSnark::setup(&pp)
        .map_err(|e| DirectChunkedError::InvalidProofShape(format!("CPU Spartan setup: {e:?}")))?;
    let compressed = postcard::from_bytes::<DirectCpuCompressedSnark>(&proof.spartan_proof)
        .map_err(|e| {
            DirectChunkedError::InvalidProofShape(format!("CPU Spartan deserialize: {e:?}"))
        })?;
    let compressed_output = compressed
        .verify(&vk, n, &z0)
        .map_err(|e| DirectChunkedError::InvalidProofShape(format!("CPU Spartan verify: {e:?}")))?;
    if output != compressed_output
        || output.len() != DIRECT_CPU_Z_ARITY
        || output[CPU_CLAIM_SLOT]
            != nova_from_fr(&folded).map_err(|e| {
                DirectChunkedError::InvalidProofShape(format!("CPU claim conversion: {e:?}"))
            })?
        || output[CPU_TRANSCRIPT_STATE_SLOT]
            != nova_from_fr(&field_from_digest(&cpu_t.state)).map_err(|e| {
                DirectChunkedError::InvalidProofShape(format!("CPU transcript conversion: {e:?}"))
            })?
        || output[CPU_TRANSCRIPT_ROUND_SLOT] != NovaScalar::from(cpu_t.n_rounds as u64)
        || output[RAM_ROOT_SLOT]
            != nova_from_fr(&proof.statement.final_ram_root).map_err(|e| {
                DirectChunkedError::InvalidProofShape(format!("RAM root conversion: {e:?}"))
            })?
        || output
            [REGISTER_STATE_OFFSET..REGISTER_STATE_OFFSET + proof.statement.final_registers.len()]
            != proof
                .statement
                .final_registers
                .iter()
                .copied()
                .map(NovaScalar::from)
                .collect::<Vec<_>>()
        || proof.initial_z != z0.iter().copied().map(nova_to_storage).collect::<Vec<_>>()
        || proof.final_z
            != output
                .iter()
                .copied()
                .map(nova_to_storage)
                .collect::<Vec<_>>()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D5 closure mismatch".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::r1cs::inputs::JoltR1CSInputs;
    use common::constants::REGISTER_COUNT;
    use tracer::instruction::{
        format::{
            format_j::{FormatJ, RegisterStateFormatJ},
            format_load::{FormatLoad, RegisterStateFormatLoad},
            format_s::{FormatS, RegisterStateFormatS},
        },
        jal::JAL,
        ld::LD,
        sd::SD,
        RAMRead, RAMWrite, RISCVCycle,
    };
    fn boundary(
        cycle: usize,
        regs: [u64; REGISTER_COUNT as usize],
        terminated: bool,
    ) -> tracer::MachineBoundaryState {
        tracer::MachineBoundaryState {
            global_cycle: cycle,
            emulator_trace_len: cycle,
            pc: 0x8000_0000 + 4 * cycle as u64,
            registers: regs.map(|x| x as i64),
            terminated,
        }
    }
    fn store_cycle(address: u64, value: u64) -> Cycle {
        RISCVCycle::<SD> {
            instruction: SD {
                address: 0x8000_0000,
                operands: FormatS {
                    rs1: 1,
                    rs2: 2,
                    imm: 0,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            register_state: RegisterStateFormatS {
                rs1: address,
                rs2: value,
            },
            ram_access: RAMWrite {
                address,
                pre_value: 0,
                post_value: value,
            },
        }
        .into()
    }
    fn load_cycle(address: u64, pre: u64, value: u64) -> Cycle {
        RISCVCycle::<LD> {
            instruction: LD {
                address: 0x8000_0004,
                operands: FormatLoad {
                    rd: 3,
                    rs1: 1,
                    imm: 0,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            register_state: RegisterStateFormatLoad {
                rd: (pre, value),
                rs1: address,
            },
            ram_access: RAMRead { address, value },
        }
        .into()
    }
    fn terminating_jump_cycle(pre: u64) -> Cycle {
        let address = 0x8000_0008u64;
        RISCVCycle::<JAL> {
            instruction: JAL {
                address,
                operands: FormatJ {
                    rd: 4,
                    imm: 0u64.wrapping_sub(address),
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            register_state: RegisterStateFormatJ {
                rd: (pre, address + 4),
            },
            ram_access: (),
        }
        .into()
    }
    fn blocks() -> Vec<TraceBlock> {
        let ram_address = common::constants::RAM_START_ADDRESS + 0x100;
        let mut a = [0u64; REGISTER_COUNT as usize];
        a[1] = ram_address;
        a[2] = 9;
        let b = a;
        let mut c = b;
        c[3] = 9;
        c[4] = 0x8000_000c;
        vec![
            TraceBlock {
                block_index: 0,
                global_cycle_start: 0,
                active_cycles: 1,
                target_size: 1,
                start_state: boundary(0, a, false),
                end_state: boundary(1, b, false),
                cycles: vec![store_cycle(ram_address, a[2])],
                ended_at_tick_boundary: true,
            },
            TraceBlock {
                block_index: 1,
                global_cycle_start: 1,
                active_cycles: 2,
                target_size: 2,
                start_state: boundary(1, b, false),
                end_state: boundary(3, c, true),
                cycles: vec![
                    load_cycle(ram_address, b[3], 9),
                    terminating_jump_cycle(b[4]),
                ],
                ended_at_tick_boundary: true,
            },
        ]
    }
    fn preprocessing(blocks: &[TraceBlock]) -> DirectChunkedPreprocessing {
        let cycles = blocks
            .iter()
            .flat_map(|b| b.cycles.iter().cloned())
            .collect::<Vec<_>>();
        DirectChunkedPreprocessing::from_trace_cycles(b"cpu-test", 64, &cycles).unwrap()
    }

    fn cpu_subclaims(
        preprocessing: &DirectChunkedPreprocessing,
        blocks: &[TraceBlock],
    ) -> Vec<DirectCpuSubclaim> {
        let mut transcript = PoseidonTranscript::new(DIRECT_CPU_QUERY_DOMAIN);
        (0..blocks.len())
            .map(|position| {
                let lookahead = blocks
                    .get(position + 1)
                    .and_then(|next| next.cycles.first())
                    .or_else(|| {
                        blocks[position]
                            .end_state
                            .terminated
                            .then_some(&TERMINAL_CPU_LOOKAHEAD)
                    });
                derive_cpu_subclaim(
                    preprocessing,
                    &blocks[position],
                    2,
                    lookahead,
                    &mut transcript,
                )
                .unwrap()
            })
            .collect()
    }

    fn synthesize_step_for_test(
        circuit: &DirectCpuStepCircuit,
        z_values: &[NovaScalar],
    ) -> (
        nova_snark::frontend::test_cs::TestConstraintSystem<NovaScalar>,
        Vec<NovaScalar>,
    ) {
        use nova_snark::frontend::{
            num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem,
        };

        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z = z_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("z {index}")), || Ok(*value)).unwrap()
            })
            .collect::<Vec<_>>();
        let output = circuit.synthesize(&mut cs, &z).unwrap();
        let output = output
            .iter()
            .map(|value| value.get_value().unwrap())
            .collect();
        (cs, output)
    }
    #[test]
    fn d5_rejects_forged_cpu_input() {
        let blocks = blocks();
        let pp = preprocessing(&blocks);
        let mut transcript = PoseidonTranscript::new(DIRECT_CPU_QUERY_DOMAIN);
        let mut claim = derive_cpu_subclaim(
            &pp,
            &blocks[0],
            2,
            blocks[1].cycles.first(),
            &mut transcript,
        )
        .unwrap();
        claim.block.cycles[0].inputs[JoltR1CSInputs::LookupOutput.to_index()] += 1;
        let mut verifier = PoseidonTranscript::new(DIRECT_CPU_QUERY_DOMAIN);
        assert!(verify_cpu_subclaim(&claim, &mut verifier).is_err());
    }

    #[test]
    fn d5_step_circuits_are_satisfied() {
        let blocks = blocks();
        let pp = preprocessing(&blocks);
        let d4 = prepare_direct_ram_stage(&pp, 2, &blocks).unwrap();
        let cpus = cpu_subclaims(&pp, &blocks);
        let mut z_values = direct_cpu_initial_z(
            &pp,
            &d4.initial_registers,
            d4.registry_root,
            d4.initial_root,
            cpus[0].block.bytecode_root,
        );
        for position in 0..blocks.len() {
            let circuit = DirectCpuStepCircuit::for_subclaims(
                d4.lookup[position].clone(),
                d4.registers[position].clone(),
                d4.rams[position].clone(),
                cpus[position].clone(),
                position + 1 == blocks.len(),
            );
            let (cs, output) = synthesize_step_for_test(&circuit, &z_values);
            assert!(
                cs.is_satisfied(),
                "block {position}: {:?}",
                cs.which_is_unsatisfied()
            );
            z_values = output;
        }
    }

    #[test]
    fn d5_rejects_relation_bytecode_and_cross_block_attacks() {
        let blocks = blocks();
        let pp = preprocessing(&blocks);
        let d4 = prepare_direct_ram_stage(&pp, 2, &blocks).unwrap();
        let cpus = cpu_subclaims(&pp, &blocks);
        let z0 = direct_cpu_initial_z(
            &pp,
            &d4.initial_registers,
            d4.registry_root,
            d4.initial_root,
            cpus[0].block.bytecode_root,
        );

        let mut forged_register_binding = cpus[0].clone();
        forged_register_binding.block.cycles[0].rs1_address ^= 1;
        let circuit = DirectCpuStepCircuit::for_subclaims(
            d4.lookup[0].clone(),
            d4.registers[0].clone(),
            d4.rams[0].clone(),
            forged_register_binding,
            false,
        );
        let (cs, _) = synthesize_step_for_test(&circuit, &z0);
        assert!(!cs.is_satisfied());

        let mut forged_bytecode = cpus[0].clone();
        forged_bytecode.block.cycles[0].bytecode_path[0] += Fr::from(1u64);
        let circuit = DirectCpuStepCircuit::for_subclaims(
            d4.lookup[0].clone(),
            d4.registers[0].clone(),
            d4.rams[0].clone(),
            forged_bytecode,
            false,
        );
        let (cs, _) = synthesize_step_for_test(&circuit, &z0);
        assert!(!cs.is_satisfied());

        let first = DirectCpuStepCircuit::for_subclaims(
            d4.lookup[0].clone(),
            d4.registers[0].clone(),
            d4.rams[0].clone(),
            cpus[0].clone(),
            false,
        );
        let (cs, mut next_z) = synthesize_step_for_test(&first, &z0);
        assert!(cs.is_satisfied(), "{:?}", cs.which_is_unsatisfied());
        next_z[CPU_EXPECTED_PC_SLOT] += NovaScalar::one();
        let second = DirectCpuStepCircuit::for_subclaims(
            d4.lookup[1].clone(),
            d4.registers[1].clone(),
            d4.rams[1].clone(),
            cpus[1].clone(),
            true,
        );
        let (cs, _) = synthesize_step_for_test(&second, &next_z);
        assert!(!cs.is_satisfied());

        let forged_early_closure = DirectCpuStepCircuit::for_subclaims(
            d4.lookup[0].clone(),
            d4.registers[0].clone(),
            d4.rams[0].clone(),
            cpus[0].clone(),
            true,
        );
        let (cs, _) = synthesize_step_for_test(&forged_early_closure, &z0);
        assert!(!cs.is_satisfied());
    }
    #[test]
    fn d5_nova_and_spartan_close_cpu_r1cs() {
        let blocks = blocks();
        let pp = preprocessing(&blocks);
        let proof = prove_direct_cpu_stage(&pp, 2, blocks).unwrap();
        assert_eq!(
            proof.statement.relations[&DirectRelation::Cpu],
            DirectRelationState::Proven
        );
        verify_direct_cpu_stage(&pp, 2, &proof).unwrap();
        let mut bad = proof;
        bad.cpu_subclaims[0].block.block_claim += Fr::from(1u64);
        assert!(verify_direct_cpu_stage(&pp, 2, &bad).is_err());
    }
}
