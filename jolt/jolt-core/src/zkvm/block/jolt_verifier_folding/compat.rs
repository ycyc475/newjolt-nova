//! Audit-only adapter from D8 witness-bearing subclaims to the V2 statement.
//!
//! It deliberately returns no `BlockJoltProof`: D11-D14 must generate compact
//! proof messages rather than laundering D8 row witnesses through a new type.

use ark_bn254::Fr;
use sha3::{Digest, Sha3_256};
use tracer::{MachineBoundaryState, TraceBlock};

use super::types::{
    deferred_claim_root, BlockBoundaryState, BlockJoltStatement, DeferredPcsClaim, FieldElement,
    TranscriptCheckpoint, BLOCK_JOLT_PROTOCOL_VERSION, BLOCK_JOLT_WIRE_VERSION,
};
use crate::zkvm::block::{
    DirectChunkedError, DirectChunkedPreprocessing, DirectCpuSubclaim, DirectLookupSubclaim,
    DirectRamSubclaim, DirectRegisterSubclaim,
};

fn field(value: &Fr) -> FieldElement {
    FieldElement::from_fr(value)
}

fn digest_words(domain: &[u8], words: impl IntoIterator<Item = Vec<u8>>) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(domain);
    for word in words {
        hasher.update((word.len() as u64).to_le_bytes());
        hasher.update(word);
    }
    hasher.finalize().into()
}

fn machine_commitment(state: &MachineBoundaryState) -> [u8; 32] {
    let mut words = vec![
        (state.global_cycle as u64).to_le_bytes().to_vec(),
        (state.emulator_trace_len as u64).to_le_bytes().to_vec(),
        state.pc.to_le_bytes().to_vec(),
        vec![u8::from(state.terminated)],
    ];
    words.extend(
        state
            .registers
            .iter()
            .map(|value| value.to_le_bytes().to_vec()),
    );
    digest_words(b"machine-boundary", words)
}

fn register_commitment(registers: &[u64]) -> [u8; 32] {
    digest_words(
        b"register-boundary",
        registers.iter().map(|value| value.to_le_bytes().to_vec()),
    )
}

fn preprocessing_id(preprocessing: &DirectChunkedPreprocessing, capacity: usize) -> [u8; 32] {
    digest_words(
        b"preprocessing",
        [
            preprocessing.program_digest.to_vec(),
            preprocessing.lookup_table_commitment.to_vec(),
            (capacity as u64).to_le_bytes().to_vec(),
        ],
    )
}

/// Builds the V2 public block statement from a D8 block and its four subclaims.
/// This function is a migration oracle only and cannot create a V2 proof.
#[allow(clippy::too_many_arguments)]
pub fn d8_statement_adapter(
    preprocessing: &DirectChunkedPreprocessing,
    trace: &TraceBlock,
    lookup: &DirectLookupSubclaim,
    register: &DirectRegisterSubclaim,
    ram: &DirectRamSubclaim,
    cpu: &DirectCpuSubclaim,
    transcript_before: TranscriptCheckpoint,
    transcript_after: TranscriptCheckpoint,
    lookup_accumulator_before: FieldElement,
    lookup_transcript_round_before: u64,
    lookup_accumulator_after: FieldElement,
    lookup_transcript_round_after: u64,
    deferred_pcs_claims: &[DeferredPcsClaim],
) -> Result<BlockJoltStatement, DirectChunkedError> {
    let index = trace.block_index;
    let start = trace.global_cycle_start;
    let active = trace.active_cycles;
    let capacity = lookup.block.cycle_capacity;
    if register.block.block_index != index
        || ram.block.block_index != index
        || cpu.block.block_index != index
        || lookup.block.block_index != index
        || register.block.global_cycle_start != start
        || ram.block.global_cycle_start != start
        || cpu.block.global_cycle_start != start
        || lookup.block.global_cycle_start != start
        || register.block.active_cycles != active
        || ram.block.active_cycles != active
        || cpu.block.active_cycles != active
        || lookup.block.active_cycles != active
        || register.block.cycle_capacity != capacity
        || ram.block.cycle_capacity != capacity
        || cpu.block.cycle_capacity != capacity
        || trace.end_state.terminated != lookup.block.terminated
        || trace.end_state.terminated != register.block.terminated
        || trace.end_state.terminated != ram.block.terminated
        || trace.end_state.terminated != cpu.block.terminated
    {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: index,
            reason: "D8 subclaims do not describe one common V2 block".to_string(),
        });
    }

    let statement = BlockJoltStatement {
        wire_version: BLOCK_JOLT_WIRE_VERSION,
        preprocessing_id: preprocessing_id(preprocessing, capacity),
        program_digest: preprocessing.program_digest,
        lookup_table_commitment: preprocessing.lookup_table_commitment,
        bytecode_commitment: field(&cpu.block.bytecode_root),
        block_index: index as u64,
        global_cycle_start: start as u64,
        global_cycle_end: (start + active) as u64,
        active_cycles: active as u64,
        cycle_capacity: capacity as u64,
        terminal: trace.end_state.terminated,
        start_pc: trace.start_state.pc,
        end_pc: trace.end_state.pc,
        start: BlockBoundaryState {
            machine_state: machine_commitment(&trace.start_state),
            register_state: register_commitment(&register.block.start_registers),
            ram_root: field(&ram.block.start_root),
        },
        end: BlockBoundaryState {
            machine_state: machine_commitment(&trace.end_state),
            register_state: register_commitment(&register.block.end_registers),
            ram_root: field(&ram.block.end_root),
        },
        lookup_accumulator_before,
        lookup_transcript_round_before,
        lookup_accumulator_after,
        lookup_transcript_round_after,
        transcript_before,
        transcript_after,
        deferred_pcs_claim_root: deferred_claim_root(deferred_pcs_claims),
    };
    statement
        .validate_shape()
        .map_err(|reason| DirectChunkedError::InvalidBlock {
            block_index: index,
            reason,
        })?;
    Ok(statement)
}
