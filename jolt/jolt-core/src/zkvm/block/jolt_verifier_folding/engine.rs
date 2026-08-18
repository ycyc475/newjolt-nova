//! D14 unified host-side block-Jolt prover and verifier.
//!
//! The master transcript binds four independently domain-separated native Jolt
//! verifier transcripts in the fixed Lookup -> Register -> RAM -> CPU order.
//! Only compact proof messages survive this API boundary; the trace block and
//! RAM map are prover-local and are discarded before the next call.

use std::collections::BTreeMap;

use serde::Serialize;
use sha3::{Digest, Sha3_256};
use tracer::{instruction::Cycle, MachineBoundaryState, TraceBlock};

use crate::transcripts::{PoseidonTranscript, Transcript};

use super::super::{direct::validate_block, DirectChunkedError, DirectChunkedPreprocessing};
use super::{
    cpu::{bytecode_root, prove_block_cpu_r1cs_with_commitment},
    deferred_claim_root,
    lookup::prove_block_lookup_lasso_with_commitment,
    new_block_cpu_transcript, new_block_lookup_transcript, new_block_ram_transcript,
    new_block_register_transcript, prove_block_cpu_r1cs, prove_block_lookup_lasso, prove_block_ram,
    prove_block_register,
    ram::prove_block_ram_with_commitment,
    register::prove_block_register_with_commitment,
    verify_block_cpu_r1cs, verify_block_lookup_lasso, verify_block_ram, verify_block_register,
    BlockBoundaryState, BlockJoltProof, BlockJoltStatement, DeferredPcsClaim, FieldElement,
    StreamingRecursiveState, TranscriptCheckpoint, BLOCK_JOLT_PROTOCOL_VERSION,
    BLOCK_JOLT_WIRE_VERSION,
};

const MASTER_TRANSCRIPT_DOMAIN: &[u8] = b"block-jolt-master-v2";
const PREPROCESSING_ID_DOMAIN: &[u8] = b"block-jolt-preprocessing-v2";
const MACHINE_STATE_DOMAIN: &[u8] = b"block-jolt-machine-state-v2";
const DEFERRED_ACCUMULATOR_DOMAIN: &[u8] = b"block-jolt-deferred-acc-v2";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockJoltHostConfig {
    pub cycle_capacity: usize,
    pub ram_k: usize,
}

impl BlockJoltHostConfig {
    pub fn validate(self) -> Result<(), DirectChunkedError> {
        if self.cycle_capacity < 2 || !self.cycle_capacity.is_power_of_two() {
            return Err(DirectChunkedError::InvalidConfiguration(
                "block-Jolt cycle capacity must be a power of two of at least two".to_string(),
            ));
        }
        if self.ram_k == 0 || !self.ram_k.is_power_of_two() {
            return Err(DirectChunkedError::InvalidConfiguration(
                "block-Jolt RAM K must be a non-zero power of two".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedBlockJoltTransition {
    pub statement: BlockJoltStatement,
    pub proof: BlockJoltProof,
    pub state_after: StreamingRecursiveState,
}

fn master_transcript() -> PoseidonTranscript {
    PoseidonTranscript::new(MASTER_TRANSCRIPT_DOMAIN)
}

fn checkpoint(transcript: &PoseidonTranscript) -> TranscriptCheckpoint {
    TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    }
}

fn machine_state_commitment(state: &MachineBoundaryState) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(MACHINE_STATE_DOMAIN);
    hasher.update((state.global_cycle as u64).to_le_bytes());
    hasher.update((state.emulator_trace_len as u64).to_le_bytes());
    hasher.update(state.pc.to_le_bytes());
    for register in state.registers {
        hasher.update(register.to_le_bytes());
    }
    hasher.update([u8::from(state.terminated)]);
    hasher.finalize().into()
}

fn preprocessing_id(
    preprocessing: &DirectChunkedPreprocessing,
    config: BlockJoltHostConfig,
    bytecode_root: FieldElement,
) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(PREPROCESSING_ID_DOMAIN);
    hasher.update(preprocessing.program_digest);
    hasher.update(preprocessing.lookup_table_commitment);
    hasher.update(bytecode_root.0);
    hasher.update((config.cycle_capacity as u64).to_le_bytes());
    hasher.update((config.ram_k as u64).to_le_bytes());
    hasher.update((preprocessing.max_padded_trace_length as u64).to_le_bytes());
    hasher.finalize().into()
}

fn absorb_serialized<T: Serialize>(
    transcript: &mut PoseidonTranscript,
    label: &'static [u8],
    value: &T,
) -> Result<(), DirectChunkedError> {
    let encoded = postcard::to_stdvec(value).map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!(
            "master transcript serialization failed: {error}"
        ))
    })?;
    transcript.append_bytes(label, &encoded);
    Ok(())
}

fn absorb_master_header(transcript: &mut PoseidonTranscript, statement: &BlockJoltStatement) {
    transcript.append_bytes(b"protocol", BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    for (label, value) in [
        (b"preprocessing_id".as_slice(), statement.preprocessing_id),
        (b"program_digest".as_slice(), statement.program_digest),
        (
            b"lookup_table".as_slice(),
            statement.lookup_table_commitment,
        ),
        (b"machine_start".as_slice(), statement.start.machine_state),
        (b"machine_end".as_slice(), statement.end.machine_state),
        (b"register_start".as_slice(), statement.start.register_state),
        (b"register_end".as_slice(), statement.end.register_state),
    ] {
        transcript.append_bytes(label, &value);
    }
    for (label, value) in [
        (b"wire_version".as_slice(), statement.wire_version as u64),
        (b"block_index".as_slice(), statement.block_index),
        (b"cycle_start".as_slice(), statement.global_cycle_start),
        (b"cycle_end".as_slice(), statement.global_cycle_end),
        (b"active_cycles".as_slice(), statement.active_cycles),
        (b"cycle_capacity".as_slice(), statement.cycle_capacity),
        (b"terminal".as_slice(), u64::from(statement.terminal)),
        (b"start_pc".as_slice(), statement.start_pc),
        (b"end_pc".as_slice(), statement.end_pc),
    ] {
        transcript.append_u64(label, value);
    }
    for (label, value) in [
        (b"bytecode_root".as_slice(), statement.bytecode_commitment),
        (b"ram_start".as_slice(), statement.start.ram_root),
        (b"ram_end".as_slice(), statement.end.ram_root),
        (
            b"lookup_acc_before".as_slice(),
            statement.lookup_accumulator_before,
        ),
        (
            b"lookup_acc_after".as_slice(),
            statement.lookup_accumulator_after,
        ),
    ] {
        transcript.append_scalar(label, &value.to_fr());
    }
}

fn absorb_master_proof(
    transcript: &mut PoseidonTranscript,
    statement: &BlockJoltStatement,
    proof: &BlockJoltProof,
) -> Result<(), DirectChunkedError> {
    absorb_master_header(transcript, statement);
    for (label, commitment) in [
        (
            b"lookup_commitment".as_slice(),
            proof.lookup.query_commitment.0,
        ),
        (
            b"register_commitment".as_slice(),
            proof.register.access_commitment.0,
        ),
        (b"ram_commitment".as_slice(), proof.ram.access_commitment.0),
        (b"cpu_commitment".as_slice(), proof.cpu.row_commitment),
    ] {
        transcript.append_bytes(label, &commitment);
    }
    absorb_serialized(transcript, b"lookup_proof", &proof.lookup)?;
    absorb_serialized(transcript, b"register_proof", &proof.register)?;
    absorb_serialized(transcript, b"ram_proof", &proof.ram)?;
    absorb_serialized(transcript, b"cpu_proof", &proof.cpu)?;
    transcript.append_bytes(b"deferred_pcs_root", &statement.deferred_pcs_claim_root);
    Ok(())
}

fn ordered_deferred_claims(proof: &BlockJoltProof) -> Vec<DeferredPcsClaim> {
    let mut claims = Vec::new();
    claims.extend(proof.lookup.sumcheck.opening_claims.iter().cloned());
    claims.extend(proof.register.sumcheck.opening_claims.iter().cloned());
    claims.extend(proof.ram.relation_proof.opening_claims.iter().cloned());
    claims.extend(proof.cpu.relation_proof.opening_claims.iter().cloned());
    claims
}

fn advance_deferred_accumulator(previous: [u8; 32], statement: &BlockJoltStatement) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(DEFERRED_ACCUMULATOR_DOMAIN);
    hasher.update(previous);
    hasher.update(statement.digest());
    hasher.update(statement.deferred_pcs_claim_root);
    hasher.finalize().into()
}

fn initial_state(
    statement: &BlockJoltStatement,
) -> Result<StreamingRecursiveState, DirectChunkedError> {
    if statement.block_index != 0 || statement.global_cycle_start != 0 {
        return Err(DirectChunkedError::InvalidProofShape(
            "the first block-Jolt statement must start at block and cycle zero".to_string(),
        ));
    }
    Ok(StreamingRecursiveState {
        wire_version: BLOCK_JOLT_WIRE_VERSION,
        preprocessing_id: statement.preprocessing_id,
        program_digest: statement.program_digest,
        lookup_table_commitment: statement.lookup_table_commitment,
        next_block_index: 0,
        next_global_cycle: 0,
        boundary: statement.start.clone(),
        lookup_accumulator: statement.lookup_accumulator_before,
        lookup_transcript_round: statement.lookup_transcript_round_before,
        transcript: statement.transcript_before,
        deferred_pcs_accumulator: [0; 32],
        total_active_cycles: 0,
        terminated: false,
    })
}

fn advance_state(
    state: &StreamingRecursiveState,
    statement: &BlockJoltStatement,
) -> Result<StreamingRecursiveState, DirectChunkedError> {
    state.validate_next(statement).map_err(|reason| {
        DirectChunkedError::InvalidProofShape(format!(
            "block-Jolt recursive state transition failed: {reason}"
        ))
    })?;
    Ok(StreamingRecursiveState {
        wire_version: state.wire_version,
        preprocessing_id: state.preprocessing_id,
        program_digest: state.program_digest,
        lookup_table_commitment: state.lookup_table_commitment,
        next_block_index: state.next_block_index + 1,
        next_global_cycle: statement.global_cycle_end,
        boundary: statement.end.clone(),
        lookup_accumulator: statement.lookup_accumulator_after,
        lookup_transcript_round: statement.lookup_transcript_round_after,
        transcript: statement.transcript_after,
        deferred_pcs_accumulator: advance_deferred_accumulator(
            state.deferred_pcs_accumulator,
            statement,
        ),
        total_active_cycles: state.total_active_cycles + statement.active_cycles,
        terminated: statement.terminal,
    })
}

#[derive(Clone)]
pub struct BlockJoltVerifier {
    preprocessing: DirectChunkedPreprocessing,
    config: BlockJoltHostConfig,
    expected_preprocessing_id: [u8; 32],
    expected_bytecode_root: FieldElement,
    master_transcript: PoseidonTranscript,
    lookup_transcript: PoseidonTranscript,
    state: Option<StreamingRecursiveState>,
}

impl BlockJoltVerifier {
    pub fn new(
        preprocessing: DirectChunkedPreprocessing,
        config: BlockJoltHostConfig,
    ) -> Result<Self, DirectChunkedError> {
        config.validate()?;
        let expected_bytecode_root = bytecode_root(&preprocessing)?;
        let expected_preprocessing_id =
            preprocessing_id(&preprocessing, config, expected_bytecode_root);
        Ok(Self {
            preprocessing,
            config,
            expected_preprocessing_id,
            expected_bytecode_root,
            master_transcript: master_transcript(),
            lookup_transcript: new_block_lookup_transcript(),
            state: None,
        })
    }

    pub fn state(&self) -> Option<&StreamingRecursiveState> {
        self.state.as_ref()
    }

    pub fn verify_block(
        &mut self,
        statement: &BlockJoltStatement,
        proof: &BlockJoltProof,
    ) -> Result<StreamingRecursiveState, DirectChunkedError> {
        if statement.preprocessing_id != self.expected_preprocessing_id
            || statement.program_digest != self.preprocessing.program_digest
            || statement.lookup_table_commitment != self.preprocessing.lookup_table_commitment
            || statement.bytecode_commitment != self.expected_bytecode_root
            || statement.cycle_capacity != self.config.cycle_capacity as u64
        {
            return Err(DirectChunkedError::InvalidProofShape(
                "block-Jolt statement uses unexpected preprocessing or circuit shape".to_string(),
            ));
        }
        proof
            .validate_structure(statement)
            .map_err(DirectChunkedError::InvalidProofShape)?;
        if checkpoint(&self.master_transcript) != statement.transcript_before
            || FieldElement(self.lookup_transcript.state) != statement.lookup_accumulator_before
            || self.lookup_transcript.n_rounds as u64 != statement.lookup_transcript_round_before
            || proof.lookup.accumulator_before != statement.lookup_accumulator_before
        {
            return Err(DirectChunkedError::InvalidProofShape(
                "block-Jolt transcript input state mismatch".to_string(),
            ));
        }

        let current = match &self.state {
            Some(state) => state.clone(),
            None => initial_state(statement)?,
        };
        current.validate_next(statement).map_err(|reason| {
            DirectChunkedError::InvalidProofShape(format!(
                "block-Jolt statement is not contiguous: {reason}"
            ))
        })?;

        // Verification is transactional: malformed input must not advance a
        // reusable verifier's transcript state.
        let mut lookup_transcript = self.lookup_transcript.clone();
        let mut master_transcript = self.master_transcript.clone();

        verify_block_lookup_lasso(
            &proof.lookup,
            statement.block_index as usize,
            statement.global_cycle_start as usize,
            statement.active_cycles as usize,
            self.config.cycle_capacity,
            &mut lookup_transcript,
        )?;
        if FieldElement(lookup_transcript.state) != statement.lookup_accumulator_after {
            return Err(DirectChunkedError::InvalidProofShape(
                "block-Jolt lookup accumulator output mismatch".to_string(),
            ));
        }
        if lookup_transcript.n_rounds as u64 != statement.lookup_transcript_round_after {
            return Err(DirectChunkedError::InvalidProofShape(
                "block-Jolt lookup transcript round output mismatch".to_string(),
            ));
        }

        let mut register_transcript = new_block_register_transcript();
        verify_block_register(
            &proof.register,
            statement.block_index as usize,
            statement.global_cycle_start as usize,
            statement.active_cycles as usize,
            self.config.cycle_capacity,
            statement.start.register_state,
            statement.end.register_state,
            &mut register_transcript,
        )?;
        let mut ram_transcript = new_block_ram_transcript();
        verify_block_ram(
            &self.preprocessing,
            &proof.ram,
            statement.block_index as usize,
            statement.global_cycle_start as usize,
            statement.active_cycles as usize,
            self.config.cycle_capacity,
            statement.start.ram_root,
            statement.end.ram_root,
            &mut ram_transcript,
        )?;
        let mut cpu_transcript = new_block_cpu_transcript();
        verify_block_cpu_r1cs(
            &proof.cpu,
            statement.block_index as usize,
            statement.global_cycle_start as usize,
            statement.active_cycles as usize,
            self.config.cycle_capacity,
            statement.start_pc,
            statement.end_pc,
            statement.terminal,
            statement.bytecode_commitment,
            &mut cpu_transcript,
        )?;
        if proof.cpu.start_pc != statement.start_pc
            || proof.cpu.end_pc != statement.end_pc
            || proof.ram.ram_k != self.config.ram_k as u64
        {
            return Err(DirectChunkedError::InvalidProofShape(
                "block-Jolt CPU/RAM public shape mismatch".to_string(),
            ));
        }

        let claims = ordered_deferred_claims(proof);
        if claims != proof.deferred_pcs_claims
            || deferred_claim_root(&claims) != statement.deferred_pcs_claim_root
        {
            return Err(DirectChunkedError::InvalidProofShape(
                "block-Jolt deferred PCS claim sequence mismatch".to_string(),
            ));
        }
        absorb_master_proof(&mut master_transcript, statement, proof)?;
        if checkpoint(&master_transcript) != statement.transcript_after {
            return Err(DirectChunkedError::InvalidProofShape(
                "block-Jolt master transcript output mismatch".to_string(),
            ));
        }
        let next = advance_state(&current, statement)?;
        self.lookup_transcript = lookup_transcript;
        self.master_transcript = master_transcript;
        self.state = Some(next.clone());
        Ok(next)
    }

    pub fn finish(&self) -> Result<&StreamingRecursiveState, DirectChunkedError> {
        let state = self.state.as_ref().ok_or(DirectChunkedError::EmptyTrace)?;
        if !state.terminated {
            return Err(DirectChunkedError::InvalidProofShape(
                "block-Jolt stream ended without a terminal block".to_string(),
            ));
        }
        Ok(state)
    }
}

pub struct BlockJoltProver {
    preprocessing: DirectChunkedPreprocessing,
    config: BlockJoltHostConfig,
    expected_preprocessing_id: [u8; 32],
    expected_bytecode_root: FieldElement,
    master_transcript: PoseidonTranscript,
    lookup_transcript: PoseidonTranscript,
    ram_state: BTreeMap<u64, u64>,
    state: Option<StreamingRecursiveState>,
}

impl BlockJoltProver {
    pub fn new(
        preprocessing: DirectChunkedPreprocessing,
        config: BlockJoltHostConfig,
        initial_ram: BTreeMap<u64, u64>,
    ) -> Result<Self, DirectChunkedError> {
        config.validate()?;
        let expected_bytecode_root = bytecode_root(&preprocessing)?;
        let expected_preprocessing_id =
            preprocessing_id(&preprocessing, config, expected_bytecode_root);
        Ok(Self {
            preprocessing,
            config,
            expected_preprocessing_id,
            expected_bytecode_root,
            master_transcript: master_transcript(),
            lookup_transcript: new_block_lookup_transcript(),
            ram_state: initial_ram,
            state: None,
        })
    }

    pub fn state(&self) -> Option<&StreamingRecursiveState> {
        self.state.as_ref()
    }

    pub(super) fn preprocessing(&self) -> &DirectChunkedPreprocessing {
        &self.preprocessing
    }

    pub(super) fn config(&self) -> BlockJoltHostConfig {
        self.config
    }

    pub(super) fn ram_state(&self) -> &BTreeMap<u64, u64> {
        &self.ram_state
    }

    pub fn prove_block(
        &mut self,
        block: &TraceBlock,
        lookahead: Option<&Cycle>,
    ) -> Result<VerifiedBlockJoltTransition, DirectChunkedError> {
        self.prove_block_bound(block, lookahead, None)
    }

    pub(super) fn prove_block_with_commitment(
        &mut self,
        block: &TraceBlock,
        lookahead: Option<&Cycle>,
        commitment_id: [u8; 32],
    ) -> Result<VerifiedBlockJoltTransition, DirectChunkedError> {
        self.prove_block_bound(block, lookahead, Some(commitment_id))
    }

    fn prove_block_bound(
        &mut self,
        block: &TraceBlock,
        lookahead: Option<&Cycle>,
        commitment_id: Option<[u8; 32]>,
    ) -> Result<VerifiedBlockJoltTransition, DirectChunkedError> {
        validate_block(block, self.config.cycle_capacity)?;
        let master_before = self.master_transcript.clone();
        let lookup_before = self.lookup_transcript.clone();
        let mut lookup_transcript = lookup_before.clone();
        let (lookup, _, _) = match commitment_id {
            Some(id) => prove_block_lookup_lasso_with_commitment(
                block,
                self.config.cycle_capacity,
                &mut lookup_transcript,
                id,
            )?,
            None => {
                prove_block_lookup_lasso(block, self.config.cycle_capacity, &mut lookup_transcript)?
            }
        };
        let mut register_transcript = new_block_register_transcript();
        let (register, _, _) = match commitment_id {
            Some(id) => prove_block_register_with_commitment(
                block,
                self.config.cycle_capacity,
                &mut register_transcript,
                id,
            )?,
            None => {
                prove_block_register(block, self.config.cycle_capacity, &mut register_transcript)?
            }
        };
        let mut ram_transcript = new_block_ram_transcript();
        let (ram, final_ram, _, _) = match commitment_id {
            Some(id) => prove_block_ram_with_commitment(
                &self.preprocessing,
                block,
                self.config.cycle_capacity,
                self.config.ram_k,
                &self.ram_state,
                &mut ram_transcript,
                Some(id),
            )?,
            None => prove_block_ram(
                &self.preprocessing,
                block,
                self.config.cycle_capacity,
                self.config.ram_k,
                &self.ram_state,
                &mut ram_transcript,
            )?,
        };
        let mut cpu_transcript = new_block_cpu_transcript();
        let (cpu, _, _) = match commitment_id {
            Some(id) => prove_block_cpu_r1cs_with_commitment(
                &self.preprocessing,
                block,
                self.config.cycle_capacity,
                lookahead,
                &mut cpu_transcript,
                Some(id),
            )?,
            None => prove_block_cpu_r1cs(
                &self.preprocessing,
                block,
                self.config.cycle_capacity,
                lookahead,
                &mut cpu_transcript,
            )?,
        };

        let start = BlockBoundaryState {
            machine_state: machine_state_commitment(&block.start_state),
            register_state: register.state_before,
            ram_root: ram.root_before,
        };
        let end = BlockBoundaryState {
            machine_state: machine_state_commitment(&block.end_state),
            register_state: register.state_after,
            ram_root: ram.root_after,
        };
        let transcript_before = checkpoint(&master_before);
        let mut proof = BlockJoltProof {
            wire_version: BLOCK_JOLT_WIRE_VERSION,
            statement_digest: [0; 32],
            transcript_before,
            lookup,
            register,
            ram,
            cpu,
            transcript_after: transcript_before,
            deferred_pcs_claims: Vec::new(),
            proof_digest: [0; 32],
        };
        proof.deferred_pcs_claims = ordered_deferred_claims(&proof);
        let deferred_pcs_claim_root = deferred_claim_root(&proof.deferred_pcs_claims);
        let mut statement = BlockJoltStatement {
            wire_version: BLOCK_JOLT_WIRE_VERSION,
            preprocessing_id: self.expected_preprocessing_id,
            program_digest: self.preprocessing.program_digest,
            lookup_table_commitment: self.preprocessing.lookup_table_commitment,
            bytecode_commitment: self.expected_bytecode_root,
            block_index: block.block_index as u64,
            global_cycle_start: block.global_cycle_start as u64,
            global_cycle_end: (block.global_cycle_start + block.active_cycles) as u64,
            active_cycles: block.active_cycles as u64,
            cycle_capacity: self.config.cycle_capacity as u64,
            terminal: block.end_state.terminated,
            start_pc: block.start_state.pc,
            end_pc: block.end_state.pc,
            start,
            end,
            lookup_accumulator_before: proof.lookup.accumulator_before,
            lookup_transcript_round_before: proof.lookup.accumulator_round_before,
            lookup_accumulator_after: proof.lookup.accumulator_after,
            lookup_transcript_round_after: proof.lookup.accumulator_round_after,
            transcript_before,
            transcript_after: transcript_before,
            deferred_pcs_claim_root,
        };
        let mut master_after = master_before.clone();
        absorb_master_proof(&mut master_after, &statement, &proof)?;
        statement.transcript_after = checkpoint(&master_after);
        proof.transcript_after = statement.transcript_after;
        proof.statement_digest = statement.digest();
        proof.seal();
        proof
            .validate_structure(&statement)
            .map_err(DirectChunkedError::InvalidProofShape)?;

        let mut verifier = BlockJoltVerifier {
            preprocessing: self.preprocessing.clone(),
            config: self.config,
            expected_preprocessing_id: self.expected_preprocessing_id,
            expected_bytecode_root: self.expected_bytecode_root,
            master_transcript: master_before,
            lookup_transcript: lookup_before,
            state: self.state.clone(),
        };
        let state_after = verifier.verify_block(&statement, &proof)?;
        if verifier.master_transcript.state != master_after.state
            || verifier.master_transcript.n_rounds != master_after.n_rounds
            || verifier.lookup_transcript.state != lookup_transcript.state
            || verifier.lookup_transcript.n_rounds != lookup_transcript.n_rounds
        {
            return Err(DirectChunkedError::InvalidProofShape(
                "block-Jolt prover/verifier transcript self-check diverged".to_string(),
            ));
        }

        self.master_transcript = master_after;
        self.lookup_transcript = lookup_transcript;
        self.ram_state = final_ram;
        self.state = Some(state_after.clone());
        Ok(VerifiedBlockJoltTransition {
            statement,
            proof,
            state_after,
        })
    }

    pub fn finish(&self) -> Result<&StreamingRecursiveState, DirectChunkedError> {
        let state = self.state.as_ref().ok_or(DirectChunkedError::EmptyTrace)?;
        if !state.terminated {
            return Err(DirectChunkedError::InvalidProofShape(
                "block-Jolt stream ended without a terminal block".to_string(),
            ));
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::constants::REGISTER_COUNT;
    use tracer::instruction::{
        and::AND,
        format::format_r::{FormatR, RegisterStateFormatR},
        RISCVCycle,
    };

    fn and_cycle(address: u64, rd: u8) -> Cycle {
        RISCVCycle::<AND> {
            instruction: AND {
                address,
                operands: FormatR { rd, rs1: 1, rs2: 2 },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            register_state: RegisterStateFormatR {
                rd: (0, 0x0a),
                rs1: 0xaa,
                rs2: 0x0f,
            },
            ram_access: (),
        }
        .into()
    }

    fn fixture() -> (DirectChunkedPreprocessing, TraceBlock, Cycle) {
        let cycle = and_cycle(0x8000_0000, 3);
        let lookahead = and_cycle(0x8000_0004, 4);
        let preprocessing = DirectChunkedPreprocessing::from_trace_cycles(
            b"d14-unified-block-jolt",
            8,
            &[cycle.clone(), lookahead.clone()],
        )
        .unwrap();
        let mut start_registers = [0i64; REGISTER_COUNT as usize];
        start_registers[1] = 0xaa;
        start_registers[2] = 0x0f;
        let mut end_registers = start_registers;
        end_registers[3] = 0x0a;
        let block = TraceBlock {
            block_index: 0,
            global_cycle_start: 0,
            active_cycles: 1,
            target_size: 2,
            start_state: MachineBoundaryState {
                global_cycle: 0,
                emulator_trace_len: 0,
                pc: 0x8000_0000,
                registers: start_registers,
                terminated: false,
            },
            end_state: MachineBoundaryState {
                global_cycle: 1,
                emulator_trace_len: 1,
                pc: 0x8000_0004,
                registers: end_registers,
                terminated: false,
            },
            cycles: vec![cycle],
            ended_at_tick_boundary: true,
        };
        (preprocessing, block, lookahead)
    }

    fn config() -> BlockJoltHostConfig {
        BlockJoltHostConfig {
            cycle_capacity: 2,
            ram_k: 2,
        }
    }

    fn two_block_fixture() -> (
        DirectChunkedPreprocessing,
        TraceBlock,
        Cycle,
        TraceBlock,
        Cycle,
    ) {
        let cycle0 = and_cycle(0x8000_0000, 3);
        let cycle1 = and_cycle(0x8000_0004, 4);
        let cycle2 = and_cycle(0x8000_0008, 5);
        let preprocessing = DirectChunkedPreprocessing::from_trace_cycles(
            b"d14-two-block-stream",
            8,
            &[cycle0.clone(), cycle1.clone(), cycle2.clone()],
        )
        .unwrap();
        let mut registers0 = [0i64; REGISTER_COUNT as usize];
        registers0[1] = 0xaa;
        registers0[2] = 0x0f;
        let mut registers1 = registers0;
        registers1[3] = 0x0a;
        let mut registers2 = registers1;
        registers2[4] = 0x0a;
        let block0 = TraceBlock {
            block_index: 0,
            global_cycle_start: 0,
            active_cycles: 1,
            target_size: 2,
            start_state: MachineBoundaryState {
                global_cycle: 0,
                emulator_trace_len: 0,
                pc: 0x8000_0000,
                registers: registers0,
                terminated: false,
            },
            end_state: MachineBoundaryState {
                global_cycle: 1,
                emulator_trace_len: 1,
                pc: 0x8000_0004,
                registers: registers1,
                terminated: false,
            },
            cycles: vec![cycle0],
            ended_at_tick_boundary: true,
        };
        let block1 = TraceBlock {
            block_index: 1,
            global_cycle_start: 1,
            active_cycles: 1,
            target_size: 2,
            start_state: block0.end_state.clone(),
            end_state: MachineBoundaryState {
                global_cycle: 2,
                emulator_trace_len: 2,
                pc: 0x8000_0008,
                registers: registers2,
                terminated: false,
            },
            cycles: vec![cycle1.clone()],
            ended_at_tick_boundary: true,
        };
        (preprocessing, block0, cycle1, block1, cycle2)
    }

    #[test]
    fn d14_unified_block_prover_and_host_verifier_round_trip() {
        let (preprocessing, block, lookahead) = fixture();
        let mut prover =
            BlockJoltProver::new(preprocessing.clone(), config(), BTreeMap::new()).unwrap();
        let transition = prover.prove_block(&block, Some(&lookahead)).unwrap();
        let mut verifier = BlockJoltVerifier::new(preprocessing, config()).unwrap();
        let verified = verifier
            .verify_block(&transition.statement, &transition.proof)
            .unwrap();
        assert_eq!(verified, transition.state_after);
        assert_eq!(verified.next_block_index, 1);
        assert_eq!(verified.total_active_cycles, 1);
        assert!(!verified.terminated);
        assert!(prover.finish().is_err());
        assert!(verifier.finish().is_err());
    }

    #[test]
    fn d14_two_block_stream_preserves_every_recursive_boundary() {
        let (preprocessing, block0, lookahead0, block1, lookahead1) = two_block_fixture();
        let mut prover =
            BlockJoltProver::new(preprocessing.clone(), config(), BTreeMap::new()).unwrap();
        let transition0 = prover.prove_block(&block0, Some(&lookahead0)).unwrap();
        let transition1 = prover.prove_block(&block1, Some(&lookahead1)).unwrap();
        assert_eq!(
            transition0.statement.end, transition1.statement.start,
            "machine/register/RAM boundaries must be contiguous"
        );
        assert_eq!(
            transition0.statement.lookup_accumulator_after,
            transition1.statement.lookup_accumulator_before
        );
        assert_eq!(
            transition0.statement.transcript_after,
            transition1.statement.transcript_before
        );

        let mut verifier = BlockJoltVerifier::new(preprocessing, config()).unwrap();
        verifier
            .verify_block(&transition0.statement, &transition0.proof)
            .unwrap();
        let final_state = verifier
            .verify_block(&transition1.statement, &transition1.proof)
            .unwrap();
        assert_eq!(final_state, transition1.state_after);
        assert_eq!(final_state.next_block_index, 2);
        assert_eq!(final_state.next_global_cycle, 2);
        assert_eq!(final_state.total_active_cycles, 2);
    }

    #[test]
    fn d14_rejects_relation_deferred_transcript_and_replay_attacks() {
        let (preprocessing, block, lookahead) = fixture();
        let mut prover =
            BlockJoltProver::new(preprocessing.clone(), config(), BTreeMap::new()).unwrap();
        let transition = prover.prove_block(&block, Some(&lookahead)).unwrap();

        let mut bad_relation = transition.proof.clone();
        bad_relation.lookup.sumcheck.compressed_proof[0] ^= 1;
        bad_relation.seal();
        let mut verifier = BlockJoltVerifier::new(preprocessing.clone(), config()).unwrap();
        assert!(verifier
            .verify_block(&transition.statement, &bad_relation)
            .is_err());
        // A rejected proof must not poison the reusable verifier state.
        verifier
            .verify_block(&transition.statement, &transition.proof)
            .unwrap();

        let mut bad_deferred = transition.proof.clone();
        bad_deferred.deferred_pcs_claims[0].claimed_value.0[0] ^= 1;
        bad_deferred.seal();
        let mut verifier = BlockJoltVerifier::new(preprocessing.clone(), config()).unwrap();
        assert!(verifier
            .verify_block(&transition.statement, &bad_deferred)
            .is_err());

        let mut bad_transcript_statement = transition.statement.clone();
        bad_transcript_statement.transcript_after.state[0] ^= 1;
        let mut rebound = transition.proof.clone();
        rebound.statement_digest = bad_transcript_statement.digest();
        rebound.transcript_after = bad_transcript_statement.transcript_after;
        rebound.seal();
        let mut verifier = BlockJoltVerifier::new(preprocessing.clone(), config()).unwrap();
        assert!(verifier
            .verify_block(&bad_transcript_statement, &rebound)
            .is_err());

        let mut bad_terminal_statement = transition.statement.clone();
        bad_terminal_statement.terminal = true;
        let mut rebound = transition.proof.clone();
        rebound.statement_digest = bad_terminal_statement.digest();
        rebound.seal();
        let mut verifier = BlockJoltVerifier::new(preprocessing.clone(), config()).unwrap();
        assert!(verifier
            .verify_block(&bad_terminal_statement, &rebound)
            .is_err());

        let mut verifier = BlockJoltVerifier::new(preprocessing.clone(), config()).unwrap();
        verifier
            .verify_block(&transition.statement, &transition.proof)
            .unwrap();
        assert!(verifier
            .verify_block(&transition.statement, &transition.proof)
            .is_err());

        let different = DirectChunkedPreprocessing::from_trace_cycles(
            b"different-program",
            8,
            &[block.cycles[0].clone(), lookahead],
        )
        .unwrap();
        let mut verifier = BlockJoltVerifier::new(different, config()).unwrap();
        assert!(verifier
            .verify_block(&transition.statement, &transition.proof)
            .is_err());
    }

    #[test]
    fn d14_production_proof_types_exclude_row_level_witnesses() {
        let types_source = include_str!("types.rs");
        for forbidden in [
            "TraceBlock",
            "DirectLookupBlockWitness",
            "DirectRegisterBlockWitness",
            "DirectRamBlockWitness",
            "DirectCpuBlockWitness",
            "MerklePath",
        ] {
            assert!(
                !types_source.contains(forbidden),
                "production proof types must not contain {forbidden}"
            );
        }
    }
}
