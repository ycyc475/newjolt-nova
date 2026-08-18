//! D12 compact per-block register proof.

use tracer::TraceBlock;

use crate::transcripts::{PoseidonTranscript, Transcript};

use super::super::{direct_register, DirectChunkedError};
use super::{RegisterBlockProof, TranscriptCheckpoint};

const BLOCK_REGISTER_TRANSCRIPT_DOMAIN: &[u8] = b"direct-register-rw-v1";

pub fn new_block_register_transcript() -> PoseidonTranscript {
    PoseidonTranscript::new(BLOCK_REGISTER_TRANSCRIPT_DOMAIN)
}

pub fn prove_block_register(
    block: &TraceBlock,
    capacity: usize,
    transcript: &mut PoseidonTranscript,
) -> Result<
    (
        RegisterBlockProof,
        TranscriptCheckpoint,
        TranscriptCheckpoint,
    ),
    DirectChunkedError,
> {
    direct_register::prove_compact_register_block(block, capacity, transcript)
}

pub(super) fn prove_block_register_with_commitment(
    block: &TraceBlock,
    capacity: usize,
    transcript: &mut PoseidonTranscript,
    commitment_id: [u8; 32],
) -> Result<
    (
        RegisterBlockProof,
        TranscriptCheckpoint,
        TranscriptCheckpoint,
    ),
    DirectChunkedError,
> {
    direct_register::prove_compact_register_block_with_commitment(
        block,
        capacity,
        transcript,
        Some(commitment_id),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_block_register(
    proof: &RegisterBlockProof,
    block_index: usize,
    global_cycle_start: usize,
    active_cycles: usize,
    capacity: usize,
    expected_state_before: [u8; 32],
    expected_state_after: [u8; 32],
    transcript: &mut PoseidonTranscript,
) -> Result<(TranscriptCheckpoint, TranscriptCheckpoint), DirectChunkedError> {
    direct_register::verify_compact_register_block(
        proof,
        block_index,
        global_cycle_start,
        active_cycles,
        capacity,
        expected_state_before,
        expected_state_after,
        transcript,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::constants::REGISTER_COUNT;
    use tracer::{
        instruction::{
            and::AND,
            format::format_r::{FormatR, RegisterStateFormatR},
            RISCVCycle,
        },
        MachineBoundaryState,
    };

    fn block() -> TraceBlock {
        let mut start_registers = [0i64; REGISTER_COUNT as usize];
        start_registers[1] = 0xaa;
        start_registers[2] = 0x0f;
        let mut end_registers = start_registers;
        end_registers[3] = 0x0a;
        let cycle = RISCVCycle::<AND> {
            instruction: AND {
                address: 0x8000_0000,
                operands: FormatR {
                    rd: 3,
                    rs1: 1,
                    rs2: 2,
                },
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
        .into();
        TraceBlock {
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
                terminated: true,
            },
            cycles: vec![cycle],
            ended_at_tick_boundary: true,
        }
    }

    #[test]
    fn d12_compact_register_sumcheck_round_trip() {
        let mut prover_transcript = new_block_register_transcript();
        let (proof, before, after) =
            prove_block_register(&block(), 2, &mut prover_transcript).unwrap();
        let mut verifier_transcript = new_block_register_transcript();
        let verified = verify_block_register(
            &proof,
            0,
            0,
            1,
            2,
            proof.state_before,
            proof.state_after,
            &mut verifier_transcript,
        )
        .unwrap();
        assert_eq!(verified, (before, after));
        assert!(!proof.sumcheck.compressed_proof.is_empty());
    }

    #[test]
    fn d12_compact_register_rejects_boundary_and_sumcheck_tampering() {
        let mut prover_transcript = new_block_register_transcript();
        let (proof, _, _) = prove_block_register(&block(), 2, &mut prover_transcript).unwrap();
        let expected_before = proof.state_before;
        let expected_after = proof.state_after;

        let mut bad_boundary = proof.clone();
        bad_boundary.state_after[0] ^= 1;
        let mut verifier_transcript = new_block_register_transcript();
        assert!(verify_block_register(
            &bad_boundary,
            0,
            0,
            1,
            2,
            expected_before,
            expected_after,
            &mut verifier_transcript,
        )
        .is_err());

        let mut bad_sumcheck = proof;
        bad_sumcheck.sumcheck.compressed_proof[0] ^= 1;
        let mut verifier_transcript = new_block_register_transcript();
        assert!(verify_block_register(
            &bad_sumcheck,
            0,
            0,
            1,
            2,
            expected_before,
            expected_after,
            &mut verifier_transcript,
        )
        .is_err());
    }
}
