//! D11 compact per-block Lasso proof.

use tracer::TraceBlock;

use crate::transcripts::{PoseidonTranscript, Transcript};

use super::super::{direct_lookup, DirectChunkedError};
use super::{LookupBlockProof, TranscriptCheckpoint};

const BLOCK_LOOKUP_TRANSCRIPT_DOMAIN: &[u8] = b"direct-lasso-v1";

pub fn new_block_lookup_transcript() -> PoseidonTranscript {
    PoseidonTranscript::new(BLOCK_LOOKUP_TRANSCRIPT_DOMAIN)
}

pub fn prove_block_lookup_lasso(
    block: &TraceBlock,
    capacity: usize,
    transcript: &mut PoseidonTranscript,
) -> Result<(LookupBlockProof, TranscriptCheckpoint, TranscriptCheckpoint), DirectChunkedError> {
    direct_lookup::prove_compact_lookup_block(block, capacity, transcript)
}

pub(super) fn prove_block_lookup_lasso_with_commitment(
    block: &TraceBlock,
    capacity: usize,
    transcript: &mut PoseidonTranscript,
    commitment_id: [u8; 32],
) -> Result<(LookupBlockProof, TranscriptCheckpoint, TranscriptCheckpoint), DirectChunkedError> {
    direct_lookup::prove_compact_lookup_block_with_commitment(
        block,
        capacity,
        transcript,
        Some(commitment_id),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_block_lookup_lasso(
    proof: &LookupBlockProof,
    block_index: usize,
    global_cycle_start: usize,
    active_cycles: usize,
    capacity: usize,
    transcript: &mut PoseidonTranscript,
) -> Result<(TranscriptCheckpoint, TranscriptCheckpoint), DirectChunkedError> {
    direct_lookup::verify_compact_lookup_block(
        proof,
        block_index,
        global_cycle_start,
        active_cycles,
        capacity,
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

    fn boundary(cycle: usize, terminated: bool) -> MachineBoundaryState {
        MachineBoundaryState {
            global_cycle: cycle,
            emulator_trace_len: cycle,
            pc: 0x8000_0000 + cycle as u64 * 4,
            registers: [0; REGISTER_COUNT as usize],
            terminated,
        }
    }

    fn block() -> TraceBlock {
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
            start_state: boundary(0, false),
            end_state: boundary(1, true),
            cycles: vec![cycle],
            ended_at_tick_boundary: true,
        }
    }

    #[test]
    fn d11_compact_lasso_proof_round_trip_without_trace_rows() {
        let mut prover_transcript = new_block_lookup_transcript();
        let (proof, before, after) =
            prove_block_lookup_lasso(&block(), 2, &mut prover_transcript).unwrap();
        assert_eq!(
            proof.sumcheck.rounds as usize,
            proof.sumcheck.challenges.len()
        );
        assert!(!proof.sumcheck.compressed_proof.is_empty());
        assert!(!proof.sumcheck.opening_claims.is_empty());

        let encoded = postcard::to_stdvec(&proof).unwrap();
        let decoded: LookupBlockProof = postcard::from_bytes(&encoded).unwrap();
        let mut verifier_transcript = new_block_lookup_transcript();
        let verified =
            verify_block_lookup_lasso(&decoded, 0, 0, 1, 2, &mut verifier_transcript).unwrap();
        assert_eq!(verified, (before, after));
    }

    #[test]
    fn d11_compact_lasso_rejects_transcript_and_opening_tampering() {
        let mut prover_transcript = new_block_lookup_transcript();
        let (proof, _, _) = prove_block_lookup_lasso(&block(), 2, &mut prover_transcript).unwrap();

        let mut bad_query = proof.clone();
        bad_query.query_commitment.0[0] ^= 1;
        let mut verifier_transcript = new_block_lookup_transcript();
        assert!(
            verify_block_lookup_lasso(&bad_query, 0, 0, 1, 2, &mut verifier_transcript).is_err()
        );

        let mut bad_opening = proof;
        bad_opening.output_claims[0].0[0] ^= 1;
        let mut verifier_transcript = new_block_lookup_transcript();
        assert!(
            verify_block_lookup_lasso(&bad_opening, 0, 0, 1, 2, &mut verifier_transcript).is_err()
        );
    }
}
