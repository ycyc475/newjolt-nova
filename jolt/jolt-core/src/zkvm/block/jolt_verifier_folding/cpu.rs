//! D13 compact per-block CPU/R1CS proof.
//!
//! This module runs Jolt's native Spartan outer relation directly over one
//! fixed-capacity trace block.  The returned object contains the original
//! univariate-skip and remaining Sumcheck messages plus deferred polynomial
//! openings; it never contains `Cycle`, `TraceBlock`, or a row-level witness.

use std::{io::Cursor, sync::Arc};

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::{Digest, Sha3_256};
use tracer::{instruction::Cycle, TraceBlock};

use crate::{
    field::JoltField,
    poly::opening_proof::{
        OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
        VerifierOpeningAccumulator, BIG_ENDIAN,
    },
    subprotocols::{
        streaming_schedule::LinearOnlySchedule,
        sumcheck::{BatchedSumcheck, ClearSumcheckProof},
        univariate_skip::{prove_uniskip_round, UniSkipFirstRoundProof},
    },
    transcripts::{PoseidonTranscript, Transcript},
    utils::math::Math,
    zkvm::{
        r1cs::{
            constraints::{OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE},
            inputs::{R1CSCycleInputs, ALL_R1CS_INPUTS, NUM_R1CS_INPUTS},
            key::UniformSpartanKey,
        },
        spartan::outer::{
            OuterRemainingStreamingSumcheck, OuterRemainingSumcheckVerifier, OuterSharedState,
            OuterUniSkipParams, OuterUniSkipProver, OuterUniSkipVerifier,
        },
        witness::VirtualPolynomial,
    },
};

use super::super::{
    direct::validate_block, direct_cpu::bytecode_tree, DirectChunkedError,
    DirectChunkedPreprocessing,
};
use super::{
    BlockRelation, CompactSumcheckProof, CpuBlockRelationProof, DeferredPcsClaim, FieldElement,
    TranscriptCheckpoint, BLOCK_JOLT_PROTOCOL_VERSION,
};

const BLOCK_CPU_TRANSCRIPT_DOMAIN: &[u8] = b"block-jolt-cpu-r1cs-v2";
const BLOCK_CPU_ROW_DOMAIN: &[u8] = b"block-jolt-cpu-row-commitment-v2";
const OUTER_REMAINING_DEGREE_BOUND: usize = 3;

pub fn new_block_cpu_transcript() -> PoseidonTranscript {
    PoseidonTranscript::new(BLOCK_CPU_TRANSCRIPT_DOMAIN)
}

fn padded_trace(block: &TraceBlock, capacity: usize) -> Arc<Vec<Cycle>> {
    let mut trace = block.cycles.clone();
    trace.resize(capacity, Cycle::NoOp);
    Arc::new(trace)
}

fn row_inputs(
    preprocessing: &DirectChunkedPreprocessing,
    trace: &[Cycle],
    active_cycles: usize,
    lookahead: Option<&Cycle>,
    row: usize,
) -> R1CSCycleInputs {
    let next = if lookahead.is_some() && row + 1 == active_cycles {
        lookahead
    } else {
        trace.get(row + 1)
    };
    R1CSCycleInputs::from_cycle_with_next::<Fr>(&preprocessing.bytecode, &trace[row], next)
}

fn row_commitment(
    preprocessing: &DirectChunkedPreprocessing,
    block: &TraceBlock,
    trace: &[Cycle],
    lookahead: Option<&Cycle>,
) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(BLOCK_CPU_ROW_DOMAIN);
    hasher.update((block.block_index as u64).to_le_bytes());
    hasher.update((block.global_cycle_start as u64).to_le_bytes());
    hasher.update((block.active_cycles as u64).to_le_bytes());
    hasher.update((trace.len() as u64).to_le_bytes());
    for row in 0..trace.len() {
        let inputs = row_inputs(preprocessing, trace, block.active_cycles, lookahead, row);
        for input in ALL_R1CS_INPUTS {
            hasher.update(inputs.get_input_value(input).to_le_bytes());
        }
    }
    hasher.finalize().into()
}

fn append_header(
    transcript: &mut PoseidonTranscript,
    block_index: usize,
    global_cycle_start: usize,
    active_cycles: usize,
    capacity: usize,
    start_pc: u64,
    end_pc: u64,
    terminal: bool,
    row_commitment: &[u8; 32],
    bytecode_root: FieldElement,
) {
    transcript.append_bytes(b"protocol", BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    for (label, value) in [
        (b"block_index".as_slice(), block_index as u64),
        (b"cycle_start".as_slice(), global_cycle_start as u64),
        (b"active_cycles".as_slice(), active_cycles as u64),
        (b"cycle_capacity".as_slice(), capacity as u64),
        (b"start_pc".as_slice(), start_pc),
        (b"end_pc".as_slice(), end_pc),
        (b"terminal".as_slice(), u64::from(terminal)),
    ] {
        transcript.append_u64(label, value);
    }
    transcript.append_bytes(b"cpu_row_commitment", row_commitment);
    transcript.append_scalar(b"bytecode_root", &bytecode_root.to_fr());
}

fn compact_cpu_id(index: usize) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(b"cpu-r1cs-input");
    hasher.update((index as u64).to_le_bytes());
    hasher.finalize().into()
}

pub(super) fn cpu_deferred_claims(
    row_commitment: [u8; 32],
    openings: &[(Vec<FieldElement>, FieldElement)],
) -> Vec<DeferredPcsClaim> {
    openings
        .iter()
        .enumerate()
        .map(|(index, (opening_point, claimed_value))| DeferredPcsClaim {
            relation: BlockRelation::CpuR1cs,
            polynomial_id: compact_cpu_id(index),
            commitment_id: row_commitment,
            opening_point: opening_point.clone(),
            claimed_value: *claimed_value,
        })
        .collect()
}

fn serialize_proof<T: CanonicalSerialize>(
    proof: &T,
    label: &str,
) -> Result<Vec<u8>, DirectChunkedError> {
    let mut encoded = Vec::new();
    proof.serialize_compressed(&mut encoded).map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!(
            "compact CPU {label} serialization failed: {error}"
        ))
    })?;
    Ok(encoded)
}

fn deserialize_exact<T: CanonicalDeserialize>(
    encoded: &[u8],
    label: &str,
) -> Result<T, DirectChunkedError> {
    let mut cursor = Cursor::new(encoded);
    let proof = T::deserialize_compressed(&mut cursor).map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!(
            "compact CPU {label} deserialization failed: {error}"
        ))
    })?;
    if cursor.position() != encoded.len() as u64 {
        return Err(DirectChunkedError::InvalidProofShape(format!(
            "compact CPU {label} has trailing bytes"
        )));
    }
    Ok(proof)
}

fn challenge_to_field(challenge: <Fr as JoltField>::Challenge) -> FieldElement {
    let value: Fr = challenge.into();
    FieldElement::from_fr(&value)
}

fn point_to_fields(point: &OpeningPoint<BIG_ENDIAN, Fr>) -> Vec<FieldElement> {
    point.r.iter().copied().map(challenge_to_field).collect()
}

pub(super) fn bytecode_root(
    preprocessing: &DirectChunkedPreprocessing,
) -> Result<FieldElement, DirectChunkedError> {
    if preprocessing.bytecode.bytecode.is_empty() {
        return Err(DirectChunkedError::InvalidConfiguration(
            "CPU/R1CS block proof requires materialized bytecode preprocessing".to_string(),
        ));
    }
    Ok(FieldElement::from_fr(
        bytecode_tree(preprocessing)
            .last()
            .and_then(|level| level.first())
            .expect("non-empty bytecode tree"),
    ))
}

/// Generates the original Jolt Spartan outer proof for one block.  `lookahead`
/// is the first cycle of the next block for non-terminal blocks.
pub fn prove_block_cpu_r1cs(
    preprocessing: &DirectChunkedPreprocessing,
    block: &TraceBlock,
    capacity: usize,
    lookahead: Option<&Cycle>,
    transcript: &mut PoseidonTranscript,
) -> Result<
    (
        CpuBlockRelationProof,
        TranscriptCheckpoint,
        TranscriptCheckpoint,
    ),
    DirectChunkedError,
> {
    validate_block(block, capacity)?;
    if !block.end_state.terminated && lookahead.is_none() {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: block.block_index,
            reason: "non-terminal CPU block is missing its next-block lookahead row".to_string(),
        });
    }
    if block.end_state.terminated && lookahead.is_some() {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: block.block_index,
            reason: "terminal CPU block must not consume a next-block lookahead row".to_string(),
        });
    }
    let before = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    let trace = padded_trace(block, capacity);
    let rows = row_commitment(preprocessing, block, &trace, lookahead);
    let bytecode_root = bytecode_root(preprocessing)?;
    let start_pc = block.start_state.pc;
    let end_pc = block.end_state.pc;
    append_header(
        transcript,
        block.block_index,
        block.global_cycle_start,
        block.active_cycles,
        capacity,
        start_pc,
        end_pc,
        block.end_state.terminated,
        &rows,
        bytecode_root,
    );

    let key = UniformSpartanKey::<Fr>::new(capacity);
    let log_t = capacity.log_2();
    let mut accumulator = ProverOpeningAccumulator::<Fr>::new(log_t);
    let uni_params = OuterUniSkipParams::new(&key, transcript);
    let mut uni_prover = OuterUniSkipProver::initialize_block(
        uni_params.clone(),
        &trace,
        &preprocessing.bytecode,
        block.active_cycles,
        lookahead,
    );
    let uni_proof = prove_uniskip_round(&mut uni_prover, &mut accumulator, transcript);
    let (uni_point, uni_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::UnivariateSkip,
        SumcheckId::SpartanOuter,
    );
    if uni_point.len() != 1 {
        return Err(DirectChunkedError::InvalidProofShape(
            "Spartan outer uniskip produced an invalid opening point".to_string(),
        ));
    }
    let uni_challenge = uni_point.r[0];

    let shared = OuterSharedState::new_block(
        Arc::clone(&trace),
        &preprocessing.bytecode,
        &uni_params,
        &accumulator,
        block.active_cycles,
        lookahead,
    );
    let schedule = LinearOnlySchedule::new(uni_params.tau.len() - 1);
    let mut outer: OuterRemainingStreamingSumcheck<Fr, _> =
        OuterRemainingStreamingSumcheck::new(shared, schedule);
    let transcript_before_sumcheck = transcript.clone();
    let (clear, challenges, initial_claim) =
        BatchedSumcheck::prove(vec![&mut outer], &mut accumulator, transcript);

    let mut batching_transcript = transcript_before_sumcheck;
    batching_transcript.append_scalar(b"sumcheck_claim", &uni_claim);
    let batching_coefficient = batching_transcript.challenge_scalar::<Fr>();
    if initial_claim != uni_claim * batching_coefficient {
        return Err(DirectChunkedError::InvalidProofShape(
            "Spartan outer batched initial claim is inconsistent".to_string(),
        ));
    }
    let challenge_fields = challenges
        .iter()
        .copied()
        .map(challenge_to_field)
        .collect::<Vec<_>>();
    let final_claim = clear
        .compressed_polys
        .iter()
        .zip(&challenges)
        .fold(initial_claim, |claim, (poly, challenge)| {
            poly.eval_from_hint(&claim, challenge)
        });

    let mut endpoint_openings = Vec::with_capacity(NUM_R1CS_INPUTS);
    for input in ALL_R1CS_INPUTS {
        let (point, claim) =
            accumulator.get_virtual_polynomial_opening((&input).into(), SumcheckId::SpartanOuter);
        endpoint_openings.push((point_to_fields(&point), FieldElement::from_fr(&claim)));
    }
    let deferred = cpu_deferred_claims(rows, &endpoint_openings);
    let after = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    Ok((
        CpuBlockRelationProof {
            row_commitment: rows,
            bytecode_root,
            start_pc,
            end_pc,
            terminal: block.end_state.terminated,
            uniskip_proof: serialize_proof(&uni_proof, "uniskip proof")?,
            uniskip_challenge: challenge_to_field(uni_challenge),
            uniskip_claim: FieldElement::from_fr(&uni_claim),
            batching_coefficient: FieldElement::from_fr(&batching_coefficient),
            relation_proof: CompactSumcheckProof {
                rounds: challenges.len() as u32,
                degree_bound: OUTER_REMAINING_DEGREE_BOUND as u32,
                initial_claim: FieldElement::from_fr(&initial_claim),
                final_claim: FieldElement::from_fr(&final_claim),
                compressed_proof: serialize_proof(&clear, "remaining sumcheck")?,
                challenges: challenge_fields,
                opening_claims: deferred,
            },
        },
        before,
        after,
    ))
}

fn seed_verifier_openings(
    accumulator: &mut VerifierOpeningAccumulator<Fr>,
    uni_claim: Fr,
    input_claims: &[Fr],
) {
    let empty = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
    accumulator.openings.insert(
        OpeningId::virt(VirtualPolynomial::UnivariateSkip, SumcheckId::SpartanOuter),
        (empty.clone(), uni_claim),
    );
    for (input, claim) in ALL_R1CS_INPUTS.iter().zip(input_claims) {
        accumulator.openings.insert(
            OpeningId::virt(VirtualPolynomial::from(input), SumcheckId::SpartanOuter),
            (empty.clone(), *claim),
        );
    }
}

/// Verifies a compact CPU/R1CS block proof without receiving any execution
/// rows.  Polynomial commitment openings are returned as authenticated,
/// explicitly deferred endpoints for the D17 PCS closure.
#[allow(clippy::too_many_arguments)]
pub fn verify_block_cpu_r1cs(
    proof: &CpuBlockRelationProof,
    block_index: usize,
    global_cycle_start: usize,
    active_cycles: usize,
    capacity: usize,
    expected_start_pc: u64,
    expected_end_pc: u64,
    expected_terminal: bool,
    expected_bytecode_root: FieldElement,
    transcript: &mut PoseidonTranscript,
) -> Result<(TranscriptCheckpoint, TranscriptCheckpoint), DirectChunkedError> {
    if capacity == 0
        || !capacity.is_power_of_two()
        || active_cycles == 0
        || active_cycles > capacity
        || proof.start_pc != expected_start_pc
        || proof.end_pc != expected_end_pc
        || proof.terminal != expected_terminal
        || proof.bytecode_root != expected_bytecode_root
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact CPU block shape or public boundary mismatch".to_string(),
        ));
    }
    if proof.relation_proof.rounds as usize != capacity.log_2() + 1
        || proof.relation_proof.degree_bound as usize != OUTER_REMAINING_DEGREE_BOUND
        || proof.relation_proof.challenges.len() != capacity.log_2() + 1
        || proof.relation_proof.opening_claims.len() != NUM_R1CS_INPUTS
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact CPU Spartan outer proof shape mismatch".to_string(),
        ));
    }

    let before = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    append_header(
        transcript,
        block_index,
        global_cycle_start,
        active_cycles,
        capacity,
        proof.start_pc,
        proof.end_pc,
        proof.terminal,
        &proof.row_commitment,
        proof.bytecode_root,
    );

    let input_claims = proof
        .relation_proof
        .opening_claims
        .iter()
        .map(|claim| claim.claimed_value.to_fr())
        .collect::<Vec<_>>();
    let mut accumulator = VerifierOpeningAccumulator::<Fr>::new(capacity.log_2(), false);
    seed_verifier_openings(&mut accumulator, proof.uniskip_claim.to_fr(), &input_claims);

    let key = UniformSpartanKey::<Fr>::new(capacity);
    let uni_proof: UniSkipFirstRoundProof<Fr, PoseidonTranscript> =
        deserialize_exact(&proof.uniskip_proof, "uniskip proof")?;
    let uni_verifier = OuterUniSkipVerifier::new(&key, transcript);
    let uni_challenge = UniSkipFirstRoundProof::verify::<
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        _,
    >(&uni_proof, &uni_verifier, &mut accumulator, transcript)
    .map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!(
            "compact CPU uniskip verification failed: {error}"
        ))
    })?;
    if challenge_to_field(uni_challenge) != proof.uniskip_challenge {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact CPU uniskip challenge mismatch".to_string(),
        ));
    }

    let outer =
        OuterRemainingSumcheckVerifier::new(key, capacity, &uni_verifier.params, &accumulator);
    let mut batching_transcript = transcript.clone();
    batching_transcript.append_scalar(b"sumcheck_claim", &proof.uniskip_claim.to_fr());
    let expected_batching = batching_transcript.challenge_scalar::<Fr>();
    let expected_initial = proof.uniskip_claim.to_fr() * expected_batching;
    if proof.batching_coefficient.to_fr() != expected_batching
        || proof.relation_proof.initial_claim.to_fr() != expected_initial
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact CPU batched initial claim mismatch".to_string(),
        ));
    }

    let clear: ClearSumcheckProof<Fr, PoseidonTranscript> =
        deserialize_exact(&proof.relation_proof.compressed_proof, "remaining sumcheck")?;
    let challenges =
        BatchedSumcheck::verify_standard(&clear, vec![&outer], &mut accumulator, transcript)
            .map_err(|error| {
                DirectChunkedError::InvalidProofShape(format!(
                    "compact CPU Spartan outer verification failed: {error}"
                ))
            })?;
    let supplied_challenges = proof
        .relation_proof
        .challenges
        .iter()
        .map(|value| value.to_fr())
        .collect::<Vec<_>>();
    let challenge_values = challenges
        .iter()
        .copied()
        .map(Into::<Fr>::into)
        .collect::<Vec<_>>();
    let final_claim = clear
        .compressed_polys
        .iter()
        .zip(&challenges)
        .fold(expected_initial, |claim, (poly, challenge)| {
            poly.eval_from_hint(&claim, challenge)
        });

    let mut endpoint_openings = Vec::with_capacity(NUM_R1CS_INPUTS);
    for input in ALL_R1CS_INPUTS {
        let (point, claim) =
            accumulator.get_virtual_polynomial_opening((&input).into(), SumcheckId::SpartanOuter);
        endpoint_openings.push((point_to_fields(&point), FieldElement::from_fr(&claim)));
    }
    let expected_deferred = cpu_deferred_claims(proof.row_commitment, &endpoint_openings);
    let after = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    if challenge_values != supplied_challenges
        || final_claim != proof.relation_proof.final_claim.to_fr()
        || proof.relation_proof.opening_claims != expected_deferred
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact CPU Fiat-Shamir or deferred-opening mismatch".to_string(),
        ));
    }
    Ok((before, after))
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
        let cycle: Cycle = RISCVCycle::<AND> {
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
        let mut start_registers = [0i64; REGISTER_COUNT as usize];
        start_registers[1] = 0xaa;
        start_registers[2] = 0x0f;
        let mut end_registers = start_registers;
        end_registers[3] = 0x0a;
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
                terminated: false,
            },
            cycles: vec![cycle],
            ended_at_tick_boundary: true,
        }
    }

    fn lookahead() -> Cycle {
        RISCVCycle::<AND> {
            instruction: AND {
                address: 0x8000_0004,
                operands: FormatR {
                    rd: 4,
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
        .into()
    }

    fn preprocessing(block: &TraceBlock, lookahead: &Cycle) -> DirectChunkedPreprocessing {
        let mut cycles = block.cycles.clone();
        cycles.push(lookahead.clone());
        DirectChunkedPreprocessing::from_trace_cycles(b"d13-cpu-r1cs", 8, &cycles).unwrap()
    }

    #[test]
    fn d13_compact_cpu_spartan_outer_round_trip_without_rows() {
        let block = block();
        let lookahead = lookahead();
        let preprocessing = preprocessing(&block, &lookahead);
        let expected_root = bytecode_root(&preprocessing).unwrap();
        let mut prover_transcript = new_block_cpu_transcript();
        let (proof, before, after) = prove_block_cpu_r1cs(
            &preprocessing,
            &block,
            2,
            Some(&lookahead),
            &mut prover_transcript,
        )
        .unwrap();
        assert_eq!(proof.relation_proof.opening_claims.len(), NUM_R1CS_INPUTS);
        assert!(!proof.uniskip_proof.is_empty());
        assert!(!proof.relation_proof.compressed_proof.is_empty());

        let encoded = postcard::to_stdvec(&proof).unwrap();
        let decoded: CpuBlockRelationProof = postcard::from_bytes(&encoded).unwrap();
        let mut verifier_transcript = new_block_cpu_transcript();
        let verified = verify_block_cpu_r1cs(
            &decoded,
            0,
            0,
            1,
            2,
            block.start_state.pc,
            block.end_state.pc,
            block.end_state.terminated,
            expected_root,
            &mut verifier_transcript,
        )
        .unwrap();
        assert_eq!(verified, (before, after));
    }

    #[test]
    fn d13_compact_cpu_rejects_uniskip_opening_and_boundary_tampering() {
        let block = block();
        let lookahead = lookahead();
        let preprocessing = preprocessing(&block, &lookahead);
        let expected_root = bytecode_root(&preprocessing).unwrap();
        let mut prover_transcript = new_block_cpu_transcript();
        let (proof, _, _) = prove_block_cpu_r1cs(
            &preprocessing,
            &block,
            2,
            Some(&lookahead),
            &mut prover_transcript,
        )
        .unwrap();

        let mut bad_uniskip = proof.clone();
        bad_uniskip.uniskip_proof[0] ^= 1;
        let mut transcript = new_block_cpu_transcript();
        assert!(verify_block_cpu_r1cs(
            &bad_uniskip,
            0,
            0,
            1,
            2,
            block.start_state.pc,
            block.end_state.pc,
            block.end_state.terminated,
            expected_root,
            &mut transcript,
        )
        .is_err());

        let mut bad_opening = proof.clone();
        bad_opening.relation_proof.opening_claims[0].claimed_value.0[0] ^= 1;
        let mut transcript = new_block_cpu_transcript();
        assert!(verify_block_cpu_r1cs(
            &bad_opening,
            0,
            0,
            1,
            2,
            block.start_state.pc,
            block.end_state.pc,
            block.end_state.terminated,
            expected_root,
            &mut transcript,
        )
        .is_err());

        let mut bad_boundary = proof;
        bad_boundary.end_pc += 4;
        let mut transcript = new_block_cpu_transcript();
        assert!(verify_block_cpu_r1cs(
            &bad_boundary,
            0,
            0,
            1,
            2,
            block.start_state.pc,
            block.end_state.pc,
            block.end_state.terminated,
            expected_root,
            &mut transcript,
        )
        .is_err());
    }
}
