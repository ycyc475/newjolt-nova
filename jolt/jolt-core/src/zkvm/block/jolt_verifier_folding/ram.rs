//! D12 compact per-block RAM read/write proof.

use std::collections::BTreeMap;

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use sha3::{Digest, Sha3_256};
use tracer::{instruction::RAMAccess, TraceBlock};

use crate::{
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{
            AbstractVerifierOpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator,
            SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::{BatchedSumcheck, ClearSumcheckProof},
    transcripts::{PoseidonTranscript, Transcript},
    utils::math::Math,
    zkvm::{
        config::{OneHotParams, ReadWriteConfig},
        ram::{
            read_write_checking::{
                RamReadWriteCheckingParams, RamReadWriteCheckingProver,
                RamReadWriteCheckingVerifier,
            },
            remap_address,
        },
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

use super::super::{direct::validate_block, DirectChunkedError, DirectChunkedPreprocessing};
use super::{
    BlockRelation, CompactSumcheckProof, DeferredPcsClaim, FieldElement, RamBlockProof,
    TranscriptCheckpoint, BLOCK_JOLT_PROTOCOL_VERSION,
};

const BLOCK_RAM_TRANSCRIPT_DOMAIN: &[u8] = b"block-jolt-ram-rw-v2";
const BLOCK_RAM_ACCESS_DOMAIN: &[u8] = b"block-jolt-ram-access-v2";
const BLOCK_RAM_STATE_DOMAIN: &[u8] = b"block-jolt-ram-state-v2";
const RAM_OUTPUT_COUNT: usize = 3;

pub fn new_block_ram_transcript() -> PoseidonTranscript {
    PoseidonTranscript::new(BLOCK_RAM_TRANSCRIPT_DOMAIN)
}

fn input_ids() -> [OpeningId; 2] {
    [
        OpeningId::virt(VirtualPolynomial::RamReadValue, SumcheckId::SpartanOuter),
        OpeningId::virt(VirtualPolynomial::RamWriteValue, SumcheckId::SpartanOuter),
    ]
}

fn output_ids() -> [OpeningId; RAM_OUTPUT_COUNT] {
    [
        OpeningId::virt(VirtualPolynomial::RamRa, SumcheckId::RamReadWriteChecking),
        OpeningId::virt(VirtualPolynomial::RamVal, SumcheckId::RamReadWriteChecking),
        OpeningId::committed(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        ),
    ]
}

fn seed_prover(accumulator: &mut ProverOpeningAccumulator<Fr>, point: &[Fr], claims: [Fr; 2]) {
    let point =
        OpeningPoint::<BIG_ENDIAN, Fr>::new(point.iter().copied().map(Into::into).collect());
    accumulator.append_virtual(
        VirtualPolynomial::RamReadValue,
        SumcheckId::SpartanOuter,
        point.clone(),
        claims[0],
    );
    accumulator.append_virtual(
        VirtualPolynomial::RamWriteValue,
        SumcheckId::SpartanOuter,
        point,
        claims[1],
    );
}

fn seed_verifier(
    accumulator: &mut VerifierOpeningAccumulator<Fr>,
    point: &[Fr],
    input_claims: [Fr; 2],
    output_claims: [Fr; RAM_OUTPUT_COUNT],
) {
    let empty = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
    for (id, claim) in output_ids().into_iter().zip(output_claims) {
        accumulator.openings.insert(id, (empty.clone(), claim));
    }
    for (id, claim) in input_ids().into_iter().zip(input_claims) {
        accumulator.openings.insert(id, (empty.clone(), claim));
    }
    let point =
        OpeningPoint::<BIG_ENDIAN, Fr>::new(point.iter().copied().map(Into::into).collect());
    accumulator.append_virtual(
        VirtualPolynomial::RamReadValue,
        SumcheckId::SpartanOuter,
        point.clone(),
    );
    accumulator.append_virtual(
        VirtualPolynomial::RamWriteValue,
        SumcheckId::SpartanOuter,
        point,
    );
}

fn padded_trace(block: &TraceBlock, capacity: usize) -> Vec<tracer::instruction::Cycle> {
    let mut trace = block.cycles.clone();
    trace.resize(capacity, tracer::instruction::Cycle::NoOp);
    trace
}

fn ram_input_claims(block: &TraceBlock, capacity: usize, point: &[Fr]) -> [Fr; 2] {
    let weights = EqPolynomial::<Fr>::evals(point);
    let mut claims = [Fr::zero(); 2];
    for (weight, cycle) in weights.into_iter().zip(padded_trace(block, capacity)) {
        match cycle.ram_access() {
            RAMAccess::Read(read) => {
                claims[0] += weight * Fr::from(read.value);
                claims[1] += weight * Fr::from(read.value);
            }
            RAMAccess::Write(write) => {
                claims[0] += weight * Fr::from(write.pre_value);
                claims[1] += weight * Fr::from(write.post_value);
            }
            RAMAccess::NoOp => {}
        }
    }
    claims
}

fn access_commitment(block: &TraceBlock, capacity: usize) -> FieldElement {
    let mut transcript = PoseidonTranscript::new(BLOCK_RAM_ACCESS_DOMAIN);
    transcript.append_scalar(b"block_index", &Fr::from(block.block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(block.global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(block.active_cycles as u64));
    for cycle in padded_trace(block, capacity) {
        let values = match cycle.ram_access() {
            RAMAccess::Read(read) => [1, read.address, read.value, read.value],
            RAMAccess::Write(write) => [2, write.address, write.pre_value, write.post_value],
            RAMAccess::NoOp => [0; 4],
        };
        for value in values {
            transcript.append_scalar(b"ram_access_word", &Fr::from(value));
        }
    }
    FieldElement(transcript.state)
}

fn state_vector(
    ram_k: usize,
    memory: &BTreeMap<u64, u64>,
    preprocessing: &DirectChunkedPreprocessing,
) -> Result<Vec<u64>, DirectChunkedError> {
    let mut state = vec![0u64; ram_k];
    for (address, value) in memory {
        let index = remap_address(*address, &preprocessing.memory_layout).ok_or_else(|| {
            DirectChunkedError::InvalidConfiguration(format!(
                "RAM address {address:#x} is outside verifier memory layout"
            ))
        })? as usize;
        if index >= ram_k {
            return Err(DirectChunkedError::InvalidConfiguration(format!(
                "RAM address {address:#x} remaps beyond ram_k={ram_k}"
            )));
        }
        state[index] = *value;
    }
    Ok(state)
}

fn state_commitment(values: &[u64]) -> FieldElement {
    let mut transcript = PoseidonTranscript::new(BLOCK_RAM_STATE_DOMAIN);
    transcript.append_scalar(b"ram_k", &Fr::from(values.len() as u64));
    for value in values {
        transcript.append_scalar(b"ram_value", &Fr::from(*value));
    }
    FieldElement(transcript.state)
}

fn registry_commitment(
    ram_k: usize,
    memory: &BTreeMap<u64, u64>,
    preprocessing: &DirectChunkedPreprocessing,
) -> Result<FieldElement, DirectChunkedError> {
    let mut transcript = PoseidonTranscript::new(b"block-jolt-ram-registry-v2");
    transcript.append_scalar(b"ram_k", &Fr::from(ram_k as u64));
    for address in memory.keys() {
        let index = remap_address(*address, &preprocessing.memory_layout).ok_or_else(|| {
            DirectChunkedError::InvalidConfiguration(format!(
                "RAM address {address:#x} is outside verifier memory layout"
            ))
        })?;
        transcript.append_scalar(b"ram_address", &Fr::from(*address));
        transcript.append_scalar(b"ram_index", &Fr::from(index));
    }
    Ok(FieldElement(transcript.state))
}

fn apply_block(
    block: &TraceBlock,
    memory: &BTreeMap<u64, u64>,
) -> Result<BTreeMap<u64, u64>, DirectChunkedError> {
    let mut next = memory.clone();
    for (row, cycle) in block.cycles.iter().enumerate() {
        match cycle.ram_access() {
            RAMAccess::Read(read) => {
                let expected = next.get(&read.address).copied().unwrap_or_default();
                if expected != read.value {
                    return Err(DirectChunkedError::InvalidBlock {
                        block_index: block.block_index,
                        reason: format!(
                            "RAM read row {row} has value {}, expected {expected}",
                            read.value
                        ),
                    });
                }
            }
            RAMAccess::Write(write) => {
                let expected = next.get(&write.address).copied().unwrap_or_default();
                if expected != write.pre_value {
                    return Err(DirectChunkedError::InvalidBlock {
                        block_index: block.block_index,
                        reason: format!(
                            "RAM write row {row} has pre-value {}, expected {expected}",
                            write.pre_value
                        ),
                    });
                }
                next.insert(write.address, write.post_value);
            }
            RAMAccess::NoOp => {}
        }
    }
    Ok(next)
}

fn append_header(
    transcript: &mut PoseidonTranscript,
    block_index: usize,
    global_cycle_start: usize,
    active_cycles: usize,
    ram_k: usize,
    access_commitment: FieldElement,
    registry_root: FieldElement,
    root_before: FieldElement,
    root_after: FieldElement,
) {
    for (label, value) in [
        (b"block_index".as_slice(), Fr::from(block_index as u64)),
        (
            b"cycle_start".as_slice(),
            Fr::from(global_cycle_start as u64),
        ),
        (b"active_cycles".as_slice(), Fr::from(active_cycles as u64)),
        (b"ram_k".as_slice(), Fr::from(ram_k as u64)),
        (b"access_commitment".as_slice(), access_commitment.to_fr()),
        (b"registry_root".as_slice(), registry_root.to_fr()),
        (b"root_before".as_slice(), root_before.to_fr()),
        (b"root_after".as_slice(), root_after.to_fr()),
    ] {
        transcript.append_scalar(label, &value);
    }
}

fn opening_id(label: &[u8], index: usize) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(b"ram-opening");
    hasher.update(label);
    hasher.update((index as u64).to_le_bytes());
    hasher.finalize().into()
}

pub(super) fn deferred_claims(
    commitment: FieldElement,
    reduction_point: &[FieldElement],
    inputs: &[FieldElement; 2],
    sumcheck_point: &[FieldElement],
    outputs: &[FieldElement],
) -> Vec<DeferredPcsClaim> {
    let mut claims = inputs
        .iter()
        .enumerate()
        .map(|(index, value)| DeferredPcsClaim {
            relation: BlockRelation::Ram,
            polynomial_id: opening_id(b"input", index),
            commitment_id: commitment.0,
            opening_point: reduction_point.to_vec(),
            claimed_value: *value,
        })
        .collect::<Vec<_>>();
    let log_t = reduction_point.len();
    let log_k = sumcheck_point.len() - log_t;
    let r_cycle = sumcheck_point[..log_t]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let r_address = sumcheck_point[log_t..log_t + log_k]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    claims.extend(outputs.iter().enumerate().map(|(index, value)| {
        let opening_point = if index == RAM_OUTPUT_COUNT - 1 {
            r_cycle.clone()
        } else {
            [r_address.clone(), r_cycle.clone()].concat()
        };
        DeferredPcsClaim {
            relation: BlockRelation::Ram,
            polynomial_id: opening_id(b"output", index),
            commitment_id: commitment.0,
            opening_point,
            claimed_value: *value,
        }
    }));
    claims
}

/// Proves Jolt's RAM read/write sumcheck for one fixed-capacity block.
#[allow(clippy::type_complexity)]
pub fn prove_block_ram(
    preprocessing: &DirectChunkedPreprocessing,
    block: &TraceBlock,
    capacity: usize,
    ram_k: usize,
    initial_memory: &BTreeMap<u64, u64>,
    transcript: &mut PoseidonTranscript,
) -> Result<
    (
        RamBlockProof,
        BTreeMap<u64, u64>,
        TranscriptCheckpoint,
        TranscriptCheckpoint,
    ),
    DirectChunkedError,
> {
    prove_block_ram_with_commitment(
        preprocessing,
        block,
        capacity,
        ram_k,
        initial_memory,
        transcript,
        None,
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prove_block_ram_with_commitment(
    preprocessing: &DirectChunkedPreprocessing,
    block: &TraceBlock,
    capacity: usize,
    ram_k: usize,
    initial_memory: &BTreeMap<u64, u64>,
    transcript: &mut PoseidonTranscript,
    commitment_id: Option<[u8; 32]>,
) -> Result<
    (
        RamBlockProof,
        BTreeMap<u64, u64>,
        TranscriptCheckpoint,
        TranscriptCheckpoint,
    ),
    DirectChunkedError,
> {
    validate_block(block, capacity)?;
    if ram_k == 0 || !ram_k.is_power_of_two() {
        return Err(DirectChunkedError::InvalidConfiguration(
            "block RAM K must be a non-zero power of two".to_string(),
        ));
    }
    let final_memory = apply_block(block, initial_memory)?;
    let initial_state = state_vector(ram_k, initial_memory, preprocessing)?;
    let final_state = state_vector(ram_k, &final_memory, preprocessing)?;
    let access_commitment = commitment_id
        .map(FieldElement)
        .unwrap_or_else(|| access_commitment(block, capacity));
    let registry_root = registry_commitment(ram_k, &final_memory, preprocessing)?;
    let root_before = state_commitment(&initial_state);
    let root_after = state_commitment(&final_state);
    let before = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    append_header(
        transcript,
        block.block_index,
        block.global_cycle_start,
        block.active_cycles,
        ram_k,
        access_commitment,
        registry_root,
        root_before,
        root_after,
    );
    let log_t = capacity.log_2();
    let reduction_point = transcript.challenge_vector::<Fr>(log_t);
    let input_claims = ram_input_claims(block, capacity, &reduction_point);
    let mut accumulator = ProverOpeningAccumulator::<Fr>::new(log_t);
    seed_prover(&mut accumulator, &reduction_point, input_claims);
    let one_hot = OneHotParams::new(log_t, preprocessing.bytecode.code_size.max(1), ram_k);
    let config = ReadWriteConfig::new(log_t, ram_k.log_2());
    let params =
        RamReadWriteCheckingParams::new(&accumulator, transcript, &one_hot, capacity, &config);
    let gamma = params.gamma;
    let expected_input = input_claims[0] + gamma * input_claims[1];
    let mut batching_transcript = transcript.clone();
    batching_transcript.append_scalar(b"sumcheck_claim", &expected_input);
    let batching_coefficient = batching_transcript.challenge_scalar::<Fr>();
    let padded = padded_trace(block, capacity);
    let mut prover = RamReadWriteCheckingProver::initialize(
        params,
        &padded,
        &preprocessing.bytecode,
        &preprocessing.memory_layout,
        &initial_state,
    );
    let (clear, challenges, initial_batched_claim) =
        BatchedSumcheck::prove(vec![&mut prover], &mut accumulator, transcript);
    let challenges = challenges.into_iter().map(Into::into).collect::<Vec<Fr>>();
    let output_claims = output_ids().map(|id| accumulator.get_opening(id));
    let final_claim = clear
        .compressed_polys
        .iter()
        .zip(&challenges)
        .fold(initial_batched_claim, |claim, (poly, challenge)| {
            poly.eval_from_hint(&claim, &(*challenge).into())
        });
    let mut compressed_proof = Vec::new();
    clear
        .serialize_compressed(&mut compressed_proof)
        .map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!(
                "compact RAM sumcheck serialization failed: {error}"
            ))
        })?;
    let reduction_point = reduction_point
        .iter()
        .map(FieldElement::from_fr)
        .collect::<Vec<_>>();
    let input_fields = input_claims.map(|value| FieldElement::from_fr(&value));
    let challenge_fields = challenges
        .iter()
        .map(FieldElement::from_fr)
        .collect::<Vec<_>>();
    let output_fields = output_claims
        .iter()
        .map(FieldElement::from_fr)
        .collect::<Vec<_>>();
    let deferred = deferred_claims(
        access_commitment,
        &reduction_point,
        &input_fields,
        &challenge_fields,
        &output_fields,
    );
    let after = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    Ok((
        RamBlockProof {
            access_commitment,
            ram_k: ram_k as u64,
            registry_root,
            root_before,
            root_after,
            reduction_point,
            input_claims: input_fields,
            gamma: FieldElement::from_fr(&gamma),
            batching_coefficient: FieldElement::from_fr(&batching_coefficient),
            output_claims: output_fields,
            relation_proof: CompactSumcheckProof {
                rounds: challenges.len() as u32,
                degree_bound: 3,
                initial_claim: FieldElement::from_fr(&initial_batched_claim),
                final_claim: FieldElement::from_fr(&final_claim),
                compressed_proof,
                challenges: challenge_fields,
                opening_claims: deferred,
            },
        },
        final_memory,
        before,
        after,
    ))
}

/// Materializes the exact RAM polynomials referenced by the compact proof, in
/// deferred-claim order: read value, write value, ra, Val, and increment.
pub(super) fn block_ram_polynomials(
    preprocessing: &DirectChunkedPreprocessing,
    block: &TraceBlock,
    capacity: usize,
    ram_k: usize,
    initial_memory: &BTreeMap<u64, u64>,
) -> Result<Vec<MultilinearPolynomial<Fr>>, DirectChunkedError> {
    validate_block(block, capacity)?;
    if ram_k == 0 || !ram_k.is_power_of_two() {
        return Err(DirectChunkedError::InvalidConfiguration(
            "block RAM K must be a non-zero power of two".to_string(),
        ));
    }
    // Validate the same state transition used by the compact RAM prover.
    let _ = apply_block(block, initial_memory)?;
    let initial_state = state_vector(ram_k, initial_memory, preprocessing)?;
    let padded = padded_trace(block, capacity);
    let mut read_values = Vec::with_capacity(capacity);
    let mut write_values = Vec::with_capacity(capacity);
    for cycle in &padded {
        match cycle.ram_access() {
            RAMAccess::Read(read) => {
                read_values.push(Fr::from(read.value));
                write_values.push(Fr::from(read.value));
            }
            RAMAccess::Write(write) => {
                read_values.push(Fr::from(write.pre_value));
                write_values.push(Fr::from(write.post_value));
            }
            RAMAccess::NoOp => {
                read_values.push(Fr::zero());
                write_values.push(Fr::zero());
            }
        }
    }
    // The sparse read/write prover treats Val(k, j) as the value at address k
    // immediately before cycle j. Materializing only sparse access entries is
    // insufficient: a write must propagate through every later cycle until the
    // next write. Build the exact dense endpoint polynomial here so deferred
    // Dory openings agree with the accepted native RAM sumcheck even when an
    // address is not touched in adjacent cycles.
    let has_ram_access = padded
        .iter()
        .any(|cycle| !matches!(cycle.ram_access(), RAMAccess::NoOp));
    let mut running = initial_state;
    let mut ra = vec![Fr::zero(); ram_k * capacity];
    let mut val = vec![Fr::zero(); ram_k * capacity];
    for (cycle_index, cycle) in padded.iter().enumerate() {
        if has_ram_access {
            for (address, value) in running.iter().enumerate() {
                val[address * capacity + cycle_index] = Fr::from(*value);
            }
        }
        match cycle.ram_access() {
            RAMAccess::Read(read) => {
                let address =
                    remap_address(read.address, &preprocessing.memory_layout).ok_or_else(|| {
                        DirectChunkedError::InvalidConfiguration(format!(
                            "RAM address {:#x} is outside verifier memory layout",
                            read.address
                        ))
                    })? as usize;
                if address >= ram_k || running[address] != read.value {
                    return Err(DirectChunkedError::InvalidBlock {
                        block_index: block.block_index,
                        reason: format!(
                            "RAM read row {cycle_index} is inconsistent while materializing Val"
                        ),
                    });
                }
                ra[address * capacity + cycle_index] = Fr::from(1u64);
            }
            RAMAccess::Write(write) => {
                let address = remap_address(write.address, &preprocessing.memory_layout)
                    .ok_or_else(|| {
                        DirectChunkedError::InvalidConfiguration(format!(
                            "RAM address {:#x} is outside verifier memory layout",
                            write.address
                        ))
                    })? as usize;
                if address >= ram_k || running[address] != write.pre_value {
                    return Err(DirectChunkedError::InvalidBlock {
                        block_index: block.block_index,
                        reason: format!(
                            "RAM write row {cycle_index} is inconsistent while materializing Val"
                        ),
                    });
                }
                ra[address * capacity + cycle_index] = Fr::from(1u64);
                running[address] = write.post_value;
            }
            RAMAccess::NoOp => {}
        }
    }
    let inc = CommittedPolynomial::RamInc.generate_witness(
        &preprocessing.bytecode,
        &preprocessing.memory_layout,
        &padded,
        None,
    );
    Ok(vec![
        read_values.into(),
        write_values.into(),
        ra.into(),
        val.into(),
        inc,
    ])
}

/// Verifies the compact RAM sumcheck without block trace or memory witness.
#[allow(clippy::too_many_arguments)]
pub fn verify_block_ram(
    preprocessing: &DirectChunkedPreprocessing,
    proof: &RamBlockProof,
    block_index: usize,
    global_cycle_start: usize,
    active_cycles: usize,
    capacity: usize,
    expected_root_before: FieldElement,
    expected_root_after: FieldElement,
    transcript: &mut PoseidonTranscript,
) -> Result<(TranscriptCheckpoint, TranscriptCheckpoint), DirectChunkedError> {
    let ram_k = proof.ram_k as usize;
    if capacity == 0
        || !capacity.is_power_of_two()
        || active_cycles == 0
        || active_cycles > capacity
        || ram_k == 0
        || !ram_k.is_power_of_two()
        || proof.root_before != expected_root_before
        || proof.root_after != expected_root_after
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact RAM shape or boundary mismatch".to_string(),
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
        ram_k,
        proof.access_commitment,
        proof.registry_root,
        proof.root_before,
        proof.root_after,
    );
    let log_t = capacity.log_2();
    let reduction_point = transcript.challenge_vector::<Fr>(log_t);
    if proof
        .reduction_point
        .iter()
        .map(|value| value.to_fr())
        .collect::<Vec<_>>()
        != reduction_point
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact RAM reduction challenge mismatch".to_string(),
        ));
    }
    let inputs = proof.input_claims.map(FieldElement::to_fr);
    let outputs: [Fr; RAM_OUTPUT_COUNT] = proof
        .output_claims
        .iter()
        .map(|value| value.to_fr())
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| {
            DirectChunkedError::InvalidProofShape(
                "compact RAM output opening count mismatch".to_string(),
            )
        })?;
    if proof.relation_proof.degree_bound != 3
        || proof.relation_proof.rounds as usize != log_t + ram_k.log_2()
        || proof.relation_proof.challenges.len() != log_t + ram_k.log_2()
        || proof.relation_proof.opening_claims
            != deferred_claims(
                proof.access_commitment,
                &proof.reduction_point,
                &proof.input_claims,
                &proof.relation_proof.challenges,
                &proof.output_claims,
            )
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact RAM sumcheck or deferred opening shape mismatch".to_string(),
        ));
    }
    let mut accumulator = VerifierOpeningAccumulator::<Fr>::new(log_t, false);
    seed_verifier(&mut accumulator, &reduction_point, inputs, outputs);
    let one_hot = OneHotParams::new(log_t, preprocessing.bytecode.code_size.max(1), ram_k);
    let config = ReadWriteConfig::new(log_t, ram_k.log_2());
    let verifier =
        RamReadWriteCheckingVerifier::new(&accumulator, transcript, &one_hot, capacity, &config);
    let gamma = proof.gamma.to_fr();
    if gamma != verifier.gamma() {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact RAM gamma mismatch".to_string(),
        ));
    }
    let expected_input = inputs[0] + gamma * inputs[1];
    let mut batching_transcript = transcript.clone();
    batching_transcript.append_scalar(b"sumcheck_claim", &expected_input);
    let expected_batching = batching_transcript.challenge_scalar::<Fr>();
    if proof.batching_coefficient.to_fr() != expected_batching
        || proof.relation_proof.initial_claim.to_fr() != expected_input * expected_batching
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact RAM batched initial claim mismatch".to_string(),
        ));
    }
    let clear = ClearSumcheckProof::<Fr, PoseidonTranscript>::deserialize_compressed(
        proof.relation_proof.compressed_proof.as_slice(),
    )
    .map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!(
            "compact RAM sumcheck deserialization failed: {error}"
        ))
    })?;
    let challenges =
        BatchedSumcheck::verify_standard(&clear, vec![&verifier], &mut accumulator, transcript)
            .map_err(|error| {
                DirectChunkedError::InvalidProofShape(format!(
                    "compact RamReadWriteChecking verification failed: {error}"
                ))
            })?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Fr>>();
    let supplied = proof
        .relation_proof
        .challenges
        .iter()
        .map(|value| value.to_fr())
        .collect::<Vec<_>>();
    let final_claim = clear.compressed_polys.iter().zip(&challenges).fold(
        proof.relation_proof.initial_claim.to_fr(),
        |claim, (poly, challenge)| poly.eval_from_hint(&claim, &(*challenge).into()),
    );
    let after = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    if challenges != supplied
        || final_claim != proof.relation_proof.final_claim.to_fr()
        || after.round < before.round
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact RAM Fiat-Shamir transcript mismatch".to_string(),
        ));
    }
    Ok((before, after))
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::constants::{RAM_START_ADDRESS, REGISTER_COUNT};
    use tracer::{
        instruction::{
            format::{
                format_load::{FormatLoad, RegisterStateFormatLoad},
                format_s::{FormatS, RegisterStateFormatS},
            },
            ld::LD,
            sd::SD,
            RAMRead, RAMWrite, RISCVCycle,
        },
        MachineBoundaryState,
    };

    use crate::poly::multilinear_polynomial::PolynomialEvaluation;

    fn block(address: u64) -> TraceBlock {
        let store = RISCVCycle::<SD> {
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
                rs2: 9,
            },
            ram_access: RAMWrite {
                address,
                pre_value: 0,
                post_value: 9,
            },
        }
        .into();
        let load = RISCVCycle::<LD> {
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
                rd: (0, 9),
                rs1: address,
            },
            ram_access: RAMRead { address, value: 9 },
        }
        .into();
        let mut start_registers = [0i64; REGISTER_COUNT as usize];
        start_registers[1] = address as i64;
        start_registers[2] = 9;
        let mut end_registers = start_registers;
        end_registers[3] = 9;
        TraceBlock {
            block_index: 0,
            global_cycle_start: 0,
            active_cycles: 2,
            target_size: 2,
            start_state: MachineBoundaryState {
                global_cycle: 0,
                emulator_trace_len: 0,
                pc: 0x8000_0000,
                registers: start_registers,
                terminated: false,
            },
            end_state: MachineBoundaryState {
                global_cycle: 2,
                emulator_trace_len: 2,
                pc: 0x8000_0008,
                registers: end_registers,
                terminated: true,
            },
            cycles: vec![store, load],
            ended_at_tick_boundary: true,
        }
    }

    fn block_with_gap_after_store(address: u64) -> TraceBlock {
        let mut block = block(address);
        let load = block.cycles.pop().unwrap();
        block.cycles.push(tracer::instruction::Cycle::NoOp);
        block.cycles.push(load);
        block.active_cycles = 3;
        block.target_size = 4;
        block.end_state.global_cycle = 3;
        block.end_state.emulator_trace_len = 3;
        block
    }

    fn no_ram_block(address: u64) -> TraceBlock {
        let mut block = block(address);
        block.cycles = vec![tracer::instruction::Cycle::NoOp];
        block.active_cycles = 1;
        block.target_size = 2;
        block.end_state.global_cycle = 1;
        block.end_state.emulator_trace_len = 1;
        block.end_state.registers = block.start_state.registers;
        block
    }

    #[test]
    fn d12_compact_ram_sumcheck_round_trip() {
        let preprocessing = DirectChunkedPreprocessing::from_program_bytes(b"ram-v2", 64);
        let address = RAM_START_ADDRESS + 0x100;
        let initial = BTreeMap::from([(address, 0)]);
        let remapped = remap_address(address, &preprocessing.memory_layout).unwrap() as usize;
        let ram_k = (remapped + 1).next_power_of_two();
        let mut prover_transcript = new_block_ram_transcript();
        let (proof, final_memory, before, after) = prove_block_ram(
            &preprocessing,
            &block(address),
            2,
            ram_k,
            &initial,
            &mut prover_transcript,
        )
        .unwrap();
        assert_eq!(final_memory[&address], 9);
        let mut verifier_transcript = new_block_ram_transcript();
        let verified = verify_block_ram(
            &preprocessing,
            &proof,
            0,
            0,
            2,
            2,
            proof.root_before,
            proof.root_after,
            &mut verifier_transcript,
        )
        .unwrap();
        assert_eq!(verified, (before, after));
    }

    #[test]
    fn d20_ram_val_opening_propagates_writes_across_idle_cycles() {
        let preprocessing = DirectChunkedPreprocessing::from_program_bytes(b"ram-v2-gap", 64);
        let address = RAM_START_ADDRESS + 0x100;
        let initial = BTreeMap::from([(address, 0)]);
        let remapped = remap_address(address, &preprocessing.memory_layout).unwrap() as usize;
        let ram_k = (remapped + 1).next_power_of_two();
        let block = block_with_gap_after_store(address);
        let mut transcript = new_block_ram_transcript();
        let (proof, _, _, _) =
            prove_block_ram(&preprocessing, &block, 4, ram_k, &initial, &mut transcript).unwrap();
        let polynomials =
            block_ram_polynomials(&preprocessing, &block, 4, ram_k, &initial).unwrap();

        for (index, (polynomial, claim)) in polynomials
            .iter()
            .zip(&proof.relation_proof.opening_claims)
            .enumerate()
        {
            let point = claim
                .opening_point
                .iter()
                .map(|value| value.to_fr())
                .collect::<Vec<_>>();
            assert_eq!(
                PolynomialEvaluation::evaluate(polynomial, &point),
                claim.claimed_value.to_fr(),
                "RAM endpoint polynomial {index} disagrees with its accepted claim"
            );
        }
    }

    #[test]
    fn d20_ram_val_opening_is_zero_for_an_access_free_block() {
        let preprocessing = DirectChunkedPreprocessing::from_program_bytes(b"ram-v2-no-access", 64);
        let address = RAM_START_ADDRESS + 0x100;
        let initial = BTreeMap::from([(address, 9)]);
        let remapped = remap_address(address, &preprocessing.memory_layout).unwrap() as usize;
        let ram_k = (remapped + 1).next_power_of_two();
        let block = no_ram_block(address);
        let mut transcript = new_block_ram_transcript();
        let (proof, _, _, _) =
            prove_block_ram(&preprocessing, &block, 2, ram_k, &initial, &mut transcript).unwrap();
        let polynomials =
            block_ram_polynomials(&preprocessing, &block, 2, ram_k, &initial).unwrap();

        assert!((0..polynomials[3].len()).all(|index| polynomials[3].get_coeff(index).is_zero()));
        for (polynomial, claim) in polynomials.iter().zip(&proof.relation_proof.opening_claims) {
            let point = claim
                .opening_point
                .iter()
                .map(|value| value.to_fr())
                .collect::<Vec<_>>();
            assert_eq!(
                PolynomialEvaluation::evaluate(polynomial, &point),
                claim.claimed_value.to_fr()
            );
        }
    }

    #[test]
    fn d12_compact_ram_rejects_root_and_opening_tampering() {
        let preprocessing = DirectChunkedPreprocessing::from_program_bytes(b"ram-v2", 64);
        let address = RAM_START_ADDRESS + 0x100;
        let initial = BTreeMap::from([(address, 0)]);
        let remapped = remap_address(address, &preprocessing.memory_layout).unwrap() as usize;
        let ram_k = (remapped + 1).next_power_of_two();
        let mut prover_transcript = new_block_ram_transcript();
        let (proof, _, _, _) = prove_block_ram(
            &preprocessing,
            &block(address),
            2,
            ram_k,
            &initial,
            &mut prover_transcript,
        )
        .unwrap();
        let expected_before = proof.root_before;
        let expected_after = proof.root_after;

        let mut bad_root = proof.clone();
        bad_root.root_after.0[0] ^= 1;
        let mut verifier_transcript = new_block_ram_transcript();
        assert!(verify_block_ram(
            &preprocessing,
            &bad_root,
            0,
            0,
            2,
            2,
            expected_before,
            expected_after,
            &mut verifier_transcript,
        )
        .is_err());

        let mut bad_opening = proof;
        bad_opening.output_claims[0].0[0] ^= 1;
        let mut verifier_transcript = new_block_ram_transcript();
        assert!(verify_block_ram(
            &preprocessing,
            &bad_opening,
            0,
            0,
            2,
            2,
            expected_before,
            expected_after,
            &mut verifier_transcript,
        )
        .is_err());
    }
}
