//! D15 fixed-shape Nova step for the compact per-block Jolt verifier.
//!
//! This circuit consumes only the compact proof messages produced by D11-D14.
//! Trace rows, Merkle paths, D8 witnesses, and host `accepted` bits are not part
//! of its witness.  It replays the four native Poseidon transcripts, verifies
//! the clear-sumcheck arithmetic and relation endpoints, and carries the exact
//! deferred-opening ledger in a circuit-native Poseidon accumulator.

use std::io::Cursor;

use ark_bn254::Fr;
use ark_ff::{Field, PrimeField};
use ark_serialize::CanonicalDeserialize;
use common::constants::REGISTER_COUNT;
use nova_snark::{
    frontend::{
        gadgets::boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, LinearCombination,
        SynthesisError,
    },
    traits::circuit::StepCircuit,
};
use strum::EnumCount;

use crate::{
    subprotocols::{sumcheck::ClearSumcheckProof, univariate_skip::UniSkipFirstRoundProof},
    transcripts::{PoseidonTranscript, Transcript},
    zkvm::{
        config::OneHotParams,
        instruction_lookups::LOG_K,
        lookup_table::LookupTables,
        r1cs::{
            constraints::{
                OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
                R1CS_CONSTRAINTS_FIRST_GROUP, R1CS_CONSTRAINTS_SECOND_GROUP,
            },
            inputs::NUM_R1CS_INPUTS,
            ops::LC,
        },
    },
};

use super::super::{
    direct_lookup::compact_lookup_deferred_claims,
    direct_register::compact_register_deferred_claims,
};
use super::super::{
    direct_lookup::{
        add_nums, alloc_u64_bits, alloc_witness_num, bit_as_num, enforce_num_equal,
        evaluate_table_mle_circuit, field_from_digest, mul_nums, nova_from_fr, poseidon_absorb,
        poseidon_challenge, scale_num, sub_nums, weighted_linear_point,
    },
    direct_register::eq_between_points,
    recursive_relations::{
        alloc_nova_constant, synthesize_recursive_clear_sumcheck_stage,
        synthesize_recursive_clear_sumcheck_transcript,
        synthesize_recursive_poseidon_transcript_transition, AllocatedRecursiveClearSumcheckStage,
        AllocatedRecursivePoseidonTranscriptState,
    },
    NovaScalar, RecursiveClearSumcheckRoundWitness, RecursiveClearSumcheckStageWitness,
    RecursiveJoltFieldElement,
};
use super::{
    cpu::cpu_deferred_claims, new_block_cpu_transcript, new_block_ram_transcript,
    new_block_register_transcript, ram::deferred_claims as ram_deferred_claims,
    BlockJoltHostConfig, BlockJoltProof, BlockJoltStatement, BlockRelation, CompactSumcheckProof,
    DeferredPcsClaim, FieldElement, BLOCK_JOLT_WIRE_VERSION,
};

pub const BLOCK_JOLT_VERIFIER_Z_ARITY: usize = 32;
const DEFERRED_TRANSCRIPT_DOMAIN: &[u8] = b"block-jolt-deferred-v3";

const WIRE_SLOT: usize = 0;
const PREPROCESSING_OFFSET: usize = 1;
const PROGRAM_OFFSET: usize = 5;
const TABLE_OFFSET: usize = 9;
const TABLE_REDUCED_SLOT: usize = 13;
const BYTECODE_SLOT: usize = 14;
pub(super) const BLOCK_SLOT: usize = 15;
pub(super) const CYCLE_SLOT: usize = 16;
pub(super) const MACHINE_OFFSET: usize = 17;
pub(super) const REGISTER_OFFSET: usize = 21;
pub(super) const RAM_ROOT_SLOT: usize = 25;
pub(super) const LOOKUP_STATE_SLOT: usize = 26;
pub(super) const LOOKUP_ROUND_SLOT: usize = 27;
pub(super) const DEFERRED_STATE_SLOT: usize = 28;
pub(super) const DEFERRED_ROUND_SLOT: usize = 29;
pub(super) const TOTAL_CYCLES_SLOT: usize = 30;
pub(super) const TERMINATED_SLOT: usize = 31;

#[derive(Clone)]
struct AllocatedDeferredClaim {
    relation: BlockRelation,
    polynomial_id: [AllocatedNum<NovaScalar>; 4],
    commitment_id: [AllocatedNum<NovaScalar>; 4],
    opening_point: Vec<AllocatedNum<NovaScalar>>,
    claimed_value: AllocatedNum<NovaScalar>,
}

fn digest_words(value: &[u8; 32]) -> [u64; 4] {
    std::array::from_fn(|index| {
        u64::from_le_bytes(value[index * 8..(index + 1) * 8].try_into().unwrap())
    })
}

fn digest_scalars(value: &[u8; 32]) -> [NovaScalar; 4] {
    digest_words(value).map(NovaScalar::from)
}

fn alloc_digest<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: &[u8; 32],
    label: &'static str,
) -> Result<[AllocatedNum<NovaScalar>; 4], SynthesisError> {
    digest_words(value)
        .into_iter()
        .enumerate()
        .map(|(index, word)| {
            alloc_u64_bits(
                cs.namespace(|| format!("{label} word {index}")),
                word,
                label,
            )
            .map(|(number, _)| number)
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| SynthesisError::AssignmentMissing)
}

fn enforce_digest_equal_to_z<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    digest: &[AllocatedNum<NovaScalar>; 4],
    z: &[AllocatedNum<NovaScalar>],
    offset: usize,
    label: &'static str,
) {
    for index in 0..4 {
        enforce_num_equal(
            cs.namespace(|| format!("{label} word {index}")),
            &digest[index],
            &z[offset + index],
            label,
        );
    }
}

fn alloc_field<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
    value: FieldElement,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    alloc_witness_num(cs, nova_from_fr(&value.to_fr())?)
}

fn alloc_fields<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    values: &[FieldElement],
    label: &'static str,
) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| alloc_field(cs.namespace(|| format!("{label} {index}")), *value))
        .collect()
}

fn allocated_constant_transcript<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    transcript: PoseidonTranscript,
) -> Result<AllocatedRecursivePoseidonTranscriptState, SynthesisError> {
    Ok(AllocatedRecursivePoseidonTranscriptState {
        state: alloc_nova_constant(
            cs.namespace(|| "constant transcript state"),
            nova_from_fr(&field_from_digest(&transcript.state))?,
        )?,
        n_rounds: alloc_nova_constant(
            cs.namespace(|| "constant transcript round"),
            NovaScalar::from(transcript.n_rounds as u64),
        )?,
    })
}

fn compact_sumcheck_witness(
    proof: &CompactSumcheckProof,
    stage_index: usize,
) -> Result<RecursiveClearSumcheckStageWitness, SynthesisError> {
    let mut cursor = Cursor::new(&proof.compressed_proof);
    let clear = ClearSumcheckProof::<Fr, PoseidonTranscript>::deserialize_compressed(&mut cursor)
        .map_err(|error| SynthesisError::Unsatisfiable(error.to_string()))?;
    if cursor.position() != proof.compressed_proof.len() as u64
        || clear.compressed_polys.len() != proof.challenges.len()
    {
        return Err(SynthesisError::Unsatisfiable(
            "compact sumcheck serialization or challenge dimensions are invalid".to_string(),
        ));
    }
    let rounds = clear
        .compressed_polys
        .iter()
        .zip(&proof.challenges)
        .map(|(poly, challenge)| {
            Ok(RecursiveClearSumcheckRoundWitness {
                coefficients_except_linear: poly
                    .coeffs_except_linear_term
                    .iter()
                    .map(|value| {
                        RecursiveJoltFieldElement::from_field(*value)
                            .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                challenge: RecursiveJoltFieldElement::from_field(challenge.to_fr())
                    .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RecursiveClearSumcheckStageWitness {
        stage_index,
        degree_bound: proof.degree_bound as usize,
        initial_claim: RecursiveJoltFieldElement::from_field(proof.initial_claim.to_fr())
            .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?,
        rounds,
        expected_final_claim: RecursiveJoltFieldElement::from_field(proof.final_claim.to_fr())
            .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))?,
    })
}

fn derive_challenges<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    mut transcript: AllocatedRecursivePoseidonTranscriptState,
    supplied: &[FieldElement],
    label: &'static str,
) -> Result<
    (
        AllocatedRecursivePoseidonTranscriptState,
        Vec<AllocatedNum<NovaScalar>>,
    ),
    SynthesisError,
> {
    let mut challenges = Vec::with_capacity(supplied.len());
    for (index, supplied) in supplied.iter().enumerate() {
        transcript = poseidon_challenge(
            cs.namespace(|| format!("{label} challenge {index}")),
            &transcript,
        )?;
        let claimed = alloc_field(
            cs.namespace(|| format!("{label} supplied challenge {index}")),
            *supplied,
        )?;
        enforce_num_equal(
            cs.namespace(|| format!("{label} challenge binding {index}")),
            &transcript.state,
            &claimed,
            "Fiat-Shamir challenge binding",
        );
        challenges.push(claimed);
    }
    Ok((transcript, challenges))
}

fn append_opening_claims<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    mut transcript: AllocatedRecursivePoseidonTranscriptState,
    claims: &[AllocatedNum<NovaScalar>],
    label: &'static str,
) -> Result<AllocatedRecursivePoseidonTranscriptState, SynthesisError> {
    for (index, claim) in claims.iter().enumerate() {
        transcript = poseidon_absorb(
            cs.namespace(|| format!("{label} opening claim {index}")),
            &transcript,
            claim,
            "opening_claim",
        )?;
    }
    Ok(transcript)
}

fn initial_and_batching_claim<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    transcript: AllocatedRecursivePoseidonTranscriptState,
    input_claims: &[AllocatedNum<NovaScalar>],
    gamma: &AllocatedNum<NovaScalar>,
    supplied_batching: FieldElement,
    sumcheck: &AllocatedRecursiveClearSumcheckStage,
) -> Result<
    (
        AllocatedRecursivePoseidonTranscriptState,
        AllocatedNum<NovaScalar>,
    ),
    SynthesisError,
> {
    let mut power = alloc_nova_constant(cs.namespace(|| "gamma power one"), NovaScalar::one())?;
    let mut input = alloc_nova_constant(cs.namespace(|| "input claim zero"), NovaScalar::zero())?;
    for (index, claim) in input_claims.iter().enumerate() {
        let term = mul_nums(
            cs.namespace(|| format!("weighted input claim {index}")),
            &power,
            claim,
            "weighted input claim",
        )?;
        input = add_nums(
            cs.namespace(|| format!("input claim sum {index}")),
            &input,
            &term,
            "input claim sum",
        )?;
        power = mul_nums(
            cs.namespace(|| format!("gamma power {index}")),
            &power,
            gamma,
            "gamma power",
        )?;
    }
    let mut transcript = poseidon_absorb(
        cs.namespace(|| "append sumcheck input claim"),
        &transcript,
        &input,
        "sumcheck_claim",
    )?;
    transcript = poseidon_challenge(cs.namespace(|| "derive batching coefficient"), &transcript)?;
    let batching = alloc_field(
        cs.namespace(|| "supplied batching coefficient"),
        supplied_batching,
    )?;
    enforce_num_equal(
        cs.namespace(|| "batching coefficient binding"),
        &transcript.state,
        &batching,
        "batching coefficient binding",
    );
    let expected_initial = mul_nums(
        cs.namespace(|| "batched initial claim"),
        &input,
        &batching,
        "batched initial claim",
    )?;
    enforce_num_equal(
        cs.namespace(|| "sumcheck initial claim binding"),
        &sumcheck.initial_claim,
        &expected_initial,
        "sumcheck initial claim binding",
    );
    Ok((transcript, batching))
}

fn allocated_claims_from_canonical<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    canonical: &[DeferredPcsClaim],
    values: &[AllocatedNum<NovaScalar>],
    label: &'static str,
) -> Result<Vec<AllocatedDeferredClaim>, SynthesisError> {
    if canonical.len() != values.len() {
        return Err(SynthesisError::Unsatisfiable(
            "canonical deferred claim/value count mismatch".to_string(),
        ));
    }
    canonical
        .iter()
        .zip(values)
        .enumerate()
        .map(|(index, (claim, value))| {
            let claimed = alloc_field(
                cs.namespace(|| format!("{label} canonical value {index}")),
                claim.claimed_value,
            )?;
            enforce_num_equal(
                cs.namespace(|| format!("{label} value binding {index}")),
                &claimed,
                value,
                "deferred value binding",
            );
            Ok(AllocatedDeferredClaim {
                relation: claim.relation,
                polynomial_id: alloc_digest(
                    cs.namespace(|| format!("{label} polynomial id {index}")),
                    &claim.polynomial_id,
                    "polynomial id",
                )?,
                commitment_id: alloc_digest(
                    cs.namespace(|| format!("{label} commitment id {index}")),
                    &claim.commitment_id,
                    "commitment id",
                )?,
                opening_point: alloc_fields(
                    cs.namespace(|| format!("{label} opening point {index}")),
                    &claim.opening_point,
                    "opening point",
                )?,
                claimed_value: value.clone(),
            })
        })
        .collect()
}

fn absorb_deferred_claims<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    mut accumulator: AllocatedRecursivePoseidonTranscriptState,
    claims: &[AllocatedDeferredClaim],
) -> Result<AllocatedRecursivePoseidonTranscriptState, SynthesisError> {
    for (claim_index, claim) in claims.iter().enumerate() {
        let relation = alloc_nova_constant(
            cs.namespace(|| format!("deferred claim {claim_index} relation")),
            NovaScalar::from(claim.relation.tag() as u64),
        )?;
        accumulator = poseidon_absorb(
            cs.namespace(|| format!("deferred claim {claim_index} relation absorb")),
            &accumulator,
            &relation,
            "claim_word",
        )?;
        for (kind, words) in [
            ("polynomial", &claim.polynomial_id),
            ("commitment", &claim.commitment_id),
        ] {
            for (word_index, word) in words.iter().enumerate() {
                accumulator = poseidon_absorb(
                    cs.namespace(|| {
                        format!("deferred claim {claim_index} {kind} word {word_index}")
                    }),
                    &accumulator,
                    word,
                    "claim_word",
                )?;
            }
        }
        let point_len = alloc_nova_constant(
            cs.namespace(|| format!("deferred claim {claim_index} point length")),
            NovaScalar::from(claim.opening_point.len() as u64),
        )?;
        accumulator = poseidon_absorb(
            cs.namespace(|| format!("deferred claim {claim_index} point length absorb")),
            &accumulator,
            &point_len,
            "claim_word",
        )?;
        for (coordinate, value) in claim.opening_point.iter().enumerate() {
            accumulator = poseidon_absorb(
                cs.namespace(|| format!("deferred claim {claim_index} point {coordinate}")),
                &accumulator,
                value,
                "claim_word",
            )?;
        }
        accumulator = poseidon_absorb(
            cs.namespace(|| format!("deferred claim {claim_index} value")),
            &accumulator,
            &claim.claimed_value,
            "claim_word",
        )?;
    }
    Ok(accumulator)
}

fn eq_points<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
    left: &[AllocatedNum<NovaScalar>],
    right: &[AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    eq_between_points(cs, left, right)
}

fn lookup_endpoint<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    r_reduction: &[AllocatedNum<NovaScalar>],
    gamma: &AllocatedNum<NovaScalar>,
    batching: &AllocatedNum<NovaScalar>,
    challenges: &[AllocatedNum<NovaScalar>],
    openings: &[AllocatedNum<NovaScalar>],
    one_hot: &OneHotParams,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let table_count = LookupTables::<{ common::constants::XLEN }>::COUNT;
    let ra_count = LOG_K / one_hot.lookups_ra_virtual_log_k_chunk;
    if challenges.len() != LOG_K + r_reduction.len() || openings.len() != table_count + ra_count + 1
    {
        return Err(SynthesisError::Unsatisfiable(
            "compact lookup endpoint dimensions are invalid".to_string(),
        ));
    }
    let (r_address, cycle_low_to_high) = challenges.split_at(LOG_K);
    let r_cycle = cycle_low_to_high.iter().rev().cloned().collect::<Vec<_>>();
    let one = alloc_nova_constant(cs.namespace(|| "lookup endpoint one"), NovaScalar::one())?;
    let mut val = alloc_nova_constant(cs.namespace(|| "lookup Val zero"), NovaScalar::zero())?;
    for table in 0..table_count {
        let table_eval = evaluate_table_mle_circuit(
            cs.namespace(|| format!("lookup fixed table {table}")),
            table,
            r_address,
        )?;
        let term = mul_nums(
            cs.namespace(|| format!("lookup table {table} contribution")),
            &table_eval,
            &openings[table],
            "lookup table contribution",
        )?;
        val = add_nums(
            cs.namespace(|| format!("lookup Val sum {table}")),
            &val,
            &term,
            "lookup Val sum",
        )?;
    }
    let left = weighted_linear_point(
        cs.namespace(|| "lookup left operand"),
        r_address,
        Some(0),
        "lookup left operand",
    )?;
    let right = weighted_linear_point(
        cs.namespace(|| "lookup right operand"),
        r_address,
        Some(1),
        "lookup right operand",
    )?;
    let identity = weighted_linear_point(
        cs.namespace(|| "lookup identity"),
        r_address,
        None,
        "lookup identity",
    )?;
    let gamma_right = mul_nums(
        cs.namespace(|| "lookup gamma right"),
        gamma,
        &right,
        "lookup gamma right",
    )?;
    let operands = add_nums(
        cs.namespace(|| "lookup operands"),
        &left,
        &gamma_right,
        "lookup operands",
    )?;
    let raf_flag = &openings[table_count + ra_count];
    let one_minus_raf = sub_nums(
        cs.namespace(|| "lookup one minus RAF"),
        &one,
        raf_flag,
        "lookup one minus RAF",
    )?;
    let regular = mul_nums(
        cs.namespace(|| "lookup regular RAF"),
        &one_minus_raf,
        &operands,
        "lookup regular RAF",
    )?;
    let gamma_identity = mul_nums(
        cs.namespace(|| "lookup gamma identity"),
        gamma,
        &identity,
        "lookup gamma identity",
    )?;
    let identity_branch = mul_nums(
        cs.namespace(|| "lookup identity RAF"),
        raf_flag,
        &gamma_identity,
        "lookup identity RAF",
    )?;
    let raf = add_nums(
        cs.namespace(|| "lookup RAF claim"),
        &regular,
        &identity_branch,
        "lookup RAF claim",
    )?;
    let gamma_raf = mul_nums(
        cs.namespace(|| "lookup gamma RAF"),
        gamma,
        &raf,
        "lookup gamma RAF",
    )?;
    let val_and_raf = add_nums(
        cs.namespace(|| "lookup Val plus RAF"),
        &val,
        &gamma_raf,
        "lookup Val plus RAF",
    )?;
    let eq = eq_points(
        cs.namespace(|| "lookup reduction equality"),
        r_reduction,
        &r_cycle,
    )?;
    let mut ra = one;
    for (index, opening) in openings[table_count..table_count + ra_count]
        .iter()
        .enumerate()
    {
        ra = mul_nums(
            cs.namespace(|| format!("lookup RA product {index}")),
            &ra,
            opening,
            "lookup RA product",
        )?;
    }
    let endpoint = mul_nums(
        cs.namespace(|| "lookup eq times RA"),
        &eq,
        &ra,
        "lookup eq times RA",
    )?;
    let endpoint = mul_nums(
        cs.namespace(|| "lookup endpoint"),
        &endpoint,
        &val_and_raf,
        "lookup endpoint",
    )?;
    mul_nums(
        cs.namespace(|| "lookup batched endpoint"),
        &endpoint,
        batching,
        "lookup batched endpoint",
    )
}

fn register_endpoint<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    log_t: usize,
    r_reduction: &[AllocatedNum<NovaScalar>],
    gamma: &AllocatedNum<NovaScalar>,
    batching: &AllocatedNum<NovaScalar>,
    challenges: &[AllocatedNum<NovaScalar>],
    openings: &[AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let log_registers = (REGISTER_COUNT as usize).ilog2() as usize;
    if challenges.len() != log_t + log_registers || openings.len() != 5 {
        return Err(SynthesisError::Unsatisfiable(
            "compact register endpoint dimensions are invalid".to_string(),
        ));
    }
    let r_cycle = challenges[..log_t]
        .iter()
        .rev()
        .cloned()
        .collect::<Vec<_>>();
    let eq = eq_points(
        cs.namespace(|| "register endpoint equality"),
        &r_cycle,
        r_reduction,
    )?;
    let inc_plus_val = add_nums(
        cs.namespace(|| "register inc plus val"),
        &openings[4],
        &openings[0],
        "register inc plus val",
    )?;
    let rd = mul_nums(
        cs.namespace(|| "register rd term"),
        &openings[3],
        &inc_plus_val,
        "register rd term",
    )?;
    let rs1 = mul_nums(
        cs.namespace(|| "register rs1 term"),
        &openings[1],
        &openings[0],
        "register rs1 term",
    )?;
    let rs2 = mul_nums(
        cs.namespace(|| "register rs2 term"),
        &openings[2],
        &openings[0],
        "register rs2 term",
    )?;
    let gamma_rs1 = mul_nums(
        cs.namespace(|| "register gamma rs1"),
        gamma,
        &rs1,
        "register gamma rs1",
    )?;
    let gamma_squared = mul_nums(
        cs.namespace(|| "register gamma squared"),
        gamma,
        gamma,
        "register gamma squared",
    )?;
    let gamma_squared_rs2 = mul_nums(
        cs.namespace(|| "register gamma squared rs2"),
        &gamma_squared,
        &rs2,
        "register gamma squared rs2",
    )?;
    let relation = add_nums(
        cs.namespace(|| "register rd plus rs1"),
        &rd,
        &gamma_rs1,
        "register rd plus rs1",
    )?;
    let relation = add_nums(
        cs.namespace(|| "register relation"),
        &relation,
        &gamma_squared_rs2,
        "register relation",
    )?;
    let relation = mul_nums(
        cs.namespace(|| "register equality relation"),
        &eq,
        &relation,
        "register equality relation",
    )?;
    mul_nums(
        cs.namespace(|| "register batched endpoint"),
        &relation,
        batching,
        "register batched endpoint",
    )
}

fn ram_endpoint<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    log_t: usize,
    r_reduction: &[AllocatedNum<NovaScalar>],
    gamma: &AllocatedNum<NovaScalar>,
    batching: &AllocatedNum<NovaScalar>,
    challenges: &[AllocatedNum<NovaScalar>],
    openings: &[AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    if challenges.len() < log_t || openings.len() != 3 {
        return Err(SynthesisError::Unsatisfiable(
            "compact RAM endpoint dimensions are invalid".to_string(),
        ));
    }
    let r_cycle = challenges[..log_t]
        .iter()
        .rev()
        .cloned()
        .collect::<Vec<_>>();
    let eq = eq_points(
        cs.namespace(|| "RAM endpoint equality"),
        &r_cycle,
        r_reduction,
    )?;
    let val_plus_inc = add_nums(
        cs.namespace(|| "RAM val plus inc"),
        &openings[1],
        &openings[2],
        "RAM val plus inc",
    )?;
    let gamma_update = mul_nums(
        cs.namespace(|| "RAM gamma update"),
        gamma,
        &val_plus_inc,
        "RAM gamma update",
    )?;
    let value_relation = add_nums(
        cs.namespace(|| "RAM value relation"),
        &openings[1],
        &gamma_update,
        "RAM value relation",
    )?;
    let relation = mul_nums(
        cs.namespace(|| "RAM RA relation"),
        &openings[0],
        &value_relation,
        "RAM RA relation",
    )?;
    let relation = mul_nums(
        cs.namespace(|| "RAM equality relation"),
        &eq,
        &relation,
        "RAM equality relation",
    )?;
    mul_nums(
        cs.namespace(|| "RAM batched endpoint"),
        &relation,
        batching,
        "RAM batched endpoint",
    )
}

fn packed_label_with_len(label: &[u8], len: usize) -> Result<NovaScalar, SynthesisError> {
    if label.len() > 24 {
        return Err(SynthesisError::Unsatisfiable(
            "Poseidon packed transcript label is too long".to_string(),
        ));
    }
    let mut packed = [0u8; 32];
    packed[..label.len()].copy_from_slice(label);
    packed[24..].copy_from_slice(&(len as u64).to_be_bytes());
    nova_from_fr(&Fr::from_le_bytes_mod_order(&packed))
}

fn poseidon_absorb_vector<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    mut transcript: AllocatedRecursivePoseidonTranscriptState,
    label: &[u8],
    values: &[AllocatedNum<NovaScalar>],
) -> Result<AllocatedRecursivePoseidonTranscriptState, SynthesisError> {
    let packed = alloc_nova_constant(
        cs.namespace(|| "packed vector label"),
        packed_label_with_len(label, values.len())?,
    )?;
    transcript = synthesize_recursive_poseidon_transcript_transition(
        cs.namespace(|| "absorb packed vector label"),
        &transcript,
        &packed,
    )?;
    for (index, value) in values.iter().enumerate() {
        transcript = synthesize_recursive_poseidon_transcript_transition(
            cs.namespace(|| format!("absorb vector item {index}")),
            &transcript,
            value,
        )?;
    }
    Ok(transcript)
}

fn poseidon_absorb_bytes<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    mut transcript: AllocatedRecursivePoseidonTranscriptState,
    label: &[u8],
    value: &AllocatedNum<NovaScalar>,
    len: usize,
) -> Result<AllocatedRecursivePoseidonTranscriptState, SynthesisError> {
    let packed = alloc_nova_constant(
        cs.namespace(|| "packed bytes label"),
        packed_label_with_len(label, len)?,
    )?;
    transcript = synthesize_recursive_poseidon_transcript_transition(
        cs.namespace(|| "absorb packed bytes label"),
        &transcript,
        &packed,
    )?;
    synthesize_recursive_poseidon_transcript_transition(
        cs.namespace(|| "absorb bytes value"),
        &transcript,
        value,
    )
}

fn reduce_digest<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    words: &[AllocatedNum<NovaScalar>; 4],
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let two_64 = nova_from_fr(&Fr::from(2u64).pow([64]))?;
    let coefficients = [
        NovaScalar::one(),
        two_64,
        two_64.square(),
        two_64.square() * two_64,
    ];
    let reduced = AllocatedNum::alloc(cs.namespace(|| format!("{label} value")), || {
        words.iter().zip(coefficients).try_fold(
            NovaScalar::zero(),
            |accumulator, (word, coefficient)| {
                Ok::<_, SynthesisError>(
                    accumulator
                        + word.get_value().ok_or(SynthesisError::AssignmentMissing)? * coefficient,
                )
            },
        )
    })?;
    let relation = words.iter().zip(coefficients).fold(
        LinearCombination::<NovaScalar>::zero(),
        |lc, (word, coefficient)| lc + (coefficient, word.get_variable()),
    );
    cs.enforce(
        || format!("{label} relation"),
        |_| relation - reduced.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(reduced)
}

fn eval_poly_at<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    coefficients: &[AllocatedNum<NovaScalar>],
    point: &AllocatedNum<NovaScalar>,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let mut result = coefficients
        .last()
        .ok_or_else(|| SynthesisError::Unsatisfiable("empty univariate polynomial".to_string()))?
        .clone();
    for (index, coefficient) in coefficients[..coefficients.len() - 1]
        .iter()
        .rev()
        .enumerate()
    {
        let product = mul_nums(
            cs.namespace(|| format!("{label} Horner product {index}")),
            &result,
            point,
            "univariate Horner product",
        )?;
        result = add_nums(
            cs.namespace(|| format!("{label} Horner sum {index}")),
            &product,
            coefficient,
            "univariate Horner sum",
        )?;
    }
    Ok(result)
}

fn signed_scalar(value: i128) -> Result<NovaScalar, SynthesisError> {
    let magnitude = value.unsigned_abs();
    let field = Fr::from_le_bytes_mod_order(&magnitude.to_le_bytes());
    let scalar = nova_from_fr(&field)?;
    Ok(if value < 0 { -scalar } else { scalar })
}

fn lc_eval<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    lc: &LC,
    inputs: &[AllocatedNum<NovaScalar>],
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let output = AllocatedNum::alloc(cs.namespace(|| format!("{label} value")), || {
        (0..lc.num_terms()).try_fold(
            signed_scalar(lc.const_term().unwrap_or_default())?,
            |accumulator, index| {
                let term = lc.term(index).ok_or_else(|| {
                    SynthesisError::Unsatisfiable("missing R1CS LC term".to_string())
                })?;
                Ok::<_, SynthesisError>(
                    accumulator
                        + signed_scalar(term.coeff)?
                            * inputs[term.input_index]
                                .get_value()
                                .ok_or(SynthesisError::AssignmentMissing)?,
                )
            },
        )
    })?;
    let mut relation = LinearCombination::<NovaScalar>::zero()
        + (
            signed_scalar(lc.const_term().unwrap_or_default())?,
            CS::one(),
        );
    for index in 0..lc.num_terms() {
        let term = lc
            .term(index)
            .ok_or_else(|| SynthesisError::Unsatisfiable("missing R1CS LC term".to_string()))?;
        relation = relation
            + (
                signed_scalar(term.coeff)?,
                inputs[term.input_index].get_variable(),
            );
    }
    cs.enforce(
        || format!("{label} relation"),
        |_| relation - output.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(output)
}

fn lagrange_basis<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    point: &AllocatedNum<NovaScalar>,
    index: usize,
    n: usize,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let start = -(((n - 1) / 2) as i128);
    let node = start + index as i128;
    let mut result = alloc_nova_constant(cs.namespace(|| "Lagrange one"), NovaScalar::one())?;
    let mut denominator = 1i128;
    for other in 0..n {
        if other == index {
            continue;
        }
        let other_node = start + other as i128;
        denominator *= node - other_node;
        let constant = alloc_nova_constant(
            cs.namespace(|| format!("{label} node {other}")),
            signed_scalar(other_node)?,
        )?;
        let difference = sub_nums(
            cs.namespace(|| format!("{label} difference {other}")),
            point,
            &constant,
            "Lagrange difference",
        )?;
        result = mul_nums(
            cs.namespace(|| format!("{label} product {other}")),
            &result,
            &difference,
            "Lagrange product",
        )?;
    }
    let magnitude = Fr::from_le_bytes_mod_order(&denominator.unsigned_abs().to_le_bytes());
    let exact_inverse = magnitude
        .inverse()
        .ok_or_else(|| SynthesisError::Unsatisfiable("zero Lagrange denominator".to_string()))?;
    let exact_inverse = if denominator < 0 {
        -exact_inverse
    } else {
        exact_inverse
    };
    scale_num(
        cs.namespace(|| format!("{label} denominator inverse")),
        &result,
        nova_from_fr(&exact_inverse)?,
        "Lagrange denominator inverse",
    )
}

fn lagrange_kernel<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    left: &AllocatedNum<NovaScalar>,
    right: &AllocatedNum<NovaScalar>,
    n: usize,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let mut result = alloc_nova_constant(cs.namespace(|| "kernel zero"), NovaScalar::zero())?;
    for index in 0..n {
        let left_basis = lagrange_basis(
            cs.namespace(|| format!("kernel left basis {index}")),
            left,
            index,
            n,
            "kernel left basis",
        )?;
        let right_basis = lagrange_basis(
            cs.namespace(|| format!("kernel right basis {index}")),
            right,
            index,
            n,
            "kernel right basis",
        )?;
        let term = mul_nums(
            cs.namespace(|| format!("kernel term {index}")),
            &left_basis,
            &right_basis,
            "kernel term",
        )?;
        result = add_nums(
            cs.namespace(|| format!("kernel sum {index}")),
            &result,
            &term,
            "kernel sum",
        )?;
    }
    Ok(result)
}

fn weighted_lc_group<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    inputs: &[AllocatedNum<NovaScalar>],
    weights: &[AllocatedNum<NovaScalar>],
    first_group: bool,
    a_side: bool,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let rows = if first_group {
        R1CS_CONSTRAINTS_FIRST_GROUP.as_slice()
    } else {
        R1CS_CONSTRAINTS_SECOND_GROUP.as_slice()
    };
    let mut result = alloc_nova_constant(cs.namespace(|| "weighted LC zero"), NovaScalar::zero())?;
    for (index, row) in rows.iter().enumerate() {
        let lc = if a_side { &row.cons.a } else { &row.cons.b };
        let evaluation = lc_eval(
            cs.namespace(|| format!("R1CS LC row {index}")),
            lc,
            inputs,
            "R1CS LC evaluation",
        )?;
        let term = mul_nums(
            cs.namespace(|| format!("weighted R1CS LC row {index}")),
            &weights[index],
            &evaluation,
            "weighted R1CS LC",
        )?;
        result = add_nums(
            cs.namespace(|| format!("weighted R1CS LC sum {index}")),
            &result,
            &term,
            "weighted R1CS LC sum",
        )?;
    }
    Ok(result)
}

fn cpu_endpoint<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    tau: &[AllocatedNum<NovaScalar>],
    r0: &AllocatedNum<NovaScalar>,
    batching: &AllocatedNum<NovaScalar>,
    challenges: &[AllocatedNum<NovaScalar>],
    inputs: &[AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    if inputs.len() != NUM_R1CS_INPUTS || tau.len() != challenges.len() + 1 || challenges.is_empty()
    {
        return Err(SynthesisError::Unsatisfiable(
            "compact CPU endpoint dimensions are invalid".to_string(),
        ));
    }
    let weights = (0..OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE)
        .map(|index| {
            lagrange_basis(
                cs.namespace(|| format!("CPU R1CS row weight {index}")),
                r0,
                index,
                OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
                "CPU row weight",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let az0 = weighted_lc_group(
        cs.namespace(|| "CPU Az first group"),
        inputs,
        &weights,
        true,
        true,
    )?;
    let bz0 = weighted_lc_group(
        cs.namespace(|| "CPU Bz first group"),
        inputs,
        &weights,
        true,
        false,
    )?;
    let az1 = weighted_lc_group(
        cs.namespace(|| "CPU Az second group"),
        inputs,
        &weights,
        false,
        true,
    )?;
    let bz1 = weighted_lc_group(
        cs.namespace(|| "CPU Bz second group"),
        inputs,
        &weights,
        false,
        false,
    )?;
    let r_stream = &challenges[0];
    let az_delta = sub_nums(
        cs.namespace(|| "CPU Az group delta"),
        &az1,
        &az0,
        "CPU Az group delta",
    )?;
    let bz_delta = sub_nums(
        cs.namespace(|| "CPU Bz group delta"),
        &bz1,
        &bz0,
        "CPU Bz group delta",
    )?;
    let az_blend = mul_nums(
        cs.namespace(|| "CPU Az group blend"),
        r_stream,
        &az_delta,
        "CPU Az group blend",
    )?;
    let bz_blend = mul_nums(
        cs.namespace(|| "CPU Bz group blend"),
        r_stream,
        &bz_delta,
        "CPU Bz group blend",
    )?;
    let az = add_nums(
        cs.namespace(|| "CPU Az final"),
        &az0,
        &az_blend,
        "CPU Az final",
    )?;
    let bz = add_nums(
        cs.namespace(|| "CPU Bz final"),
        &bz0,
        &bz_blend,
        "CPU Bz final",
    )?;
    let inner = mul_nums(cs.namespace(|| "CPU AzBz"), &az, &bz, "CPU AzBz")?;
    let tau_kernel = lagrange_kernel(
        cs.namespace(|| "CPU tau high kernel"),
        tau.last().unwrap(),
        r0,
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    )?;
    let reversed_challenges = challenges.iter().rev().cloned().collect::<Vec<_>>();
    let tau_eq = eq_points(
        cs.namespace(|| "CPU tau tail equality"),
        &tau[..tau.len() - 1],
        &reversed_challenges,
    )?;
    let relation = mul_nums(
        cs.namespace(|| "CPU tau relation"),
        &tau_kernel,
        &tau_eq,
        "CPU tau relation",
    )?;
    let relation = mul_nums(
        cs.namespace(|| "CPU outer endpoint"),
        &relation,
        &inner,
        "CPU outer endpoint",
    )?;
    mul_nums(
        cs.namespace(|| "CPU batched endpoint"),
        &relation,
        batching,
        "CPU batched endpoint",
    )
}

fn cpu_uniskip<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    proof: &super::CpuBlockRelationProof,
    mut transcript: AllocatedRecursivePoseidonTranscriptState,
    tau_len: usize,
) -> Result<
    (
        AllocatedRecursivePoseidonTranscriptState,
        Vec<AllocatedNum<NovaScalar>>,
        AllocatedNum<NovaScalar>,
        AllocatedNum<NovaScalar>,
    ),
    SynthesisError,
> {
    let mut tau = Vec::with_capacity(tau_len);
    for index in 0..tau_len {
        transcript = poseidon_challenge(
            cs.namespace(|| format!("CPU tau challenge {index}")),
            &transcript,
        )?;
        tau.push(transcript.state.clone());
    }
    let mut cursor = Cursor::new(&proof.uniskip_proof);
    let native =
        UniSkipFirstRoundProof::<Fr, PoseidonTranscript>::deserialize_compressed(&mut cursor)
            .map_err(|error| SynthesisError::Unsatisfiable(error.to_string()))?;
    if cursor.position() != proof.uniskip_proof.len() as u64
        || native.uni_poly.coeffs.len() != OUTER_FIRST_ROUND_POLY_NUM_COEFFS
    {
        return Err(SynthesisError::Unsatisfiable(
            "CPU uniskip proof does not have the pinned fixed shape".to_string(),
        ));
    }
    let coefficients = native
        .uni_poly
        .coeffs
        .iter()
        .enumerate()
        .map(|(index, value)| {
            alloc_witness_num(
                cs.namespace(|| format!("CPU uniskip coefficient {index}")),
                nova_from_fr(value)?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    transcript = poseidon_absorb_vector(
        cs.namespace(|| "CPU uniskip polynomial transcript"),
        transcript,
        b"uniskip_poly",
        &coefficients,
    )?;
    transcript = poseidon_challenge(cs.namespace(|| "CPU uniskip challenge"), &transcript)?;
    let r0 = alloc_field(
        cs.namespace(|| "CPU supplied uniskip challenge"),
        proof.uniskip_challenge,
    )?;
    enforce_num_equal(
        cs.namespace(|| "CPU uniskip challenge binding"),
        &transcript.state,
        &r0,
        "CPU uniskip challenge binding",
    );
    let zero = alloc_nova_constant(cs.namespace(|| "CPU uniskip sum zero"), NovaScalar::zero())?;
    let start = -(((OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE - 1) / 2) as i128);
    let mut domain_sum = zero.clone();
    for index in 0..OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE {
        let point = alloc_nova_constant(
            cs.namespace(|| format!("CPU uniskip domain point {index}")),
            signed_scalar(start + index as i128)?,
        )?;
        let evaluation = eval_poly_at(
            cs.namespace(|| format!("CPU uniskip domain evaluation {index}")),
            &coefficients,
            &point,
            "CPU uniskip domain evaluation",
        )?;
        domain_sum = add_nums(
            cs.namespace(|| format!("CPU uniskip domain sum {index}")),
            &domain_sum,
            &evaluation,
            "CPU uniskip domain sum",
        )?;
    }
    enforce_num_equal(
        cs.namespace(|| "CPU uniskip symmetric sum is zero"),
        &domain_sum,
        &zero,
        "CPU uniskip symmetric sum is zero",
    );
    let evaluated = eval_poly_at(
        cs.namespace(|| "CPU uniskip evaluation at r0"),
        &coefficients,
        &r0,
        "CPU uniskip evaluation",
    )?;
    let claim = alloc_field(
        cs.namespace(|| "CPU supplied uniskip claim"),
        proof.uniskip_claim,
    )?;
    enforce_num_equal(
        cs.namespace(|| "CPU uniskip output claim binding"),
        &evaluated,
        &claim,
        "CPU uniskip output claim binding",
    );
    transcript = poseidon_absorb(
        cs.namespace(|| "CPU append uniskip output claim"),
        &transcript,
        &claim,
        "opening_claim",
    )?;
    Ok((transcript, tau, r0, claim))
}

fn canonical_deferred_claims(proof: &BlockJoltProof) -> Vec<DeferredPcsClaim> {
    let mut claims = compact_lookup_deferred_claims(
        proof.lookup.query_commitment,
        &proof.lookup.reduction_point,
        &proof.lookup.input_claims,
        &proof.lookup.sumcheck.challenges,
        &proof.lookup.output_claims,
    );
    claims.extend(compact_register_deferred_claims(
        proof.register.access_commitment,
        &proof.register.reduction_point,
        &proof.register.input_claims,
        &proof.register.sumcheck.challenges,
        &proof.register.output_claims,
    ));
    claims.extend(ram_deferred_claims(
        proof.ram.access_commitment,
        &proof.ram.reduction_point,
        &proof.ram.input_claims,
        &proof.ram.relation_proof.challenges,
        &proof.ram.output_claims,
    ));
    let cpu_openings = proof
        .cpu
        .relation_proof
        .opening_claims
        .iter()
        .map(|claim| (claim.opening_point.clone(), claim.claimed_value))
        .collect::<Vec<_>>();
    claims.extend(cpu_deferred_claims(proof.cpu.row_commitment, &cpu_openings));
    claims
}

/// Fixed-shape D15 relation. All fields are private so production callers can
/// only construct a step after structural validation of the compact artifact.
#[derive(Clone)]
pub struct BlockJoltVerifierStepCircuit {
    config: BlockJoltHostConfig,
    statement: BlockJoltStatement,
    proof: BlockJoltProof,
}

impl BlockJoltVerifierStepCircuit {
    pub fn new(
        config: BlockJoltHostConfig,
        statement: BlockJoltStatement,
        proof: BlockJoltProof,
    ) -> Result<Self, String> {
        config.validate().map_err(|error| error.to_string())?;
        proof.validate_structure(&statement)?;
        if statement.cycle_capacity != config.cycle_capacity as u64
            || proof.ram.ram_k != config.ram_k as u64
            || proof.lookup.accumulator_round_before != statement.lookup_transcript_round_before
            || proof.lookup.accumulator_round_after != statement.lookup_transcript_round_after
        {
            return Err("compact verifier statement/configuration mismatch".to_string());
        }
        let canonical = canonical_deferred_claims(&proof);
        if proof.deferred_pcs_claims != canonical {
            return Err("compact verifier deferred PCS sequence is not canonical".to_string());
        }
        Ok(Self {
            config,
            statement,
            proof,
        })
    }

    pub fn initial_z(&self) -> Result<Vec<NovaScalar>, String> {
        if self.statement.block_index != 0 || self.statement.global_cycle_start != 0 {
            return Err("D15 initial state requires block and cycle zero".to_string());
        }
        let deferred = PoseidonTranscript::new(DEFERRED_TRANSCRIPT_DOMAIN);
        let mut z = vec![NovaScalar::zero(); BLOCK_JOLT_VERIFIER_Z_ARITY];
        z[WIRE_SLOT] = NovaScalar::from(BLOCK_JOLT_WIRE_VERSION as u64);
        z[PREPROCESSING_OFFSET..PREPROCESSING_OFFSET + 4]
            .copy_from_slice(&digest_scalars(&self.statement.preprocessing_id));
        z[PROGRAM_OFFSET..PROGRAM_OFFSET + 4]
            .copy_from_slice(&digest_scalars(&self.statement.program_digest));
        z[TABLE_OFFSET..TABLE_OFFSET + 4]
            .copy_from_slice(&digest_scalars(&self.statement.lookup_table_commitment));
        z[TABLE_REDUCED_SLOT] =
            nova_from_fr(&field_from_digest(&self.statement.lookup_table_commitment))
                .map_err(|error| format!("table commitment field conversion failed: {error:?}"))?;
        z[BYTECODE_SLOT] = nova_from_fr(&self.statement.bytecode_commitment.to_fr())
            .map_err(|error| format!("bytecode field conversion failed: {error:?}"))?;
        z[BLOCK_SLOT] = NovaScalar::zero();
        z[CYCLE_SLOT] = NovaScalar::zero();
        z[MACHINE_OFFSET..MACHINE_OFFSET + 4]
            .copy_from_slice(&digest_scalars(&self.statement.start.machine_state));
        z[REGISTER_OFFSET..REGISTER_OFFSET + 4]
            .copy_from_slice(&digest_scalars(&self.statement.start.register_state));
        z[RAM_ROOT_SLOT] = nova_from_fr(&self.statement.start.ram_root.to_fr())
            .map_err(|error| format!("RAM root field conversion failed: {error:?}"))?;
        z[LOOKUP_STATE_SLOT] = nova_from_fr(&self.statement.lookup_accumulator_before.to_fr())
            .map_err(|error| format!("lookup state field conversion failed: {error:?}"))?;
        z[LOOKUP_ROUND_SLOT] = NovaScalar::from(self.statement.lookup_transcript_round_before);
        z[DEFERRED_STATE_SLOT] = nova_from_fr(&field_from_digest(&deferred.state))
            .map_err(|error| format!("deferred state conversion failed: {error:?}"))?;
        z[DEFERRED_ROUND_SLOT] = NovaScalar::from(deferred.n_rounds as u64);
        z[TOTAL_CYCLES_SLOT] = NovaScalar::zero();
        z[TERMINATED_SLOT] = NovaScalar::zero();
        Ok(z)
    }

    fn synthesize_lookup<CS: ConstraintSystem<NovaScalar>>(
        &self,
        mut cs: CS,
        z: &[AllocatedNum<NovaScalar>],
        active: &AllocatedNum<NovaScalar>,
    ) -> Result<
        (
            AllocatedRecursivePoseidonTranscriptState,
            Vec<AllocatedDeferredClaim>,
        ),
        SynthesisError,
    > {
        let proof = &self.proof.lookup;
        let log_t = self.config.cycle_capacity.ilog2() as usize;
        let before = alloc_field(
            cs.namespace(|| "lookup supplied state before"),
            proof.accumulator_before,
        )?;
        enforce_num_equal(
            cs.namespace(|| "lookup state before continuity"),
            &before,
            &z[LOOKUP_STATE_SLOT],
            "lookup state before continuity",
        );
        let before_round = alloc_u64_bits(
            cs.namespace(|| "lookup supplied round before"),
            proof.accumulator_round_before,
            "lookup round before",
        )?
        .0;
        enforce_num_equal(
            cs.namespace(|| "lookup round before continuity"),
            &before_round,
            &z[LOOKUP_ROUND_SLOT],
            "lookup round before continuity",
        );
        let mut transcript = AllocatedRecursivePoseidonTranscriptState {
            state: before,
            n_rounds: before_round,
        };
        let query = alloc_field(
            cs.namespace(|| "lookup query commitment"),
            proof.query_commitment,
        )?;
        for (label, value) in [
            ("table_root", &z[TABLE_REDUCED_SLOT]),
            ("block_index", &z[BLOCK_SLOT]),
            ("cycle_start", &z[CYCLE_SLOT]),
            ("active_cycles", active),
            ("query_root", &query),
        ] {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("lookup header {label}")),
                &transcript,
                value,
                label,
            )?;
        }
        let (after_reduction, reduction) = derive_challenges(
            cs.namespace(|| "lookup reduction challenges"),
            transcript,
            &proof.reduction_point,
            "lookup reduction",
        )?;
        if reduction.len() != log_t {
            return Err(SynthesisError::Unsatisfiable(
                "lookup reduction point has the wrong dimension".to_string(),
            ));
        }
        transcript = poseidon_challenge(cs.namespace(|| "lookup gamma"), &after_reduction)?;
        let gamma = alloc_field(cs.namespace(|| "lookup supplied gamma"), proof.gamma)?;
        enforce_num_equal(
            cs.namespace(|| "lookup gamma binding"),
            &transcript.state,
            &gamma,
            "lookup gamma binding",
        );
        let inputs = alloc_fields(
            cs.namespace(|| "lookup input claims"),
            &proof.input_claims,
            "lookup input claim",
        )?;
        let stage_witness = compact_sumcheck_witness(&proof.sumcheck, 0)?;
        let stage = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "lookup sumcheck arithmetic"),
            &stage_witness,
            0,
            LOG_K + log_t,
            proof.sumcheck.degree_bound as usize,
        )?;
        let (sumcheck_transcript, batching) = initial_and_batching_claim(
            cs.namespace(|| "lookup input relation"),
            transcript,
            &inputs,
            &gamma,
            proof.batching_coefficient,
            &stage,
        )?;
        transcript = synthesize_recursive_clear_sumcheck_transcript(
            cs.namespace(|| "lookup sumcheck transcript"),
            &sumcheck_transcript,
            &stage,
        )?;
        let outputs = alloc_fields(
            cs.namespace(|| "lookup output claims"),
            &proof.output_claims,
            "lookup output claim",
        )?;
        let one_hot = OneHotParams::new(log_t, 1, 1);
        let endpoint = lookup_endpoint(
            cs.namespace(|| "lookup endpoint relation"),
            &reduction,
            &gamma,
            &batching,
            &stage.challenges,
            &outputs,
            &one_hot,
        )?;
        enforce_num_equal(
            cs.namespace(|| "lookup endpoint binding"),
            &stage.final_claim,
            &endpoint,
            "lookup endpoint binding",
        );
        transcript = append_opening_claims(
            cs.namespace(|| "lookup reduction opening transcript"),
            transcript,
            &inputs,
            "lookup reduction",
        )?;
        transcript = append_opening_claims(
            cs.namespace(|| "lookup output opening transcript"),
            transcript,
            &outputs,
            "lookup output",
        )?;
        let after = alloc_field(
            cs.namespace(|| "lookup supplied state after"),
            proof.accumulator_after,
        )?;
        enforce_num_equal(
            cs.namespace(|| "lookup state after binding"),
            &transcript.state,
            &after,
            "lookup state after binding",
        );
        let after_round = alloc_u64_bits(
            cs.namespace(|| "lookup supplied round after"),
            proof.accumulator_round_after,
            "lookup round after",
        )?
        .0;
        enforce_num_equal(
            cs.namespace(|| "lookup round after binding"),
            &transcript.n_rounds,
            &after_round,
            "lookup round after binding",
        );
        let canonical = compact_lookup_deferred_claims(
            proof.query_commitment,
            &proof.reduction_point,
            &proof.input_claims,
            &proof.sumcheck.challenges,
            &proof.output_claims,
        );
        let values = inputs
            .iter()
            .chain(outputs.iter())
            .cloned()
            .collect::<Vec<_>>();
        let claims = allocated_claims_from_canonical(
            cs.namespace(|| "lookup deferred claims"),
            &canonical,
            &values,
            "lookup deferred",
        )?;
        Ok((transcript, claims))
    }

    fn synthesize_register<CS: ConstraintSystem<NovaScalar>>(
        &self,
        mut cs: CS,
        z: &[AllocatedNum<NovaScalar>],
        active: &AllocatedNum<NovaScalar>,
    ) -> Result<Vec<AllocatedDeferredClaim>, SynthesisError> {
        let proof = &self.proof.register;
        let log_t = self.config.cycle_capacity.ilog2() as usize;
        let mut transcript = allocated_constant_transcript(
            cs.namespace(|| "register transcript initial"),
            new_block_register_transcript(),
        )?;
        let access = alloc_field(
            cs.namespace(|| "register access commitment"),
            proof.access_commitment,
        )?;
        for (label, value) in [
            ("block_index", &z[BLOCK_SLOT]),
            ("cycle_start", &z[CYCLE_SLOT]),
            ("active_cycles", active),
            ("register_query_root", &access),
        ] {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("register header {label}")),
                &transcript,
                value,
                label,
            )?;
        }
        let (after_reduction, reduction) = derive_challenges(
            cs.namespace(|| "register reduction challenges"),
            transcript,
            &proof.reduction_point,
            "register reduction",
        )?;
        if reduction.len() != log_t {
            return Err(SynthesisError::Unsatisfiable(
                "register reduction point has the wrong dimension".to_string(),
            ));
        }
        transcript = poseidon_challenge(cs.namespace(|| "register gamma"), &after_reduction)?;
        let gamma = alloc_field(cs.namespace(|| "register supplied gamma"), proof.gamma)?;
        enforce_num_equal(
            cs.namespace(|| "register gamma binding"),
            &transcript.state,
            &gamma,
            "register gamma binding",
        );
        let inputs = alloc_fields(
            cs.namespace(|| "register input claims"),
            &proof.input_claims,
            "register input claim",
        )?;
        let stage_witness = compact_sumcheck_witness(&proof.sumcheck, 1)?;
        let stage = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "register sumcheck arithmetic"),
            &stage_witness,
            1,
            log_t + (REGISTER_COUNT as usize).ilog2() as usize,
            3,
        )?;
        let (sumcheck_transcript, batching) = initial_and_batching_claim(
            cs.namespace(|| "register input relation"),
            transcript,
            &inputs,
            &gamma,
            proof.batching_coefficient,
            &stage,
        )?;
        transcript = synthesize_recursive_clear_sumcheck_transcript(
            cs.namespace(|| "register sumcheck transcript"),
            &sumcheck_transcript,
            &stage,
        )?;
        let outputs = alloc_fields(
            cs.namespace(|| "register output claims"),
            &proof.output_claims,
            "register output claim",
        )?;
        let endpoint = register_endpoint(
            cs.namespace(|| "register endpoint relation"),
            log_t,
            &reduction,
            &gamma,
            &batching,
            &stage.challenges,
            &outputs,
        )?;
        enforce_num_equal(
            cs.namespace(|| "register endpoint binding"),
            &stage.final_claim,
            &endpoint,
            "register endpoint binding",
        );
        transcript = append_opening_claims(
            cs.namespace(|| "register reduction openings"),
            transcript,
            &inputs,
            "register reduction",
        )?;
        let _ = append_opening_claims(
            cs.namespace(|| "register output openings"),
            transcript,
            &outputs,
            "register output",
        )?;
        let canonical = compact_register_deferred_claims(
            proof.access_commitment,
            &proof.reduction_point,
            &proof.input_claims,
            &proof.sumcheck.challenges,
            &proof.output_claims,
        );
        let values = inputs
            .iter()
            .chain(outputs.iter())
            .cloned()
            .collect::<Vec<_>>();
        allocated_claims_from_canonical(
            cs.namespace(|| "register deferred claims"),
            &canonical,
            &values,
            "register deferred",
        )
    }

    fn synthesize_ram<CS: ConstraintSystem<NovaScalar>>(
        &self,
        mut cs: CS,
        z: &[AllocatedNum<NovaScalar>],
        active: &AllocatedNum<NovaScalar>,
    ) -> Result<Vec<AllocatedDeferredClaim>, SynthesisError> {
        let proof = &self.proof.ram;
        let log_t = self.config.cycle_capacity.ilog2() as usize;
        let log_k = self.config.ram_k.ilog2() as usize;
        let mut transcript = allocated_constant_transcript(
            cs.namespace(|| "RAM transcript initial"),
            new_block_ram_transcript(),
        )?;
        let ram_k = alloc_nova_constant(
            cs.namespace(|| "RAM K"),
            NovaScalar::from(self.config.ram_k as u64),
        )?;
        let access = alloc_field(
            cs.namespace(|| "RAM access commitment"),
            proof.access_commitment,
        )?;
        let registry = alloc_field(cs.namespace(|| "RAM registry root"), proof.registry_root)?;
        let root_before = alloc_field(cs.namespace(|| "RAM root before"), proof.root_before)?;
        enforce_num_equal(
            cs.namespace(|| "RAM root before continuity"),
            &root_before,
            &z[RAM_ROOT_SLOT],
            "RAM root before continuity",
        );
        let root_after = alloc_field(cs.namespace(|| "RAM root after"), proof.root_after)?;
        for (label, value) in [
            ("block_index", &z[BLOCK_SLOT]),
            ("cycle_start", &z[CYCLE_SLOT]),
            ("active_cycles", active),
            ("ram_k", &ram_k),
            ("access_commitment", &access),
            ("registry_root", &registry),
            ("root_before", &root_before),
            ("root_after", &root_after),
        ] {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("RAM header {label}")),
                &transcript,
                value,
                label,
            )?;
        }
        let (after_reduction, reduction) = derive_challenges(
            cs.namespace(|| "RAM reduction challenges"),
            transcript,
            &proof.reduction_point,
            "RAM reduction",
        )?;
        if reduction.len() != log_t {
            return Err(SynthesisError::Unsatisfiable(
                "RAM reduction point has the wrong dimension".to_string(),
            ));
        }
        transcript = poseidon_challenge(cs.namespace(|| "RAM gamma"), &after_reduction)?;
        let gamma = alloc_field(cs.namespace(|| "RAM supplied gamma"), proof.gamma)?;
        enforce_num_equal(
            cs.namespace(|| "RAM gamma binding"),
            &transcript.state,
            &gamma,
            "RAM gamma binding",
        );
        let inputs = alloc_fields(
            cs.namespace(|| "RAM input claims"),
            &proof.input_claims,
            "RAM input claim",
        )?;
        let stage_witness = compact_sumcheck_witness(&proof.relation_proof, 2)?;
        let stage = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "RAM sumcheck arithmetic"),
            &stage_witness,
            2,
            log_t + log_k,
            3,
        )?;
        let (sumcheck_transcript, batching) = initial_and_batching_claim(
            cs.namespace(|| "RAM input relation"),
            transcript,
            &inputs,
            &gamma,
            proof.batching_coefficient,
            &stage,
        )?;
        transcript = synthesize_recursive_clear_sumcheck_transcript(
            cs.namespace(|| "RAM sumcheck transcript"),
            &sumcheck_transcript,
            &stage,
        )?;
        let outputs = alloc_fields(
            cs.namespace(|| "RAM output claims"),
            &proof.output_claims,
            "RAM output claim",
        )?;
        let endpoint = ram_endpoint(
            cs.namespace(|| "RAM endpoint relation"),
            log_t,
            &reduction,
            &gamma,
            &batching,
            &stage.challenges,
            &outputs,
        )?;
        enforce_num_equal(
            cs.namespace(|| "RAM endpoint binding"),
            &stage.final_claim,
            &endpoint,
            "RAM endpoint binding",
        );
        transcript = append_opening_claims(
            cs.namespace(|| "RAM reduction openings"),
            transcript,
            &inputs,
            "RAM reduction",
        )?;
        let _ = append_opening_claims(
            cs.namespace(|| "RAM output openings"),
            transcript,
            &outputs,
            "RAM output",
        )?;
        let canonical = ram_deferred_claims(
            proof.access_commitment,
            &proof.reduction_point,
            &proof.input_claims,
            &proof.relation_proof.challenges,
            &proof.output_claims,
        );
        let values = inputs
            .iter()
            .chain(outputs.iter())
            .cloned()
            .collect::<Vec<_>>();
        allocated_claims_from_canonical(
            cs.namespace(|| "RAM deferred claims"),
            &canonical,
            &values,
            "RAM deferred",
        )
    }

    fn synthesize_cpu<CS: ConstraintSystem<NovaScalar>>(
        &self,
        mut cs: CS,
        z: &[AllocatedNum<NovaScalar>],
        active: &AllocatedNum<NovaScalar>,
        terminal: &AllocatedNum<NovaScalar>,
    ) -> Result<Vec<AllocatedDeferredClaim>, SynthesisError> {
        let proof = &self.proof.cpu;
        let log_t = self.config.cycle_capacity.ilog2() as usize;
        if proof.relation_proof.opening_claims.len() != NUM_R1CS_INPUTS {
            return Err(SynthesisError::Unsatisfiable(
                "CPU opening-claim count does not match the pinned R1CS shape".to_string(),
            ));
        }

        let mut transcript = allocated_constant_transcript(
            cs.namespace(|| "CPU transcript initial"),
            new_block_cpu_transcript(),
        )?;
        let protocol = alloc_nova_constant(
            cs.namespace(|| "CPU protocol bytes"),
            nova_from_fr(&Fr::from_le_bytes_mod_order(
                super::BLOCK_JOLT_PROTOCOL_VERSION.as_bytes(),
            ))?,
        )?;
        transcript = poseidon_absorb_bytes(
            cs.namespace(|| "CPU protocol transcript"),
            transcript,
            b"protocol",
            &protocol,
            super::BLOCK_JOLT_PROTOCOL_VERSION.len(),
        )?;
        let capacity = alloc_nova_constant(
            cs.namespace(|| "CPU cycle capacity"),
            NovaScalar::from(self.config.cycle_capacity as u64),
        )?;
        let start_pc = alloc_u64_bits(
            cs.namespace(|| "CPU start PC"),
            proof.start_pc,
            "CPU start PC",
        )?
        .0;
        let end_pc = alloc_u64_bits(cs.namespace(|| "CPU end PC"), proof.end_pc, "CPU end PC")?.0;
        for (label, value) in [
            ("block_index", &z[BLOCK_SLOT]),
            ("cycle_start", &z[CYCLE_SLOT]),
            ("active_cycles", active),
            ("cycle_capacity", &capacity),
            ("start_pc", &start_pc),
            ("end_pc", &end_pc),
            ("terminal", terminal),
        ] {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("CPU header {label}")),
                &transcript,
                value,
                label,
            )?;
        }
        let row_words = alloc_digest(
            cs.namespace(|| "CPU row commitment words"),
            &proof.row_commitment,
            "CPU row commitment",
        )?;
        let row_commitment = reduce_digest(
            cs.namespace(|| "CPU row commitment reduction"),
            &row_words,
            "CPU row commitment reduction",
        )?;
        transcript = poseidon_absorb_bytes(
            cs.namespace(|| "CPU row commitment transcript"),
            transcript,
            b"cpu_row_commitment",
            &row_commitment,
            32,
        )?;
        let bytecode = alloc_field(
            cs.namespace(|| "CPU supplied bytecode root"),
            proof.bytecode_root,
        )?;
        enforce_num_equal(
            cs.namespace(|| "CPU bytecode root binding"),
            &bytecode,
            &z[BYTECODE_SLOT],
            "CPU bytecode root binding",
        );
        transcript = poseidon_absorb(
            cs.namespace(|| "CPU bytecode root transcript"),
            &transcript,
            &bytecode,
            "bytecode_root",
        )?;

        let (transcript_after_uniskip, tau, r0, uniskip_claim) = cpu_uniskip(
            cs.namespace(|| "CPU uniskip verifier"),
            proof,
            transcript,
            log_t + 2,
        )?;
        let stage_witness = compact_sumcheck_witness(&proof.relation_proof, 3)?;
        let stage = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "CPU remaining sumcheck arithmetic"),
            &stage_witness,
            3,
            log_t + 1,
            3,
        )?;
        let mut sumcheck_transcript = poseidon_absorb(
            cs.namespace(|| "CPU remaining input claim"),
            &transcript_after_uniskip,
            &uniskip_claim,
            "sumcheck_claim",
        )?;
        sumcheck_transcript = poseidon_challenge(
            cs.namespace(|| "CPU remaining batching challenge"),
            &sumcheck_transcript,
        )?;
        let batching = alloc_field(
            cs.namespace(|| "CPU supplied batching coefficient"),
            proof.batching_coefficient,
        )?;
        enforce_num_equal(
            cs.namespace(|| "CPU batching coefficient binding"),
            &sumcheck_transcript.state,
            &batching,
            "CPU batching coefficient binding",
        );
        let expected_initial = mul_nums(
            cs.namespace(|| "CPU batched initial claim"),
            &uniskip_claim,
            &batching,
            "CPU batched initial claim",
        )?;
        enforce_num_equal(
            cs.namespace(|| "CPU remaining initial claim binding"),
            &stage.initial_claim,
            &expected_initial,
            "CPU remaining initial claim binding",
        );
        let transcript = synthesize_recursive_clear_sumcheck_transcript(
            cs.namespace(|| "CPU remaining sumcheck transcript"),
            &sumcheck_transcript,
            &stage,
        )?;
        let openings = proof
            .relation_proof
            .opening_claims
            .iter()
            .enumerate()
            .map(|(index, claim)| {
                alloc_field(
                    cs.namespace(|| format!("CPU R1CS opening {index}")),
                    claim.claimed_value,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let endpoint = cpu_endpoint(
            cs.namespace(|| "CPU endpoint relation"),
            &tau,
            &r0,
            &batching,
            &stage.challenges,
            &openings,
        )?;
        enforce_num_equal(
            cs.namespace(|| "CPU endpoint binding"),
            &stage.final_claim,
            &endpoint,
            "CPU endpoint binding",
        );
        let _ = append_opening_claims(
            cs.namespace(|| "CPU R1CS opening transcript"),
            transcript,
            &openings,
            "CPU R1CS",
        )?;
        let native_openings = proof
            .relation_proof
            .opening_claims
            .iter()
            .map(|claim| (claim.opening_point.clone(), claim.claimed_value))
            .collect::<Vec<_>>();
        let canonical = cpu_deferred_claims(proof.row_commitment, &native_openings);
        allocated_claims_from_canonical(
            cs.namespace(|| "CPU deferred claims"),
            &canonical,
            &openings,
            "CPU deferred",
        )
    }
}

impl StepCircuit<NovaScalar> for BlockJoltVerifierStepCircuit {
    fn arity(&self) -> usize {
        BLOCK_JOLT_VERIFIER_Z_ARITY
    }

    fn synthesize<CS: ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<NovaScalar>],
    ) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
        if z.len() != BLOCK_JOLT_VERIFIER_Z_ARITY {
            return Err(SynthesisError::Unsatisfiable(
                "compact block-Jolt verifier state has the wrong arity".to_string(),
            ));
        }
        let wire = alloc_nova_constant(
            cs.namespace(|| "block-Jolt wire version"),
            NovaScalar::from(BLOCK_JOLT_WIRE_VERSION as u64),
        )?;
        enforce_num_equal(
            cs.namespace(|| "wire-version continuity"),
            &wire,
            &z[WIRE_SLOT],
            "wire-version continuity",
        );
        let preprocessing = alloc_digest(
            cs.namespace(|| "preprocessing identity"),
            &self.statement.preprocessing_id,
            "preprocessing identity",
        )?;
        enforce_digest_equal_to_z(
            cs.namespace(|| "preprocessing continuity"),
            &preprocessing,
            z,
            PREPROCESSING_OFFSET,
            "preprocessing continuity",
        );
        let program = alloc_digest(
            cs.namespace(|| "program digest"),
            &self.statement.program_digest,
            "program digest",
        )?;
        enforce_digest_equal_to_z(
            cs.namespace(|| "program continuity"),
            &program,
            z,
            PROGRAM_OFFSET,
            "program continuity",
        );
        let table = alloc_digest(
            cs.namespace(|| "lookup table commitment"),
            &self.statement.lookup_table_commitment,
            "lookup table commitment",
        )?;
        enforce_digest_equal_to_z(
            cs.namespace(|| "lookup table continuity"),
            &table,
            z,
            TABLE_OFFSET,
            "lookup table continuity",
        );
        let table_reduced = reduce_digest(
            cs.namespace(|| "lookup table reduction"),
            &table,
            "lookup table reduction",
        )?;
        enforce_num_equal(
            cs.namespace(|| "lookup table reduced continuity"),
            &table_reduced,
            &z[TABLE_REDUCED_SLOT],
            "lookup table reduced continuity",
        );
        let bytecode = alloc_field(
            cs.namespace(|| "statement bytecode root"),
            self.statement.bytecode_commitment,
        )?;
        enforce_num_equal(
            cs.namespace(|| "bytecode-root continuity"),
            &bytecode,
            &z[BYTECODE_SLOT],
            "bytecode-root continuity",
        );

        let block = alloc_u64_bits(
            cs.namespace(|| "statement block index"),
            self.statement.block_index,
            "statement block index",
        )?
        .0;
        enforce_num_equal(
            cs.namespace(|| "block-index continuity"),
            &block,
            &z[BLOCK_SLOT],
            "block-index continuity",
        );
        let cycle_start = alloc_u64_bits(
            cs.namespace(|| "statement cycle start"),
            self.statement.global_cycle_start,
            "statement cycle start",
        )?
        .0;
        enforce_num_equal(
            cs.namespace(|| "cycle-start continuity"),
            &cycle_start,
            &z[CYCLE_SLOT],
            "cycle-start continuity",
        );
        let active = alloc_u64_bits(
            cs.namespace(|| "statement active cycles"),
            self.statement.active_cycles,
            "statement active cycles",
        )?
        .0;
        let active_selectors = (1..=self.config.cycle_capacity)
            .map(|candidate| {
                AllocatedBit::alloc(
                    cs.namespace(|| format!("active cycles equals {candidate}")),
                    Some(self.statement.active_cycles as usize == candidate),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let selector_sum = active_selectors
            .iter()
            .fold(LinearCombination::<NovaScalar>::zero(), |lc, selector| {
                lc + selector.get_variable()
            });
        cs.enforce(
            || "active-cycle selector is one-hot",
            |_| selector_sum - CS::one(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        let selected_active = active_selectors.iter().enumerate().fold(
            LinearCombination::<NovaScalar>::zero(),
            |lc, (index, selector)| {
                lc + (
                    NovaScalar::from((index + 1) as u64),
                    selector.get_variable(),
                )
            },
        );
        cs.enforce(
            || "active cycles are in the fixed-capacity range",
            |_| selected_active - active.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        let cycle_end = alloc_u64_bits(
            cs.namespace(|| "statement cycle end"),
            self.statement.global_cycle_end,
            "statement cycle end",
        )?
        .0;
        let computed_cycle_end = add_nums(
            cs.namespace(|| "computed cycle end"),
            &cycle_start,
            &active,
            "computed cycle end",
        )?;
        enforce_num_equal(
            cs.namespace(|| "cycle interval binding"),
            &computed_cycle_end,
            &cycle_end,
            "cycle interval binding",
        );
        let terminal_bit = AllocatedBit::alloc(
            cs.namespace(|| "statement terminal bit"),
            Some(self.statement.terminal),
        )?;
        let terminal = bit_as_num(&terminal_bit);
        let zero = alloc_nova_constant(cs.namespace(|| "state zero"), NovaScalar::zero())?;
        enforce_num_equal(
            cs.namespace(|| "no block follows termination"),
            &z[TERMINATED_SLOT],
            &zero,
            "no block follows termination",
        );

        let machine_start = alloc_digest(
            cs.namespace(|| "machine state before"),
            &self.statement.start.machine_state,
            "machine state before",
        )?;
        enforce_digest_equal_to_z(
            cs.namespace(|| "machine-state continuity"),
            &machine_start,
            z,
            MACHINE_OFFSET,
            "machine-state continuity",
        );
        let machine_end = alloc_digest(
            cs.namespace(|| "machine state after"),
            &self.statement.end.machine_state,
            "machine state after",
        )?;
        let register_start = alloc_digest(
            cs.namespace(|| "register state before"),
            &self.proof.register.state_before,
            "register state before",
        )?;
        enforce_digest_equal_to_z(
            cs.namespace(|| "register-state continuity"),
            &register_start,
            z,
            REGISTER_OFFSET,
            "register-state continuity",
        );
        let register_end = alloc_digest(
            cs.namespace(|| "register state after"),
            &self.proof.register.state_after,
            "register state after",
        )?;

        let (lookup_after, mut deferred_claims) =
            self.synthesize_lookup(cs.namespace(|| "compact lookup verifier"), z, &active)?;
        deferred_claims.extend(self.synthesize_register(
            cs.namespace(|| "compact register verifier"),
            z,
            &active,
        )?);
        deferred_claims.extend(self.synthesize_ram(
            cs.namespace(|| "compact RAM verifier"),
            z,
            &active,
        )?);
        deferred_claims.extend(self.synthesize_cpu(
            cs.namespace(|| "compact CPU verifier"),
            z,
            &active,
            &terminal,
        )?);
        if deferred_claims.len() != self.proof.deferred_pcs_claims.len() {
            return Err(SynthesisError::Unsatisfiable(
                "compact verifier deferred-claim shape changed".to_string(),
            ));
        }
        let deferred_before = AllocatedRecursivePoseidonTranscriptState {
            state: z[DEFERRED_STATE_SLOT].clone(),
            n_rounds: z[DEFERRED_ROUND_SLOT].clone(),
        };
        let deferred_after = absorb_deferred_claims(
            cs.namespace(|| "deferred PCS accumulator"),
            deferred_before,
            &deferred_claims,
        )?;
        let one = alloc_nova_constant(cs.namespace(|| "block increment"), NovaScalar::one())?;
        let next_block = add_nums(
            cs.namespace(|| "next block index"),
            &block,
            &one,
            "next block index",
        )?;
        let total_cycles = add_nums(
            cs.namespace(|| "total active cycles"),
            &z[TOTAL_CYCLES_SLOT],
            &active,
            "total active cycles",
        )?;
        let ram_after = alloc_field(
            cs.namespace(|| "RAM state after"),
            self.proof.ram.root_after,
        )?;

        let mut output = vec![zero; BLOCK_JOLT_VERIFIER_Z_ARITY];
        output[WIRE_SLOT] = wire;
        output[PREPROCESSING_OFFSET..PREPROCESSING_OFFSET + 4].clone_from_slice(&preprocessing);
        output[PROGRAM_OFFSET..PROGRAM_OFFSET + 4].clone_from_slice(&program);
        output[TABLE_OFFSET..TABLE_OFFSET + 4].clone_from_slice(&table);
        output[TABLE_REDUCED_SLOT] = table_reduced;
        output[BYTECODE_SLOT] = bytecode;
        output[BLOCK_SLOT] = next_block;
        output[CYCLE_SLOT] = cycle_end;
        output[MACHINE_OFFSET..MACHINE_OFFSET + 4].clone_from_slice(&machine_end);
        output[REGISTER_OFFSET..REGISTER_OFFSET + 4].clone_from_slice(&register_end);
        output[RAM_ROOT_SLOT] = ram_after;
        output[LOOKUP_STATE_SLOT] = lookup_after.state;
        output[LOOKUP_ROUND_SLOT] = lookup_after.n_rounds;
        output[DEFERRED_STATE_SLOT] = deferred_after.state;
        output[DEFERRED_ROUND_SLOT] = deferred_after.n_rounds;
        output[TOTAL_CYCLES_SLOT] = total_cycles;
        output[TERMINATED_SLOT] = terminal;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use common::constants::REGISTER_COUNT;
    use nova_snark::frontend::{
        num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem,
    };
    use tracer::{
        instruction::{
            and::AND,
            format::format_r::{FormatR, RegisterStateFormatR},
            Cycle, RISCVCycle,
        },
        MachineBoundaryState, TraceBlock,
    };

    use super::*;
    use crate::zkvm::block::{BlockJoltProver, DirectChunkedPreprocessing};

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
            b"d15-compact-verifier",
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

    fn transition() -> super::super::VerifiedBlockJoltTransition {
        let (preprocessing, block, lookahead) = fixture();
        let mut prover = BlockJoltProver::new(preprocessing, config(), BTreeMap::new()).unwrap();
        prover.prove_block(&block, Some(&lookahead)).unwrap()
    }

    fn synthesize(
        circuit: &BlockJoltVerifierStepCircuit,
    ) -> Result<TestConstraintSystem<NovaScalar>, SynthesisError> {
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z = circuit
            .initial_z()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("z {index}")), || Ok(value))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = circuit.synthesize(&mut cs, &z)?;
        assert_eq!(output.len(), BLOCK_JOLT_VERIFIER_Z_ARITY);
        Ok(cs)
    }

    #[test]
    fn d15_compact_verifier_accepts_real_m1_relation_messages() {
        let transition = transition();
        let circuit =
            BlockJoltVerifierStepCircuit::new(config(), transition.statement, transition.proof)
                .unwrap();
        let cs = synthesize(&circuit).unwrap();
        assert!(
            cs.is_satisfied(),
            "D15 compact verifier failed at {:?}",
            cs.which_is_unsatisfied()
        );
    }

    #[test]
    fn d15_compact_verifier_rejects_sumcheck_and_state_tampering() {
        let transition = transition();
        let mut bad_proof = transition.proof.clone();
        bad_proof.lookup.sumcheck.final_claim.0[0] ^= 1;
        bad_proof.seal();
        let bad_circuit =
            BlockJoltVerifierStepCircuit::new(config(), transition.statement.clone(), bad_proof)
                .unwrap();
        match synthesize(&bad_circuit) {
            Ok(cs) => assert!(!cs.is_satisfied()),
            Err(_) => {}
        }

        let honest =
            BlockJoltVerifierStepCircuit::new(config(), transition.statement, transition.proof)
                .unwrap();
        let mut z = honest.initial_z().unwrap();
        z[LOOKUP_ROUND_SLOT] += NovaScalar::one();
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let allocated = z
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("bad z {index}")), || Ok(value))
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        honest.synthesize(&mut cs, &allocated).unwrap();
        assert!(!cs.is_satisfied());
    }
}
