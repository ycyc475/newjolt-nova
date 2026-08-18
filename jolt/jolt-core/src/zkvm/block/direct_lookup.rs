//! Block-native lookup relation for the direct Jolt-Nova path.
//!
//! Jolt's current lookup implementation is the sparse-dense Shout/Twist
//! `InstructionReadRaf` relation.  This module invokes that real prover and
//! verifier directly on each fixed-capacity trace block.  No global Jolt proof,
//! verified receipt, or host-provided lookup claim is accepted as input.

use std::{cell::RefCell, marker::PhantomData, sync::Arc};

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use nova_snark::{
    frontend::{
        gadgets::boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, LinearCombination,
        SynthesisError,
    },
    traits::circuit::StepCircuit,
};
use sha3::{Digest, Sha3_256};
use strum::EnumCount;
#[cfg(test)]
use strum::IntoEnumIterator;
use tracer::{instruction::Cycle, TraceBlock};

use crate::{
    poly::{
        eq_poly::EqPolynomial,
        opening_proof::{
            AbstractVerifierOpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator,
            SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::{BatchedSumcheck, ClearSumcheckProof},
    transcripts::{PoseidonTranscript, Transcript},
    utils::math::Math,
    zkvm::{
        config::OneHotParams,
        instruction::{
            Flags, InstructionLookup, InterleavedBitsMarker, JoltTraceCycle, LookupQuery,
        },
        instruction_lookups::{
            read_raf_checking::{
                InstructionReadRafSumcheckParams, InstructionReadRafSumcheckProver,
                InstructionReadRafSumcheckVerifier,
            },
            LOG_K,
        },
        lookup_table::LookupTables,
        witness::VirtualPolynomial,
    },
};

use super::{
    direct::{fixed_lookup_registry_commitment, validate_block},
    recursive_relations::{
        alloc_nova_constant, ark_bn254_scalar_as_nova_scalar,
        synthesize_recursive_clear_sumcheck_stage, synthesize_recursive_clear_sumcheck_transcript,
        synthesize_recursive_poseidon_transcript_transition,
        AllocatedRecursivePoseidonTranscriptState,
    },
    BlockRelation, CompactSumcheckProof, DeferredPcsClaim, DirectChunkedError,
    DirectChunkedPreprocessing, DirectRelation, DirectRelationState, FieldElement,
    LookupBlockProof, NovaPrimaryEngine, NovaPrimarySpartanSnark, NovaScalar, NovaSecondaryEngine,
    NovaSecondarySpartanSnark, RecursiveClearSumcheckRoundWitness,
    RecursiveClearSumcheckStageWitness, RecursiveJoltFieldElement, TranscriptCheckpoint,
};

pub(super) const NO_LOOKUP_TABLE_ID: u8 = LookupTables::<{ common::constants::XLEN }>::COUNT as u8;
const DIRECT_LOOKUP_QUERY_DOMAIN: &[u8] = b"direct-query-v1";
pub(super) const DIRECT_LOOKUP_TRANSCRIPT_DOMAIN: &[u8] = b"direct-lasso-v1";
pub(super) const DIRECT_LOOKUP_Z_ARITY: usize = 10;

type DirectLookupNovaSnark = nova_snark::nova::RecursiveSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    DirectLookupStepCircuit,
>;
type DirectLookupCompressedSnark = nova_snark::nova::CompressedSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    DirectLookupStepCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;
type DirectLookupPublicParams =
    nova_snark::nova::PublicParams<NovaPrimaryEngine, NovaSecondaryEngine, DirectLookupStepCircuit>;

/// Lookup-facing values derived from one final Jolt trace cycle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectLookupCycleWitness {
    pub active: bool,
    pub table_id: u8,
    pub lookup_index: u128,
    pub left_operand: u64,
    pub right_operand: u128,
    pub output: u64,
    pub raf_identity_path: bool,
}

impl Default for DirectLookupCycleWitness {
    fn default() -> Self {
        Self {
            active: false,
            table_id: NO_LOOKUP_TABLE_ID,
            lookup_index: 0,
            left_operand: 0,
            right_operand: 0,
            output: 0,
            raf_identity_path: false,
        }
    }
}

impl DirectLookupCycleWitness {
    fn from_cycle(cycle: &Cycle, block_index: usize) -> Result<Self, DirectChunkedError> {
        let jolt =
            JoltTraceCycle::try_new(cycle).map_err(|kind| DirectChunkedError::InvalidBlock {
                block_index,
                reason: format!("lookup cycle is not a final Jolt row: {kind:?}"),
            })?;
        let table: Option<LookupTables<{ common::constants::XLEN }>> = jolt.lookup_table();
        let table_id = table
            .map(|table| LookupTables::enum_index(&table) as u8)
            .unwrap_or(NO_LOOKUP_TABLE_ID);
        let lookup_index = LookupQuery::<{ common::constants::XLEN }>::to_lookup_index(&jolt);
        let (left_operand, right_operand) =
            LookupQuery::<{ common::constants::XLEN }>::to_lookup_operands(&jolt);
        let output = LookupQuery::<{ common::constants::XLEN }>::to_lookup_output(&jolt);
        if let Some(table) = table {
            let expected = table.materialize_entry(lookup_index);
            if expected != output {
                return Err(DirectChunkedError::InvalidBlock {
                    block_index,
                    reason: format!(
                        "lookup implementation/table mismatch: table {table_id}, index {lookup_index}, output {output}, expected {expected}"
                    ),
                });
            }
        } else if output != 0 {
            return Err(DirectChunkedError::InvalidBlock {
                block_index,
                reason: "cycle without a lookup table has non-zero lookup output".to_string(),
            });
        }
        Ok(Self {
            active: true,
            table_id,
            lookup_index,
            left_operand,
            right_operand,
            output,
            raf_identity_path: !jolt.circuit_flags().is_interleaved_operands(),
        })
    }
}

/// Fixed-shape block witness. Inactive suffix slots are canonical all-zero
/// lookup rows and are constrained as padding in the recursive circuit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectLookupBlockWitness {
    pub block_index: usize,
    pub global_cycle_start: usize,
    pub active_cycles: usize,
    pub cycle_capacity: usize,
    pub terminated: bool,
    pub cycles: Vec<DirectLookupCycleWitness>,
}

impl DirectLookupBlockWitness {
    pub fn from_trace_block(
        block: &TraceBlock,
        cycle_capacity: usize,
    ) -> Result<Self, DirectChunkedError> {
        if block.active_cycles != block.cycles.len()
            || block.active_cycles == 0
            || block.active_cycles > cycle_capacity
            || !cycle_capacity.is_power_of_two()
        {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: "lookup block does not fit the fixed power-of-two circuit shape"
                    .to_string(),
            });
        }
        let mut cycles = block
            .cycles
            .iter()
            .map(|cycle| DirectLookupCycleWitness::from_cycle(cycle, block.block_index))
            .collect::<Result<Vec<_>, _>>()?;
        cycles.resize(cycle_capacity, DirectLookupCycleWitness::default());
        Ok(Self {
            block_index: block.block_index,
            global_cycle_start: block.global_cycle_start,
            active_cycles: block.active_cycles,
            cycle_capacity,
            terminated: block.end_state.terminated,
            cycles,
        })
    }
}

/// Lossless native block subclaim. The sumcheck is Jolt's real
/// `InstructionReadRaf` clear proof; all endpoint openings are recomputed from
/// `cycles` by the verifier and later by the Nova step circuit.
#[derive(Clone)]
pub struct DirectLookupSubclaim {
    pub block: DirectLookupBlockWitness,
    pub query_root: Fr,
    pub r_reduction: Vec<Fr>,
    pub input_claims: [Fr; 3],
    pub gamma: Fr,
    pub batching_coefficient: Fr,
    pub initial_batched_claim: Fr,
    pub final_sumcheck_claim: Fr,
    pub proof: ClearSumcheckProof<Fr, PoseidonTranscript>,
    pub sumcheck_challenges: Vec<Fr>,
    pub degree_bound: usize,
    pub opening_claims: Vec<Fr>,
    pub transcript_state_after: [u8; 32],
    pub transcript_round_after: u32,
}

/// Public closure statement for a lookup-only D2 proof.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectLookupStageStatement {
    pub program_digest: [u8; 32],
    pub lookup_table_commitment: [u8; 32],
    pub block_count: usize,
    pub total_cycles: usize,
    pub final_global_cycle: usize,
    pub terminated: bool,
    pub relations: std::collections::BTreeMap<DirectRelation, DirectRelationState>,
}

/// D2 artifact. Nova/Spartan byte fields are populated in the recursive part
/// of this module; native subclaims are retained for audit and adversarial
/// regression tests, not trusted during final verification.
#[derive(Clone)]
pub struct DirectLookupStageProof {
    pub statement: DirectLookupStageStatement,
    pub subclaims: Vec<DirectLookupSubclaim>,
    pub nova_recursive_snark: Vec<u8>,
    pub spartan_proof: Vec<u8>,
    pub initial_z: Vec<[u8; 32]>,
    pub final_z: Vec<[u8; 32]>,
}

#[derive(Clone, Default)]
pub(super) struct DirectLookupStepCircuit {
    subclaim: Option<DirectLookupSubclaim>,
    final_step: bool,
}

impl DirectLookupStepCircuit {
    pub(super) fn for_subclaim(subclaim: DirectLookupSubclaim, final_step: bool) -> Self {
        Self {
            subclaim: Some(subclaim),
            final_step,
        }
    }
}

pub(super) fn nova_from_fr(value: &Fr) -> Result<NovaScalar, SynthesisError> {
    ark_bn254_scalar_as_nova_scalar(value)
}

pub(super) fn nova_digest_limbs(value: &[u8; 32]) -> [NovaScalar; 2] {
    let low = u128::from_le_bytes(value[..16].try_into().expect("fixed digest limb width"));
    let high = u128::from_le_bytes(value[16..].try_into().expect("fixed digest limb width"));
    [
        nova_from_fr(&field_from_u128(low)).expect("128-bit digest limb is canonical"),
        nova_from_fr(&field_from_u128(high)).expect("128-bit digest limb is canonical"),
    ]
}

pub(super) fn nova_to_storage(value: NovaScalar) -> [u8; 32] {
    value.to_bytes()
}

pub(super) fn nova_power_of_two(exponent: usize) -> NovaScalar {
    let mut value = NovaScalar::one();
    for _ in 0..exponent {
        value = value + value;
    }
    value
}

pub(super) fn direct_initial_z(preprocessing: &DirectChunkedPreprocessing) -> Vec<NovaScalar> {
    let transcript = PoseidonTranscript::new(DIRECT_LOOKUP_TRANSCRIPT_DOMAIN);
    let program = nova_digest_limbs(&preprocessing.program_digest);
    let tables = nova_digest_limbs(&preprocessing.lookup_table_commitment);
    vec![
        NovaScalar::zero(),
        NovaScalar::zero(),
        NovaScalar::zero(),
        program[0],
        program[1],
        tables[0],
        tables[1],
        nova_from_fr(&field_from_digest(&transcript.state))
            .expect("Poseidon transcript state is a canonical scalar"),
        NovaScalar::zero(),
        NovaScalar::zero(),
    ]
}

pub(super) struct AllocatedDirectLookupCycle {
    pub(super) active: AllocatedBit,
    pub(super) table_id: AllocatedNum<NovaScalar>,
    pub(super) lookup_index_bits: Vec<AllocatedBit>,
    pub(super) lookup_index_low: AllocatedNum<NovaScalar>,
    pub(super) lookup_index_high: AllocatedNum<NovaScalar>,
    pub(super) left_operand: AllocatedNum<NovaScalar>,
    pub(super) left_operand_bits: Vec<AllocatedBit>,
    pub(super) right_operand: AllocatedNum<NovaScalar>,
    pub(super) right_operand_bits: Vec<AllocatedBit>,
    pub(super) right_operand_low: AllocatedNum<NovaScalar>,
    pub(super) right_operand_high: AllocatedNum<NovaScalar>,
    pub(super) output: AllocatedNum<NovaScalar>,
    pub(super) raf_identity_path: AllocatedBit,
    pub(super) table_selectors: Vec<AllocatedBit>,
}

pub(super) fn alloc_witness_num<CS: ConstraintSystem<NovaScalar>>(
    cs: CS,
    value: NovaScalar,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    AllocatedNum::alloc(cs, || Ok(value))
}

pub(super) fn enforce_num_equal<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    left: &AllocatedNum<NovaScalar>,
    right: &AllocatedNum<NovaScalar>,
    label: &'static str,
) {
    cs.enforce(
        || label,
        |lc| lc + left.get_variable() - right.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
}

pub(super) fn add_nums<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    left: &AllocatedNum<NovaScalar>,
    right: &AllocatedNum<NovaScalar>,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let output = AllocatedNum::alloc(cs.namespace(|| format!("{label} output")), || {
        Ok(left.get_value().ok_or(SynthesisError::AssignmentMissing)?
            + right.get_value().ok_or(SynthesisError::AssignmentMissing)?)
    })?;
    cs.enforce(
        || label,
        |lc| lc + left.get_variable() + right.get_variable() - output.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(output)
}

pub(super) fn mul_nums<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    left: &AllocatedNum<NovaScalar>,
    right: &AllocatedNum<NovaScalar>,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let output = AllocatedNum::alloc(cs.namespace(|| format!("{label} output")), || {
        Ok(left.get_value().ok_or(SynthesisError::AssignmentMissing)?
            * right.get_value().ok_or(SynthesisError::AssignmentMissing)?)
    })?;
    cs.enforce(
        || label,
        |lc| lc + left.get_variable(),
        |lc| lc + right.get_variable(),
        |lc| lc + output.get_variable(),
    );
    Ok(output)
}

pub(super) fn sub_nums<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    left: &AllocatedNum<NovaScalar>,
    right: &AllocatedNum<NovaScalar>,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let output = AllocatedNum::alloc(cs.namespace(|| format!("{label} output")), || {
        Ok(left.get_value().ok_or(SynthesisError::AssignmentMissing)?
            - right.get_value().ok_or(SynthesisError::AssignmentMissing)?)
    })?;
    cs.enforce(
        || label,
        |lc| lc + left.get_variable() - right.get_variable() - output.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(output)
}

pub(super) fn scale_num<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    input: &AllocatedNum<NovaScalar>,
    coefficient: NovaScalar,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let output = AllocatedNum::alloc(cs.namespace(|| format!("{label} output")), || {
        Ok(input.get_value().ok_or(SynthesisError::AssignmentMissing)? * coefficient)
    })?;
    cs.enforce(
        || label,
        |lc| lc + (coefficient, input.get_variable()) - output.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(output)
}

pub(super) fn bit_as_num(bit: &AllocatedBit) -> AllocatedNum<NovaScalar> {
    AllocatedNum::from_parts(
        bit.get_variable(),
        bit.get_value().map(|value| {
            if value {
                NovaScalar::one()
            } else {
                NovaScalar::zero()
            }
        }),
    )
}

pub(super) fn alloc_u64_bits<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: u64,
    label: &'static str,
) -> Result<(AllocatedNum<NovaScalar>, Vec<AllocatedBit>), SynthesisError> {
    let bits = (0..64)
        .map(|bit| {
            AllocatedBit::alloc(
                cs.namespace(|| format!("{label} bit {bit}")),
                Some(((value >> bit) & 1) == 1),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let number = alloc_witness_num(cs.namespace(|| label), NovaScalar::from(value))?;
    let packed = bits.iter().enumerate().fold(
        LinearCombination::<NovaScalar>::zero(),
        |lc, (bit, allocated)| {
            let coefficient = nova_power_of_two(bit);
            lc + (coefficient, allocated.get_variable())
        },
    );
    cs.enforce(
        || format!("{label} bits pack"),
        |_| packed - number.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok((number, bits))
}

fn enforce_inactive_zero<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    active: &AllocatedBit,
    value: &AllocatedNum<NovaScalar>,
    label: &'static str,
) {
    cs.enforce(
        || label,
        |lc| lc + CS::one() - active.get_variable(),
        |lc| lc + value.get_variable(),
        |lc| lc,
    );
}

fn allocate_lookup_cycle<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    cycle: &DirectLookupCycleWitness,
) -> Result<AllocatedDirectLookupCycle, SynthesisError> {
    let active = AllocatedBit::alloc(cs.namespace(|| "active"), Some(cycle.active))?;
    let raf_identity_path = AllocatedBit::alloc(
        cs.namespace(|| "RAF identity path"),
        Some(cycle.raf_identity_path),
    )?;
    let table_id = alloc_witness_num(
        cs.namespace(|| "table id"),
        NovaScalar::from(cycle.table_id as u64),
    )?;
    let table_selectors = (0..=NO_LOOKUP_TABLE_ID as usize)
        .map(|table| {
            AllocatedBit::alloc(
                cs.namespace(|| format!("table selector {table}")),
                Some(cycle.table_id as usize == table),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selector_sum = table_selectors
        .iter()
        .fold(LinearCombination::<NovaScalar>::zero(), |lc, selector| {
            lc + selector.get_variable()
        });
    cs.enforce(
        || "exactly one table selector",
        |_| selector_sum - CS::one(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    let encoded_table = table_selectors.iter().enumerate().fold(
        LinearCombination::<NovaScalar>::zero(),
        |lc, (table, selector)| lc + (NovaScalar::from(table as u64), selector.get_variable()),
    );
    cs.enforce(
        || "table selector encodes table id",
        |_| encoded_table - table_id.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );

    let (lookup_index_low, mut lookup_index_bits) = alloc_u64_bits(
        cs.namespace(|| "lookup index low"),
        cycle.lookup_index as u64,
        "lookup index low",
    )?;
    let (lookup_index_high, lookup_index_high_bits) = alloc_u64_bits(
        cs.namespace(|| "lookup index high"),
        (cycle.lookup_index >> 64) as u64,
        "lookup index high",
    )?;
    lookup_index_bits.extend(lookup_index_high_bits);
    let lookup_index = alloc_witness_num(
        cs.namespace(|| "lookup index"),
        nova_from_fr(&field_from_u128(cycle.lookup_index))?,
    )?;
    cs.enforce(
        || "lookup index halves pack",
        |lc| {
            lc + lookup_index_low.get_variable()
                + (nova_power_of_two(64), lookup_index_high.get_variable())
                - lookup_index.get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );

    let (left_operand, left_operand_bits) = alloc_u64_bits(
        cs.namespace(|| "left operand"),
        cycle.left_operand,
        "left operand",
    )?;
    let (right_operand_low, mut right_operand_bits) = alloc_u64_bits(
        cs.namespace(|| "right operand low"),
        cycle.right_operand as u64,
        "right operand low",
    )?;
    let (right_operand_high, right_operand_high_bits) = alloc_u64_bits(
        cs.namespace(|| "right operand high"),
        (cycle.right_operand >> 64) as u64,
        "right operand high",
    )?;
    right_operand_bits.extend(right_operand_high_bits);
    let right_operand = alloc_witness_num(
        cs.namespace(|| "right operand"),
        nova_from_fr(&field_from_u128(cycle.right_operand))?,
    )?;
    cs.enforce(
        || "right operand halves pack",
        |lc| {
            lc + right_operand_low.get_variable()
                + (nova_power_of_two(64), right_operand_high.get_variable())
                - right_operand.get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );
    let (output, _) = alloc_u64_bits(
        cs.namespace(|| "lookup output"),
        cycle.output,
        "lookup output",
    )?;

    for (label, value) in [
        ("inactive lookup index is zero", &lookup_index),
        ("inactive left operand is zero", &left_operand),
        ("inactive right operand is zero", &right_operand),
        ("inactive output is zero", &output),
    ] {
        enforce_inactive_zero(cs.namespace(|| label), &active, value, label);
    }
    let raf_number = bit_as_num(&raf_identity_path);
    enforce_inactive_zero(
        cs.namespace(|| "inactive RAF flag is zero"),
        &active,
        &raf_number,
        "inactive RAF flag is zero",
    );
    let no_table = &table_selectors[NO_LOOKUP_TABLE_ID as usize];
    cs.enforce(
        || "inactive row selects no-table",
        |lc| lc + CS::one() - active.get_variable(),
        |lc| lc + CS::one() - no_table.get_variable(),
        |lc| lc,
    );
    cs.enforce(
        || "no-table row has zero output",
        |lc| lc + no_table.get_variable(),
        |lc| lc + output.get_variable(),
        |lc| lc,
    );

    Ok(AllocatedDirectLookupCycle {
        active,
        table_id,
        lookup_index_bits,
        lookup_index_low,
        lookup_index_high,
        left_operand,
        left_operand_bits,
        right_operand,
        right_operand_bits,
        right_operand_low,
        right_operand_high,
        output,
        raf_identity_path,
        table_selectors,
    })
}

pub(super) fn field_from_u128(value: u128) -> Fr {
    Fr::from_le_bytes_mod_order(&value.to_le_bytes())
}

pub(super) fn field_from_digest(value: &[u8; 32]) -> Fr {
    Fr::from_le_bytes_mod_order(value)
}

pub(super) fn allocated_poseidon_initial<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    domain: &'static [u8],
) -> Result<AllocatedRecursivePoseidonTranscriptState, SynthesisError> {
    let transcript = PoseidonTranscript::new(domain);
    Ok(AllocatedRecursivePoseidonTranscriptState {
        state: alloc_nova_constant(
            cs.namespace(|| "initial Poseidon state"),
            nova_from_fr(&field_from_digest(&transcript.state))?,
        )?,
        n_rounds: alloc_nova_constant(
            cs.namespace(|| "initial Poseidon round"),
            NovaScalar::zero(),
        )?,
    })
}

pub(super) fn poseidon_absorb<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    state: &AllocatedRecursivePoseidonTranscriptState,
    value: &AllocatedNum<NovaScalar>,
    label: &'static str,
) -> Result<AllocatedRecursivePoseidonTranscriptState, SynthesisError> {
    let label_field = Fr::from_le_bytes_mod_order(label.as_bytes());
    let label = alloc_nova_constant(
        cs.namespace(|| "Poseidon label"),
        nova_from_fr(&label_field)?,
    )?;
    let after_label = synthesize_recursive_poseidon_transcript_transition(
        cs.namespace(|| "absorb Poseidon label"),
        state,
        &label,
    )?;
    synthesize_recursive_poseidon_transcript_transition(
        cs.namespace(|| "absorb Poseidon value"),
        &after_label,
        value,
    )
}

pub(super) fn poseidon_challenge<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    state: &AllocatedRecursivePoseidonTranscriptState,
) -> Result<AllocatedRecursivePoseidonTranscriptState, SynthesisError> {
    let zero = alloc_nova_constant(
        cs.namespace(|| "Poseidon challenge zero"),
        NovaScalar::zero(),
    )?;
    synthesize_recursive_poseidon_transcript_transition(
        cs.namespace(|| "derive Poseidon challenge"),
        state,
        &zero,
    )
}

fn append_lookup_cycle(transcript: &mut PoseidonTranscript, cycle: &DirectLookupCycleWitness) {
    let values = [
        Fr::from(u64::from(cycle.active)),
        Fr::from(cycle.table_id as u64),
        Fr::from(cycle.lookup_index as u64),
        Fr::from((cycle.lookup_index >> 64) as u64),
        Fr::from(cycle.left_operand),
        Fr::from(cycle.right_operand as u64),
        Fr::from((cycle.right_operand >> 64) as u64),
        Fr::from(cycle.output),
        Fr::from(u64::from(cycle.raf_identity_path)),
    ];
    for value in values {
        transcript.append_scalar(b"query_word", &value);
    }
}

fn query_root(block: &DirectLookupBlockWitness) -> Fr {
    let mut transcript = PoseidonTranscript::new(DIRECT_LOOKUP_QUERY_DOMAIN);
    transcript.append_scalar(b"block_index", &Fr::from(block.block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(block.global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(block.active_cycles as u64));
    for cycle in &block.cycles {
        append_lookup_cycle(&mut transcript, cycle);
    }
    field_from_digest(&transcript.state)
}

fn append_block_header(
    transcript: &mut PoseidonTranscript,
    table_root: Fr,
    block: &DirectLookupBlockWitness,
    root: Fr,
) {
    append_compact_block_header(
        transcript,
        table_root,
        block.block_index,
        block.global_cycle_start,
        block.active_cycles,
        root,
    );
}

fn append_compact_block_header(
    transcript: &mut PoseidonTranscript,
    table_root: Fr,
    block_index: usize,
    global_cycle_start: usize,
    active_cycles: usize,
    root: Fr,
) {
    // The same preprocessing commitment is reused by every fixed-shape step.
    // Reabsorbing it keeps the Nova circuit shape independent of block index.
    transcript.append_scalar(b"table_root", &table_root);
    transcript.append_scalar(b"block_index", &Fr::from(block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(active_cycles as u64));
    transcript.append_scalar(b"query_root", &root);
}

fn mle_claims(block: &DirectLookupBlockWitness, point: &[Fr]) -> [Fr; 3] {
    let weights = EqPolynomial::<Fr>::evals(point);
    let mut claims = [Fr::zero(); 3];
    for (weight, cycle) in weights.into_iter().zip(&block.cycles) {
        claims[0] += weight * Fr::from(cycle.output);
        claims[1] += weight * Fr::from(cycle.left_operand);
        claims[2] += weight * field_from_u128(cycle.right_operand);
    }
    claims
}

fn synthesize_query_root<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    block_index: &AllocatedNum<NovaScalar>,
    cycle_start: &AllocatedNum<NovaScalar>,
    active_cycles: &AllocatedNum<NovaScalar>,
    allocated_cycles: &[AllocatedDirectLookupCycle],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let mut state = allocated_poseidon_initial(
        cs.namespace(|| "query commitment initial state"),
        DIRECT_LOOKUP_QUERY_DOMAIN,
    )?;
    for (label, value) in [
        ("block_index", block_index),
        ("cycle_start", cycle_start),
        ("active_cycles", active_cycles),
    ] {
        state = poseidon_absorb(cs.namespace(|| label), &state, value, label)?;
    }
    for (index, cycle) in allocated_cycles.iter().enumerate() {
        let active = bit_as_num(&cycle.active);
        let raf = bit_as_num(&cycle.raf_identity_path);
        for (word_index, word) in [
            &active,
            &cycle.table_id,
            &cycle.lookup_index_low,
            &cycle.lookup_index_high,
            &cycle.left_operand,
            &cycle.right_operand_low,
            &cycle.right_operand_high,
            &cycle.output,
            &raf,
        ]
        .into_iter()
        .enumerate()
        {
            state = poseidon_absorb(
                cs.namespace(|| format!("cycle {index} query word {word_index}")),
                &state,
                word,
                "query_word",
            )?;
        }
    }
    Ok(state.state)
}

pub(super) fn eq_weight_for_index<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    point: &[AllocatedNum<NovaScalar>],
    index: usize,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let one = alloc_nova_constant(cs.namespace(|| "eq one"), NovaScalar::one())?;
    let mut weight = one.clone();
    for (coordinate, challenge) in point.iter().enumerate() {
        let bit = (index >> (point.len() - 1 - coordinate)) & 1;
        let factor = if bit == 1 {
            challenge.clone()
        } else {
            sub_nums(
                cs.namespace(|| format!("{label} one minus {coordinate}")),
                &one,
                challenge,
                "eq one minus coordinate",
            )?
        };
        weight = mul_nums(
            cs.namespace(|| format!("{label} factor {coordinate}")),
            &weight,
            &factor,
            "eq factor product",
        )?;
    }
    Ok(weight)
}

pub(super) fn mle_from_values<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    values: &[AllocatedNum<NovaScalar>],
    point: &[AllocatedNum<NovaScalar>],
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    if values.len() != 1usize << point.len() {
        return Err(SynthesisError::Unsatisfiable(
            "direct lookup MLE dimension mismatch".to_string(),
        ));
    }
    let mut result = alloc_nova_constant(cs.namespace(|| "MLE zero"), NovaScalar::zero())?;
    for (index, value) in values.iter().enumerate() {
        let weight = eq_weight_for_index(
            cs.namespace(|| format!("{label} weight {index}")),
            point,
            index,
            "MLE weight",
        )?;
        let term = mul_nums(
            cs.namespace(|| format!("{label} term {index}")),
            &weight,
            value,
            "MLE weighted term",
        )?;
        result = add_nums(
            cs.namespace(|| format!("{label} sum {index}")),
            &result,
            &term,
            "MLE accumulate",
        )?;
    }
    Ok(result)
}

fn one_minus<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: &AllocatedNum<NovaScalar>,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let one = alloc_nova_constant(cs.namespace(|| "table one"), NovaScalar::one())?;
    sub_nums(cs.namespace(|| "one minus"), &one, value, "one minus")
}

fn add<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    a: &AllocatedNum<NovaScalar>,
    b: &AllocatedNum<NovaScalar>,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    add_nums(cs.namespace(|| "table add"), a, b, "table add")
}

fn mul<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    a: &AllocatedNum<NovaScalar>,
    b: &AllocatedNum<NovaScalar>,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    mul_nums(cs.namespace(|| "table mul"), a, b, "table mul")
}

fn sub<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    a: &AllocatedNum<NovaScalar>,
    b: &AllocatedNum<NovaScalar>,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    sub_nums(cs.namespace(|| "table sub"), a, b, "table sub")
}

fn scale<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    value: &AllocatedNum<NovaScalar>,
    coefficient: u128,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    scale_num(
        cs.namespace(|| "table scale"),
        value,
        nova_from_fr(&field_from_u128(coefficient))?,
        "table scale",
    )
}

struct DeferredNamespaces<CS>(PhantomData<CS>);

impl<CS> DeferredNamespaces<CS> {
    fn namespace<NR, N>(&self, _name: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }
}

pub(super) fn evaluate_table_mle_circuit<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    table_id: usize,
    r: &[AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    const X: usize = common::constants::XLEN;
    if r.len() != 2 * X {
        return Err(SynthesisError::Unsatisfiable(
            "direct table MLE point has invalid width".to_string(),
        ));
    }
    let zero = alloc_nova_constant(cs.namespace(|| "table zero"), NovaScalar::zero())?;
    let one = alloc_nova_constant(cs.namespace(|| "table one"), NovaScalar::one())?;
    let root_cs = RefCell::new(cs);
    let cs = DeferredNamespaces::<CS>(PhantomData);
    let namespace_counter = RefCell::new(0usize);
    let one_minus = |_: (), value: &AllocatedNum<NovaScalar>| {
        let mut root = root_cs.borrow_mut();
        let index = {
            let mut counter = namespace_counter.borrow_mut();
            let index = *counter;
            *counter += 1;
            index
        };
        one_minus(root.namespace(|| format!("table one minus {index}")), value)
    };
    let add = |_: (), a: &AllocatedNum<NovaScalar>, b: &AllocatedNum<NovaScalar>| {
        let mut root = root_cs.borrow_mut();
        let index = {
            let mut counter = namespace_counter.borrow_mut();
            let index = *counter;
            *counter += 1;
            index
        };
        add(root.namespace(|| format!("table add {index}")), a, b)
    };
    let mul = |_: (), a: &AllocatedNum<NovaScalar>, b: &AllocatedNum<NovaScalar>| {
        let mut root = root_cs.borrow_mut();
        let index = {
            let mut counter = namespace_counter.borrow_mut();
            let index = *counter;
            *counter += 1;
            index
        };
        mul(root.namespace(|| format!("table mul {index}")), a, b)
    };
    let sub = |_: (), a: &AllocatedNum<NovaScalar>, b: &AllocatedNum<NovaScalar>| {
        let mut root = root_cs.borrow_mut();
        let index = {
            let mut counter = namespace_counter.borrow_mut();
            let index = *counter;
            *counter += 1;
            index
        };
        sub(root.namespace(|| format!("table sub {index}")), a, b)
    };
    let scale = |_: (), value: &AllocatedNum<NovaScalar>, coefficient: u128| {
        let mut root = root_cs.borrow_mut();
        let index = {
            let mut counter = namespace_counter.borrow_mut();
            let index = *counter;
            *counter += 1;
            index
        };
        scale(
            root.namespace(|| format!("table scale {index}")),
            value,
            coefficient,
        )
    };
    let scale_field = |_: (), value: &AllocatedNum<NovaScalar>, coefficient: NovaScalar| {
        let mut root = root_cs.borrow_mut();
        let index = {
            let mut counter = namespace_counter.borrow_mut();
            let index = *counter;
            *counter += 1;
            index
        };
        scale_num(
            root.namespace(|| format!("table field scale {index}")),
            value,
            coefficient,
            "table field scale",
        )
    };
    let linear_bits = |_: (),
                       _: &[AllocatedNum<NovaScalar>],
                       indices: Vec<(usize, u128)>|
     -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
        let mut result = zero.clone();
        for (index, coefficient) in indices {
            let term = scale((), &r[index], coefficient)?;
            result = add((), &result, &term)?;
        }
        Ok(result)
    };
    let eq_eval = |_: (), _: &[AllocatedNum<NovaScalar>]| {
        let mut result = one.clone();
        for i in 0..X {
            let xy = mul((), &r[2 * i], &r[2 * i + 1])?;
            let nx = one_minus((), &r[2 * i])?;
            let ny = one_minus((), &r[2 * i + 1])?;
            let nxny = mul((), &nx, &ny)?;
            let factor = add((), &xy, &nxny)?;
            result = mul((), &result, &factor)?;
        }
        Ok::<_, SynthesisError>(result)
    };
    let unsigned_lt = |_: (), _: &[AllocatedNum<NovaScalar>]| {
        let mut result = zero.clone();
        let mut eq = one.clone();
        for i in 0..X {
            let nx = one_minus((), &r[2 * i])?;
            let nx_y = mul((), &nx, &r[2 * i + 1])?;
            let term = mul((), &nx_y, &eq)?;
            result = add((), &result, &term)?;
            let xy = mul((), &r[2 * i], &r[2 * i + 1])?;
            let ny = one_minus((), &r[2 * i + 1])?;
            let nxny = mul((), &nx, &ny)?;
            let factor = add((), &xy, &nxny)?;
            eq = mul((), &eq, &factor)?;
        }
        Ok::<_, SynthesisError>(result)
    };
    match table_id {
        0 => linear_bits(
            cs.namespace(|| "RangeCheck"),
            r,
            (0..X).map(|i| (X + i, 1u128 << (X - 1 - i))).collect(),
        ),
        1 => linear_bits(
            cs.namespace(|| "RangeCheckAligned"),
            r,
            (0..X - 1).map(|i| (X + i, 1u128 << (X - 1 - i))).collect(),
        ),
        2 | 3 | 4 | 5 => {
            let mut result = zero.clone();
            for i in 0..X {
                let x = &r[2 * i];
                let y = &r[2 * i + 1];
                let bit = match table_id {
                    2 => mul(cs.namespace(|| "and"), x, y)?,
                    3 => mul(
                        cs.namespace(|| "andn"),
                        x,
                        &one_minus(cs.namespace(|| "andn ny"), y)?,
                    )?,
                    4 => sub(
                        cs.namespace(|| "or"),
                        &add(cs.namespace(|| "or add"), x, y)?,
                        &mul(cs.namespace(|| "or mul"), x, y)?,
                    )?,
                    _ => add(
                        cs.namespace(|| "xor"),
                        &mul(
                            cs.namespace(|| "xor nx y"),
                            &one_minus(cs.namespace(|| "xor nx"), x)?,
                            y,
                        )?,
                        &mul(
                            cs.namespace(|| "xor x ny"),
                            x,
                            &one_minus(cs.namespace(|| "xor ny"), y)?,
                        )?,
                    )?,
                };
                let term = scale(cs.namespace(|| "bit weight"), &bit, 1u128 << (X - 1 - i))?;
                result = add(cs.namespace(|| "bitwise sum"), &result, &term)?;
            }
            Ok(result)
        }
        6 => eq_eval(cs.namespace(|| "Equal"), r),
        7 => {
            let lt = unsigned_lt(cs.namespace(|| "signed lt body"), r)?;
            let signed_lt = add(
                cs.namespace(|| "signed lt"),
                &sub(cs.namespace(|| "sign difference"), &r[0], &r[1])?,
                &lt,
            )?;
            sub(cs.namespace(|| "signed ge"), &one, &signed_lt)
        }
        8 => sub(
            cs.namespace(|| "unsigned ge"),
            &one,
            &unsigned_lt(cs.namespace(|| "unsigned lt"), r)?,
        ),
        9 => sub(
            cs.namespace(|| "not equal"),
            &one,
            &eq_eval(cs.namespace(|| "eq"), r)?,
        ),
        10 => add(
            cs.namespace(|| "signed lt"),
            &sub(cs.namespace(|| "sign difference"), &r[0], &r[1])?,
            &unsigned_lt(cs.namespace(|| "lt body"), r)?,
        ),
        11 => unsigned_lt(cs.namespace(|| "unsigned lt"), r),
        12 => scale(cs.namespace(|| "movsign"), &r[0], (1u128 << X) - 1),
        13 => linear_bits(
            cs.namespace(|| "upper word"),
            r,
            (0..X).map(|i| (i, 1u128 << (X - 1 - i))).collect(),
        ),
        14 => add(
            cs.namespace(|| "less than equal"),
            &unsigned_lt(cs.namespace(|| "lt"), r)?,
            &eq_eval(cs.namespace(|| "eq"), r)?,
        ),
        15 => {
            let mut divisor_zero = one.clone();
            for i in 0..X {
                divisor_zero = mul(
                    cs.namespace(|| "divisor zero"),
                    &divisor_zero,
                    &one_minus(cs.namespace(|| "divisor zero bit"), &r[2 * i + 1])?,
                )?;
            }
            add(
                cs.namespace(|| "valid remainder"),
                &unsigned_lt(cs.namespace(|| "remainder lt"), r)?,
                &divisor_zero,
            )
        }
        16 => {
            let mut divisor_zero = one.clone();
            let mut valid_zero = one.clone();
            for i in 0..X {
                let nx = one_minus(cs.namespace(|| "div0 nx"), &r[2 * i])?;
                divisor_zero = mul(cs.namespace(|| "div0 divisor"), &divisor_zero, &nx)?;
                valid_zero = mul(
                    cs.namespace(|| "div0 valid"),
                    &valid_zero,
                    &mul(cs.namespace(|| "div0 nx y"), &nx, &r[2 * i + 1])?,
                )?;
            }
            add(
                cs.namespace(|| "valid div0"),
                &sub(cs.namespace(|| "not divisor zero"), &one, &divisor_zero)?,
                &valid_zero,
            )
        }
        17 => one_minus(cs.namespace(|| "halfword aligned"), &r[2 * X - 1]),
        18 => mul(
            cs.namespace(|| "word aligned"),
            &one_minus(cs.namespace(|| "word lsb0"), &r[2 * X - 1])?,
            &one_minus(cs.namespace(|| "word lsb1"), &r[2 * X - 2])?,
        ),
        19 => linear_bits(
            cs.namespace(|| "lower half word"),
            r,
            (0..X / 2)
                .map(|i| (X + X / 2 + i, 1u128 << (X / 2 - 1 - i)))
                .collect(),
        ),
        20 => {
            let lower = linear_bits(
                cs.namespace(|| "sign extend lower"),
                r,
                (0..X / 2)
                    .map(|i| (X + X / 2 + i, 1u128 << (X / 2 - 1 - i)))
                    .collect(),
            )?;
            let extension = scale(
                cs.namespace(|| "sign extension"),
                &r[X + X / 2],
                ((1u128 << (X / 2)) - 1) << (X / 2),
            )?;
            add(cs.namespace(|| "sign extend"), &lower, &extension)
        }
        21 | 22 => {
            let bits = if table_id == 21 { X.log_2() } else { 5 };
            let mut result = one.clone();
            for i in 0..bits {
                let coefficient = (1u64 << (1usize << i)) - 1;
                let scaled = scale_field(
                    cs.namespace(|| "pow2 factor scale"),
                    &r[2 * X - i - 1],
                    NovaScalar::from(coefficient),
                )?;
                result = mul(
                    cs.namespace(|| "pow2 product"),
                    &result,
                    &add(cs.namespace(|| "pow2 factor"), &one, &scaled)?,
                )?;
            }
            Ok(result)
        }
        23 => {
            let log_w = X.log_2();
            let mut result = zero.clone();
            for shift in 0..X {
                let mut weight = one.clone();
                for i in 0..log_w {
                    let coordinate = &r[2 * X - log_w + (log_w - i - 1)];
                    let factor = if (shift >> i) & 1 == 1 {
                        coordinate.clone()
                    } else {
                        one_minus(cs.namespace(|| "mask eq complement"), coordinate)?
                    };
                    weight = mul(cs.namespace(|| "mask eq"), &weight, &factor)?;
                }
                let mask = ((1u128 << (X - shift)) - 1) << shift;
                result = add(
                    cs.namespace(|| "mask sum"),
                    &result,
                    &scale(cs.namespace(|| "mask scale"), &weight, mask)?,
                )?;
            }
            Ok(result)
        }
        24 => {
            let mut output_bits = Vec::with_capacity(X);
            for (output_byte, input_byte) in [3usize, 2, 1, 0, 7, 6, 5, 4].into_iter().enumerate() {
                for bit in 0..8 {
                    let coordinate = 2 * X - 1 - (input_byte * 8 + bit);
                    output_bits.push((coordinate, 1u128 << (output_byte * 8 + bit)));
                }
            }
            linear_bits(cs.namespace(|| "rev8w"), r, output_bits)
        }
        25 | 26 => {
            let start = 0;
            let mut result = zero.clone();
            let mut sign_extension = zero.clone();
            for i in start..X {
                result = add(
                    cs.namespace(|| "shift recurrence add"),
                    &mul(
                        cs.namespace(|| "shift recurrence mul"),
                        &result,
                        &add(cs.namespace(|| "one plus y"), &one, &r[2 * i + 1])?,
                    )?,
                    &mul(cs.namespace(|| "shift xy"), &r[2 * i], &r[2 * i + 1])?,
                )?;
                if table_id == 26 && i != 0 {
                    sign_extension = add(
                        cs.namespace(|| "SRA extension"),
                        &sign_extension,
                        &scale(
                            cs.namespace(|| "SRA extension bit"),
                            &one_minus(cs.namespace(|| "SRA one minus y"), &r[2 * i + 1])?,
                            1u128 << i,
                        )?,
                    )?;
                }
            }
            if table_id == 26 {
                add(
                    cs.namespace(|| "SRA result"),
                    &result,
                    &mul(
                        cs.namespace(|| "SRA signed extension"),
                        &r[0],
                        &sign_extension,
                    )?,
                )
            } else {
                Ok(result)
            }
        }
        27 | 28 => {
            let start = if table_id == 27 { 0 } else { X / 2 };
            let mut product = one.clone();
            let mut first = zero.clone();
            let mut second = zero.clone();
            for i in start..X {
                first = add(
                    cs.namespace(|| "rotr first"),
                    &mul(
                        cs.namespace(|| "rotr first product"),
                        &first,
                        &add(cs.namespace(|| "rotr one plus y"), &one, &r[2 * i + 1])?,
                    )?,
                    &mul(cs.namespace(|| "rotr xy"), &r[2 * i], &r[2 * i + 1])?,
                )?;
                let second_factor = mul(
                    cs.namespace(|| "rotr second factor"),
                    &mul(
                        cs.namespace(|| "rotr x not y"),
                        &r[2 * i],
                        &one_minus(cs.namespace(|| "rotr not y"), &r[2 * i + 1])?,
                    )?,
                    &product,
                )?;
                second = add(
                    cs.namespace(|| "rotr second"),
                    &second,
                    &scale(
                        cs.namespace(|| "rotr weight"),
                        &second_factor,
                        1u128 << (X - 1 - i),
                    )?,
                )?;
                product = mul(
                    cs.namespace(|| "rotr product"),
                    &product,
                    &add(cs.namespace(|| "rotr product factor"), &one, &r[2 * i + 1])?,
                )?;
            }
            add(cs.namespace(|| "rotr result"), &first, &second)
        }
        29 => {
            let divisor = linear_bits(
                cs.namespace(|| "change divisor value"),
                r,
                (0..X).map(|i| (2 * i + 1, 1u128 << (X - 1 - i))).collect(),
            )?;
            let mut x_product = r[0].clone();
            for i in 1..X {
                x_product = mul(
                    cs.namespace(|| "change divisor x product"),
                    &x_product,
                    &one_minus(cs.namespace(|| "change divisor nx"), &r[2 * i])?,
                )?;
            }
            let mut y_product = one.clone();
            for i in 0..X {
                y_product = mul(
                    cs.namespace(|| "change divisor y product"),
                    &y_product,
                    &r[2 * i + 1],
                )?;
            }
            let exceptional = mul(
                cs.namespace(|| "change divisor exceptional"),
                &x_product,
                &y_product,
            )?;
            add(
                cs.namespace(|| "change divisor"),
                &divisor,
                &scale_field(
                    cs.namespace(|| "change divisor adjustment"),
                    &exceptional,
                    NovaScalar::from(2) - nova_power_of_two(X),
                )?,
            )
        }
        30 => {
            let divisor = linear_bits(
                cs.namespace(|| "change divisor W value"),
                r,
                (X / 2..X)
                    .map(|i| (2 * i + 1, 1u128 << (X - 1 - i)))
                    .collect(),
            )?;
            let mut x_product = r[X].clone();
            for i in X / 2 + 1..X {
                x_product = mul(
                    cs.namespace(|| "change divisor W x"),
                    &x_product,
                    &one_minus(cs.namespace(|| "change divisor W nx"), &r[2 * i])?,
                )?;
            }
            let mut y_product = one.clone();
            for i in X / 2..X {
                y_product = mul(
                    cs.namespace(|| "change divisor W y"),
                    &y_product,
                    &r[2 * i + 1],
                )?;
            }
            let exceptional = mul(
                cs.namespace(|| "change divisor W exceptional"),
                &x_product,
                &y_product,
            )?;
            let sign_extension = scale_field(
                cs.namespace(|| "change divisor W sign extension"),
                &r[X + 1],
                nova_power_of_two(X) - nova_power_of_two(X / 2),
            )?;
            add(
                cs.namespace(|| "change divisor W total"),
                &add(
                    cs.namespace(|| "change divisor W base"),
                    &divisor,
                    &scale_field(
                        cs.namespace(|| "change divisor W adjustment"),
                        &exceptional,
                        NovaScalar::from(2) - nova_power_of_two(X),
                    )?,
                )?,
                &sign_extension,
            )
        }
        31 => {
            let mut result = one.clone();
            for value in &r[..X] {
                result = mul(
                    cs.namespace(|| "overflow bits zero"),
                    &result,
                    &one_minus(cs.namespace(|| "overflow bit complement"), value)?,
                )?;
            }
            Ok(result)
        }
        32..=39 => {
            let (word, rotation) = if table_id < 36 {
                (false, [32usize, 24, 16, 63][table_id - 32])
            } else {
                (true, [16usize, 12, 8, 7][table_id - 36])
            };
            let start = if word { X / 2 } else { 0 };
            let width = if word { X / 2 } else { X };
            let mut result = zero.clone();
            for i in start..X {
                let xor = add(
                    cs.namespace(|| "xor rotate bit"),
                    &mul(
                        cs.namespace(|| "xor rotate nx y"),
                        &one_minus(cs.namespace(|| "xor rotate nx"), &r[2 * i])?,
                        &r[2 * i + 1],
                    )?,
                    &mul(
                        cs.namespace(|| "xor rotate x ny"),
                        &r[2 * i],
                        &one_minus(cs.namespace(|| "xor rotate ny"), &r[2 * i + 1])?,
                    )?,
                )?;
                let position = i - start;
                let bit_position = width - 1 - ((position + rotation) % width);
                result = add(
                    cs.namespace(|| "xor rotate sum"),
                    &result,
                    &scale(
                        cs.namespace(|| "xor rotate weight"),
                        &xor,
                        1u128 << bit_position,
                    )?,
                )?;
            }
            Ok(result)
        }
        _ => Err(SynthesisError::Unsatisfiable(
            "unknown fixed lookup table identifier".to_string(),
        )),
    }
}

pub(super) fn recursive_field(value: Fr) -> Result<RecursiveJoltFieldElement, SynthesisError> {
    RecursiveJoltFieldElement::from_field(value)
        .map_err(|reason| SynthesisError::Unsatisfiable(reason.to_string()))
}

fn sumcheck_witness(
    subclaim: &DirectLookupSubclaim,
) -> Result<RecursiveClearSumcheckStageWitness, SynthesisError> {
    if subclaim.proof.compressed_polys.len() != subclaim.sumcheck_challenges.len() {
        return Err(SynthesisError::Unsatisfiable(
            "direct lookup sumcheck proof/challenge length mismatch".to_string(),
        ));
    }
    let rounds = subclaim
        .proof
        .compressed_polys
        .iter()
        .zip(&subclaim.sumcheck_challenges)
        .map(|(poly, challenge)| {
            Ok(RecursiveClearSumcheckRoundWitness {
                coefficients_except_linear: poly
                    .coeffs_except_linear_term
                    .iter()
                    .copied()
                    .map(recursive_field)
                    .collect::<Result<Vec<_>, _>>()?,
                challenge: recursive_field(*challenge)?,
            })
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;
    Ok(RecursiveClearSumcheckStageWitness {
        stage_index: 0,
        degree_bound: subclaim.degree_bound,
        initial_claim: recursive_field(subclaim.initial_batched_claim)?,
        rounds,
        expected_final_claim: recursive_field(subclaim.final_sumcheck_claim)?,
    })
}

pub(super) fn eq_from_allocated_bits<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    point: &[AllocatedNum<NovaScalar>],
    bits_le: &[AllocatedBit],
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    if point.len() != bits_le.len() {
        return Err(SynthesisError::Unsatisfiable(
            "direct lookup bit-equality dimension mismatch".to_string(),
        ));
    }
    let one = alloc_nova_constant(cs.namespace(|| "bit equality one"), NovaScalar::one())?;
    let two = alloc_nova_constant(cs.namespace(|| "bit equality two"), NovaScalar::from(2))?;
    let mut weight = one.clone();
    for (coordinate, challenge) in point.iter().enumerate() {
        let bit = bit_as_num(&bits_le[point.len() - 1 - coordinate]);
        let bit_times_challenge = mul_nums(
            cs.namespace(|| format!("{label} bit times challenge {coordinate}")),
            &bit,
            challenge,
            "bit times challenge",
        )?;
        let twice = mul_nums(
            cs.namespace(|| format!("{label} twice product {coordinate}")),
            &two,
            &bit_times_challenge,
            "twice bit challenge",
        )?;
        let one_minus_bit = sub_nums(
            cs.namespace(|| format!("{label} one minus bit {coordinate}")),
            &one,
            &bit,
            "one minus bit",
        )?;
        let base = sub_nums(
            cs.namespace(|| format!("{label} base {coordinate}")),
            &one_minus_bit,
            challenge,
            "bit equality base",
        )?;
        let factor = add_nums(
            cs.namespace(|| format!("{label} factor {coordinate}")),
            &base,
            &twice,
            "bit equality factor",
        )?;
        weight = mul_nums(
            cs.namespace(|| format!("{label} product {coordinate}")),
            &weight,
            &factor,
            "bit equality product",
        )?;
    }
    Ok(weight)
}

pub(super) fn weighted_linear_point<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    point: &[AllocatedNum<NovaScalar>],
    parity: Option<usize>,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let mut result = alloc_nova_constant(cs.namespace(|| "linear point zero"), NovaScalar::zero())?;
    match parity {
        Some(parity) => {
            for i in 0..point.len() / 2 {
                let coordinate = 2 * i + parity;
                let term = scale_num(
                    cs.namespace(|| format!("{label} term {i}")),
                    &point[coordinate],
                    nova_power_of_two(point.len() / 2 - 1 - i),
                    "linear point term",
                )?;
                result = add_nums(
                    cs.namespace(|| format!("{label} sum {i}")),
                    &result,
                    &term,
                    "linear point sum",
                )?;
            }
        }
        None => {
            for (i, coordinate) in point.iter().enumerate() {
                let term = scale_num(
                    cs.namespace(|| format!("{label} term {i}")),
                    coordinate,
                    nova_power_of_two(point.len() - 1 - i),
                    "linear point term",
                )?;
                result = add_nums(
                    cs.namespace(|| format!("{label} sum {i}")),
                    &result,
                    &term,
                    "linear point sum",
                )?;
            }
        }
    }
    Ok(result)
}

fn synthesize_instruction_read_raf_endpoint<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    cycles: &[AllocatedDirectLookupCycle],
    r_reduction: &[AllocatedNum<NovaScalar>],
    gamma: &AllocatedNum<NovaScalar>,
    batching: &AllocatedNum<NovaScalar>,
    sumcheck_challenges: &[AllocatedNum<NovaScalar>],
    opening_claims: &[AllocatedNum<NovaScalar>],
    one_hot: &OneHotParams,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let table_count = LookupTables::<{ common::constants::XLEN }>::COUNT;
    let ra_count = LOG_K / one_hot.lookups_ra_virtual_log_k_chunk;
    if sumcheck_challenges.len() != LOG_K + r_reduction.len()
        || opening_claims.len() != table_count + ra_count + 1
    {
        return Err(SynthesisError::Unsatisfiable(
            "direct lookup endpoint witness has invalid dimensions".to_string(),
        ));
    }
    let (r_address, r_cycle_binding) = sumcheck_challenges.split_at(LOG_K);
    let r_cycle = r_cycle_binding.iter().rev().cloned().collect::<Vec<_>>();

    let cycle_weights = (0..cycles.len())
        .map(|index| {
            eq_weight_for_index(
                cs.namespace(|| format!("cycle weight {index}")),
                &r_cycle,
                index,
                "cycle weight",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    for table in 0..table_count {
        let mut claim = alloc_nova_constant(
            cs.namespace(|| format!("table flag {table} zero")),
            NovaScalar::zero(),
        )?;
        for (cycle_index, (cycle, weight)) in cycles.iter().zip(&cycle_weights).enumerate() {
            let selector = bit_as_num(&cycle.table_selectors[table]);
            let term = mul_nums(
                cs.namespace(|| format!("table {table} cycle {cycle_index} term")),
                weight,
                &selector,
                "table flag term",
            )?;
            claim = add_nums(
                cs.namespace(|| format!("table {table} cycle {cycle_index} sum")),
                &claim,
                &term,
                "table flag sum",
            )?;
        }
        enforce_num_equal(
            cs.namespace(|| format!("table flag opening {table}")),
            &claim,
            &opening_claims[table],
            "table flag opening",
        );
    }

    for chunk_index in 0..ra_count {
        let start = chunk_index * one_hot.lookups_ra_virtual_log_k_chunk;
        let end = start + one_hot.lookups_ra_virtual_log_k_chunk;
        let bit_shift = one_hot.lookups_ra_virtual_log_k_chunk * (ra_count - 1 - chunk_index);
        let mut claim = alloc_nova_constant(
            cs.namespace(|| format!("RA chunk {chunk_index} zero")),
            NovaScalar::zero(),
        )?;
        for (cycle_index, (cycle, cycle_weight)) in cycles.iter().zip(&cycle_weights).enumerate() {
            let index_weight = eq_from_allocated_bits(
                cs.namespace(|| format!("RA chunk {chunk_index} cycle {cycle_index} index")),
                &r_address[start..end],
                &cycle.lookup_index_bits
                    [bit_shift..bit_shift + one_hot.lookups_ra_virtual_log_k_chunk],
                "RA chunk index",
            )?;
            let term = mul_nums(
                cs.namespace(|| format!("RA chunk {chunk_index} cycle {cycle_index} term")),
                cycle_weight,
                &index_weight,
                "RA opening term",
            )?;
            claim = add_nums(
                cs.namespace(|| format!("RA chunk {chunk_index} cycle {cycle_index} sum")),
                &claim,
                &term,
                "RA opening sum",
            )?;
        }
        enforce_num_equal(
            cs.namespace(|| format!("RA opening {chunk_index}")),
            &claim,
            &opening_claims[table_count + chunk_index],
            "RA opening",
        );
    }

    let mut raf_flag_claim =
        alloc_nova_constant(cs.namespace(|| "RAF flag zero"), NovaScalar::zero())?;
    for (cycle_index, (cycle, cycle_weight)) in cycles.iter().zip(&cycle_weights).enumerate() {
        let raf = bit_as_num(&cycle.raf_identity_path);
        let term = mul_nums(
            cs.namespace(|| format!("RAF cycle {cycle_index} term")),
            cycle_weight,
            &raf,
            "RAF flag term",
        )?;
        raf_flag_claim = add_nums(
            cs.namespace(|| format!("RAF cycle {cycle_index} sum")),
            &raf_flag_claim,
            &term,
            "RAF flag sum",
        )?;
    }
    enforce_num_equal(
        cs.namespace(|| "RAF flag opening"),
        &raf_flag_claim,
        &opening_claims[table_count + ra_count],
        "RAF flag opening",
    );

    let mut val_claim = alloc_nova_constant(cs.namespace(|| "Val zero"), NovaScalar::zero())?;
    for table in 0..table_count {
        let table_eval = evaluate_table_mle_circuit(
            cs.namespace(|| format!("fixed table {table} evaluation")),
            table,
            r_address,
        )?;
        let term = mul_nums(
            cs.namespace(|| format!("fixed table {table} weighted")),
            &table_eval,
            &opening_claims[table],
            "weighted table evaluation",
        )?;
        val_claim = add_nums(
            cs.namespace(|| format!("Val table {table} sum")),
            &val_claim,
            &term,
            "Val table sum",
        )?;
    }

    let left = weighted_linear_point(
        cs.namespace(|| "left operand polynomial"),
        r_address,
        Some(0),
        "left operand polynomial",
    )?;
    let right = weighted_linear_point(
        cs.namespace(|| "right operand polynomial"),
        r_address,
        Some(1),
        "right operand polynomial",
    )?;
    let identity = weighted_linear_point(
        cs.namespace(|| "identity polynomial"),
        r_address,
        None,
        "identity polynomial",
    )?;
    let gamma_right = mul_nums(
        cs.namespace(|| "gamma times right"),
        gamma,
        &right,
        "gamma times right",
    )?;
    let operands = add_nums(
        cs.namespace(|| "left plus gamma right"),
        &left,
        &gamma_right,
        "left plus gamma right",
    )?;
    let one = alloc_nova_constant(cs.namespace(|| "endpoint one"), NovaScalar::one())?;
    let one_minus_raf = sub_nums(
        cs.namespace(|| "one minus RAF flag"),
        &one,
        &raf_flag_claim,
        "one minus RAF flag",
    )?;
    let regular_raf = mul_nums(
        cs.namespace(|| "regular RAF branch"),
        &one_minus_raf,
        &operands,
        "regular RAF branch",
    )?;
    let gamma_identity = mul_nums(
        cs.namespace(|| "gamma identity"),
        gamma,
        &identity,
        "gamma identity",
    )?;
    let identity_raf = mul_nums(
        cs.namespace(|| "identity RAF branch"),
        &raf_flag_claim,
        &gamma_identity,
        "identity RAF branch",
    )?;
    let raf_claim = add_nums(
        cs.namespace(|| "combined RAF claim"),
        &regular_raf,
        &identity_raf,
        "combined RAF claim",
    )?;
    let gamma_raf = mul_nums(
        cs.namespace(|| "gamma times RAF claim"),
        gamma,
        &raf_claim,
        "gamma times RAF claim",
    )?;
    let value_and_raf = add_nums(
        cs.namespace(|| "Val plus gamma RAF"),
        &val_claim,
        &gamma_raf,
        "Val plus gamma RAF",
    )?;

    let mut eq_reduction = one.clone();
    for (coordinate, (left_coordinate, right_coordinate)) in
        r_reduction.iter().zip(&r_cycle).enumerate()
    {
        let lr = mul_nums(
            cs.namespace(|| format!("reduction equality product {coordinate}")),
            left_coordinate,
            right_coordinate,
            "reduction equality product",
        )?;
        let twice_lr = scale_num(
            cs.namespace(|| format!("reduction equality twice {coordinate}")),
            &lr,
            NovaScalar::from(2),
            "reduction equality twice",
        )?;
        let one_minus_left = sub_nums(
            cs.namespace(|| format!("reduction equality one minus left {coordinate}")),
            &one,
            left_coordinate,
            "reduction equality one minus left",
        )?;
        let base = sub_nums(
            cs.namespace(|| format!("reduction equality base {coordinate}")),
            &one_minus_left,
            right_coordinate,
            "reduction equality base",
        )?;
        let factor = add_nums(
            cs.namespace(|| format!("reduction equality factor {coordinate}")),
            &base,
            &twice_lr,
            "reduction equality factor",
        )?;
        eq_reduction = mul_nums(
            cs.namespace(|| format!("reduction equality accumulate {coordinate}")),
            &eq_reduction,
            &factor,
            "reduction equality accumulate",
        )?;
    }
    let mut ra_claim = one;
    for (index, opening) in opening_claims[table_count..table_count + ra_count]
        .iter()
        .enumerate()
    {
        ra_claim = mul_nums(
            cs.namespace(|| format!("RA claim product {index}")),
            &ra_claim,
            opening,
            "RA claim product",
        )?;
    }
    let endpoint = mul_nums(
        cs.namespace(|| "endpoint equality times RA"),
        &eq_reduction,
        &ra_claim,
        "endpoint equality times RA",
    )?;
    let endpoint = mul_nums(
        cs.namespace(|| "endpoint relation"),
        &endpoint,
        &value_and_raf,
        "endpoint relation",
    )?;
    mul_nums(
        cs.namespace(|| "batched endpoint relation"),
        &endpoint,
        batching,
        "batched endpoint relation",
    )
}

impl DirectLookupStepCircuit {
    pub(super) fn synthesize_with_observations<CS: ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<NovaScalar>],
    ) -> Result<
        (
            Vec<AllocatedNum<NovaScalar>>,
            Vec<AllocatedDirectLookupCycle>,
        ),
        SynthesisError,
    > {
        if z.len() != DIRECT_LOOKUP_Z_ARITY {
            return Err(SynthesisError::Unsatisfiable(
                "direct lookup Nova state has invalid arity".to_string(),
            ));
        }
        let subclaim = self
            .subclaim
            .as_ref()
            .ok_or(SynthesisError::AssignmentMissing)?;
        let block = &subclaim.block;
        if block.cycle_capacity == 0
            || !block.cycle_capacity.is_power_of_two()
            || block.cycles.len() != block.cycle_capacity
        {
            return Err(SynthesisError::Unsatisfiable(
                "direct lookup block has invalid fixed circuit shape".to_string(),
            ));
        }
        let log_t = block.cycle_capacity.log_2();
        let one_hot = OneHotParams::new(log_t, 1, 1);
        let expected_rounds = LOG_K + log_t;
        let expected_degree = (LOG_K / one_hot.lookups_ra_virtual_log_k_chunk) + 2;
        let expected_openings = LookupTables::<{ common::constants::XLEN }>::COUNT
            + LOG_K / one_hot.lookups_ra_virtual_log_k_chunk
            + 1;
        if subclaim.r_reduction.len() != log_t
            || subclaim.input_claims.len() != 3
            || subclaim.sumcheck_challenges.len() != expected_rounds
            || subclaim.proof.compressed_polys.len() != expected_rounds
            || subclaim.degree_bound != expected_degree
            || subclaim.opening_claims.len() != expected_openings
        {
            return Err(SynthesisError::Unsatisfiable(
                "direct lookup subclaim has invalid recursive dimensions".to_string(),
            ));
        }

        let zero = alloc_nova_constant(cs.namespace(|| "state zero"), NovaScalar::zero())?;
        let one = alloc_nova_constant(cs.namespace(|| "state one"), NovaScalar::one())?;
        enforce_num_equal(
            cs.namespace(|| "input is not already closed"),
            &z[9],
            &zero,
            "input is not already closed",
        );
        let table_identity = nova_digest_limbs(&fixed_lookup_registry_commitment());
        for (limb, expected) in table_identity.into_iter().enumerate() {
            let expected = alloc_nova_constant(
                cs.namespace(|| format!("fixed lookup registry limb {limb}")),
                expected,
            )?;
            enforce_num_equal(
                cs.namespace(|| format!("fixed lookup registry limb {limb} is immutable")),
                &z[5 + limb],
                &expected,
                "fixed lookup registry limb is immutable",
            );
        }

        let block_index = alloc_witness_num(
            cs.namespace(|| "block index"),
            NovaScalar::from(block.block_index as u64),
        )?;
        let cycle_start = alloc_witness_num(
            cs.namespace(|| "global cycle start"),
            NovaScalar::from(block.global_cycle_start as u64),
        )?;
        let active_cycles = alloc_witness_num(
            cs.namespace(|| "active cycle count"),
            NovaScalar::from(block.active_cycles as u64),
        )?;
        enforce_num_equal(
            cs.namespace(|| "sequential block index"),
            &block_index,
            &z[0],
            "sequential block index",
        );
        enforce_num_equal(
            cs.namespace(|| "continuous global cycle"),
            &cycle_start,
            &z[2],
            "continuous global cycle",
        );

        let cycles = block
            .cycles
            .iter()
            .enumerate()
            .map(|(index, cycle)| {
                allocate_lookup_cycle(cs.namespace(|| format!("lookup cycle {index}")), cycle)
            })
            .collect::<Result<Vec<_>, _>>()?;
        cs.enforce(
            || "first row is active",
            |lc| lc + cycles[0].active.get_variable() - CS::one(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        for index in 1..cycles.len() {
            cs.enforce(
                || format!("active rows form a prefix {index}"),
                |lc| lc + cycles[index].active.get_variable(),
                |lc| lc + CS::one() - cycles[index - 1].active.get_variable(),
                |lc| lc,
            );
        }
        let active_sum = cycles
            .iter()
            .fold(LinearCombination::<NovaScalar>::zero(), |lc, cycle| {
                lc + cycle.active.get_variable()
            });
        cs.enforce(
            || "active bit sum equals active cycle count",
            |_| active_sum - active_cycles.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );

        let query_commitment = synthesize_query_root(
            cs.namespace(|| "block query commitment"),
            &block_index,
            &cycle_start,
            &active_cycles,
            &cycles,
        )?;
        let claimed_query_commitment = alloc_witness_num(
            cs.namespace(|| "claimed query commitment"),
            nova_from_fr(&subclaim.query_root)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "query commitment is trace-derived"),
            &query_commitment,
            &claimed_query_commitment,
            "query commitment is trace-derived",
        );

        let mut transcript = AllocatedRecursivePoseidonTranscriptState {
            state: z[7].clone(),
            n_rounds: z[8].clone(),
        };
        let table_root = alloc_nova_constant(
            cs.namespace(|| "lookup table Poseidon commitment"),
            nova_from_fr(&field_from_digest(&fixed_lookup_registry_commitment()))?,
        )?;
        for (label, value) in [
            ("table_root", &table_root),
            ("block_index", &block_index),
            ("cycle_start", &cycle_start),
            ("active_cycles", &active_cycles),
            ("query_root", &query_commitment),
        ] {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("lookup header {label}")),
                &transcript,
                value,
                label,
            )?;
        }

        let mut r_reduction = Vec::with_capacity(log_t);
        for index in 0..log_t {
            transcript = poseidon_challenge(
                cs.namespace(|| format!("lookup reduction challenge {index}")),
                &transcript,
            )?;
            let claimed = alloc_witness_num(
                cs.namespace(|| format!("claimed lookup reduction challenge {index}")),
                nova_from_fr(&subclaim.r_reduction[index])?,
            )?;
            enforce_num_equal(
                cs.namespace(|| format!("lookup reduction challenge binding {index}")),
                &transcript.state,
                &claimed,
                "lookup reduction challenge binding",
            );
            r_reduction.push(transcript.state.clone());
        }

        transcript = poseidon_challenge(cs.namespace(|| "lookup gamma"), &transcript)?;
        let gamma = transcript.state.clone();
        let claimed_gamma = alloc_witness_num(
            cs.namespace(|| "claimed lookup gamma"),
            nova_from_fr(&subclaim.gamma)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "lookup gamma binding"),
            &gamma,
            &claimed_gamma,
            "lookup gamma binding",
        );

        let output_values = cycles
            .iter()
            .map(|cycle| cycle.output.clone())
            .collect::<Vec<_>>();
        let left_values = cycles
            .iter()
            .map(|cycle| cycle.left_operand.clone())
            .collect::<Vec<_>>();
        let right_values = cycles
            .iter()
            .map(|cycle| cycle.right_operand.clone())
            .collect::<Vec<_>>();
        let derived_claims = [
            mle_from_values(
                cs.namespace(|| "lookup output reduction opening"),
                &output_values,
                &r_reduction,
                "lookup output reduction opening",
            )?,
            mle_from_values(
                cs.namespace(|| "left operand reduction opening"),
                &left_values,
                &r_reduction,
                "left operand reduction opening",
            )?,
            mle_from_values(
                cs.namespace(|| "right operand reduction opening"),
                &right_values,
                &r_reduction,
                "right operand reduction opening",
            )?,
        ];
        for (index, derived) in derived_claims.iter().enumerate() {
            let claimed = alloc_witness_num(
                cs.namespace(|| format!("claimed reduction opening {index}")),
                nova_from_fr(&subclaim.input_claims[index])?,
            )?;
            enforce_num_equal(
                cs.namespace(|| format!("reduction opening binding {index}")),
                derived,
                &claimed,
                "reduction opening binding",
            );
        }
        let gamma_squared = mul_nums(
            cs.namespace(|| "lookup gamma squared"),
            &gamma,
            &gamma,
            "lookup gamma squared",
        )?;
        let gamma_left = mul_nums(
            cs.namespace(|| "gamma times left opening"),
            &gamma,
            &derived_claims[1],
            "gamma times left opening",
        )?;
        let gamma_squared_right = mul_nums(
            cs.namespace(|| "gamma squared times right opening"),
            &gamma_squared,
            &derived_claims[2],
            "gamma squared times right opening",
        )?;
        let input_claim = add_nums(
            cs.namespace(|| "lookup input claim first sum"),
            &derived_claims[0],
            &gamma_left,
            "lookup input claim first sum",
        )?;
        let input_claim = add_nums(
            cs.namespace(|| "lookup input claim"),
            &input_claim,
            &gamma_squared_right,
            "lookup input claim",
        )?;
        transcript = poseidon_absorb(
            cs.namespace(|| "append lookup sumcheck claim"),
            &transcript,
            &input_claim,
            "sumcheck_claim",
        )?;
        transcript = poseidon_challenge(
            cs.namespace(|| "lookup sumcheck batching coefficient"),
            &transcript,
        )?;
        let batching = transcript.state.clone();
        let claimed_batching = alloc_witness_num(
            cs.namespace(|| "claimed batching coefficient"),
            nova_from_fr(&subclaim.batching_coefficient)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "batching coefficient binding"),
            &batching,
            &claimed_batching,
            "batching coefficient binding",
        );
        let initial_batched_claim = mul_nums(
            cs.namespace(|| "initial batched claim"),
            &input_claim,
            &batching,
            "initial batched claim",
        )?;

        let witness = sumcheck_witness(subclaim)?;
        let allocated_sumcheck = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "InstructionReadRaf sumcheck arithmetic"),
            &witness,
            0,
            expected_rounds,
            expected_degree,
        )?;
        enforce_num_equal(
            cs.namespace(|| "sumcheck initial claim is lookup-derived"),
            &allocated_sumcheck.initial_claim,
            &initial_batched_claim,
            "sumcheck initial claim is lookup-derived",
        );
        transcript = synthesize_recursive_clear_sumcheck_transcript(
            cs.namespace(|| "InstructionReadRaf sumcheck transcript"),
            &transcript,
            &allocated_sumcheck,
        )?;

        let opening_claims = subclaim
            .opening_claims
            .iter()
            .enumerate()
            .map(|(index, value)| {
                alloc_witness_num(
                    cs.namespace(|| format!("lookup endpoint opening {index}")),
                    nova_from_fr(value)?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expected_endpoint = synthesize_instruction_read_raf_endpoint(
            cs.namespace(|| "InstructionReadRaf endpoint relation"),
            &cycles,
            &r_reduction,
            &gamma,
            &batching,
            &allocated_sumcheck.challenges,
            &opening_claims,
            &one_hot,
        )?;
        enforce_num_equal(
            cs.namespace(|| "sumcheck endpoint is lookup-derived"),
            &allocated_sumcheck.final_claim,
            &expected_endpoint,
            "sumcheck endpoint is lookup-derived",
        );
        // These three openings were seeded before BatchedSumcheck::prove and
        // therefore precede the newly cached endpoint openings in Jolt's
        // pending-claim FIFO. The duplicate SpartanProductVirtualization
        // output aliases the first output opening and is not absorbed twice.
        for (index, opening) in derived_claims.iter().enumerate() {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("append reduction opening {index}")),
                &transcript,
                opening,
                "opening_claim",
            )?;
        }
        for (index, opening) in opening_claims.iter().enumerate() {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("append lookup opening {index}")),
                &transcript,
                opening,
                "opening_claim",
            )?;
        }
        let claimed_transcript_state = alloc_witness_num(
            cs.namespace(|| "claimed final lookup transcript state"),
            nova_from_fr(&field_from_digest(&subclaim.transcript_state_after))?,
        )?;
        let claimed_transcript_round = alloc_witness_num(
            cs.namespace(|| "claimed final lookup transcript round"),
            NovaScalar::from(subclaim.transcript_round_after as u64),
        )?;
        enforce_num_equal(
            cs.namespace(|| "final lookup transcript state binding"),
            &transcript.state,
            &claimed_transcript_state,
            "final lookup transcript state binding",
        );
        enforce_num_equal(
            cs.namespace(|| "final lookup transcript round binding"),
            &transcript.n_rounds,
            &claimed_transcript_round,
            "final lookup transcript round binding",
        );

        let next_block_count = add_nums(
            cs.namespace(|| "increment block count"),
            &z[0],
            &one,
            "increment block count",
        )?;
        let next_total_cycles = add_nums(
            cs.namespace(|| "accumulate total cycles"),
            &z[1],
            &active_cycles,
            "accumulate total cycles",
        )?;
        let next_global_cycle = add_nums(
            cs.namespace(|| "advance global cycle"),
            &z[2],
            &active_cycles,
            "advance global cycle",
        )?;
        let final_step = AllocatedBit::alloc(cs.namespace(|| "final step"), Some(self.final_step))?;
        let terminated =
            AllocatedBit::alloc(cs.namespace(|| "terminated trace"), Some(block.terminated))?;
        let final_step_num = bit_as_num(&final_step);
        let terminated_num = bit_as_num(&terminated);
        enforce_num_equal(
            cs.namespace(|| "lookup closure equals trace termination"),
            &final_step_num,
            &terminated_num,
            "lookup closure equals trace termination",
        );

        Ok((
            vec![
                next_block_count,
                next_total_cycles,
                next_global_cycle,
                z[3].clone(),
                z[4].clone(),
                z[5].clone(),
                z[6].clone(),
                transcript.state,
                transcript.n_rounds,
                terminated_num,
            ],
            cycles,
        ))
    }
}

impl StepCircuit<NovaScalar> for DirectLookupStepCircuit {
    fn arity(&self) -> usize {
        DIRECT_LOOKUP_Z_ARITY
    }

    fn synthesize<CS: ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<NovaScalar>],
    ) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
        self.synthesize_with_observations(cs, z)
            .map(|(output, _)| output)
    }
}

fn seed_prover_openings(
    accumulator: &mut ProverOpeningAccumulator<Fr>,
    r_reduction: &[Fr],
    claims: [Fr; 3],
) {
    let point =
        OpeningPoint::<BIG_ENDIAN, Fr>::new(r_reduction.iter().copied().map(Into::into).collect());
    accumulator.append_virtual(
        VirtualPolynomial::LookupOutput,
        SumcheckId::InstructionClaimReduction,
        point.clone(),
        claims[0],
    );
    accumulator.append_virtual(
        VirtualPolynomial::LeftLookupOperand,
        SumcheckId::InstructionClaimReduction,
        point.clone(),
        claims[1],
    );
    accumulator.append_virtual(
        VirtualPolynomial::RightLookupOperand,
        SumcheckId::InstructionClaimReduction,
        point.clone(),
        claims[2],
    );
    accumulator.append_virtual(
        VirtualPolynomial::LookupOutput,
        SumcheckId::SpartanProductVirtualization,
        point,
        claims[0],
    );
}

fn padded_trace(block: &TraceBlock, capacity: usize) -> Arc<Vec<Cycle>> {
    let mut trace = block.cycles.clone();
    trace.resize(capacity, Cycle::NoOp);
    Arc::new(trace)
}

fn native_output_openings(
    block: &DirectLookupBlockWitness,
    sumcheck_challenges: &[Fr],
    one_hot: &OneHotParams,
) -> Vec<(OpeningId, Fr)> {
    let (r_address, r_cycle_binding) = sumcheck_challenges.split_at(LOG_K);
    let r_cycle = r_cycle_binding.iter().copied().rev().collect::<Vec<_>>();
    let cycle_weights = EqPolynomial::<Fr>::evals(&r_cycle);
    let mut result = Vec::new();

    for table_id in 0..LookupTables::<{ common::constants::XLEN }>::COUNT {
        let claim = cycle_weights
            .iter()
            .zip(&block.cycles)
            .filter(|(_, cycle)| cycle.table_id as usize == table_id)
            .map(|(weight, _)| *weight)
            .sum();
        result.push((
            OpeningId::virt(
                VirtualPolynomial::LookupTableFlag(table_id),
                SumcheckId::InstructionReadRaf,
            ),
            claim,
        ));
    }

    for (chunk_index, chunk_point) in r_address
        .chunks(one_hot.lookups_ra_virtual_log_k_chunk)
        .enumerate()
    {
        let chunk_weights = EqPolynomial::<Fr>::evals(chunk_point);
        let shift = one_hot.lookups_ra_virtual_log_k_chunk
            * (LOG_K / one_hot.lookups_ra_virtual_log_k_chunk - 1 - chunk_index);
        let mask = (1u128 << one_hot.lookups_ra_virtual_log_k_chunk) - 1;
        let claim = cycle_weights
            .iter()
            .zip(&block.cycles)
            .map(|(cycle_weight, cycle)| {
                let index = ((cycle.lookup_index >> shift) & mask) as usize;
                *cycle_weight * chunk_weights[index]
            })
            .sum();
        result.push((
            OpeningId::virt(
                VirtualPolynomial::InstructionRa(chunk_index),
                SumcheckId::InstructionReadRaf,
            ),
            claim,
        ));
    }

    let raf_claim = cycle_weights
        .iter()
        .zip(&block.cycles)
        .filter(|(_, cycle)| cycle.raf_identity_path)
        .map(|(weight, _)| *weight)
        .sum();
    result.push((
        OpeningId::virt(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        ),
        raf_claim,
    ));
    result
}

fn native_output_opening_ids(one_hot: &OneHotParams) -> Vec<OpeningId> {
    let mut ids = (0..LookupTables::<{ common::constants::XLEN }>::COUNT)
        .map(|table_id| {
            OpeningId::virt(
                VirtualPolynomial::LookupTableFlag(table_id),
                SumcheckId::InstructionReadRaf,
            )
        })
        .collect::<Vec<_>>();
    ids.extend(
        (0..LOG_K / one_hot.lookups_ra_virtual_log_k_chunk).map(|chunk_index| {
            OpeningId::virt(
                VirtualPolynomial::InstructionRa(chunk_index),
                SumcheckId::InstructionReadRaf,
            )
        }),
    );
    ids.push(OpeningId::virt(
        VirtualPolynomial::InstructionRafFlag,
        SumcheckId::InstructionReadRaf,
    ));
    ids
}

fn seed_verifier_openings(
    accumulator: &mut VerifierOpeningAccumulator<Fr>,
    r_reduction: &[Fr],
    claims: [Fr; 3],
    outputs: &[(OpeningId, Fr)],
) {
    let empty = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
    for (id, claim) in outputs {
        accumulator.openings.insert(*id, (empty.clone(), *claim));
    }
    for (id, claim) in [
        (
            OpeningId::virt(
                VirtualPolynomial::LookupOutput,
                SumcheckId::InstructionClaimReduction,
            ),
            claims[0],
        ),
        (
            OpeningId::virt(
                VirtualPolynomial::LeftLookupOperand,
                SumcheckId::InstructionClaimReduction,
            ),
            claims[1],
        ),
        (
            OpeningId::virt(
                VirtualPolynomial::RightLookupOperand,
                SumcheckId::InstructionClaimReduction,
            ),
            claims[2],
        ),
        (
            OpeningId::virt(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanProductVirtualization,
            ),
            claims[0],
        ),
    ] {
        accumulator.openings.insert(id, (empty.clone(), claim));
    }

    let point =
        OpeningPoint::<BIG_ENDIAN, Fr>::new(r_reduction.iter().copied().map(Into::into).collect());
    accumulator.append_virtual(
        VirtualPolynomial::LookupOutput,
        SumcheckId::InstructionClaimReduction,
        point.clone(),
    );
    accumulator.append_virtual(
        VirtualPolynomial::LeftLookupOperand,
        SumcheckId::InstructionClaimReduction,
        point.clone(),
    );
    accumulator.append_virtual(
        VirtualPolynomial::RightLookupOperand,
        SumcheckId::InstructionClaimReduction,
        point.clone(),
    );
    accumulator.append_virtual(
        VirtualPolynomial::LookupOutput,
        SumcheckId::SpartanProductVirtualization,
        point,
    );
}

pub(super) fn prove_native_subclaim(
    block: &TraceBlock,
    capacity: usize,
    transcript: &mut PoseidonTranscript,
) -> Result<DirectLookupSubclaim, DirectChunkedError> {
    let witness = DirectLookupBlockWitness::from_trace_block(block, capacity)?;
    let root = query_root(&witness);
    let table_root = field_from_digest(&fixed_lookup_registry_commitment());
    append_block_header(transcript, table_root, &witness, root);

    let log_t = capacity.log_2();
    let r_reduction = transcript.challenge_vector::<Fr>(log_t);
    let claims = mle_claims(&witness, &r_reduction);
    let mut accumulator = ProverOpeningAccumulator::<Fr>::new(log_t);
    seed_prover_openings(&mut accumulator, &r_reduction, claims);
    let one_hot = OneHotParams::new(log_t, 1, 1);
    let params = InstructionReadRafSumcheckParams::new(log_t, &one_hot, &accumulator, transcript);
    let gamma = params.gamma;
    let degree_bound = (LOG_K / one_hot.lookups_ra_virtual_log_k_chunk) + 2;
    let input_claim = claims[0] + gamma * claims[1] + (gamma * gamma) * claims[2];
    let mut batching_transcript = transcript.clone();
    batching_transcript.append_scalar(b"sumcheck_claim", &input_claim);
    let batching_coefficient = batching_transcript.challenge_scalar::<Fr>();
    let mut prover =
        InstructionReadRafSumcheckProver::initialize(params, padded_trace(block, capacity));
    let (proof, challenges, initial_batched_claim) =
        BatchedSumcheck::prove(vec![&mut prover], &mut accumulator, transcript);
    let opening_claims = native_output_openings(
        &witness,
        &challenges
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<Fr>>(),
        &one_hot,
    )
    .into_iter()
    .map(|(_, claim)| claim)
    .collect();
    let final_sumcheck_claim = proof
        .compressed_polys
        .iter()
        .zip(challenges.iter())
        .fold(initial_batched_claim, |claim, (poly, challenge)| {
            poly.eval_from_hint(&claim, challenge)
        });
    let challenges = challenges.into_iter().map(Into::into).collect::<Vec<Fr>>();
    let subclaim = DirectLookupSubclaim {
        block: witness,
        query_root: root,
        r_reduction,
        input_claims: claims,
        gamma,
        batching_coefficient,
        initial_batched_claim,
        final_sumcheck_claim,
        proof,
        sumcheck_challenges: challenges,
        degree_bound,
        opening_claims,
        transcript_state_after: transcript.state,
        transcript_round_after: transcript.n_rounds,
    };
    Ok(subclaim)
}

pub(super) fn verify_native_subclaim(
    subclaim: &DirectLookupSubclaim,
    transcript: &mut PoseidonTranscript,
) -> Result<(), DirectChunkedError> {
    if subclaim.block.cycle_capacity == 0
        || !subclaim.block.cycle_capacity.is_power_of_two()
        || subclaim.block.cycles.len() != subclaim.block.cycle_capacity
        || subclaim.block.active_cycles == 0
        || subclaim.block.active_cycles > subclaim.block.cycle_capacity
        || subclaim.block.cycles[..subclaim.block.active_cycles]
            .iter()
            .any(|cycle| !cycle.active)
        || subclaim.block.cycles[subclaim.block.active_cycles..]
            .iter()
            .any(|cycle| cycle != &DirectLookupCycleWitness::default())
    {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: subclaim.block.block_index,
            reason: "lookup subclaim has non-canonical active/padding layout".to_string(),
        });
    }

    let root = query_root(&subclaim.block);
    if root != subclaim.query_root {
        return Err(DirectChunkedError::InvalidProofShape(
            "lookup query commitment mismatch".to_string(),
        ));
    }
    let table_root = field_from_digest(&fixed_lookup_registry_commitment());
    append_block_header(transcript, table_root, &subclaim.block, root);
    let log_t = subclaim.block.cycle_capacity.log_2();
    let r_reduction = transcript.challenge_vector::<Fr>(log_t);
    let claims = mle_claims(&subclaim.block, &r_reduction);
    if r_reduction != subclaim.r_reduction || claims != subclaim.input_claims {
        return Err(DirectChunkedError::InvalidProofShape(
            "lookup reduction point or input opening mismatch".to_string(),
        ));
    }

    let one_hot = OneHotParams::new(log_t, 1, 1);
    let expected_degree_bound = (LOG_K / one_hot.lookups_ra_virtual_log_k_chunk) + 2;
    if subclaim.degree_bound != expected_degree_bound
        || subclaim.proof.compressed_polys.len() != LOG_K + log_t
        || subclaim.sumcheck_challenges.len() != LOG_K + log_t
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "lookup sumcheck has a non-canonical degree or round count".to_string(),
        ));
    }
    let outputs = native_output_openings(&subclaim.block, &subclaim.sumcheck_challenges, &one_hot);
    if outputs.iter().map(|(_, claim)| *claim).collect::<Vec<_>>() != subclaim.opening_claims {
        return Err(DirectChunkedError::InvalidProofShape(
            "lookup output opening vector mismatch".to_string(),
        ));
    }
    let mut accumulator = VerifierOpeningAccumulator::<Fr>::new(log_t, false);
    seed_verifier_openings(&mut accumulator, &r_reduction, claims, &outputs);
    let verifier =
        InstructionReadRafSumcheckVerifier::new(log_t, &one_hot, &accumulator, transcript);
    let expected_input =
        claims[0] + subclaim.gamma * claims[1] + (subclaim.gamma * subclaim.gamma) * claims[2];
    let mut batching_transcript = transcript.clone();
    batching_transcript.append_scalar(b"sumcheck_claim", &expected_input);
    let expected_batching = batching_transcript.challenge_scalar::<Fr>();
    if expected_batching != subclaim.batching_coefficient
        || expected_input * expected_batching != subclaim.initial_batched_claim
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "lookup batched sumcheck initial claim mismatch".to_string(),
        ));
    }
    let challenges = BatchedSumcheck::verify_standard(
        &subclaim.proof,
        vec![&verifier],
        &mut accumulator,
        transcript,
    )
    .map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!(
            "native InstructionReadRaf verification failed: {error}"
        ))
    })?
    .into_iter()
    .map(Into::into)
    .collect::<Vec<Fr>>();
    if challenges != subclaim.sumcheck_challenges
        || subclaim
            .proof
            .compressed_polys
            .iter()
            .zip(subclaim.sumcheck_challenges.iter())
            .fold(
                subclaim.initial_batched_claim,
                |claim, (poly, challenge)| {
                    let challenge = (*challenge).into();
                    poly.eval_from_hint(&claim, &challenge)
                },
            )
            != subclaim.final_sumcheck_claim
        || transcript.state != subclaim.transcript_state_after
        || transcript.n_rounds != subclaim.transcript_round_after
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "lookup Fiat-Shamir transcript mismatch".to_string(),
        ));
    }
    Ok(())
}

fn compact_lookup_id(label: &[u8], index: usize) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(super::BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(b"lookup-opening");
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update((index as u64).to_le_bytes());
    hasher.finalize().into()
}

pub(super) fn compact_lookup_deferred_claims(
    query_commitment: FieldElement,
    reduction_point: &[FieldElement],
    input_claims: &[FieldElement; 3],
    sumcheck_point: &[FieldElement],
    output_claims: &[FieldElement],
) -> Vec<DeferredPcsClaim> {
    let mut claims = input_claims
        .iter()
        .enumerate()
        .map(|(index, value)| DeferredPcsClaim {
            relation: BlockRelation::LookupLasso,
            polynomial_id: compact_lookup_id(b"input", index),
            commitment_id: query_commitment.0,
            opening_point: reduction_point.to_vec(),
            claimed_value: *value,
        })
        .collect::<Vec<_>>();
    let one_hot = OneHotParams::new(reduction_point.len(), 1, 1);
    let table_count = LookupTables::<{ common::constants::XLEN }>::COUNT;
    let ra_count = LOG_K / one_hot.lookups_ra_virtual_log_k_chunk;
    let r_address = &sumcheck_point[..LOG_K];
    let r_cycle = sumcheck_point[LOG_K..]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    claims.extend(output_claims.iter().enumerate().map(|(index, value)| {
        let opening_point = if index < table_count || index == table_count + ra_count {
            r_cycle.clone()
        } else {
            let chunk_index = index - table_count;
            let start = chunk_index * one_hot.lookups_ra_virtual_log_k_chunk;
            let end = start + one_hot.lookups_ra_virtual_log_k_chunk;
            [r_address[start..end].to_vec(), r_cycle.clone()].concat()
        };
        DeferredPcsClaim {
            relation: BlockRelation::LookupLasso,
            polynomial_id: compact_lookup_id(b"output", index),
            commitment_id: query_commitment.0,
            opening_point,
            claimed_value: *value,
        }
    }));
    claims
}

/// Generates the compact D11 Lasso proof. The D8 cycle witness exists only
/// while the native Jolt prover is running and is absent from the returned type.
pub(super) fn prove_compact_lookup_block(
    block: &TraceBlock,
    capacity: usize,
    transcript: &mut PoseidonTranscript,
) -> Result<(LookupBlockProof, TranscriptCheckpoint, TranscriptCheckpoint), DirectChunkedError> {
    let before = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    let subclaim = prove_native_subclaim(block, capacity, transcript)?;
    let after = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    let mut compressed_proof = Vec::new();
    subclaim
        .proof
        .serialize_compressed(&mut compressed_proof)
        .map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!(
                "compact lookup sumcheck serialization failed: {error}"
            ))
        })?;
    let query_commitment = FieldElement::from_fr(&subclaim.query_root);
    let reduction_point = subclaim
        .r_reduction
        .iter()
        .map(FieldElement::from_fr)
        .collect::<Vec<_>>();
    let input_claims = subclaim
        .input_claims
        .map(|value| FieldElement::from_fr(&value));
    let challenges = subclaim
        .sumcheck_challenges
        .iter()
        .map(FieldElement::from_fr)
        .collect::<Vec<_>>();
    let output_claims = subclaim
        .opening_claims
        .iter()
        .map(FieldElement::from_fr)
        .collect::<Vec<_>>();
    let deferred = compact_lookup_deferred_claims(
        query_commitment,
        &reduction_point,
        &input_claims,
        &challenges,
        &output_claims,
    );
    let proof = LookupBlockProof {
        query_commitment,
        table_commitment: fixed_lookup_registry_commitment(),
        accumulator_before: FieldElement(before.state),
        accumulator_round_before: before.round,
        accumulator_after: FieldElement(after.state),
        accumulator_round_after: after.round,
        reduction_point,
        input_claims,
        gamma: FieldElement::from_fr(&subclaim.gamma),
        batching_coefficient: FieldElement::from_fr(&subclaim.batching_coefficient),
        output_claims,
        sumcheck: CompactSumcheckProof {
            rounds: subclaim.sumcheck_challenges.len() as u32,
            degree_bound: subclaim.degree_bound as u32,
            initial_claim: FieldElement::from_fr(&subclaim.initial_batched_claim),
            final_claim: FieldElement::from_fr(&subclaim.final_sumcheck_claim),
            compressed_proof,
            challenges,
            opening_claims: deferred,
        },
    };
    Ok((proof, before, after))
}

/// Verifies the real clear InstructionReadRaf sumcheck without receiving trace
/// rows. Endpoint polynomial openings remain explicit deferred PCS claims.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_compact_lookup_block(
    proof: &LookupBlockProof,
    block_index: usize,
    global_cycle_start: usize,
    active_cycles: usize,
    capacity: usize,
    transcript: &mut PoseidonTranscript,
) -> Result<(TranscriptCheckpoint, TranscriptCheckpoint), DirectChunkedError> {
    if capacity == 0
        || !capacity.is_power_of_two()
        || active_cycles == 0
        || active_cycles > capacity
        || proof.table_commitment != fixed_lookup_registry_commitment()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact lookup block shape or table commitment mismatch".to_string(),
        ));
    }
    let before = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    if proof.accumulator_before != FieldElement(before.state) {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact lookup transcript input mismatch".to_string(),
        ));
    }
    let root = proof.query_commitment.to_fr();
    let table_root = field_from_digest(&proof.table_commitment);
    append_compact_block_header(
        transcript,
        table_root,
        block_index,
        global_cycle_start,
        active_cycles,
        root,
    );

    let log_t = capacity.log_2();
    let r_reduction = transcript.challenge_vector::<Fr>(log_t);
    let supplied_reduction = proof
        .reduction_point
        .iter()
        .map(|value| value.to_fr())
        .collect::<Vec<_>>();
    if supplied_reduction != r_reduction {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact lookup reduction challenge mismatch".to_string(),
        ));
    }
    let input_claims = proof.input_claims.map(FieldElement::to_fr);
    let output_claims = proof
        .output_claims
        .iter()
        .map(|value| value.to_fr())
        .collect::<Vec<_>>();
    let one_hot = OneHotParams::new(log_t, 1, 1);
    let expected_degree_bound = (LOG_K / one_hot.lookups_ra_virtual_log_k_chunk) + 2;
    let output_ids = native_output_opening_ids(&one_hot);
    if proof.sumcheck.degree_bound as usize != expected_degree_bound
        || proof.sumcheck.rounds as usize != LOG_K + log_t
        || proof.sumcheck.challenges.len() != LOG_K + log_t
        || output_claims.len() != output_ids.len()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact lookup sumcheck shape mismatch".to_string(),
        ));
    }
    let outputs = output_ids
        .into_iter()
        .zip(output_claims.iter().copied())
        .collect::<Vec<_>>();
    let expected_deferred = compact_lookup_deferred_claims(
        proof.query_commitment,
        &proof.reduction_point,
        &proof.input_claims,
        &proof.sumcheck.challenges,
        &proof.output_claims,
    );
    if proof.sumcheck.opening_claims != expected_deferred {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact lookup deferred opening set mismatch".to_string(),
        ));
    }

    let mut accumulator = VerifierOpeningAccumulator::<Fr>::new(log_t, false);
    seed_verifier_openings(&mut accumulator, &r_reduction, input_claims, &outputs);
    let verifier =
        InstructionReadRafSumcheckVerifier::new(log_t, &one_hot, &accumulator, transcript);
    let gamma = proof.gamma.to_fr();
    if gamma != verifier.gamma() {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact lookup gamma mismatch".to_string(),
        ));
    }
    let expected_input =
        input_claims[0] + gamma * input_claims[1] + gamma * gamma * input_claims[2];
    let mut batching_transcript = transcript.clone();
    batching_transcript.append_scalar(b"sumcheck_claim", &expected_input);
    let expected_batching = batching_transcript.challenge_scalar::<Fr>();
    if proof.batching_coefficient.to_fr() != expected_batching
        || proof.sumcheck.initial_claim.to_fr() != expected_input * expected_batching
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact lookup batched initial claim mismatch".to_string(),
        ));
    }
    let clear = ClearSumcheckProof::<Fr, PoseidonTranscript>::deserialize_compressed(
        proof.sumcheck.compressed_proof.as_slice(),
    )
    .map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!(
            "compact lookup sumcheck deserialization failed: {error}"
        ))
    })?;
    let challenges =
        BatchedSumcheck::verify_standard(&clear, vec![&verifier], &mut accumulator, transcript)
            .map_err(|error| {
                DirectChunkedError::InvalidProofShape(format!(
                    "compact InstructionReadRaf verification failed: {error}"
                ))
            })?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Fr>>();
    let supplied_challenges = proof
        .sumcheck
        .challenges
        .iter()
        .map(|value| value.to_fr())
        .collect::<Vec<_>>();
    let final_claim = clear.compressed_polys.iter().zip(&challenges).fold(
        proof.sumcheck.initial_claim.to_fr(),
        |claim, (poly, challenge)| {
            let challenge = (*challenge).into();
            poly.eval_from_hint(&claim, &challenge)
        },
    );
    let after = TranscriptCheckpoint {
        state: transcript.state,
        round: transcript.n_rounds as u64,
    };
    if challenges != supplied_challenges
        || final_claim != proof.sumcheck.final_claim.to_fr()
        || proof.accumulator_after != FieldElement(after.state)
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "compact lookup Fiat-Shamir transcript mismatch".to_string(),
        ));
    }
    Ok((before, after))
}

pub(super) fn prove_and_verify_native_lookup_blocks<I>(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    blocks: I,
) -> Result<Vec<DirectLookupSubclaim>, DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    if preprocessing.lookup_table_commitment != fixed_lookup_registry_commitment() {
        return Err(DirectChunkedError::InvalidConfiguration(
            "direct preprocessing lookup registry commitment mismatch".to_string(),
        ));
    }
    let mut prover_transcript = PoseidonTranscript::new(DIRECT_LOOKUP_TRANSCRIPT_DOMAIN);
    let mut verifier_transcript = PoseidonTranscript::new(DIRECT_LOOKUP_TRANSCRIPT_DOMAIN);
    let mut subclaims = Vec::new();
    for block in blocks {
        validate_block(&block, capacity)?;
        let subclaim = prove_native_subclaim(&block, capacity, &mut prover_transcript)?;
        verify_native_subclaim(&subclaim, &mut verifier_transcript)?;
        subclaims.push(subclaim);
    }
    if subclaims.is_empty() {
        return Err(DirectChunkedError::EmptyTrace);
    }
    Ok(subclaims)
}

fn nova_stage_error(context: &str, error: impl std::fmt::Debug) -> DirectChunkedError {
    DirectChunkedError::InvalidProofShape(format!("{context}: {error:?}"))
}

pub(super) fn validate_lookup_subclaim_sequence(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    subclaims: &[DirectLookupSubclaim],
) -> Result<(usize, usize), DirectChunkedError> {
    if subclaims.is_empty() {
        return Err(DirectChunkedError::EmptyTrace);
    }
    if capacity == 0 || !capacity.is_power_of_two() {
        return Err(DirectChunkedError::InvalidConfiguration(
            "direct lookup capacity must be a non-zero power of two".to_string(),
        ));
    }
    let mut total_cycles = 0usize;
    for (position, subclaim) in subclaims.iter().enumerate() {
        let block = &subclaim.block;
        if block.cycle_capacity != capacity || block.block_index != position {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: format!("expected block {position} with fixed lookup capacity {capacity}"),
            });
        }
        if block.global_cycle_start != total_cycles {
            return Err(DirectChunkedError::DiscontinuousBlocks {
                previous_block: position.saturating_sub(1),
                next_block: position,
                reason: "lookup global cycle boundary is discontinuous".to_string(),
            });
        }
        let is_last = position + 1 == subclaims.len();
        if block.terminated != is_last {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: if is_last {
                    "final lookup block must terminate the trace".to_string()
                } else {
                    "only the final lookup block may terminate the trace".to_string()
                },
            });
        }
        total_cycles = total_cycles
            .checked_add(block.active_cycles)
            .ok_or_else(|| DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: "lookup cycle counter overflow".to_string(),
            })?;
        if total_cycles > preprocessing.max_padded_trace_length {
            return Err(DirectChunkedError::TraceTooLong {
                observed: total_cycles,
                maximum: preprocessing.max_padded_trace_length,
            });
        }
    }
    Ok((subclaims.len(), total_cycles))
}

fn setup_direct_lookup_public_params(
    circuit: &DirectLookupStepCircuit,
) -> Result<DirectLookupPublicParams, DirectChunkedError> {
    DirectLookupPublicParams::setup(
        circuit,
        &*nova_snark::traits::snark::default_ck_hint::<NovaPrimaryEngine>(),
        &*nova_snark::traits::snark::default_ck_hint::<NovaSecondaryEngine>(),
    )
    .map_err(|error| nova_stage_error("direct lookup Nova public-parameter setup failed", error))
}

/// Builds and self-verifies the complete D2 artifact:
/// trace blocks -> native Jolt InstructionReadRaf subclaims -> Nova folding ->
/// Spartan-compressed final recursive proof.
pub fn prove_direct_lookup_stage<I>(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    blocks: I,
) -> Result<DirectLookupStageProof, DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    if preprocessing.lookup_table_commitment != fixed_lookup_registry_commitment() {
        return Err(DirectChunkedError::InvalidConfiguration(
            "direct preprocessing lookup registry commitment mismatch".to_string(),
        ));
    }
    let subclaims = prove_and_verify_native_lookup_blocks(preprocessing, capacity, blocks)?;
    let (block_count, total_cycles) =
        validate_lookup_subclaim_sequence(preprocessing, capacity, &subclaims)?;
    let z0 = direct_initial_z(preprocessing);
    let first_circuit =
        DirectLookupStepCircuit::for_subclaim(subclaims[0].clone(), block_count == 1);
    let pp = setup_direct_lookup_public_params(&first_circuit)?;
    let mut recursive = DirectLookupNovaSnark::new(&pp, &first_circuit, &z0)
        .map_err(|error| nova_stage_error("direct lookup Nova initialization failed", error))?;
    recursive
        .prove_step(&pp, &first_circuit)
        .map_err(|error| nova_stage_error("direct lookup Nova first step failed", error))?;
    for (position, subclaim) in subclaims.iter().enumerate().skip(1) {
        let circuit =
            DirectLookupStepCircuit::for_subclaim(subclaim.clone(), position + 1 == block_count);
        recursive.prove_step(&pp, &circuit).map_err(|error| {
            nova_stage_error(&format!("direct lookup Nova step {position} failed"), error)
        })?;
    }
    let recursive_output = recursive
        .verify(&pp, block_count, &z0)
        .map_err(|error| nova_stage_error("direct lookup Nova self-verification failed", error))?;
    if recursive_output.len() != DIRECT_LOOKUP_Z_ARITY
        || recursive_output[0] != NovaScalar::from(block_count as u64)
        || recursive_output[1] != NovaScalar::from(total_cycles as u64)
        || recursive_output[2] != NovaScalar::from(total_cycles as u64)
        || recursive_output[3] != z0[3]
        || recursive_output[4] != z0[4]
        || recursive_output[5] != z0[5]
        || recursive_output[6] != z0[6]
        || recursive_output[9] != NovaScalar::one()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct lookup Nova output does not close the public stage statement".to_string(),
        ));
    }

    let (pk, vk) = DirectLookupCompressedSnark::setup(&pp)
        .map_err(|error| nova_stage_error("direct lookup Spartan key setup failed", error))?;
    let compressed = DirectLookupCompressedSnark::prove(&pp, &pk, &recursive)
        .map_err(|error| nova_stage_error("direct lookup Spartan proving failed", error))?;
    let compressed_output = compressed.verify(&vk, block_count, &z0).map_err(|error| {
        nova_stage_error("direct lookup Spartan self-verification failed", error)
    })?;
    if compressed_output != recursive_output {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct lookup Spartan and Nova outputs differ".to_string(),
        ));
    }

    let nova_recursive_snark = postcard::to_stdvec(&recursive)
        .map_err(|error| nova_stage_error("direct lookup Nova serialization failed", error))?;
    let spartan_proof = postcard::to_stdvec(&compressed)
        .map_err(|error| nova_stage_error("direct lookup Spartan serialization failed", error))?;
    let relations = DirectRelation::ALL
        .into_iter()
        .map(|relation| {
            let state = match relation {
                DirectRelation::Lookup => DirectRelationState::Proven,
                DirectRelation::Pcs => DirectRelationState::Deferred,
                DirectRelation::Register | DirectRelation::Ram | DirectRelation::Cpu => {
                    DirectRelationState::Unsupported
                }
            };
            (relation, state)
        })
        .collect();
    let proof = DirectLookupStageProof {
        statement: DirectLookupStageStatement {
            program_digest: preprocessing.program_digest,
            lookup_table_commitment: preprocessing.lookup_table_commitment,
            block_count,
            total_cycles,
            final_global_cycle: total_cycles,
            terminated: true,
            relations,
        },
        subclaims,
        nova_recursive_snark,
        spartan_proof,
        initial_z: z0.iter().copied().map(nova_to_storage).collect(),
        final_z: recursive_output
            .iter()
            .copied()
            .map(nova_to_storage)
            .collect(),
    };
    verify_direct_lookup_stage(preprocessing, capacity, &proof)?;
    Ok(proof)
}

/// Verifies both retained native audit subclaims and the recursive/Spartan D2
/// proof. No host-provided `accepted` bit or opaque receipt is trusted.
pub fn verify_direct_lookup_stage(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    proof: &DirectLookupStageProof,
) -> Result<(), DirectChunkedError> {
    if preprocessing.lookup_table_commitment != fixed_lookup_registry_commitment()
        || proof.statement.program_digest != preprocessing.program_digest
        || proof.statement.lookup_table_commitment != preprocessing.lookup_table_commitment
        || !proof.statement.terminated
        || proof.statement.block_count != proof.subclaims.len()
        || proof.statement.final_global_cycle != proof.statement.total_cycles
        || proof.statement.relations.get(&DirectRelation::Lookup)
            != Some(&DirectRelationState::Proven)
        || proof.statement.relations.get(&DirectRelation::Pcs)
            != Some(&DirectRelationState::Deferred)
        || DirectRelation::ALL.into_iter().any(|relation| {
            let expected = match relation {
                DirectRelation::Lookup => DirectRelationState::Proven,
                DirectRelation::Pcs => DirectRelationState::Deferred,
                DirectRelation::Register | DirectRelation::Ram | DirectRelation::Cpu => {
                    DirectRelationState::Unsupported
                }
            };
            proof.statement.relations.get(&relation) != Some(&expected)
        })
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct lookup public statement is inconsistent".to_string(),
        ));
    }
    let (block_count, total_cycles) =
        validate_lookup_subclaim_sequence(preprocessing, capacity, &proof.subclaims)?;
    if block_count != proof.statement.block_count || total_cycles != proof.statement.total_cycles {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct lookup statement counters do not match block subclaims".to_string(),
        ));
    }

    let mut transcript = PoseidonTranscript::new(DIRECT_LOOKUP_TRANSCRIPT_DOMAIN);
    for subclaim in &proof.subclaims {
        verify_native_subclaim(subclaim, &mut transcript)?;
    }
    let native_transcript_state = nova_from_fr(&field_from_digest(&transcript.state))
        .map_err(|error| nova_stage_error("lookup transcript field conversion failed", error))?;
    let native_transcript_round = NovaScalar::from(transcript.n_rounds as u64);

    let z0 = direct_initial_z(preprocessing);
    let expected_initial = z0.iter().copied().map(nova_to_storage).collect::<Vec<_>>();
    if proof.initial_z != expected_initial
        || proof.initial_z.len() != DIRECT_LOOKUP_Z_ARITY
        || proof.final_z.len() != DIRECT_LOOKUP_Z_ARITY
        || proof.nova_recursive_snark.is_empty()
        || proof.spartan_proof.is_empty()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct lookup proof envelope is incomplete".to_string(),
        ));
    }
    let setup_circuit =
        DirectLookupStepCircuit::for_subclaim(proof.subclaims[0].clone(), block_count == 1);
    let pp = setup_direct_lookup_public_params(&setup_circuit)?;
    let recursive = postcard::from_bytes::<DirectLookupNovaSnark>(&proof.nova_recursive_snark)
        .map_err(|error| nova_stage_error("direct lookup Nova deserialization failed", error))?;
    let recursive_output = recursive
        .verify(&pp, block_count, &z0)
        .map_err(|error| nova_stage_error("direct lookup Nova verification failed", error))?;
    let (_, vk) = DirectLookupCompressedSnark::setup(&pp)
        .map_err(|error| nova_stage_error("direct lookup Spartan key setup failed", error))?;
    let compressed = postcard::from_bytes::<DirectLookupCompressedSnark>(&proof.spartan_proof)
        .map_err(|error| nova_stage_error("direct lookup Spartan deserialization failed", error))?;
    let compressed_output = compressed
        .verify(&vk, block_count, &z0)
        .map_err(|error| nova_stage_error("direct lookup Spartan verification failed", error))?;
    let stored_output = recursive_output
        .iter()
        .copied()
        .map(nova_to_storage)
        .collect::<Vec<_>>();
    if compressed_output != recursive_output
        || proof.final_z != stored_output
        || recursive_output[0] != NovaScalar::from(block_count as u64)
        || recursive_output[1] != NovaScalar::from(total_cycles as u64)
        || recursive_output[2] != NovaScalar::from(total_cycles as u64)
        || recursive_output[3] != z0[3]
        || recursive_output[4] != z0[4]
        || recursive_output[5] != z0[5]
        || recursive_output[6] != z0[6]
        || recursive_output[7] != native_transcript_state
        || recursive_output[8] != native_transcript_round
        || recursive_output[9] != NovaScalar::one()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct lookup recursive output does not match its public closure".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::{test_rng, UniformRand};
    use common::constants::REGISTER_COUNT;
    use nova_snark::frontend::test_cs::TestConstraintSystem;
    use tracer::{
        instruction::{
            and::AND,
            format::format_r::{FormatR, RegisterStateFormatR},
            or::OR,
            RISCVCycle,
        },
        MachineBoundaryState,
    };

    fn boundary(cycle: usize, terminated: bool) -> MachineBoundaryState {
        MachineBoundaryState {
            global_cycle: cycle,
            emulator_trace_len: cycle,
            pc: 0x8000_0000 + (cycle as u64) * 4,
            registers: [0; REGISTER_COUNT as usize],
            terminated,
        }
    }

    fn and_cycle(x: u64, y: u64) -> Cycle {
        RISCVCycle::<AND> {
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
                rd: (0, x & y),
                rs1: x,
                rs2: y,
            },
            ram_access: (),
        }
        .into()
    }

    fn or_cycle(x: u64, y: u64) -> Cycle {
        RISCVCycle::<OR> {
            instruction: OR {
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
                rd: (0, x | y),
                rs1: x,
                rs2: y,
            },
            ram_access: (),
        }
        .into()
    }

    fn block(index: usize, start: usize, cycles: Vec<Cycle>, terminated: bool) -> TraceBlock {
        TraceBlock {
            block_index: index,
            global_cycle_start: start,
            active_cycles: cycles.len(),
            target_size: cycles.len(),
            start_state: boundary(start, false),
            end_state: boundary(start + cycles.len(), terminated),
            cycles,
            ended_at_tick_boundary: true,
        }
    }

    fn preprocessing() -> DirectChunkedPreprocessing {
        DirectChunkedPreprocessing::from_program_bytes(b"lookup-test", 64)
    }

    #[test]
    fn d2_circuit_table_evaluators_match_native_jolt() {
        let mut rng = test_rng();
        for sample in 0..2 {
            let point = (0..2 * common::constants::XLEN)
                .map(|_| Fr::rand(&mut rng))
                .collect::<Vec<_>>();
            for table in LookupTables::<{ common::constants::XLEN }>::iter() {
                let table_id = LookupTables::enum_index(&table);
                let mut cs = TestConstraintSystem::<NovaScalar>::new();
                let allocated = point
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        alloc_witness_num(
                            cs.namespace(|| format!("point {index}")),
                            nova_from_fr(value).unwrap(),
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                let output = evaluate_table_mle_circuit(
                    cs.namespace(|| format!("table {table_id} sample {sample}")),
                    table_id,
                    &allocated,
                )
                .unwrap();
                assert!(
                    cs.is_satisfied(),
                    "table {table_id}: {:?}",
                    cs.which_is_unsatisfied()
                );
                let circuit = Fr::from_le_bytes_mod_order(&output.get_value().unwrap().to_bytes());
                let native = table.evaluate_mle::<Fr, Fr>(&point);
                assert_eq!(circuit, native, "table {table_id}, sample {sample}");
            }
        }
    }

    #[test]
    fn d2_native_instruction_read_raf_round_trip() {
        let blocks = vec![
            block(0, 0, vec![and_cycle(0xaa, 0x0f)], false),
            block(1, 1, vec![or_cycle(0x1234, 0xff)], true),
        ];
        let subclaims = prove_and_verify_native_lookup_blocks(&preprocessing(), 2, blocks).unwrap();
        assert_eq!(subclaims.len(), 2);
        assert_eq!(subclaims[0].block.cycles[0].output, 0x0a);
        assert!(subclaims[0].proof.compressed_polys.len() > LOG_K);
    }

    #[test]
    fn d2_recursive_step_circuit_accepts_real_lookup_subclaim() {
        let subclaim = prove_and_verify_native_lookup_blocks(
            &preprocessing(),
            2,
            vec![block(0, 0, vec![and_cycle(0xaa, 0x0f)], true)],
        )
        .unwrap()
        .remove(0);
        let circuit = DirectLookupStepCircuit::for_subclaim(subclaim, true);
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z = direct_initial_z(&preprocessing())
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("z {index}")), || Ok(value)).unwrap()
            })
            .collect::<Vec<_>>();
        let output = circuit.synthesize(&mut cs, &z).unwrap();
        assert_eq!(output.len(), DIRECT_LOOKUP_Z_ARITY);
        assert!(
            cs.is_satisfied(),
            "recursive lookup circuit failed at {:?}",
            cs.which_is_unsatisfied()
        );
        assert_eq!(output[0].get_value(), Some(NovaScalar::one()));
        assert_eq!(output[1].get_value(), Some(NovaScalar::one()));
        assert_eq!(output[9].get_value(), Some(NovaScalar::one()));
    }

    #[test]
    fn d2_nova_folds_two_blocks_and_spartan_closes_lookup_stage() {
        let preprocessing = preprocessing();
        let blocks = vec![
            block(0, 0, vec![and_cycle(0xaa, 0x0f)], false),
            block(1, 1, vec![and_cycle(0x1234, 0xff)], true),
        ];
        let proof = prove_direct_lookup_stage(&preprocessing, 2, blocks).unwrap();
        assert_eq!(proof.statement.block_count, 2);
        assert_eq!(proof.statement.total_cycles, 2);
        assert!(!proof.nova_recursive_snark.is_empty());
        assert!(!proof.spartan_proof.is_empty());
        verify_direct_lookup_stage(&preprocessing, 2, &proof).unwrap();

        let mut bad_statement = proof.clone();
        bad_statement.statement.total_cycles += 1;
        assert!(verify_direct_lookup_stage(&preprocessing, 2, &bad_statement).is_err());

        let mut bad_spartan = proof.clone();
        bad_spartan.spartan_proof[0] ^= 1;
        assert!(verify_direct_lookup_stage(&preprocessing, 2, &bad_spartan).is_err());

        let mut bad_continuity = proof;
        bad_continuity.subclaims[1].block.global_cycle_start += 1;
        assert!(verify_direct_lookup_stage(&preprocessing, 2, &bad_continuity).is_err());
    }

    #[test]
    fn d2_recursive_circuit_rejects_tampered_query_commitment() {
        let mut subclaim = prove_and_verify_native_lookup_blocks(
            &preprocessing(),
            2,
            vec![block(0, 0, vec![and_cycle(7, 3)], true)],
        )
        .unwrap()
        .remove(0);
        subclaim.query_root += Fr::from(1u64);
        let circuit = DirectLookupStepCircuit::for_subclaim(subclaim, true);
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z = direct_initial_z(&preprocessing())
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("z {index}")), || Ok(value)).unwrap()
            })
            .collect::<Vec<_>>();
        circuit.synthesize(&mut cs, &z).unwrap();
        assert!(!cs.is_satisfied());
    }

    #[test]
    fn d2_recursive_step_has_fixed_shape_across_distinct_blocks() {
        let subclaims = prove_and_verify_native_lookup_blocks(
            &preprocessing(),
            2,
            vec![
                block(0, 0, vec![and_cycle(7, 3)], false),
                block(1, 1, vec![or_cycle(0xfeed, 0x55)], true),
            ],
        )
        .unwrap();
        let mut counts = Vec::new();
        for (index, subclaim) in subclaims.into_iter().enumerate() {
            let circuit = DirectLookupStepCircuit::for_subclaim(subclaim, index == 1);
            let mut cs = TestConstraintSystem::<NovaScalar>::new();
            let z = direct_initial_z(&preprocessing())
                .into_iter()
                .enumerate()
                .map(|(slot, value)| {
                    AllocatedNum::alloc(cs.namespace(|| format!("z {slot}")), || Ok(value)).unwrap()
                })
                .collect::<Vec<_>>();
            circuit.synthesize(&mut cs, &z).unwrap();
            counts.push(cs.num_constraints());
        }
        assert_eq!(counts[0], counts[1]);
    }

    #[test]
    fn d2_initial_state_binds_full_width_preprocessing_digests() {
        let base = preprocessing();
        let mut changed_program = base.clone();
        changed_program.program_digest[31] ^= 0x80;
        let mut changed_tables = base.clone();
        changed_tables.lookup_table_commitment[31] ^= 0x80;

        let base_z = direct_initial_z(&base);
        let program_z = direct_initial_z(&changed_program);
        let table_z = direct_initial_z(&changed_tables);
        assert_eq!(base_z[3], program_z[3]);
        assert_ne!(base_z[4], program_z[4]);
        assert_eq!(base_z[5], table_z[5]);
        assert_ne!(base_z[6], table_z[6]);
    }

    #[test]
    fn d2_tampered_query_or_sumcheck_is_rejected() {
        let mut subclaim = prove_and_verify_native_lookup_blocks(
            &preprocessing(),
            2,
            vec![block(0, 0, vec![and_cycle(7, 3)], true)],
        )
        .unwrap()
        .remove(0);
        subclaim.block.cycles[0].lookup_index ^= 1;
        let mut transcript = PoseidonTranscript::new(DIRECT_LOOKUP_TRANSCRIPT_DOMAIN);
        assert!(verify_native_subclaim(&subclaim, &mut transcript).is_err());

        let mut subclaim = prove_and_verify_native_lookup_blocks(
            &preprocessing(),
            2,
            vec![block(0, 0, vec![and_cycle(7, 3)], true)],
        )
        .unwrap()
        .remove(0);
        subclaim.proof.compressed_polys[0].coeffs_except_linear_term[0] += Fr::from(1u64);
        let mut transcript = PoseidonTranscript::new(DIRECT_LOOKUP_TRANSCRIPT_DOMAIN);
        assert!(verify_native_subclaim(&subclaim, &mut transcript).is_err());
    }

    #[test]
    fn d2_rejects_valid_subclaim_replacement_from_another_trace() {
        let preprocessing = preprocessing();
        let mut proof = prove_direct_lookup_stage(
            &preprocessing,
            2,
            vec![block(0, 0, vec![and_cycle(7, 3)], true)],
        )
        .unwrap();
        let replacement = prove_and_verify_native_lookup_blocks(
            &preprocessing,
            2,
            vec![block(0, 0, vec![or_cycle(0xfeed, 0x55)], true)],
        )
        .unwrap();
        proof.subclaims = replacement;
        assert!(verify_direct_lookup_stage(&preprocessing, 2, &proof).is_err());
    }
}
