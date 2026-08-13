//! D6: one Fiat--Shamir transcript and one PCS closure for the direct path.
//!
//! D2--D5 constrain the block-local lookup, register, authenticated RAM, and
//! CPU/R1CS relations.  This module commits to the fixed-shape global CPU/R1CS
//! witness with Dory.  Every Nova step evaluates its own rows at the same
//! Fiat--Shamir point from the *allocated D5 row variables* and accumulates the
//! result.  The terminal step enforces equality with the Dory opening.  Thus a
//! valid Dory proof for an unrelated digest or trace cannot close this circuit.

use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

use ark_bn254::Fr;
use ark_std::Zero;
use nova_snark::{
    frontend::{
        gadgets::{boolean::AllocatedBit, num::AllocatedNum},
        ConstraintSystem, SynthesisError,
    },
    traits::circuit::StepCircuit,
};
use sha3::{Digest as ShaDigest, Sha3_256};
use tracer::TraceBlock;

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{
                bind_opening_inputs, ArkDoryProof, ArkGT, DoryCommitmentScheme, DoryContext,
                DoryGlobals,
            },
        },
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{PoseidonTranscript, Transcript},
};

use super::{
    direct_cpu::{
        bytecode_tree, direct_cpu_initial_z, prepare_direct_cpu_stage, verify_cpu_subclaim,
        DirectCpuStageStatement, DirectCpuStepCircuit, DirectCpuSubclaim, CPU_CLAIM_SLOT,
        CPU_TRANSCRIPT_ROUND_SLOT, CPU_TRANSCRIPT_STATE_SLOT, DIRECT_CPU_Z_ARITY, R1CS_INPUT_COUNT,
    },
    direct_lookup::{
        add_nums, alloc_u64_bits, enforce_num_equal, eq_from_allocated_bits, eq_weight_for_index,
        field_from_digest, mul_nums, nova_from_fr, nova_to_storage, poseidon_absorb, sub_nums,
    },
    direct_ram::{
        ram_memory_root, ram_registry_root, verify_ram_subclaim, RAM_ROOT_SLOT,
        RAM_TRANSCRIPT_ROUND_SLOT, RAM_TRANSCRIPT_STATE_SLOT,
    },
    direct_register::{
        validate_combined_sequence, verify_native_register_subclaim, REGISTER_STATE_OFFSET,
        REGISTER_TRANSCRIPT_ROUND_SLOT, REGISTER_TRANSCRIPT_STATE_SLOT,
    },
    recursive_relations::{alloc_nova_constant, AllocatedRecursivePoseidonTranscriptState},
    DirectChunkedError, DirectChunkedPreprocessing, DirectLookupSubclaim, DirectRamSubclaim,
    DirectRegisterSubclaim, DirectRelation, DirectRelationState, NovaPrimaryEngine,
    NovaPrimarySpartanSnark, NovaScalar, NovaSecondaryEngine, NovaSecondarySpartanSnark,
};

const DIRECT_GLOBAL_TRANSCRIPT_DOMAIN: &[u8] = b"direct-global-fs-v1";
const DIRECT_PCS_OPENING_DOMAIN: &[u8] = b"direct-dory-close-v1";
const PCS_LANE_WIDTH: usize = R1CS_INPUT_COUNT.next_power_of_two();
static DIRECT_DORY_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

type DirectPcsNovaSnark =
    nova_snark::nova::RecursiveSNARK<NovaPrimaryEngine, NovaSecondaryEngine, DirectPcsStepCircuit>;
type DirectPcsCompressedSnark = nova_snark::nova::CompressedSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    DirectPcsStepCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;
type DirectPcsPublicParams =
    nova_snark::nova::PublicParams<NovaPrimaryEngine, NovaSecondaryEngine, DirectPcsStepCircuit>;

/// Inputs to the transparent D6 path. Advice is carried in clear in this proof
/// format so that a standalone verifier can reconstruct initial RAM. D8/BlindFold
/// is responsible for replacing this clear binding with a hiding equivalent.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectExecutionInputs {
    pub public_inputs: Vec<u8>,
    pub public_outputs: Vec<u8>,
    pub trusted_advice: Vec<u8>,
    pub untrusted_advice: Vec<u8>,
    pub panic: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectPcsStageStatement {
    pub cpu: DirectCpuStageStatement,
    pub public_inputs: Vec<u8>,
    pub public_outputs: Vec<u8>,
    pub advice_commitment: [u8; 32],
    pub panic: bool,
    pub pcs_num_vars: usize,
    pub pcs_opening: Fr,
    pub global_transcript_state: [u8; 32],
    pub global_transcript_round: u32,
    pub relations: BTreeMap<DirectRelation, DirectRelationState>,
}

#[derive(Clone)]
pub struct DirectPcsStageProof {
    pub statement: DirectPcsStageStatement,
    pub execution: DirectExecutionInputs,
    pub lookup_subclaims: Vec<DirectLookupSubclaim>,
    pub register_subclaims: Vec<DirectRegisterSubclaim>,
    pub ram_subclaims: Vec<DirectRamSubclaim>,
    pub cpu_subclaims: Vec<DirectCpuSubclaim>,
    pub pcs_commitment: ArkGT,
    pub pcs_opening_point: Vec<Fr>,
    pub pcs_opening_proof: ArkDoryProof,
    pub nova_recursive_snark: Vec<u8>,
    pub spartan_proof: Vec<u8>,
    pub initial_z: Vec<[u8; 32]>,
    pub final_z: Vec<[u8; 32]>,
}

#[derive(Clone, Default)]
struct DirectPcsStepCircuit {
    cpu: DirectCpuStepCircuit,
    point_len: usize,
    block_bits: usize,
    block_index: u64,
    final_step: bool,
}

impl DirectPcsStepCircuit {
    fn for_subclaims(
        lookup: DirectLookupSubclaim,
        register: DirectRegisterSubclaim,
        ram: DirectRamSubclaim,
        cpu: DirectCpuSubclaim,
        block_count: usize,
        final_step: bool,
    ) -> Self {
        let capacity = cpu.block.cycle_capacity;
        let block_index = cpu.block.block_index as u64;
        Self {
            cpu: DirectCpuStepCircuit::for_subclaims(lookup, register, ram, cpu, final_step),
            point_len: pcs_num_vars(block_count, capacity),
            block_bits: block_count.next_power_of_two().ilog2() as usize,
            block_index,
            final_step,
        }
    }

    fn opening_point_range(&self) -> std::ops::Range<usize> {
        DIRECT_CPU_Z_ARITY..DIRECT_CPU_Z_ARITY + self.point_len
    }

    fn expected_opening_slot(&self) -> usize {
        DIRECT_CPU_Z_ARITY + self.point_len
    }

    fn running_opening_slot(&self) -> usize {
        self.expected_opening_slot() + 1
    }

    fn transcript_state_slot(&self) -> usize {
        self.running_opening_slot() + 1
    }

    fn transcript_round_slot(&self) -> usize {
        self.transcript_state_slot() + 1
    }
}

impl StepCircuit<NovaScalar> for DirectPcsStepCircuit {
    fn arity(&self) -> usize {
        DIRECT_CPU_Z_ARITY + self.point_len + 4
    }

    fn synthesize<CS: ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<NovaScalar>],
    ) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
        if z.len() != self.arity() || self.point_len < self.block_bits + 6 {
            return Err(SynthesisError::Unsatisfiable(
                "D6 PCS state shape mismatch".to_string(),
            ));
        }
        let (cpu_output, rows, block_index, terminated) = self
            .cpu
            .synthesize_with_observations(cs, &z[..DIRECT_CPU_Z_ARITY])?;
        let point = &z[self.opening_point_range()];
        let row_bits = self.point_len - self.block_bits - 6;
        if rows.len() != 1usize << row_bits {
            return Err(SynthesisError::Unsatisfiable(
                "D6 PCS row dimension mismatch".to_string(),
            ));
        }

        let (_, index_bits) = alloc_u64_bits(
            cs.namespace(|| "D6 block index bits"),
            self.block_index,
            "D6 block index",
        )?;
        enforce_num_equal(
            cs.namespace(|| "D6 block index binding"),
            &block_index,
            &z[0],
            "D6 block index binding",
        );
        let block_weight = eq_from_allocated_bits(
            cs.namespace(|| "D6 block equality"),
            &point[..self.block_bits],
            &index_bits[..self.block_bits],
            "D6 block equality",
        )?;
        let row_point = &point[self.block_bits..self.block_bits + row_bits];
        let lane_point = &point[self.block_bits + row_bits..];
        let lane_weights = (0..R1CS_INPUT_COUNT)
            .map(|lane| {
                eq_weight_for_index(
                    cs.namespace(|| format!("D6 lane weight {lane}")),
                    lane_point,
                    lane,
                    "D6 lane weight",
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut block_evaluation = alloc_nova_constant(
            cs.namespace(|| "D6 block evaluation zero"),
            NovaScalar::zero(),
        )?;
        for (row_index, row) in rows.iter().enumerate() {
            let mut lane_evaluation = alloc_nova_constant(
                cs.namespace(|| format!("D6 lane evaluation zero {row_index}")),
                NovaScalar::zero(),
            )?;
            for (lane, input) in row.inputs.iter().enumerate() {
                let term = mul_nums(
                    cs.namespace(|| format!("D6 lane term {row_index} {lane}")),
                    input,
                    &lane_weights[lane],
                    "D6 lane term",
                )?;
                lane_evaluation = add_nums(
                    cs.namespace(|| format!("D6 lane sum {row_index} {lane}")),
                    &lane_evaluation,
                    &term,
                    "D6 lane sum",
                )?;
            }
            let row_weight = eq_weight_for_index(
                cs.namespace(|| format!("D6 row weight {row_index}")),
                row_point,
                row_index,
                "D6 row weight",
            )?;
            let row_term = mul_nums(
                cs.namespace(|| format!("D6 row term {row_index}")),
                &lane_evaluation,
                &row_weight,
                "D6 row term",
            )?;
            block_evaluation = add_nums(
                cs.namespace(|| format!("D6 row sum {row_index}")),
                &block_evaluation,
                &row_term,
                "D6 row sum",
            )?;
        }
        let contribution = mul_nums(
            cs.namespace(|| "D6 block contribution"),
            &block_weight,
            &block_evaluation,
            "D6 block contribution",
        )?;
        let running = add_nums(
            cs.namespace(|| "D6 running opening"),
            &z[self.running_opening_slot()],
            &contribution,
            "D6 running opening",
        )?;
        let final_bit =
            AllocatedBit::alloc(cs.namespace(|| "D6 final step"), Some(self.final_step))?;
        let final_num = super::direct_lookup::bit_as_num(&final_bit);
        enforce_num_equal(
            cs.namespace(|| "D6 final termination"),
            &final_num,
            &terminated,
            "D6 final termination",
        );
        let opening_difference = sub_nums(
            cs.namespace(|| "D6 opening difference"),
            &running,
            &z[self.expected_opening_slot()],
            "D6 opening difference",
        )?;
        cs.enforce(
            || "D6 terminal opening closure",
            |lc| lc + final_bit.get_variable(),
            |lc| lc + opening_difference.get_variable(),
            |lc| lc,
        );

        let mut transcript = AllocatedRecursivePoseidonTranscriptState {
            state: z[self.transcript_state_slot()].clone(),
            n_rounds: z[self.transcript_round_slot()].clone(),
        };
        for (label, value) in [
            ("block", &block_index),
            ("d2_state", &cpu_output[7]),
            ("d2_round", &cpu_output[8]),
            ("d3_state", &cpu_output[REGISTER_TRANSCRIPT_STATE_SLOT]),
            ("d3_round", &cpu_output[REGISTER_TRANSCRIPT_ROUND_SLOT]),
            ("d4_state", &cpu_output[RAM_TRANSCRIPT_STATE_SLOT]),
            ("d4_round", &cpu_output[RAM_TRANSCRIPT_ROUND_SLOT]),
            ("d5_state", &cpu_output[CPU_TRANSCRIPT_STATE_SLOT]),
            ("d5_round", &cpu_output[CPU_TRANSCRIPT_ROUND_SLOT]),
            ("ram_root", &cpu_output[RAM_ROOT_SLOT]),
            ("cpu_claim", &cpu_output[CPU_CLAIM_SLOT]),
        ] {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("D6 global transcript {label}")),
                &transcript,
                value,
                label,
            )?;
        }

        let mut output = cpu_output;
        output.extend_from_slice(point);
        output.push(z[self.expected_opening_slot()].clone());
        output.push(running);
        output.push(transcript.state);
        output.push(transcript.n_rounds);
        Ok(output)
    }
}

fn pcs_num_vars(block_count: usize, capacity: usize) -> usize {
    block_count.next_power_of_two().ilog2() as usize
        + capacity.ilog2() as usize
        + PCS_LANE_WIDTH.ilog2() as usize
}

fn direct_dory_lock() -> std::sync::MutexGuard<'static, ()> {
    DIRECT_DORY_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn advice_commitment(execution: &DirectExecutionInputs) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"direct-clear-advice-v1");
    hasher.update((execution.trusted_advice.len() as u64).to_le_bytes());
    hasher.update(&execution.trusted_advice);
    hasher.update((execution.untrusted_advice.len() as u64).to_le_bytes());
    hasher.update(&execution.untrusted_advice);
    hasher.finalize().into()
}

fn validate_execution_inputs(
    preprocessing: &DirectChunkedPreprocessing,
    execution: &DirectExecutionInputs,
) -> Result<(), DirectChunkedError> {
    let layout = &preprocessing.memory_layout;
    for (label, observed, maximum) in [
        (
            "public input",
            execution.public_inputs.len(),
            layout.max_input_size,
        ),
        (
            "public output",
            execution.public_outputs.len(),
            layout.max_output_size,
        ),
        (
            "trusted advice",
            execution.trusted_advice.len(),
            layout.max_trusted_advice_size,
        ),
        (
            "untrusted advice",
            execution.untrusted_advice.len(),
            layout.max_untrusted_advice_size,
        ),
    ] {
        if observed as u64 > maximum {
            return Err(DirectChunkedError::InvalidConfiguration(format!(
                "{label} has {observed} bytes, exceeding configured maximum {maximum}"
            )));
        }
    }
    Ok(())
}

fn byte_at(
    preprocessing: &DirectChunkedPreprocessing,
    execution: &DirectExecutionInputs,
    address: u64,
) -> u8 {
    let layout = &preprocessing.memory_layout;
    let region_byte = |start: u64, bytes: &[u8]| {
        address
            .checked_sub(start)
            .and_then(|offset| bytes.get(offset as usize))
            .copied()
    };
    if let Some(value) = region_byte(layout.input_start, &execution.public_inputs) {
        return value;
    }
    if let Some(value) = region_byte(layout.trusted_advice_start, &execution.trusted_advice) {
        return value;
    }
    if let Some(value) = region_byte(layout.untrusted_advice_start, &execution.untrusted_advice) {
        return value;
    }
    if address < common::constants::RAM_START_ADDRESS {
        return 0;
    }
    let image_start = (preprocessing.program_image_start / 8) * 8;
    let Some(offset) = address.checked_sub(image_start) else {
        return 0;
    };
    let Some(word) = preprocessing.program_image_words.get((offset / 8) as usize) else {
        return 0;
    };
    word.to_le_bytes()[(offset % 8) as usize]
}

fn initial_word(
    preprocessing: &DirectChunkedPreprocessing,
    execution: &DirectExecutionInputs,
    address: u64,
) -> u64 {
    let mut bytes = [0u8; 8];
    for (offset, byte) in bytes.iter_mut().enumerate() {
        *byte = byte_at(preprocessing, execution, address + offset as u64);
    }
    u64::from_le_bytes(bytes)
}

fn canonical_initial_values(
    preprocessing: &DirectChunkedPreprocessing,
    execution: &DirectExecutionInputs,
    addresses: &[u64],
) -> Vec<u64> {
    addresses
        .iter()
        .map(|address| initial_word(preprocessing, execution, *address))
        .collect()
}

fn validate_public_outputs(
    preprocessing: &DirectChunkedPreprocessing,
    execution: &DirectExecutionInputs,
    rams: &[DirectRamSubclaim],
) -> Result<(), DirectChunkedError> {
    let layout = &preprocessing.memory_layout;
    let mut output = vec![0u8; layout.max_output_size as usize];
    let mut panic = false;
    for subclaim in rams {
        for cycle in subclaim
            .block
            .cycles
            .iter()
            .take(subclaim.block.active_cycles)
        {
            if cycle.kind != 2 {
                continue;
            }
            for (offset, byte) in cycle.write_value.to_le_bytes().into_iter().enumerate() {
                let address = cycle.address + offset as u64;
                if address >= layout.output_start && address < layout.output_end {
                    output[(address - layout.output_start) as usize] = byte;
                }
                if address == layout.panic && byte != 0 {
                    panic = true;
                }
            }
        }
    }
    if output[..execution.public_outputs.len()] != execution.public_outputs
        || output[execution.public_outputs.len()..]
            .iter()
            .any(|byte| *byte != 0)
        || panic != execution.panic
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 public output or panic binding mismatch".to_string(),
        ));
    }
    Ok(())
}

fn witness_polynomial(cpus: &[DirectCpuSubclaim], capacity: usize) -> MultilinearPolynomial<Fr> {
    let block_domain = cpus.len().next_power_of_two();
    let mut coefficients = vec![Fr::zero(); block_domain * capacity * PCS_LANE_WIDTH];
    for cpu in cpus {
        for (row, cycle) in cpu.block.cycles.iter().enumerate() {
            let base = (cpu.block.block_index * capacity + row) * PCS_LANE_WIDTH;
            for (lane, value) in cycle.inputs.iter().enumerate() {
                coefficients[base + lane] = Fr::from_i128(*value);
            }
        }
    }
    MultilinearPolynomial::from(coefficients)
}

fn all_relations_proven() -> BTreeMap<DirectRelation, DirectRelationState> {
    DirectRelation::ALL
        .into_iter()
        .map(|relation| (relation, DirectRelationState::Proven))
        .collect()
}

fn cpu_relations() -> BTreeMap<DirectRelation, DirectRelationState> {
    DirectRelation::ALL
        .into_iter()
        .map(|relation| {
            (
                relation,
                if relation == DirectRelation::Pcs {
                    DirectRelationState::Deferred
                } else {
                    DirectRelationState::Proven
                },
            )
        })
        .collect()
}

fn seed_global_transcript(
    statement: &DirectPcsStageStatement,
    commitment: &ArkGT,
) -> PoseidonTranscript {
    let mut transcript = PoseidonTranscript::new(DIRECT_GLOBAL_TRANSCRIPT_DOMAIN);
    transcript.append_bytes(b"program", &statement.cpu.program_digest);
    transcript.append_bytes(b"tables", &statement.cpu.lookup_table_commitment);
    transcript.append_scalar(b"bytecode", &statement.cpu.bytecode_root);
    transcript.append_u64(b"blocks", statement.cpu.block_count as u64);
    transcript.append_u64(b"cycles", statement.cpu.total_cycles as u64);
    transcript.append_scalar(b"registry", &statement.cpu.ram_registry_root);
    transcript.append_scalar(b"initial_ram", &statement.cpu.initial_ram_root);
    transcript.append_scalar(b"final_ram", &statement.cpu.final_ram_root);
    transcript.append_bytes(b"public_in", &statement.public_inputs);
    transcript.append_bytes(b"public_out", &statement.public_outputs);
    transcript.append_bytes(b"advice", &statement.advice_commitment);
    transcript.append_u64(b"panic", u64::from(statement.panic));
    transcript.append_serializable(b"trace_commitment", commitment);
    transcript
}

fn update_global_transcript(
    transcript: &mut PoseidonTranscript,
    position: usize,
    lookup: &DirectLookupSubclaim,
    register: &DirectRegisterSubclaim,
    ram: &DirectRamSubclaim,
    cpu: &DirectCpuSubclaim,
    running_cpu_claim: Fr,
) {
    let values = [
        (b"block" as &'static [u8], Fr::from(position as u64)),
        (
            b"d2_state",
            field_from_digest(&lookup.transcript_state_after),
        ),
        (b"d2_round", Fr::from(lookup.transcript_round_after as u64)),
        (
            b"d3_state",
            field_from_digest(&register.transcript_state_after),
        ),
        (
            b"d3_round",
            Fr::from(register.transcript_round_after as u64),
        ),
        (b"d4_state", field_from_digest(&ram.transcript_state_after)),
        (b"d4_round", Fr::from(ram.transcript_round_after as u64)),
        (b"d5_state", field_from_digest(&cpu.transcript_state_after)),
        (b"d5_round", Fr::from(cpu.transcript_round_after as u64)),
        (b"ram_root", ram.block.end_root),
        (b"cpu_claim", running_cpu_claim),
    ];
    for (label, value) in values {
        transcript.append_scalar(label, &value);
    }
}

fn dory_transcript(
    final_state: &[u8; 32],
    final_round: u32,
    commitment: &ArkGT,
) -> PoseidonTranscript {
    let mut transcript = PoseidonTranscript::new(DIRECT_PCS_OPENING_DOMAIN);
    transcript.append_bytes(b"global_state", final_state);
    transcript.append_u64(b"global_round", final_round as u64);
    transcript.append_serializable(b"trace_commitment", commitment);
    transcript
}

fn pcs_initial_z(
    preprocessing: &DirectChunkedPreprocessing,
    statement: &DirectPcsStageStatement,
    opening_point: &[Fr],
    initial_global: &PoseidonTranscript,
) -> Result<Vec<NovaScalar>, DirectChunkedError> {
    let mut z = direct_cpu_initial_z(
        preprocessing,
        &statement.cpu.initial_registers,
        statement.cpu.ram_registry_root,
        statement.cpu.initial_ram_root,
        statement.cpu.bytecode_root,
    );
    for coordinate in opening_point {
        z.push(nova_from_fr(coordinate).map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!("D6 opening point conversion: {error:?}"))
        })?);
    }
    z.push(nova_from_fr(&statement.pcs_opening).map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!("D6 opening conversion: {error:?}"))
    })?);
    z.push(NovaScalar::zero());
    z.push(
        nova_from_fr(&field_from_digest(&initial_global.state)).map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!("D6 transcript conversion: {error:?}"))
        })?,
    );
    z.push(NovaScalar::from(initial_global.n_rounds as u64));
    Ok(z)
}

fn setup_pcs_pp(
    circuit: &DirectPcsStepCircuit,
) -> Result<DirectPcsPublicParams, DirectChunkedError> {
    DirectPcsPublicParams::setup(
        circuit,
        &*nova_snark::traits::snark::default_ck_hint::<NovaPrimaryEngine>(),
        &*nova_snark::traits::snark::default_ck_hint::<NovaSecondaryEngine>(),
    )
    .map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!("D6 Nova setup failed: {error:?}"))
    })
}

fn validate_subclaim_envelope(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    proof: &DirectPcsStageProof,
) -> Result<(), DirectChunkedError> {
    let statement = &proof.statement.cpu;
    let n = statement.block_count;
    if n == 0
        || statement.program_digest != preprocessing.program_digest
        || statement.lookup_table_commitment != preprocessing.lookup_table_commitment
        || statement.bytecode_root != bytecode_tree(preprocessing).last().unwrap()[0]
        || statement.relations != cpu_relations()
        || !statement.terminated
        || proof.statement.relations != all_relations_proven()
        || proof.lookup_subclaims.len() != n
        || proof.register_subclaims.len() != n
        || proof.ram_subclaims.len() != n
        || proof.cpu_subclaims.len() != n
        || proof
            .cpu_subclaims
            .iter()
            .any(|cpu| cpu.block.cycle_capacity != capacity)
        || statement.ram_registry_root != ram_registry_root(&statement.ram_addresses)
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 proof envelope mismatch".to_string(),
        ));
    }
    let (block_count, total_cycles) = validate_combined_sequence(
        preprocessing,
        capacity,
        &proof.lookup_subclaims,
        &proof.register_subclaims,
    )?;
    if block_count != n
        || total_cycles != statement.total_cycles
        || proof.register_subclaims[0].block.start_registers != statement.initial_registers
        || proof.register_subclaims[n - 1].block.end_registers != statement.final_registers
        || proof.ram_subclaims[0].block.start_root != statement.initial_ram_root
        || proof.ram_subclaims[n - 1].block.end_root != statement.final_ram_root
        || proof.ram_subclaims[0].block.registry_root != statement.ram_registry_root
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 relation boundary mismatch".to_string(),
        ));
    }
    let mut lookup_t =
        PoseidonTranscript::new(super::direct_lookup::DIRECT_LOOKUP_TRANSCRIPT_DOMAIN);
    let mut register_t =
        PoseidonTranscript::new(super::direct_register::DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
    let mut ram_t = PoseidonTranscript::new(super::direct_ram::DIRECT_RAM_QUERY_DOMAIN);
    let mut cpu_t = PoseidonTranscript::new(super::direct_cpu::DIRECT_CPU_QUERY_DOMAIN);
    for position in 0..n {
        super::direct_lookup::verify_native_subclaim(
            &proof.lookup_subclaims[position],
            &mut lookup_t,
        )?;
        verify_native_register_subclaim(&proof.register_subclaims[position], &mut register_t)?;
        verify_ram_subclaim(&proof.ram_subclaims[position], &mut ram_t)?;
        verify_cpu_subclaim(&proof.cpu_subclaims[position], &mut cpu_t)?;
        if proof.cpu_subclaims[position].block.block_index != position
            || proof.ram_subclaims[position].block.block_index != position
            || (position > 0
                && proof.ram_subclaims[position - 1].block.end_root
                    != proof.ram_subclaims[position].block.start_root)
        {
            return Err(DirectChunkedError::InvalidProofShape(
                "D6 block or RAM continuity mismatch".to_string(),
            ));
        }
    }
    let folded: Fr = proof
        .cpu_subclaims
        .iter()
        .map(|cpu| cpu.block.block_claim)
        .sum();
    if folded != statement.folded_cpu_claim {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 folded CPU claim mismatch".to_string(),
        ));
    }
    Ok(())
}

pub fn prove_direct_pcs_stage<I>(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    execution: DirectExecutionInputs,
    blocks: I,
) -> Result<DirectPcsStageProof, DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    validate_execution_inputs(preprocessing, &execution)?;
    let blocks = blocks.into_iter().collect::<Vec<_>>();
    let prepared = prepare_direct_cpu_stage(preprocessing, capacity, &blocks)?;
    let d4 = prepared.ram;
    let cpus = prepared.cpus;
    let initial_values = canonical_initial_values(preprocessing, &execution, &d4.addresses);
    let expected_initial_root = ram_memory_root(&d4.addresses, &initial_values);
    if d4.initial_root != expected_initial_root {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 trace RAM does not match program/input/advice initial memory".to_string(),
        ));
    }
    validate_public_outputs(preprocessing, &execution, &d4.rams)?;

    let poly = witness_polynomial(&cpus, capacity);
    let num_vars = poly.get_num_vars();
    let (prover_setup, commitment, opening_hint) = {
        let _lock = direct_dory_lock();
        DoryGlobals::initialize_context(1, 1usize << num_vars, DoryContext::Main, None);
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let (commitment, opening_hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);
        (prover_setup, commitment, opening_hint)
    };
    let folded_cpu_claim = cpus.iter().map(|cpu| cpu.block.block_claim).sum();
    let mut statement = DirectPcsStageStatement {
        cpu: DirectCpuStageStatement {
            program_digest: preprocessing.program_digest,
            lookup_table_commitment: preprocessing.lookup_table_commitment,
            bytecode_root: cpus[0].block.bytecode_root,
            block_count: d4.block_count,
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
        public_inputs: execution.public_inputs.clone(),
        public_outputs: execution.public_outputs.clone(),
        advice_commitment: advice_commitment(&execution),
        panic: execution.panic,
        pcs_num_vars: num_vars,
        pcs_opening: Fr::zero(),
        global_transcript_state: [0; 32],
        global_transcript_round: 0,
        relations: all_relations_proven(),
    };
    let mut global = seed_global_transcript(&statement, &commitment);
    let opening_point = global.challenge_vector_optimized::<Fr>(num_vars);
    let opening_point_fields = opening_point
        .iter()
        .map(|coordinate| (*coordinate).into())
        .collect::<Vec<Fr>>();
    let opening = PolynomialEvaluation::evaluate(&poly, &opening_point);
    statement.pcs_opening = opening;

    let z0 = pcs_initial_z(preprocessing, &statement, &opening_point_fields, &global)?;
    let first = DirectPcsStepCircuit::for_subclaims(
        d4.lookup[0].clone(),
        d4.registers[0].clone(),
        d4.rams[0].clone(),
        cpus[0].clone(),
        d4.block_count,
        d4.block_count == 1,
    );
    let pp = setup_pcs_pp(&first)?;
    let mut recursive = DirectPcsNovaSnark::new(&pp, &first, &z0).map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!("D6 Nova initialization failed: {error:?}"))
    })?;
    recursive.prove_step(&pp, &first).map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!("D6 Nova first step failed: {error:?}"))
    })?;
    let mut running_cpu_claim = cpus[0].block.block_claim;
    update_global_transcript(
        &mut global,
        0,
        &d4.lookup[0],
        &d4.registers[0],
        &d4.rams[0],
        &cpus[0],
        running_cpu_claim,
    );
    for position in 1..d4.block_count {
        let circuit = DirectPcsStepCircuit::for_subclaims(
            d4.lookup[position].clone(),
            d4.registers[position].clone(),
            d4.rams[position].clone(),
            cpus[position].clone(),
            d4.block_count,
            position + 1 == d4.block_count,
        );
        recursive.prove_step(&pp, &circuit).map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!("D6 Nova step failed: {error:?}"))
        })?;
        running_cpu_claim += cpus[position].block.block_claim;
        update_global_transcript(
            &mut global,
            position,
            &d4.lookup[position],
            &d4.registers[position],
            &d4.rams[position],
            &cpus[position],
            running_cpu_claim,
        );
    }
    let output = recursive
        .verify(&pp, d4.block_count, &z0)
        .map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!(
                "D6 Nova self verification failed: {error:?}"
            ))
        })?;
    let (pk, vk) = DirectPcsCompressedSnark::setup(&pp).map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!("D6 Spartan setup failed: {error:?}"))
    })?;
    let compressed = DirectPcsCompressedSnark::prove(&pp, &pk, &recursive).map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!("D6 Spartan proving failed: {error:?}"))
    })?;
    if compressed
        .verify(&vk, d4.block_count, &z0)
        .map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!(
                "D6 Spartan self verification failed: {error:?}"
            ))
        })?
        != output
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 Nova/Spartan output mismatch".to_string(),
        ));
    }
    statement.global_transcript_state = global.state;
    statement.global_transcript_round = global.n_rounds;
    let opening_proof = {
        let _lock = direct_dory_lock();
        DoryGlobals::initialize_context(1, 1usize << num_vars, DoryContext::Main, None);
        let mut pcs_transcript = dory_transcript(&global.state, global.n_rounds, &commitment);
        bind_opening_inputs::<Fr, _>(&mut pcs_transcript, &opening_point, &opening);
        DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &opening_point,
            Some(opening_hint),
            &mut pcs_transcript,
        )
        .0
    };
    let proof = DirectPcsStageProof {
        statement,
        execution,
        lookup_subclaims: d4.lookup,
        register_subclaims: d4.registers,
        ram_subclaims: d4.rams,
        cpu_subclaims: cpus,
        pcs_commitment: commitment,
        pcs_opening_point: opening_point_fields,
        pcs_opening_proof: opening_proof,
        nova_recursive_snark: postcard::to_stdvec(&recursive).map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!("D6 Nova serialization: {error:?}"))
        })?,
        spartan_proof: postcard::to_stdvec(&compressed).map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!("D6 Spartan serialization: {error:?}"))
        })?,
        initial_z: z0.iter().copied().map(nova_to_storage).collect(),
        final_z: output.iter().copied().map(nova_to_storage).collect(),
    };
    verify_direct_pcs_stage(preprocessing, capacity, &proof)?;
    Ok(proof)
}

pub fn verify_direct_pcs_stage(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    proof: &DirectPcsStageProof,
) -> Result<(), DirectChunkedError> {
    validate_execution_inputs(preprocessing, &proof.execution)?;
    if proof.statement.public_inputs != proof.execution.public_inputs
        || proof.statement.public_outputs != proof.execution.public_outputs
        || proof.statement.advice_commitment != advice_commitment(&proof.execution)
        || proof.statement.panic != proof.execution.panic
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 execution statement mismatch".to_string(),
        ));
    }
    validate_subclaim_envelope(preprocessing, capacity, proof)?;
    let initial_values = canonical_initial_values(
        preprocessing,
        &proof.execution,
        &proof.statement.cpu.ram_addresses,
    );
    if ram_memory_root(&proof.statement.cpu.ram_addresses, &initial_values)
        != proof.statement.cpu.initial_ram_root
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 initial RAM is not bound to program/input/advice".to_string(),
        ));
    }
    validate_public_outputs(preprocessing, &proof.execution, &proof.ram_subclaims)?;

    let num_vars = pcs_num_vars(proof.statement.cpu.block_count, capacity);
    if proof.statement.pcs_num_vars != num_vars || proof.pcs_opening_point.len() != num_vars {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 PCS dimension mismatch".to_string(),
        ));
    }
    let mut global = seed_global_transcript(&proof.statement, &proof.pcs_commitment);
    let opening_point = global.challenge_vector_optimized::<Fr>(num_vars);
    let initial_global = global.clone();
    let opening_fields = opening_point
        .iter()
        .map(|coordinate| (*coordinate).into())
        .collect::<Vec<Fr>>();
    if opening_fields != proof.pcs_opening_point {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 Fiat-Shamir opening point mismatch".to_string(),
        ));
    }
    let mut running_cpu_claim = Fr::zero();
    for position in 0..proof.statement.cpu.block_count {
        running_cpu_claim += proof.cpu_subclaims[position].block.block_claim;
        update_global_transcript(
            &mut global,
            position,
            &proof.lookup_subclaims[position],
            &proof.register_subclaims[position],
            &proof.ram_subclaims[position],
            &proof.cpu_subclaims[position],
            running_cpu_claim,
        );
    }
    if global.state != proof.statement.global_transcript_state
        || global.n_rounds != proof.statement.global_transcript_round
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D6 global transcript closure mismatch".to_string(),
        ));
    }
    {
        let _lock = direct_dory_lock();
        DoryGlobals::initialize_context(1, 1usize << num_vars, DoryContext::Main, None);
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);
        let mut pcs_transcript =
            dory_transcript(&global.state, global.n_rounds, &proof.pcs_commitment);
        bind_opening_inputs::<Fr, _>(
            &mut pcs_transcript,
            &opening_point,
            &proof.statement.pcs_opening,
        );
        DoryCommitmentScheme::verify(
            &proof.pcs_opening_proof,
            &verifier_setup,
            &mut pcs_transcript,
            &opening_point,
            &proof.statement.pcs_opening,
            &proof.pcs_commitment,
        )
        .map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!(
                "D6 Dory opening verification failed: {error}"
            ))
        })?;
    }

    let z0 = pcs_initial_z(
        preprocessing,
        &proof.statement,
        &opening_fields,
        &initial_global,
    )?;
    let first = DirectPcsStepCircuit::for_subclaims(
        proof.lookup_subclaims[0].clone(),
        proof.register_subclaims[0].clone(),
        proof.ram_subclaims[0].clone(),
        proof.cpu_subclaims[0].clone(),
        proof.statement.cpu.block_count,
        proof.statement.cpu.block_count == 1,
    );
    let pp = setup_pcs_pp(&first)?;
    let recursive = postcard::from_bytes::<DirectPcsNovaSnark>(&proof.nova_recursive_snark)
        .map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!("D6 Nova deserialization: {error:?}"))
        })?;
    let output = recursive
        .verify(&pp, proof.statement.cpu.block_count, &z0)
        .map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!("D6 Nova verification failed: {error:?}"))
        })?;
    let (_, vk) = DirectPcsCompressedSnark::setup(&pp).map_err(|error| {
        DirectChunkedError::InvalidProofShape(format!("D6 Spartan setup failed: {error:?}"))
    })?;
    let compressed = postcard::from_bytes::<DirectPcsCompressedSnark>(&proof.spartan_proof)
        .map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!("D6 Spartan deserialization: {error:?}"))
        })?;
    let compressed_output = compressed
        .verify(&vk, proof.statement.cpu.block_count, &z0)
        .map_err(|error| {
            DirectChunkedError::InvalidProofShape(format!(
                "D6 Spartan verification failed: {error:?}"
            ))
        })?;
    let circuit = &first;
    if output != compressed_output
        || output[circuit.running_opening_slot()]
            != nova_from_fr(&proof.statement.pcs_opening).map_err(|error| {
                DirectChunkedError::InvalidProofShape(format!("D6 opening conversion: {error:?}"))
            })?
        || output[circuit.transcript_state_slot()]
            != nova_from_fr(&field_from_digest(&global.state)).map_err(|error| {
                DirectChunkedError::InvalidProofShape(format!(
                    "D6 transcript conversion: {error:?}"
                ))
            })?
        || output[circuit.transcript_round_slot()] != NovaScalar::from(global.n_rounds as u64)
        || output[RAM_ROOT_SLOT]
            != nova_from_fr(&proof.statement.cpu.final_ram_root).map_err(|error| {
                DirectChunkedError::InvalidProofShape(format!("D6 RAM root conversion: {error:?}"))
            })?
        || output[REGISTER_STATE_OFFSET
            ..REGISTER_STATE_OFFSET + proof.statement.cpu.final_registers.len()]
            != proof
                .statement
                .cpu
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
            "D6 recursive/PCS closure mismatch".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::constants::REGISTER_COUNT;
    use serial_test::serial;
    use tracer::{
        instruction::{
            format::{
                format_j::{FormatJ, RegisterStateFormatJ},
                format_load::{FormatLoad, RegisterStateFormatLoad},
                format_s::{FormatS, RegisterStateFormatS},
            },
            jal::JAL,
            ld::LD,
            sd::SD,
            RAMRead, RAMWrite, RISCVCycle,
        },
        MachineBoundaryState,
    };

    fn boundary(
        cycle: usize,
        registers: [u64; REGISTER_COUNT as usize],
        terminated: bool,
    ) -> MachineBoundaryState {
        MachineBoundaryState {
            global_cycle: cycle,
            emulator_trace_len: cycle,
            pc: 0x8000_0000 + 4 * cycle as u64,
            registers: registers.map(|value| value as i64),
            terminated,
        }
    }

    fn store_cycle(address: u64, value: u64) -> tracer::instruction::Cycle {
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

    fn load_cycle(address: u64, pre: u64, value: u64) -> tracer::instruction::Cycle {
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

    fn terminating_jump_cycle(pre: u64) -> tracer::instruction::Cycle {
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
        let address = common::constants::RAM_START_ADDRESS + 0x100;
        let mut start = [0u64; REGISTER_COUNT as usize];
        start[1] = address;
        start[2] = 9;
        let middle = start;
        let mut end = middle;
        end[3] = 9;
        end[4] = 0x8000_000c;
        vec![
            TraceBlock {
                block_index: 0,
                global_cycle_start: 0,
                active_cycles: 1,
                target_size: 2,
                start_state: boundary(0, start, false),
                end_state: boundary(1, middle, false),
                cycles: vec![store_cycle(address, 9)],
                ended_at_tick_boundary: true,
            },
            TraceBlock {
                block_index: 1,
                global_cycle_start: 1,
                active_cycles: 2,
                target_size: 2,
                start_state: boundary(1, middle, false),
                end_state: boundary(3, end, true),
                cycles: vec![load_cycle(address, 0, 9), terminating_jump_cycle(0)],
                ended_at_tick_boundary: true,
            },
        ]
    }

    fn preprocessing(blocks: &[TraceBlock]) -> DirectChunkedPreprocessing {
        let cycles = blocks
            .iter()
            .flat_map(|block| block.cycles.iter().cloned())
            .collect::<Vec<_>>();
        DirectChunkedPreprocessing::from_trace_cycles(b"d6-test", 64, &cycles).unwrap()
    }

    #[test]
    #[serial]
    fn d6_closes_one_global_dory_opening_with_nova_and_spartan() {
        let blocks = blocks();
        let preprocessing = preprocessing(&blocks);
        let proof =
            prove_direct_pcs_stage(&preprocessing, 2, DirectExecutionInputs::default(), blocks)
                .unwrap();
        verify_direct_pcs_stage(&preprocessing, 2, &proof).unwrap();
        assert_eq!(proof.statement.relations, all_relations_proven());
        assert_eq!(proof.statement.pcs_num_vars, 8);
    }

    #[test]
    #[serial]
    fn d6_rejects_opening_point_transcript_and_witness_attacks() {
        let blocks = blocks();
        let preprocessing = preprocessing(&blocks);
        let proof =
            prove_direct_pcs_stage(&preprocessing, 2, DirectExecutionInputs::default(), blocks)
                .unwrap();

        let mut forged_opening = proof.clone();
        forged_opening.statement.pcs_opening += Fr::from(1u64);
        assert!(verify_direct_pcs_stage(&preprocessing, 2, &forged_opening).is_err());

        let mut forged_point = proof.clone();
        forged_point.pcs_opening_point[0] += Fr::from(1u64);
        assert!(verify_direct_pcs_stage(&preprocessing, 2, &forged_point).is_err());

        let mut forged_transcript = proof.clone();
        forged_transcript.statement.global_transcript_state[0] ^= 1;
        assert!(verify_direct_pcs_stage(&preprocessing, 2, &forged_transcript).is_err());

        let mut forged_cpu = proof.clone();
        forged_cpu.cpu_subclaims[0].block.cycles[0].inputs[0] ^= 1;
        assert!(verify_direct_pcs_stage(&preprocessing, 2, &forged_cpu).is_err());

        let mut forged_advice = proof.clone();
        forged_advice.execution.trusted_advice.push(1);
        assert!(verify_direct_pcs_stage(&preprocessing, 2, &forged_advice).is_err());
    }

    #[test]
    #[serial]
    fn d6_rejects_trace_inferred_initial_ram() {
        let blocks = blocks();
        let address = common::constants::RAM_START_ADDRESS + 0x100;
        let preprocessing = preprocessing(&blocks).with_initial_memory(
            common::jolt_device::MemoryLayout::default(),
            address,
            vec![7],
        );
        assert!(prove_direct_pcs_stage(
            &preprocessing,
            2,
            DirectExecutionInputs::default(),
            blocks,
        )
        .is_err());
    }
}
