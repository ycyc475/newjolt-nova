//! Block-native register relation for the direct Jolt-Nova path.
//!
//! D3 extends the D2 lookup circuit instead of creating a parallel proof. Each
//! Nova step verifies the real Jolt `RegistersReadWriteChecking` sumcheck and
//! constrains every register read/write against the complete 128-register
//! boundary state carried by the recursive public state. No native Jolt proof,
//! verified receipt, host `accepted` bit, or digest-only register capsule is an
//! input to this protocol.

use std::{collections::BTreeMap, sync::Arc};

use ark_bn254::Fr;
use ark_std::Zero;
use common::{constants::REGISTER_COUNT, jolt_device::MemoryLayout};
use nova_snark::{
    frontend::{
        gadgets::boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, LinearCombination,
        SynthesisError,
    },
    traits::circuit::StepCircuit,
};
use tracer::{instruction::Cycle, MachineBoundaryState, TraceBlock};

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
        bytecode::BytecodePreprocessing,
        config::ReadWriteConfig,
        registers::read_write_checking::{
            RegistersReadWriteCheckingParams, RegistersReadWriteCheckingProver,
            RegistersReadWriteCheckingVerifier,
        },
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

use super::{
    direct_lookup::{
        add_nums, alloc_u64_bits, alloc_witness_num, allocated_poseidon_initial, bit_as_num,
        direct_initial_z, enforce_num_equal, eq_weight_for_index, field_from_digest,
        mle_from_values, mul_nums, nova_from_fr, nova_to_storage, poseidon_absorb,
        poseidon_challenge, recursive_field, scale_num, sub_nums,
        validate_lookup_subclaim_sequence, verify_native_subclaim, DirectLookupStepCircuit,
        DIRECT_LOOKUP_TRANSCRIPT_DOMAIN, DIRECT_LOOKUP_Z_ARITY,
    },
    recursive_relations::{
        alloc_nova_constant, synthesize_recursive_clear_sumcheck_stage,
        synthesize_recursive_clear_sumcheck_transcript, AllocatedRecursivePoseidonTranscriptState,
    },
    DirectChunkedError, DirectChunkedPreprocessing, DirectLookupSubclaim, DirectRelation,
    DirectRelationState, NovaPrimaryEngine, NovaPrimarySpartanSnark, NovaScalar,
    NovaSecondaryEngine, NovaSecondarySpartanSnark, RecursiveClearSumcheckRoundWitness,
    RecursiveClearSumcheckStageWitness,
};

const REGISTER_COUNT_USIZE: usize = REGISTER_COUNT as usize;
const LOG_REGISTER_COUNT: usize = REGISTER_COUNT.ilog2() as usize;
const DIRECT_REGISTER_QUERY_DOMAIN: &[u8] = b"direct-register-query-v1";
const DIRECT_REGISTER_TRANSCRIPT_DOMAIN: &[u8] = b"direct-register-rw-v1";
const DIRECT_REGISTER_OPENING_COUNT: usize = 5;
const DIRECT_REGISTER_Z_ARITY: usize = DIRECT_LOOKUP_Z_ARITY + REGISTER_COUNT_USIZE + 2;
const REGISTER_STATE_OFFSET: usize = DIRECT_LOOKUP_Z_ARITY;
const REGISTER_TRANSCRIPT_STATE_SLOT: usize = REGISTER_STATE_OFFSET + REGISTER_COUNT_USIZE;
const REGISTER_TRANSCRIPT_ROUND_SLOT: usize = REGISTER_TRANSCRIPT_STATE_SLOT + 1;

type DirectRegisterNovaSnark = nova_snark::nova::RecursiveSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    DirectRegisterStepCircuit,
>;
type DirectRegisterCompressedSnark = nova_snark::nova::CompressedSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    DirectRegisterStepCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;
type DirectRegisterPublicParams = nova_snark::nova::PublicParams<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    DirectRegisterStepCircuit,
>;

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct DirectRegisterReadWitness {
    pub enabled: bool,
    pub register: u8,
    pub value: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct DirectRegisterWriteWitness {
    pub enabled: bool,
    pub register: u8,
    pub pre_value: u64,
    pub post_value: u64,
}

/// Register-facing values extracted directly from one final tracer cycle.
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct DirectRegisterCycleWitness {
    pub active: bool,
    pub rs1: DirectRegisterReadWitness,
    pub rs2: DirectRegisterReadWitness,
    pub rd: DirectRegisterWriteWitness,
}

impl DirectRegisterCycleWitness {
    fn from_cycle(cycle: &Cycle) -> Self {
        let rs1 = cycle
            .rs1_read()
            .map(|(register, value)| DirectRegisterReadWitness {
                enabled: true,
                register,
                value,
            })
            .unwrap_or_default();
        let rs2 = cycle
            .rs2_read()
            .map(|(register, value)| DirectRegisterReadWitness {
                enabled: true,
                register,
                value,
            })
            .unwrap_or_default();
        let rd = cycle
            .rd_write()
            .map(
                |(register, pre_value, post_value)| DirectRegisterWriteWitness {
                    enabled: true,
                    register,
                    pre_value,
                    post_value,
                },
            )
            .unwrap_or_default();
        Self {
            active: true,
            rs1,
            rs2,
            rd,
        }
    }
}

/// Fixed-shape register witness for one trace block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectRegisterBlockWitness {
    pub block_index: usize,
    pub global_cycle_start: usize,
    pub active_cycles: usize,
    pub cycle_capacity: usize,
    pub terminated: bool,
    pub start_registers: [u64; REGISTER_COUNT_USIZE],
    pub end_registers: [u64; REGISTER_COUNT_USIZE],
    pub cycles: Vec<DirectRegisterCycleWitness>,
}

impl DirectRegisterBlockWitness {
    pub fn from_trace_block(
        block: &TraceBlock,
        cycle_capacity: usize,
    ) -> Result<Self, DirectChunkedError> {
        if cycle_capacity == 0
            || !cycle_capacity.is_power_of_two()
            || block.active_cycles == 0
            || block.active_cycles != block.cycles.len()
            || block.active_cycles > cycle_capacity
        {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: "register block does not fit the fixed power-of-two circuit shape"
                    .to_string(),
            });
        }
        let start_registers = boundary_registers(&block.start_state);
        let expected_end = boundary_registers(&block.end_state);
        let mut running = start_registers;
        let mut cycles = Vec::with_capacity(cycle_capacity);
        for cycle in &block.cycles {
            let witness = DirectRegisterCycleWitness::from_cycle(cycle);
            validate_and_apply_cycle(block.block_index, &mut running, &witness)?;
            cycles.push(witness);
        }
        if running != expected_end {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: "register accesses do not reconstruct end_state.registers".to_string(),
            });
        }
        cycles.resize(cycle_capacity, DirectRegisterCycleWitness::default());
        Ok(Self {
            block_index: block.block_index,
            global_cycle_start: block.global_cycle_start,
            active_cycles: block.active_cycles,
            cycle_capacity,
            terminated: block.end_state.terminated,
            start_registers,
            end_registers: expected_end,
            cycles,
        })
    }
}

fn boundary_registers(state: &MachineBoundaryState) -> [u64; REGISTER_COUNT_USIZE] {
    state.registers.map(|value| value as u64)
}

fn validate_read(
    block_index: usize,
    registers: &[u64; REGISTER_COUNT_USIZE],
    read: &DirectRegisterReadWitness,
    label: &str,
) -> Result<(), DirectChunkedError> {
    if !read.enabled {
        if read != &DirectRegisterReadWitness::default() {
            return Err(DirectChunkedError::InvalidBlock {
                block_index,
                reason: format!("disabled {label} register access is not canonical zero"),
            });
        }
        return Ok(());
    }
    let register = read.register as usize;
    if register >= REGISTER_COUNT_USIZE || registers[register] != read.value {
        return Err(DirectChunkedError::InvalidBlock {
            block_index,
            reason: format!("{label} read does not match the authenticated register state"),
        });
    }
    Ok(())
}

fn validate_and_apply_cycle(
    block_index: usize,
    registers: &mut [u64; REGISTER_COUNT_USIZE],
    cycle: &DirectRegisterCycleWitness,
) -> Result<(), DirectChunkedError> {
    if !cycle.active {
        if cycle != &DirectRegisterCycleWitness::default() {
            return Err(DirectChunkedError::InvalidBlock {
                block_index,
                reason: "inactive register cycle is not canonical zero".to_string(),
            });
        }
        return Ok(());
    }
    validate_read(block_index, registers, &cycle.rs1, "rs1")?;
    validate_read(block_index, registers, &cycle.rs2, "rs2")?;
    if !cycle.rd.enabled {
        if cycle.rd != DirectRegisterWriteWitness::default() {
            return Err(DirectChunkedError::InvalidBlock {
                block_index,
                reason: "disabled rd access is not canonical zero".to_string(),
            });
        }
        return Ok(());
    }
    let register = cycle.rd.register as usize;
    if register >= REGISTER_COUNT_USIZE || registers[register] != cycle.rd.pre_value {
        return Err(DirectChunkedError::InvalidBlock {
            block_index,
            reason: "rd pre-value does not match the authenticated register state".to_string(),
        });
    }
    registers[register] = cycle.rd.post_value;
    Ok(())
}

/// Real per-block Jolt register sumcheck plus all lossless endpoint openings.
#[derive(Clone)]
pub struct DirectRegisterSubclaim {
    pub block: DirectRegisterBlockWitness,
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
    pub opening_claims: [Fr; DIRECT_REGISTER_OPENING_COUNT],
    pub transcript_state_after: [u8; 32],
    pub transcript_round_after: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectRegisterStageStatement {
    pub program_digest: [u8; 32],
    pub lookup_table_commitment: [u8; 32],
    pub block_count: usize,
    pub total_cycles: usize,
    pub initial_registers: [u64; REGISTER_COUNT_USIZE],
    pub final_registers: [u64; REGISTER_COUNT_USIZE],
    pub terminated: bool,
    pub relations: BTreeMap<DirectRelation, DirectRelationState>,
}

/// Combined D2+D3 proof. Lookup and register relations share each Nova step.
#[derive(Clone)]
pub struct DirectRegisterStageProof {
    pub statement: DirectRegisterStageStatement,
    pub lookup_subclaims: Vec<DirectLookupSubclaim>,
    pub register_subclaims: Vec<DirectRegisterSubclaim>,
    pub nova_recursive_snark: Vec<u8>,
    pub spartan_proof: Vec<u8>,
    pub initial_z: Vec<[u8; 32]>,
    pub final_z: Vec<[u8; 32]>,
}

fn signed_delta(pre: u64, post: u64) -> Fr {
    let delta = post as i128 - pre as i128;
    if delta >= 0 {
        Fr::from(delta as u64)
    } else {
        -Fr::from((-delta) as u64)
    }
}

fn append_register_cycle(transcript: &mut PoseidonTranscript, cycle: &DirectRegisterCycleWitness) {
    for value in [
        Fr::from(u64::from(cycle.active)),
        Fr::from(u64::from(cycle.rs1.enabled)),
        Fr::from(cycle.rs1.register as u64),
        Fr::from(cycle.rs1.value),
        Fr::from(u64::from(cycle.rs2.enabled)),
        Fr::from(cycle.rs2.register as u64),
        Fr::from(cycle.rs2.value),
        Fr::from(u64::from(cycle.rd.enabled)),
        Fr::from(cycle.rd.register as u64),
        Fr::from(cycle.rd.pre_value),
        Fr::from(cycle.rd.post_value),
    ] {
        transcript.append_scalar(b"register_word", &value);
    }
}

fn register_query_root(block: &DirectRegisterBlockWitness) -> Fr {
    let mut transcript = PoseidonTranscript::new(DIRECT_REGISTER_QUERY_DOMAIN);
    transcript.append_scalar(b"block_index", &Fr::from(block.block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(block.global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(block.active_cycles as u64));
    for cycle in &block.cycles {
        append_register_cycle(&mut transcript, cycle);
    }
    field_from_digest(&transcript.state)
}

fn append_register_header(
    transcript: &mut PoseidonTranscript,
    block: &DirectRegisterBlockWitness,
    query_root: Fr,
) {
    transcript.append_scalar(b"block_index", &Fr::from(block.block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(block.global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(block.active_cycles as u64));
    transcript.append_scalar(b"register_query_root", &query_root);
}

fn register_input_claims(block: &DirectRegisterBlockWitness, point: &[Fr]) -> [Fr; 3] {
    let weights = EqPolynomial::<Fr>::evals(point);
    let mut claims = [Fr::zero(); 3];
    for (weight, cycle) in weights.into_iter().zip(&block.cycles) {
        claims[0] += weight * Fr::from(cycle.rd.post_value);
        claims[1] += weight * Fr::from(cycle.rs1.value);
        claims[2] += weight * Fr::from(cycle.rs2.value);
    }
    claims
}

fn normalize_register_point(challenges: &[Fr], log_t: usize) -> (Vec<Fr>, Vec<Fr>) {
    let r_cycle = challenges[..log_t].iter().rev().copied().collect();
    let r_address = challenges[log_t..log_t + LOG_REGISTER_COUNT]
        .iter()
        .rev()
        .copied()
        .collect();
    (r_address, r_cycle)
}

fn native_register_output_openings(
    block: &DirectRegisterBlockWitness,
    challenges: &[Fr],
) -> Result<[Fr; DIRECT_REGISTER_OPENING_COUNT], DirectChunkedError> {
    let log_t = block.cycle_capacity.log_2();
    if challenges.len() != log_t + LOG_REGISTER_COUNT {
        return Err(DirectChunkedError::InvalidProofShape(
            "register sumcheck challenge vector has the wrong dimension".to_string(),
        ));
    }
    let (r_address, r_cycle) = normalize_register_point(challenges, log_t);
    let address_weights = EqPolynomial::<Fr>::evals(&r_address);
    let cycle_weights = EqPolynomial::<Fr>::evals(&r_cycle);
    let mut running = block.start_registers;
    let mut val = Fr::zero();
    let mut rs1_ra = Fr::zero();
    let mut rs2_ra = Fr::zero();
    let mut rd_wa = Fr::zero();
    let mut rd_inc = Fr::zero();
    for (cycle_index, cycle) in block.cycles.iter().enumerate() {
        let cycle_weight = cycle_weights[cycle_index];
        let state_at_address = running
            .iter()
            .zip(&address_weights)
            .map(|(value, weight)| Fr::from(*value) * weight)
            .sum::<Fr>();
        val += cycle_weight * state_at_address;
        if cycle.rs1.enabled {
            rs1_ra += cycle_weight * address_weights[cycle.rs1.register as usize];
        }
        if cycle.rs2.enabled {
            rs2_ra += cycle_weight * address_weights[cycle.rs2.register as usize];
        }
        if cycle.rd.enabled {
            rd_wa += cycle_weight * address_weights[cycle.rd.register as usize];
            rd_inc += cycle_weight * signed_delta(cycle.rd.pre_value, cycle.rd.post_value);
            running[cycle.rd.register as usize] = cycle.rd.post_value;
        }
    }
    Ok([val, rs1_ra, rs2_ra, rd_wa, rd_inc])
}

fn input_ids() -> [OpeningId; 5] {
    [
        OpeningId::virt(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
        ),
        OpeningId::virt(
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
        ),
        OpeningId::virt(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        ),
        OpeningId::virt(
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
        ),
        OpeningId::virt(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        ),
    ]
}

fn output_ids() -> [OpeningId; DIRECT_REGISTER_OPENING_COUNT] {
    [
        OpeningId::virt(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        ),
        OpeningId::virt(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        ),
        OpeningId::virt(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        ),
        OpeningId::virt(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        ),
        OpeningId::committed(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        ),
    ]
}

fn seed_register_prover(
    accumulator: &mut ProverOpeningAccumulator<Fr>,
    point: &[Fr],
    claims: [Fr; 3],
) {
    let point =
        OpeningPoint::<BIG_ENDIAN, Fr>::new(point.iter().copied().map(Into::into).collect());
    accumulator.append_virtual(
        VirtualPolynomial::RdWriteValue,
        SumcheckId::RegistersClaimReduction,
        point.clone(),
        claims[0],
    );
    accumulator.append_virtual(
        VirtualPolynomial::Rs1Value,
        SumcheckId::RegistersClaimReduction,
        point.clone(),
        claims[1],
    );
    accumulator.append_virtual(
        VirtualPolynomial::Rs1Value,
        SumcheckId::InstructionInputVirtualization,
        point.clone(),
        claims[1],
    );
    accumulator.append_virtual(
        VirtualPolynomial::Rs2Value,
        SumcheckId::RegistersClaimReduction,
        point.clone(),
        claims[2],
    );
    accumulator.append_virtual(
        VirtualPolynomial::Rs2Value,
        SumcheckId::InstructionInputVirtualization,
        point,
        claims[2],
    );
}

fn seed_register_verifier(
    accumulator: &mut VerifierOpeningAccumulator<Fr>,
    point: &[Fr],
    input_claims: [Fr; 3],
    output_claims: [Fr; DIRECT_REGISTER_OPENING_COUNT],
) {
    let empty = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
    for (id, claim) in output_ids().into_iter().zip(output_claims) {
        accumulator.openings.insert(id, (empty.clone(), claim));
    }
    for (id, claim) in input_ids().into_iter().zip([
        input_claims[0],
        input_claims[1],
        input_claims[1],
        input_claims[2],
        input_claims[2],
    ]) {
        accumulator.openings.insert(id, (empty.clone(), claim));
    }
    let point =
        OpeningPoint::<BIG_ENDIAN, Fr>::new(point.iter().copied().map(Into::into).collect());
    accumulator.append_virtual(
        VirtualPolynomial::RdWriteValue,
        SumcheckId::RegistersClaimReduction,
        point.clone(),
    );
    accumulator.append_virtual(
        VirtualPolynomial::Rs1Value,
        SumcheckId::RegistersClaimReduction,
        point.clone(),
    );
    accumulator.append_virtual(
        VirtualPolynomial::Rs1Value,
        SumcheckId::InstructionInputVirtualization,
        point.clone(),
    );
    accumulator.append_virtual(
        VirtualPolynomial::Rs2Value,
        SumcheckId::RegistersClaimReduction,
        point.clone(),
    );
    accumulator.append_virtual(
        VirtualPolynomial::Rs2Value,
        SumcheckId::InstructionInputVirtualization,
        point,
    );
}

fn padded_trace(block: &TraceBlock, capacity: usize) -> Arc<Vec<Cycle>> {
    let mut trace = block.cycles.clone();
    trace.resize(capacity, Cycle::NoOp);
    Arc::new(trace)
}

fn prove_native_register_subclaim(
    block: &TraceBlock,
    capacity: usize,
    transcript: &mut PoseidonTranscript,
) -> Result<DirectRegisterSubclaim, DirectChunkedError> {
    let witness = DirectRegisterBlockWitness::from_trace_block(block, capacity)?;
    let query_root = register_query_root(&witness);
    append_register_header(transcript, &witness, query_root);
    let log_t = capacity.log_2();
    let r_reduction = transcript.challenge_vector::<Fr>(log_t);
    let input_claims = register_input_claims(&witness, &r_reduction);
    let mut accumulator = ProverOpeningAccumulator::<Fr>::new(log_t);
    seed_register_prover(&mut accumulator, &r_reduction, input_claims);
    let config = ReadWriteConfig::new(log_t, 0);
    let params = RegistersReadWriteCheckingParams::new(capacity, &accumulator, transcript, &config);
    let gamma = params.gamma;
    let input_claim = input_claims[0] + gamma * input_claims[1] + gamma * gamma * input_claims[2];
    let mut batching_transcript = transcript.clone();
    batching_transcript.append_scalar(b"sumcheck_claim", &input_claim);
    let batching_coefficient = batching_transcript.challenge_scalar::<Fr>();
    let bytecode = BytecodePreprocessing::default();
    let memory_layout = MemoryLayout::default();
    let mut prover = RegistersReadWriteCheckingProver::initialize_with_initial_registers(
        params,
        padded_trace(block, capacity),
        &bytecode,
        &memory_layout,
        &witness.start_registers,
    );
    let (proof, challenges, initial_batched_claim) =
        BatchedSumcheck::prove(vec![&mut prover], &mut accumulator, transcript);
    let challenges = challenges.into_iter().map(Into::into).collect::<Vec<Fr>>();
    let opening_claims = native_register_output_openings(&witness, &challenges)?;
    for (id, expected) in output_ids().into_iter().zip(opening_claims) {
        if accumulator.get_opening(id) != expected {
            return Err(DirectChunkedError::InvalidProofShape(
                "native register prover opening differs from trace reconstruction".to_string(),
            ));
        }
    }
    let final_sumcheck_claim = proof
        .compressed_polys
        .iter()
        .zip(&challenges)
        .fold(initial_batched_claim, |claim, (poly, challenge)| {
            poly.eval_from_hint(&claim, &(*challenge).into())
        });
    Ok(DirectRegisterSubclaim {
        block: witness,
        query_root,
        r_reduction,
        input_claims,
        gamma,
        batching_coefficient,
        initial_batched_claim,
        final_sumcheck_claim,
        proof,
        sumcheck_challenges: challenges,
        degree_bound: 3,
        opening_claims,
        transcript_state_after: transcript.state,
        transcript_round_after: transcript.n_rounds,
    })
}

fn verify_native_register_subclaim(
    subclaim: &DirectRegisterSubclaim,
    transcript: &mut PoseidonTranscript,
) -> Result<(), DirectChunkedError> {
    let block = &subclaim.block;
    if block.cycle_capacity == 0
        || !block.cycle_capacity.is_power_of_two()
        || block.cycles.len() != block.cycle_capacity
        || block.active_cycles == 0
        || block.active_cycles > block.cycle_capacity
        || block.cycles[..block.active_cycles]
            .iter()
            .any(|cycle| !cycle.active)
        || block.cycles[block.active_cycles..]
            .iter()
            .any(|cycle| cycle != &DirectRegisterCycleWitness::default())
    {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: block.block_index,
            reason: "register subclaim has non-canonical active/padding layout".to_string(),
        });
    }
    let mut running = block.start_registers;
    for cycle in &block.cycles {
        validate_and_apply_cycle(block.block_index, &mut running, cycle)?;
    }
    if running != block.end_registers {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: block.block_index,
            reason: "register subclaim does not reconstruct its end boundary".to_string(),
        });
    }
    let query_root = register_query_root(block);
    if query_root != subclaim.query_root {
        return Err(DirectChunkedError::InvalidProofShape(
            "register query commitment mismatch".to_string(),
        ));
    }
    append_register_header(transcript, block, query_root);
    let log_t = block.cycle_capacity.log_2();
    let r_reduction = transcript.challenge_vector::<Fr>(log_t);
    let input_claims = register_input_claims(block, &r_reduction);
    if r_reduction != subclaim.r_reduction || input_claims != subclaim.input_claims {
        return Err(DirectChunkedError::InvalidProofShape(
            "register reduction point or input openings mismatch".to_string(),
        ));
    }
    if subclaim.degree_bound != 3
        || subclaim.proof.compressed_polys.len() != log_t + LOG_REGISTER_COUNT
        || subclaim.sumcheck_challenges.len() != log_t + LOG_REGISTER_COUNT
        || native_register_output_openings(block, &subclaim.sumcheck_challenges)?
            != subclaim.opening_claims
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "register sumcheck dimensions or endpoint openings are non-canonical".to_string(),
        ));
    }
    let mut accumulator = VerifierOpeningAccumulator::<Fr>::new(log_t, false);
    seed_register_verifier(
        &mut accumulator,
        &r_reduction,
        input_claims,
        subclaim.opening_claims,
    );
    let config = ReadWriteConfig::new(log_t, 0);
    let verifier = RegistersReadWriteCheckingVerifier::new(
        block.cycle_capacity,
        &accumulator,
        transcript,
        &config,
    );
    let expected_input = input_claims[0]
        + subclaim.gamma * input_claims[1]
        + subclaim.gamma * subclaim.gamma * input_claims[2];
    let mut batching_transcript = transcript.clone();
    batching_transcript.append_scalar(b"sumcheck_claim", &expected_input);
    let expected_batching = batching_transcript.challenge_scalar::<Fr>();
    if expected_batching != subclaim.batching_coefficient
        || expected_input * expected_batching != subclaim.initial_batched_claim
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "register batched sumcheck initial claim mismatch".to_string(),
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
            "native RegistersReadWriteChecking verification failed: {error}"
        ))
    })?
    .into_iter()
    .map(Into::into)
    .collect::<Vec<Fr>>();
    let final_claim = subclaim
        .proof
        .compressed_polys
        .iter()
        .zip(&subclaim.sumcheck_challenges)
        .fold(
            subclaim.initial_batched_claim,
            |claim, (poly, challenge)| poly.eval_from_hint(&claim, &(*challenge).into()),
        );
    if challenges != subclaim.sumcheck_challenges
        || final_claim != subclaim.final_sumcheck_claim
        || transcript.state != subclaim.transcript_state_after
        || transcript.n_rounds != subclaim.transcript_round_after
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "register Fiat-Shamir transcript mismatch".to_string(),
        ));
    }
    Ok(())
}

#[derive(Clone)]
struct AllocatedRegisterRead {
    enabled: AllocatedBit,
    address: AllocatedNum<NovaScalar>,
    address_bits: Vec<AllocatedBit>,
    value: AllocatedNum<NovaScalar>,
}

#[derive(Clone)]
struct AllocatedRegisterWrite {
    enabled: AllocatedBit,
    address: AllocatedNum<NovaScalar>,
    address_bits: Vec<AllocatedBit>,
    pre_value: AllocatedNum<NovaScalar>,
    post_value: AllocatedNum<NovaScalar>,
}

#[derive(Clone)]
struct AllocatedRegisterCycle {
    active: AllocatedBit,
    rs1: AllocatedRegisterRead,
    rs2: AllocatedRegisterRead,
    rd: AllocatedRegisterWrite,
}

fn alloc_register_address<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    address: u8,
    label: &'static str,
) -> Result<(AllocatedNum<NovaScalar>, Vec<AllocatedBit>), SynthesisError> {
    let bits = (0..LOG_REGISTER_COUNT)
        .map(|bit| {
            AllocatedBit::alloc(
                cs.namespace(|| format!("{label} bit {bit}")),
                Some(((address >> bit) & 1) == 1),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let number = alloc_witness_num(cs.namespace(|| label), NovaScalar::from(address as u64))?;
    let packed = bits.iter().enumerate().fold(
        LinearCombination::<NovaScalar>::zero(),
        |lc, (bit, value)| lc + (NovaScalar::from(1u64 << bit), value.get_variable()),
    );
    cs.enforce(
        || format!("{label} bit packing"),
        |_| packed - number.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok((number, bits))
}

fn enforce_disabled_zero<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    enabled: &AllocatedBit,
    value: &AllocatedNum<NovaScalar>,
    label: &'static str,
) {
    cs.enforce(
        || label,
        |lc| lc + CS::one() - enabled.get_variable(),
        |lc| lc + value.get_variable(),
        |lc| lc,
    );
}

fn enforce_enabled_implies_active<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    enabled: &AllocatedBit,
    active: &AllocatedBit,
    label: &'static str,
) {
    cs.enforce(
        || label,
        |lc| lc + enabled.get_variable(),
        |lc| lc + CS::one() - active.get_variable(),
        |lc| lc,
    );
}

fn allocate_register_read<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    read: &DirectRegisterReadWitness,
    active: &AllocatedBit,
    label: &'static str,
) -> Result<AllocatedRegisterRead, SynthesisError> {
    let enabled = AllocatedBit::alloc(
        cs.namespace(|| format!("{label} enabled")),
        Some(read.enabled),
    )?;
    let (address, address_bits) = alloc_register_address(
        cs.namespace(|| format!("{label} address")),
        read.register,
        "register address",
    )?;
    let (value, _) = alloc_u64_bits(
        cs.namespace(|| format!("{label} value")),
        read.value,
        "register read value",
    )?;
    enforce_enabled_implies_active(
        cs.namespace(|| format!("{label} is active")),
        &enabled,
        active,
        "register read enabled implies active",
    );
    enforce_disabled_zero(
        cs.namespace(|| format!("{label} disabled address")),
        &enabled,
        &address,
        "disabled register read address is zero",
    );
    enforce_disabled_zero(
        cs.namespace(|| format!("{label} disabled value")),
        &enabled,
        &value,
        "disabled register read value is zero",
    );
    Ok(AllocatedRegisterRead {
        enabled,
        address,
        address_bits,
        value,
    })
}

fn allocate_register_write<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    write: &DirectRegisterWriteWitness,
    active: &AllocatedBit,
) -> Result<AllocatedRegisterWrite, SynthesisError> {
    let enabled = AllocatedBit::alloc(cs.namespace(|| "rd enabled"), Some(write.enabled))?;
    let (address, address_bits) = alloc_register_address(
        cs.namespace(|| "rd address"),
        write.register,
        "register address",
    )?;
    let (pre_value, _) = alloc_u64_bits(
        cs.namespace(|| "rd pre value"),
        write.pre_value,
        "register pre value",
    )?;
    let (post_value, _) = alloc_u64_bits(
        cs.namespace(|| "rd post value"),
        write.post_value,
        "register post value",
    )?;
    enforce_enabled_implies_active(
        cs.namespace(|| "rd is active"),
        &enabled,
        active,
        "register write enabled implies active",
    );
    for (label, value) in [
        ("disabled rd address", &address),
        ("disabled rd pre value", &pre_value),
        ("disabled rd post value", &post_value),
    ] {
        enforce_disabled_zero(cs.namespace(|| label), &enabled, value, label);
    }
    Ok(AllocatedRegisterWrite {
        enabled,
        address,
        address_bits,
        pre_value,
        post_value,
    })
}

fn allocate_register_cycle<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    cycle: &DirectRegisterCycleWitness,
) -> Result<AllocatedRegisterCycle, SynthesisError> {
    let active = AllocatedBit::alloc(cs.namespace(|| "active"), Some(cycle.active))?;
    let rs1 = allocate_register_read(cs.namespace(|| "rs1"), &cycle.rs1, &active, "rs1")?;
    let rs2 = allocate_register_read(cs.namespace(|| "rs2"), &cycle.rs2, &active, "rs2")?;
    let rd = allocate_register_write(cs.namespace(|| "rd"), &cycle.rd, &active)?;
    Ok(AllocatedRegisterCycle {
        active,
        rs1,
        rs2,
        rd,
    })
}

fn address_selector<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    bits_le: &[AllocatedBit],
    index: usize,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let one = alloc_nova_constant(cs.namespace(|| "selector one"), NovaScalar::one())?;
    let mut selector = one.clone();
    for (bit_index, bit) in bits_le.iter().enumerate() {
        let factor = if ((index >> bit_index) & 1) == 1 {
            bit_as_num(bit)
        } else {
            sub_nums(
                cs.namespace(|| format!("{label} inverted bit {bit_index}")),
                &one,
                &bit_as_num(bit),
                "one minus address bit",
            )?
        };
        selector = mul_nums(
            cs.namespace(|| format!("{label} selector product {bit_index}")),
            &selector,
            &factor,
            "address selector product",
        )?;
    }
    Ok(selector)
}

fn select_register<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    registers: &[AllocatedNum<NovaScalar>],
    selectors: &[AllocatedNum<NovaScalar>],
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let mut selected = alloc_nova_constant(cs.namespace(|| "selected zero"), NovaScalar::zero())?;
    for (index, (register, selector)) in registers.iter().zip(selectors).enumerate() {
        let term = mul_nums(
            cs.namespace(|| format!("{label} selected term {index}")),
            register,
            selector,
            "register selection term",
        )?;
        selected = add_nums(
            cs.namespace(|| format!("{label} selected sum {index}")),
            &selected,
            &term,
            "register selection sum",
        )?;
    }
    Ok(selected)
}

fn enforce_enabled_value<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    enabled: &AllocatedBit,
    selected: &AllocatedNum<NovaScalar>,
    claimed: &AllocatedNum<NovaScalar>,
    label: &'static str,
) {
    cs.enforce(
        || label,
        |lc| lc + enabled.get_variable(),
        |lc| lc + selected.get_variable() - claimed.get_variable(),
        |lc| lc,
    );
}

fn synthesize_register_query_root<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    block_index: &AllocatedNum<NovaScalar>,
    cycle_start: &AllocatedNum<NovaScalar>,
    active_cycles: &AllocatedNum<NovaScalar>,
    cycles: &[AllocatedRegisterCycle],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let mut transcript = allocated_poseidon_initial(
        cs.namespace(|| "register query initial state"),
        DIRECT_REGISTER_QUERY_DOMAIN,
    )?;
    for (label, value) in [
        ("block_index", block_index),
        ("cycle_start", cycle_start),
        ("active_cycles", active_cycles),
    ] {
        transcript = poseidon_absorb(
            cs.namespace(|| format!("register query {label}")),
            &transcript,
            value,
            label,
        )?;
    }
    for (index, cycle) in cycles.iter().enumerate() {
        let values = [
            bit_as_num(&cycle.active),
            bit_as_num(&cycle.rs1.enabled),
            cycle.rs1.address.clone(),
            cycle.rs1.value.clone(),
            bit_as_num(&cycle.rs2.enabled),
            cycle.rs2.address.clone(),
            cycle.rs2.value.clone(),
            bit_as_num(&cycle.rd.enabled),
            cycle.rd.address.clone(),
            cycle.rd.pre_value.clone(),
            cycle.rd.post_value.clone(),
        ];
        for (word, value) in values.iter().enumerate() {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("register cycle {index} word {word}")),
                &transcript,
                value,
                "register_word",
            )?;
        }
    }
    Ok(transcript.state)
}

fn sumcheck_witness(
    subclaim: &DirectRegisterSubclaim,
) -> Result<RecursiveClearSumcheckStageWitness, SynthesisError> {
    if subclaim.proof.compressed_polys.len() != subclaim.sumcheck_challenges.len() {
        return Err(SynthesisError::Unsatisfiable(
            "register sumcheck proof/challenge length mismatch".to_string(),
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
        stage_index: 1,
        degree_bound: subclaim.degree_bound,
        initial_claim: recursive_field(subclaim.initial_batched_claim)?,
        rounds,
        expected_final_claim: recursive_field(subclaim.final_sumcheck_claim)?,
    })
}

fn eq_between_points<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    left: &[AllocatedNum<NovaScalar>],
    right: &[AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    if left.len() != right.len() {
        return Err(SynthesisError::Unsatisfiable(
            "register equality points have different dimensions".to_string(),
        ));
    }
    let one = alloc_nova_constant(cs.namespace(|| "eq one"), NovaScalar::one())?;
    let two = NovaScalar::from(2u64);
    let mut result = one.clone();
    for (index, (x, y)) in left.iter().zip(right).enumerate() {
        let x_plus_y = add_nums(
            cs.namespace(|| format!("eq x plus y {index}")),
            x,
            y,
            "eq x plus y",
        )?;
        let xy = mul_nums(cs.namespace(|| format!("eq xy {index}")), x, y, "eq xy")?;
        let twice_xy = scale_num(
            cs.namespace(|| format!("eq twice xy {index}")),
            &xy,
            two,
            "eq twice xy",
        )?;
        let one_minus_sum = sub_nums(
            cs.namespace(|| format!("eq one minus sum {index}")),
            &one,
            &x_plus_y,
            "eq one minus sum",
        )?;
        let factor = add_nums(
            cs.namespace(|| format!("eq factor {index}")),
            &one_minus_sum,
            &twice_xy,
            "eq factor",
        )?;
        result = mul_nums(
            cs.namespace(|| format!("eq product {index}")),
            &result,
            &factor,
            "eq point product",
        )?;
    }
    Ok(result)
}

#[derive(Clone, Default)]
struct DirectRegisterStepCircuit {
    lookup: Option<DirectLookupSubclaim>,
    register: Option<DirectRegisterSubclaim>,
    final_step: bool,
}

impl DirectRegisterStepCircuit {
    fn for_subclaims(
        lookup: DirectLookupSubclaim,
        register: DirectRegisterSubclaim,
        final_step: bool,
    ) -> Self {
        Self {
            lookup: Some(lookup),
            register: Some(register),
            final_step,
        }
    }
}

impl StepCircuit<NovaScalar> for DirectRegisterStepCircuit {
    fn arity(&self) -> usize {
        DIRECT_REGISTER_Z_ARITY
    }

    fn synthesize<CS: ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<NovaScalar>],
    ) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
        if z.len() != DIRECT_REGISTER_Z_ARITY {
            return Err(SynthesisError::Unsatisfiable(
                "direct register Nova state has invalid arity".to_string(),
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
        let block = &register.block;
        if lookup.block.block_index != block.block_index
            || lookup.block.global_cycle_start != block.global_cycle_start
            || lookup.block.active_cycles != block.active_cycles
            || lookup.block.cycle_capacity != block.cycle_capacity
            || lookup.block.terminated != block.terminated
        {
            return Err(SynthesisError::Unsatisfiable(
                "lookup/register subclaims describe different block metadata".to_string(),
            ));
        }
        let lookup_circuit = DirectLookupStepCircuit::for_subclaim(lookup.clone(), self.final_step);
        let lookup_output = {
            let mut namespace = cs.namespace(|| "D2 lookup relation");
            lookup_circuit.synthesize(&mut namespace, &z[..DIRECT_LOOKUP_Z_ARITY])?
        };
        let log_t = block.cycle_capacity.log_2();
        let expected_rounds = log_t + LOG_REGISTER_COUNT;
        if block.cycles.len() != block.cycle_capacity
            || subclaim_dimensions_invalid(register, log_t, expected_rounds)
        {
            return Err(SynthesisError::Unsatisfiable(
                "direct register subclaim has invalid recursive dimensions".to_string(),
            ));
        }
        let zero = alloc_nova_constant(cs.namespace(|| "register zero"), NovaScalar::zero())?;
        let block_index = alloc_witness_num(
            cs.namespace(|| "register block index"),
            NovaScalar::from(block.block_index as u64),
        )?;
        let cycle_start = alloc_witness_num(
            cs.namespace(|| "register cycle start"),
            NovaScalar::from(block.global_cycle_start as u64),
        )?;
        let active_cycles = alloc_witness_num(
            cs.namespace(|| "register active cycles"),
            NovaScalar::from(block.active_cycles as u64),
        )?;
        enforce_num_equal(
            cs.namespace(|| "register block index follows folding state"),
            &block_index,
            &z[0],
            "register block index follows folding state",
        );
        enforce_num_equal(
            cs.namespace(|| "register cycle start follows folding state"),
            &cycle_start,
            &z[2],
            "register cycle start follows folding state",
        );
        let lookup_cycle_delta = sub_nums(
            cs.namespace(|| "lookup cycle delta"),
            &lookup_output[1],
            &z[1],
            "lookup cycle delta",
        )?;
        enforce_num_equal(
            cs.namespace(|| "lookup register active cycle binding"),
            &active_cycles,
            &lookup_cycle_delta,
            "lookup register active cycle binding",
        );

        let cycles = block
            .cycles
            .iter()
            .enumerate()
            .map(|(index, cycle)| {
                allocate_register_cycle(cs.namespace(|| format!("register cycle {index}")), cycle)
            })
            .collect::<Result<Vec<_>, _>>()?;
        cs.enforce(
            || "first register row is active",
            |lc| lc + cycles[0].active.get_variable() - CS::one(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        for index in 1..cycles.len() {
            cs.enforce(
                || format!("register active rows form prefix {index}"),
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
            || "register active sum",
            |_| active_sum - active_cycles.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );

        let mut running = z[REGISTER_STATE_OFFSET..REGISTER_TRANSCRIPT_STATE_SLOT].to_vec();
        for (index, (state, claimed)) in running.iter().zip(block.start_registers).enumerate() {
            let claimed = alloc_witness_num(
                cs.namespace(|| format!("claimed start register {index}")),
                NovaScalar::from(claimed),
            )?;
            enforce_num_equal(
                cs.namespace(|| format!("start register boundary {index}")),
                state,
                &claimed,
                "start register boundary",
            );
        }
        enforce_num_equal(
            cs.namespace(|| "x0 input is zero"),
            &running[0],
            &zero,
            "x0 input is zero",
        );
        let mut states_before = Vec::with_capacity(cycles.len());
        for (cycle_index, cycle) in cycles.iter().enumerate() {
            states_before.push(running.clone());
            let rs1_selectors = (0..REGISTER_COUNT_USIZE)
                .map(|register_index| {
                    address_selector(
                        cs.namespace(|| {
                            format!("cycle {cycle_index} rs1 selector {register_index}")
                        }),
                        &cycle.rs1.address_bits,
                        register_index,
                        "rs1 address",
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let rs2_selectors = (0..REGISTER_COUNT_USIZE)
                .map(|register_index| {
                    address_selector(
                        cs.namespace(|| {
                            format!("cycle {cycle_index} rs2 selector {register_index}")
                        }),
                        &cycle.rs2.address_bits,
                        register_index,
                        "rs2 address",
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let rd_selectors = (0..REGISTER_COUNT_USIZE)
                .map(|register_index| {
                    address_selector(
                        cs.namespace(|| {
                            format!("cycle {cycle_index} rd selector {register_index}")
                        }),
                        &cycle.rd.address_bits,
                        register_index,
                        "rd address",
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let rs1_selected = select_register(
                cs.namespace(|| format!("cycle {cycle_index} rs1 selection")),
                &running,
                &rs1_selectors,
                "rs1",
            )?;
            let rs2_selected = select_register(
                cs.namespace(|| format!("cycle {cycle_index} rs2 selection")),
                &running,
                &rs2_selectors,
                "rs2",
            )?;
            let rd_selected = select_register(
                cs.namespace(|| format!("cycle {cycle_index} rd selection")),
                &running,
                &rd_selectors,
                "rd",
            )?;
            enforce_enabled_value(
                cs.namespace(|| format!("cycle {cycle_index} rs1 value")),
                &cycle.rs1.enabled,
                &rs1_selected,
                &cycle.rs1.value,
                "rs1 read matches current register state",
            );
            enforce_enabled_value(
                cs.namespace(|| format!("cycle {cycle_index} rs2 value")),
                &cycle.rs2.enabled,
                &rs2_selected,
                &cycle.rs2.value,
                "rs2 read matches current register state",
            );
            enforce_enabled_value(
                cs.namespace(|| format!("cycle {cycle_index} rd pre value")),
                &cycle.rd.enabled,
                &rd_selected,
                &cycle.rd.pre_value,
                "rd pre-value matches current register state",
            );
            let delta = sub_nums(
                cs.namespace(|| format!("cycle {cycle_index} rd delta")),
                &cycle.rd.post_value,
                &cycle.rd.pre_value,
                "rd delta",
            )?;
            let enabled = bit_as_num(&cycle.rd.enabled);
            let enabled_delta = mul_nums(
                cs.namespace(|| format!("cycle {cycle_index} enabled delta")),
                &enabled,
                &delta,
                "enabled rd delta",
            )?;
            running = running
                .iter()
                .zip(&rd_selectors)
                .enumerate()
                .map(|(register_index, (old, selector))| {
                    let update = mul_nums(
                        cs.namespace(|| {
                            format!("cycle {cycle_index} update register {register_index}")
                        }),
                        selector,
                        &enabled_delta,
                        "selected register update",
                    )?;
                    add_nums(
                        cs.namespace(|| {
                            format!("cycle {cycle_index} next register {register_index}")
                        }),
                        old,
                        &update,
                        "next register state",
                    )
                })
                .collect::<Result<Vec<_>, SynthesisError>>()?;
            enforce_num_equal(
                cs.namespace(|| format!("cycle {cycle_index} x0 remains zero")),
                &running[0],
                &zero,
                "x0 remains zero",
            );
        }
        for (index, (state, claimed)) in running.iter().zip(block.end_registers).enumerate() {
            let claimed = alloc_witness_num(
                cs.namespace(|| format!("claimed end register {index}")),
                NovaScalar::from(claimed),
            )?;
            enforce_num_equal(
                cs.namespace(|| format!("end register boundary {index}")),
                state,
                &claimed,
                "end register boundary",
            );
        }

        let query_root = synthesize_register_query_root(
            cs.namespace(|| "register query root"),
            &block_index,
            &cycle_start,
            &active_cycles,
            &cycles,
        )?;
        let claimed_query_root = alloc_witness_num(
            cs.namespace(|| "claimed register query root"),
            nova_from_fr(&register.query_root)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "register query root binding"),
            &query_root,
            &claimed_query_root,
            "register query root binding",
        );
        let mut transcript = AllocatedRecursivePoseidonTranscriptState {
            state: z[REGISTER_TRANSCRIPT_STATE_SLOT].clone(),
            n_rounds: z[REGISTER_TRANSCRIPT_ROUND_SLOT].clone(),
        };
        for (label, value) in [
            ("block_index", &block_index),
            ("cycle_start", &cycle_start),
            ("active_cycles", &active_cycles),
            ("register_query_root", &query_root),
        ] {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("register header {label}")),
                &transcript,
                value,
                label,
            )?;
        }
        let mut r_reduction = Vec::with_capacity(log_t);
        for index in 0..log_t {
            transcript = poseidon_challenge(
                cs.namespace(|| format!("register reduction challenge {index}")),
                &transcript,
            )?;
            let claimed = alloc_witness_num(
                cs.namespace(|| format!("claimed register reduction challenge {index}")),
                nova_from_fr(&register.r_reduction[index])?,
            )?;
            enforce_num_equal(
                cs.namespace(|| format!("register reduction challenge binding {index}")),
                &transcript.state,
                &claimed,
                "register reduction challenge binding",
            );
            r_reduction.push(transcript.state.clone());
        }
        let rd_values = cycles
            .iter()
            .map(|cycle| cycle.rd.post_value.clone())
            .collect::<Vec<_>>();
        let rs1_values = cycles
            .iter()
            .map(|cycle| cycle.rs1.value.clone())
            .collect::<Vec<_>>();
        let rs2_values = cycles
            .iter()
            .map(|cycle| cycle.rs2.value.clone())
            .collect::<Vec<_>>();
        let derived_inputs = [
            mle_from_values(
                cs.namespace(|| "rd write reduction opening"),
                &rd_values,
                &r_reduction,
                "rd write reduction opening",
            )?,
            mle_from_values(
                cs.namespace(|| "rs1 reduction opening"),
                &rs1_values,
                &r_reduction,
                "rs1 reduction opening",
            )?,
            mle_from_values(
                cs.namespace(|| "rs2 reduction opening"),
                &rs2_values,
                &r_reduction,
                "rs2 reduction opening",
            )?,
        ];
        for (index, derived) in derived_inputs.iter().enumerate() {
            let claimed = alloc_witness_num(
                cs.namespace(|| format!("claimed register input opening {index}")),
                nova_from_fr(&register.input_claims[index])?,
            )?;
            enforce_num_equal(
                cs.namespace(|| format!("register input opening binding {index}")),
                derived,
                &claimed,
                "register input opening binding",
            );
        }
        transcript = poseidon_challenge(cs.namespace(|| "register gamma"), &transcript)?;
        let gamma = transcript.state.clone();
        let claimed_gamma = alloc_witness_num(
            cs.namespace(|| "claimed register gamma"),
            nova_from_fr(&register.gamma)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "register gamma binding"),
            &gamma,
            &claimed_gamma,
            "register gamma binding",
        );
        let gamma_squared = mul_nums(
            cs.namespace(|| "register gamma squared"),
            &gamma,
            &gamma,
            "register gamma squared",
        )?;
        let gamma_rs1 = mul_nums(
            cs.namespace(|| "gamma rs1"),
            &gamma,
            &derived_inputs[1],
            "gamma rs1",
        )?;
        let gamma_squared_rs2 = mul_nums(
            cs.namespace(|| "gamma squared rs2"),
            &gamma_squared,
            &derived_inputs[2],
            "gamma squared rs2",
        )?;
        let input_claim = add_nums(
            cs.namespace(|| "register input first sum"),
            &derived_inputs[0],
            &gamma_rs1,
            "register input first sum",
        )?;
        let input_claim = add_nums(
            cs.namespace(|| "register input claim"),
            &input_claim,
            &gamma_squared_rs2,
            "register input claim",
        )?;
        transcript = poseidon_absorb(
            cs.namespace(|| "append register sumcheck claim"),
            &transcript,
            &input_claim,
            "sumcheck_claim",
        )?;
        transcript = poseidon_challenge(
            cs.namespace(|| "register sumcheck batching coefficient"),
            &transcript,
        )?;
        let batching = transcript.state.clone();
        let claimed_batching = alloc_witness_num(
            cs.namespace(|| "claimed register batching"),
            nova_from_fr(&register.batching_coefficient)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "register batching binding"),
            &batching,
            &claimed_batching,
            "register batching binding",
        );
        let initial_batched_claim = mul_nums(
            cs.namespace(|| "register initial batched claim"),
            &input_claim,
            &batching,
            "register initial batched claim",
        )?;
        let sumcheck = sumcheck_witness(register)?;
        let allocated_sumcheck = synthesize_recursive_clear_sumcheck_stage(
            cs.namespace(|| "RegistersReadWriteChecking sumcheck arithmetic"),
            &sumcheck,
            1,
            expected_rounds,
            3,
        )?;
        enforce_num_equal(
            cs.namespace(|| "register sumcheck initial claim"),
            &allocated_sumcheck.initial_claim,
            &initial_batched_claim,
            "register sumcheck initial claim",
        );
        transcript = synthesize_recursive_clear_sumcheck_transcript(
            cs.namespace(|| "RegistersReadWriteChecking transcript"),
            &transcript,
            &allocated_sumcheck,
        )?;

        let (cycle_challenges_low_to_high, address_challenges_low_to_high) =
            allocated_sumcheck.challenges.split_at(log_t);
        let r_cycle = cycle_challenges_low_to_high
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        let r_address = address_challenges_low_to_high
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        let cycle_weights = (0..block.cycle_capacity)
            .map(|index| {
                eq_weight_for_index(
                    cs.namespace(|| format!("register cycle endpoint weight {index}")),
                    &r_cycle,
                    index,
                    "register cycle endpoint weight",
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let address_weights = (0..REGISTER_COUNT_USIZE)
            .map(|index| {
                eq_weight_for_index(
                    cs.namespace(|| format!("register address endpoint weight {index}")),
                    &r_address,
                    index,
                    "register address endpoint weight",
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut val_opening = zero.clone();
        let mut rs1_ra_opening = zero.clone();
        let mut rs2_ra_opening = zero.clone();
        let mut rd_wa_opening = zero.clone();
        let mut rd_inc_opening = zero.clone();
        for cycle_index in 0..block.cycle_capacity {
            let mut state_at_address = zero.clone();
            for register_index in 0..REGISTER_COUNT_USIZE {
                let term = mul_nums(
                    cs.namespace(|| {
                        format!("cycle {cycle_index} state MLE register {register_index}")
                    }),
                    &states_before[cycle_index][register_index],
                    &address_weights[register_index],
                    "register state address MLE term",
                )?;
                state_at_address = add_nums(
                    cs.namespace(|| format!("cycle {cycle_index} state MLE sum {register_index}")),
                    &state_at_address,
                    &term,
                    "register state address MLE sum",
                )?;
            }
            let val_term = mul_nums(
                cs.namespace(|| format!("cycle {cycle_index} val endpoint term")),
                &cycle_weights[cycle_index],
                &state_at_address,
                "register val endpoint term",
            )?;
            val_opening = add_nums(
                cs.namespace(|| format!("cycle {cycle_index} val endpoint sum")),
                &val_opening,
                &val_term,
                "register val endpoint sum",
            )?;
            for (label, access_enabled, access_bits, accumulator) in [
                (
                    "rs1",
                    &cycles[cycle_index].rs1.enabled,
                    &cycles[cycle_index].rs1.address_bits,
                    &mut rs1_ra_opening,
                ),
                (
                    "rs2",
                    &cycles[cycle_index].rs2.enabled,
                    &cycles[cycle_index].rs2.address_bits,
                    &mut rs2_ra_opening,
                ),
                (
                    "rd",
                    &cycles[cycle_index].rd.enabled,
                    &cycles[cycle_index].rd.address_bits,
                    &mut rd_wa_opening,
                ),
            ] {
                let address_eq = address_weights
                    .iter()
                    .enumerate()
                    .try_fold(zero.clone(), |sum, (register_index, weight)| {
                        let selector = address_selector(
                            cs.namespace(|| {
                                format!("cycle {cycle_index} {label} endpoint selector {register_index}")
                            }),
                            access_bits,
                            register_index,
                            "endpoint access address",
                        )?;
                        let term = mul_nums(
                            cs.namespace(|| {
                                format!("cycle {cycle_index} {label} endpoint address term {register_index}")
                            }),
                            &selector,
                            weight,
                            "endpoint access address term",
                        )?;
                        add_nums(
                            cs.namespace(|| {
                                format!("cycle {cycle_index} {label} endpoint address sum {register_index}")
                            }),
                            &sum,
                            &term,
                            "endpoint access address sum",
                        )
                    })?;
                let enabled = bit_as_num(access_enabled);
                let enabled_address = mul_nums(
                    cs.namespace(|| format!("cycle {cycle_index} {label} enabled address")),
                    &enabled,
                    &address_eq,
                    "enabled endpoint address",
                )?;
                let term = mul_nums(
                    cs.namespace(|| format!("cycle {cycle_index} {label} endpoint term")),
                    &cycle_weights[cycle_index],
                    &enabled_address,
                    "endpoint access term",
                )?;
                *accumulator = add_nums(
                    cs.namespace(|| format!("cycle {cycle_index} {label} endpoint sum")),
                    accumulator,
                    &term,
                    "endpoint access sum",
                )?;
            }
            let inc = sub_nums(
                cs.namespace(|| format!("cycle {cycle_index} endpoint rd inc")),
                &cycles[cycle_index].rd.post_value,
                &cycles[cycle_index].rd.pre_value,
                "endpoint rd inc",
            )?;
            let enabled_inc = mul_nums(
                cs.namespace(|| format!("cycle {cycle_index} enabled endpoint rd inc")),
                &bit_as_num(&cycles[cycle_index].rd.enabled),
                &inc,
                "enabled endpoint rd inc",
            )?;
            let inc_term = mul_nums(
                cs.namespace(|| format!("cycle {cycle_index} endpoint rd inc term")),
                &cycle_weights[cycle_index],
                &enabled_inc,
                "endpoint rd inc term",
            )?;
            rd_inc_opening = add_nums(
                cs.namespace(|| format!("cycle {cycle_index} endpoint rd inc sum")),
                &rd_inc_opening,
                &inc_term,
                "endpoint rd inc sum",
            )?;
        }
        let derived_openings = [
            val_opening,
            rs1_ra_opening,
            rs2_ra_opening,
            rd_wa_opening,
            rd_inc_opening,
        ];
        for (index, derived) in derived_openings.iter().enumerate() {
            let claimed = alloc_witness_num(
                cs.namespace(|| format!("claimed register endpoint opening {index}")),
                nova_from_fr(&register.opening_claims[index])?,
            )?;
            enforce_num_equal(
                cs.namespace(|| format!("register endpoint opening binding {index}")),
                derived,
                &claimed,
                "register endpoint opening binding",
            );
        }
        let inc_plus_val = add_nums(
            cs.namespace(|| "register endpoint inc plus val"),
            &derived_openings[4],
            &derived_openings[0],
            "register endpoint inc plus val",
        )?;
        let rd_term = mul_nums(
            cs.namespace(|| "register endpoint rd term"),
            &derived_openings[3],
            &inc_plus_val,
            "register endpoint rd term",
        )?;
        let rs1_term = mul_nums(
            cs.namespace(|| "register endpoint rs1 term"),
            &derived_openings[1],
            &derived_openings[0],
            "register endpoint rs1 term",
        )?;
        let rs2_term = mul_nums(
            cs.namespace(|| "register endpoint rs2 term"),
            &derived_openings[2],
            &derived_openings[0],
            "register endpoint rs2 term",
        )?;
        let gamma_rs1_term = mul_nums(
            cs.namespace(|| "register endpoint gamma rs1"),
            &gamma,
            &rs1_term,
            "register endpoint gamma rs1",
        )?;
        let gamma_squared_rs2_term = mul_nums(
            cs.namespace(|| "register endpoint gamma squared rs2"),
            &gamma_squared,
            &rs2_term,
            "register endpoint gamma squared rs2",
        )?;
        let endpoint = add_nums(
            cs.namespace(|| "register endpoint first sum"),
            &rd_term,
            &gamma_rs1_term,
            "register endpoint first sum",
        )?;
        let endpoint = add_nums(
            cs.namespace(|| "register endpoint access sum"),
            &endpoint,
            &gamma_squared_rs2_term,
            "register endpoint access sum",
        )?;
        let eq = eq_between_points(
            cs.namespace(|| "register endpoint equality"),
            &r_cycle,
            &r_reduction,
        )?;
        let endpoint = mul_nums(
            cs.namespace(|| "register endpoint equality product"),
            &eq,
            &endpoint,
            "register endpoint equality product",
        )?;
        let endpoint = mul_nums(
            cs.namespace(|| "register batched endpoint"),
            &endpoint,
            &batching,
            "register batched endpoint",
        )?;
        enforce_num_equal(
            cs.namespace(|| "register sumcheck endpoint binding"),
            &allocated_sumcheck.final_claim,
            &endpoint,
            "register sumcheck endpoint binding",
        );
        for (index, opening) in derived_inputs.iter().enumerate() {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("append register input opening {index}")),
                &transcript,
                opening,
                "opening_claim",
            )?;
        }
        for (index, opening) in derived_openings.iter().enumerate() {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("append register endpoint opening {index}")),
                &transcript,
                opening,
                "opening_claim",
            )?;
        }
        let claimed_transcript_state = alloc_witness_num(
            cs.namespace(|| "claimed final register transcript state"),
            nova_from_fr(&field_from_digest(&register.transcript_state_after))?,
        )?;
        let claimed_transcript_round = alloc_witness_num(
            cs.namespace(|| "claimed final register transcript round"),
            NovaScalar::from(register.transcript_round_after as u64),
        )?;
        enforce_num_equal(
            cs.namespace(|| "final register transcript state binding"),
            &transcript.state,
            &claimed_transcript_state,
            "final register transcript state binding",
        );
        enforce_num_equal(
            cs.namespace(|| "final register transcript round binding"),
            &transcript.n_rounds,
            &claimed_transcript_round,
            "final register transcript round binding",
        );
        let mut output = lookup_output;
        output.extend(running);
        output.push(transcript.state);
        output.push(transcript.n_rounds);
        Ok(output)
    }
}

fn subclaim_dimensions_invalid(
    subclaim: &DirectRegisterSubclaim,
    log_t: usize,
    expected_rounds: usize,
) -> bool {
    subclaim.r_reduction.len() != log_t
        || subclaim.sumcheck_challenges.len() != expected_rounds
        || subclaim.proof.compressed_polys.len() != expected_rounds
        || subclaim.degree_bound != 3
}

fn direct_register_initial_z(
    preprocessing: &DirectChunkedPreprocessing,
    initial_registers: &[u64; REGISTER_COUNT_USIZE],
) -> Vec<NovaScalar> {
    let mut z = direct_initial_z(preprocessing);
    z.extend(initial_registers.iter().copied().map(NovaScalar::from));
    let transcript = PoseidonTranscript::new(DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
    z.push(
        nova_from_fr(&field_from_digest(&transcript.state))
            .expect("register transcript state is a canonical scalar"),
    );
    z.push(NovaScalar::zero());
    z
}

fn validate_combined_sequence(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    lookup_subclaims: &[DirectLookupSubclaim],
    register_subclaims: &[DirectRegisterSubclaim],
) -> Result<(usize, usize), DirectChunkedError> {
    let (block_count, total_cycles) =
        validate_lookup_subclaim_sequence(preprocessing, capacity, lookup_subclaims)?;
    if register_subclaims.len() != block_count {
        return Err(DirectChunkedError::InvalidProofShape(
            "lookup/register subclaim counts differ".to_string(),
        ));
    }
    for position in 0..block_count {
        let lookup = &lookup_subclaims[position].block;
        let register = &register_subclaims[position].block;
        if register.block_index != position
            || register.cycle_capacity != capacity
            || register.block_index != lookup.block_index
            || register.global_cycle_start != lookup.global_cycle_start
            || register.active_cycles != lookup.active_cycles
            || register.terminated != lookup.terminated
        {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: register.block_index,
                reason: "lookup/register block metadata differs".to_string(),
            });
        }
        if position > 0
            && register_subclaims[position - 1].block.end_registers != register.start_registers
        {
            return Err(DirectChunkedError::DiscontinuousBlocks {
                previous_block: position - 1,
                next_block: position,
                reason: "register end boundary does not equal next start boundary".to_string(),
            });
        }
    }
    Ok((block_count, total_cycles))
}

fn setup_direct_register_public_params(
    circuit: &DirectRegisterStepCircuit,
) -> Result<DirectRegisterPublicParams, DirectChunkedError> {
    DirectRegisterPublicParams::setup(
        circuit,
        &*nova_snark::traits::snark::default_ck_hint::<NovaPrimaryEngine>(),
        &*nova_snark::traits::snark::default_ck_hint::<NovaSecondaryEngine>(),
    )
    .map_err(|error| register_stage_error("direct register Nova setup failed", error))
}

fn register_stage_error(context: &str, error: impl std::fmt::Debug) -> DirectChunkedError {
    DirectChunkedError::InvalidProofShape(format!("{context}: {error:?}"))
}

/// Produces one combined D2+D3 recursive proof from raw trace blocks.
pub fn prove_direct_register_stage<I>(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    blocks: I,
) -> Result<DirectRegisterStageProof, DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    let blocks = blocks.into_iter().collect::<Vec<_>>();
    if blocks.is_empty() {
        return Err(DirectChunkedError::EmptyTrace);
    }
    let lookup_subclaims = super::direct_lookup::prove_and_verify_native_lookup_blocks(
        preprocessing,
        capacity,
        blocks.clone(),
    )?;
    let mut register_prover_transcript = PoseidonTranscript::new(DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
    let mut register_verifier_transcript =
        PoseidonTranscript::new(DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
    let mut register_subclaims = Vec::with_capacity(blocks.len());
    for block in &blocks {
        let subclaim =
            prove_native_register_subclaim(block, capacity, &mut register_prover_transcript)?;
        verify_native_register_subclaim(&subclaim, &mut register_verifier_transcript)?;
        register_subclaims.push(subclaim);
    }
    let (block_count, total_cycles) = validate_combined_sequence(
        preprocessing,
        capacity,
        &lookup_subclaims,
        &register_subclaims,
    )?;
    let initial_registers = register_subclaims[0].block.start_registers;
    let final_registers = register_subclaims[block_count - 1].block.end_registers;
    let z0 = direct_register_initial_z(preprocessing, &initial_registers);
    let first_circuit = DirectRegisterStepCircuit::for_subclaims(
        lookup_subclaims[0].clone(),
        register_subclaims[0].clone(),
        block_count == 1,
    );
    let pp = setup_direct_register_public_params(&first_circuit)?;
    let mut recursive =
        DirectRegisterNovaSnark::new(&pp, &first_circuit, &z0).map_err(|error| {
            register_stage_error("direct register Nova initialization failed", error)
        })?;
    recursive
        .prove_step(&pp, &first_circuit)
        .map_err(|error| register_stage_error("direct register Nova first step failed", error))?;
    for position in 1..block_count {
        let circuit = DirectRegisterStepCircuit::for_subclaims(
            lookup_subclaims[position].clone(),
            register_subclaims[position].clone(),
            position + 1 == block_count,
        );
        recursive.prove_step(&pp, &circuit).map_err(|error| {
            register_stage_error(
                &format!("direct register Nova step {position} failed"),
                error,
            )
        })?;
    }
    let recursive_output = recursive.verify(&pp, block_count, &z0).map_err(|error| {
        register_stage_error("direct register Nova self-verification failed", error)
    })?;
    validate_recursive_output(
        &recursive_output,
        &z0,
        block_count,
        total_cycles,
        &final_registers,
        register_prover_transcript.state,
        register_prover_transcript.n_rounds,
    )?;
    let (pk, vk) = DirectRegisterCompressedSnark::setup(&pp)
        .map_err(|error| register_stage_error("direct register Spartan setup failed", error))?;
    let compressed = DirectRegisterCompressedSnark::prove(&pp, &pk, &recursive)
        .map_err(|error| register_stage_error("direct register Spartan proving failed", error))?;
    let compressed_output = compressed.verify(&vk, block_count, &z0).map_err(|error| {
        register_stage_error("direct register Spartan self-verification failed", error)
    })?;
    if compressed_output != recursive_output {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct register Nova and Spartan outputs differ".to_string(),
        ));
    }
    let relations = DirectRelation::ALL
        .into_iter()
        .map(|relation| {
            let state = match relation {
                DirectRelation::Lookup | DirectRelation::Register => DirectRelationState::Proven,
                DirectRelation::Pcs => DirectRelationState::Deferred,
                DirectRelation::Ram | DirectRelation::Cpu => DirectRelationState::Unsupported,
            };
            (relation, state)
        })
        .collect();
    let proof = DirectRegisterStageProof {
        statement: DirectRegisterStageStatement {
            program_digest: preprocessing.program_digest,
            lookup_table_commitment: preprocessing.lookup_table_commitment,
            block_count,
            total_cycles,
            initial_registers,
            final_registers,
            terminated: true,
            relations,
        },
        lookup_subclaims,
        register_subclaims,
        nova_recursive_snark: postcard::to_stdvec(&recursive)
            .map_err(|error| register_stage_error("register Nova serialization failed", error))?,
        spartan_proof: postcard::to_stdvec(&compressed).map_err(|error| {
            register_stage_error("register Spartan serialization failed", error)
        })?,
        initial_z: z0.iter().copied().map(nova_to_storage).collect(),
        final_z: recursive_output
            .iter()
            .copied()
            .map(nova_to_storage)
            .collect(),
    };
    verify_direct_register_stage(preprocessing, capacity, &proof)?;
    Ok(proof)
}

fn validate_recursive_output(
    output: &[NovaScalar],
    z0: &[NovaScalar],
    block_count: usize,
    total_cycles: usize,
    final_registers: &[u64; REGISTER_COUNT_USIZE],
    register_transcript_state: [u8; 32],
    register_transcript_round: u32,
) -> Result<(), DirectChunkedError> {
    let expected_register_transcript = nova_from_fr(&field_from_digest(&register_transcript_state))
        .map_err(|error| {
            register_stage_error("register transcript field conversion failed", error)
        })?;
    if output.len() != DIRECT_REGISTER_Z_ARITY
        || output[0] != NovaScalar::from(block_count as u64)
        || output[1] != NovaScalar::from(total_cycles as u64)
        || output[2] != NovaScalar::from(total_cycles as u64)
        || output[3] != z0[3]
        || output[4] != z0[4]
        || output[5] != z0[5]
        || output[6] != z0[6]
        || output[9] != NovaScalar::one()
        || output[REGISTER_TRANSCRIPT_STATE_SLOT] != expected_register_transcript
        || output[REGISTER_TRANSCRIPT_ROUND_SLOT]
            != NovaScalar::from(register_transcript_round as u64)
        || output[REGISTER_STATE_OFFSET..REGISTER_TRANSCRIPT_STATE_SLOT]
            != final_registers
                .iter()
                .copied()
                .map(NovaScalar::from)
                .collect::<Vec<_>>()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct register recursive output does not close the public statement".to_string(),
        ));
    }
    Ok(())
}

/// Verifies native audit subclaims and their combined Nova/Spartan proof.
pub fn verify_direct_register_stage(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    proof: &DirectRegisterStageProof,
) -> Result<(), DirectChunkedError> {
    let expected_relations = DirectRelation::ALL
        .into_iter()
        .map(|relation| {
            let state = match relation {
                DirectRelation::Lookup | DirectRelation::Register => DirectRelationState::Proven,
                DirectRelation::Pcs => DirectRelationState::Deferred,
                DirectRelation::Ram | DirectRelation::Cpu => DirectRelationState::Unsupported,
            };
            (relation, state)
        })
        .collect::<BTreeMap<_, _>>();
    if proof.statement.program_digest != preprocessing.program_digest
        || proof.statement.lookup_table_commitment != preprocessing.lookup_table_commitment
        || !proof.statement.terminated
        || proof.statement.relations != expected_relations
        || proof.statement.block_count != proof.lookup_subclaims.len()
        || proof.statement.block_count != proof.register_subclaims.len()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct register public statement is inconsistent".to_string(),
        ));
    }
    let (block_count, total_cycles) = validate_combined_sequence(
        preprocessing,
        capacity,
        &proof.lookup_subclaims,
        &proof.register_subclaims,
    )?;
    if block_count != proof.statement.block_count
        || total_cycles != proof.statement.total_cycles
        || proof.register_subclaims[0].block.start_registers != proof.statement.initial_registers
        || proof.register_subclaims[block_count - 1]
            .block
            .end_registers
            != proof.statement.final_registers
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct register statement counters or boundaries differ".to_string(),
        ));
    }
    let mut lookup_transcript = PoseidonTranscript::new(DIRECT_LOOKUP_TRANSCRIPT_DOMAIN);
    for subclaim in &proof.lookup_subclaims {
        verify_native_subclaim(subclaim, &mut lookup_transcript)?;
    }
    let mut register_transcript = PoseidonTranscript::new(DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
    for subclaim in &proof.register_subclaims {
        verify_native_register_subclaim(subclaim, &mut register_transcript)?;
    }
    let z0 = direct_register_initial_z(preprocessing, &proof.statement.initial_registers);
    if proof.initial_z != z0.iter().copied().map(nova_to_storage).collect::<Vec<_>>()
        || proof.initial_z.len() != DIRECT_REGISTER_Z_ARITY
        || proof.final_z.len() != DIRECT_REGISTER_Z_ARITY
        || proof.nova_recursive_snark.is_empty()
        || proof.spartan_proof.is_empty()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct register proof envelope is incomplete".to_string(),
        ));
    }
    let setup_circuit = DirectRegisterStepCircuit::for_subclaims(
        proof.lookup_subclaims[0].clone(),
        proof.register_subclaims[0].clone(),
        block_count == 1,
    );
    let pp = setup_direct_register_public_params(&setup_circuit)?;
    let recursive = postcard::from_bytes::<DirectRegisterNovaSnark>(&proof.nova_recursive_snark)
        .map_err(|error| register_stage_error("register Nova deserialization failed", error))?;
    let recursive_output = recursive
        .verify(&pp, block_count, &z0)
        .map_err(|error| register_stage_error("register Nova verification failed", error))?;
    let (_, vk) = DirectRegisterCompressedSnark::setup(&pp)
        .map_err(|error| register_stage_error("register Spartan setup failed", error))?;
    let compressed = postcard::from_bytes::<DirectRegisterCompressedSnark>(&proof.spartan_proof)
        .map_err(|error| register_stage_error("register Spartan deserialization failed", error))?;
    let compressed_output = compressed
        .verify(&vk, block_count, &z0)
        .map_err(|error| register_stage_error("register Spartan verification failed", error))?;
    validate_recursive_output(
        &recursive_output,
        &z0,
        block_count,
        total_cycles,
        &proof.statement.final_registers,
        register_transcript.state,
        register_transcript.n_rounds,
    )?;
    let expected_lookup_state = nova_from_fr(&field_from_digest(&lookup_transcript.state))
        .map_err(|error| {
            register_stage_error("lookup transcript field conversion failed", error)
        })?;
    if compressed_output != recursive_output
        || recursive_output[7] != expected_lookup_state
        || recursive_output[8] != NovaScalar::from(lookup_transcript.n_rounds as u64)
        || proof.final_z
            != recursive_output
                .iter()
                .copied()
                .map(nova_to_storage)
                .collect::<Vec<_>>()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "direct register recursive proof is not bound to retained subclaims".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_snark::frontend::test_cs::TestConstraintSystem;
    use tracer::instruction::{
        and::AND,
        format::format_r::{FormatR, RegisterStateFormatR},
        or::OR,
        RISCVCycle,
    };

    fn boundary(
        cycle: usize,
        registers: [u64; REGISTER_COUNT_USIZE],
        terminated: bool,
    ) -> MachineBoundaryState {
        MachineBoundaryState {
            global_cycle: cycle,
            emulator_trace_len: cycle,
            pc: 0x8000_0000 + (cycle as u64) * 4,
            registers: registers.map(|value| value as i64),
            terminated,
        }
    }

    fn and_cycle_with_registers(pre_rd: u64, x: u64, y: u64, rd: u8, rs1: u8, rs2: u8) -> Cycle {
        RISCVCycle::<AND> {
            instruction: AND {
                address: 0x8000_0000,
                operands: FormatR { rd, rs1, rs2 },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            register_state: RegisterStateFormatR {
                rd: (pre_rd, x & y),
                rs1: x,
                rs2: y,
            },
            ram_access: (),
        }
        .into()
    }

    fn and_cycle(pre_rd: u64, x: u64, y: u64) -> Cycle {
        and_cycle_with_registers(pre_rd, x, y, 3, 1, 2)
    }

    fn or_cycle_with_registers(pre_rd: u64, x: u64, y: u64, rd: u8, rs1: u8, rs2: u8) -> Cycle {
        RISCVCycle::<OR> {
            instruction: OR {
                address: 0x8000_0004,
                operands: FormatR { rd, rs1, rs2 },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            register_state: RegisterStateFormatR {
                rd: (pre_rd, x | y),
                rs1: x,
                rs2: y,
            },
            ram_access: (),
        }
        .into()
    }

    fn or_cycle(pre_rd: u64, x: u64, y: u64) -> Cycle {
        or_cycle_with_registers(pre_rd, x, y, 3, 1, 2)
    }

    fn two_blocks() -> Vec<TraceBlock> {
        let mut start = [0u64; REGISTER_COUNT_USIZE];
        start[1] = 0xaa;
        start[2] = 0x0f;
        let mut middle = start;
        middle[3] = 0x0a;
        let mut end = middle;
        end[3] = 0xaf;
        vec![
            TraceBlock {
                block_index: 0,
                global_cycle_start: 0,
                active_cycles: 1,
                target_size: 1,
                start_state: boundary(0, start, false),
                end_state: boundary(1, middle, false),
                cycles: vec![and_cycle(0, start[1], start[2])],
                ended_at_tick_boundary: true,
            },
            TraceBlock {
                block_index: 1,
                global_cycle_start: 1,
                active_cycles: 1,
                target_size: 1,
                start_state: boundary(1, middle, false),
                end_state: boundary(2, end, true),
                cycles: vec![or_cycle(middle[3], middle[1], middle[2])],
                ended_at_tick_boundary: true,
            },
        ]
    }

    fn one_block_two_cycles() -> TraceBlock {
        let mut start = [0u64; REGISTER_COUNT_USIZE];
        start[1] = 0xaa;
        start[2] = 0x0f;
        let mut end = start;
        end[3] = 0x0a;
        end[4] = 0x0f;
        TraceBlock {
            block_index: 0,
            global_cycle_start: 0,
            active_cycles: 2,
            target_size: 2,
            start_state: boundary(0, start, false),
            end_state: boundary(2, end, true),
            cycles: vec![
                and_cycle(0, start[1], start[2]),
                or_cycle_with_registers(0, 0x0a, start[2], 4, 3, 2),
            ],
            ended_at_tick_boundary: true,
        }
    }

    fn preprocessing() -> DirectChunkedPreprocessing {
        DirectChunkedPreprocessing::from_program_bytes(b"register-test", 64)
    }

    fn native_subclaims(
        blocks: &[TraceBlock],
    ) -> (Vec<DirectLookupSubclaim>, Vec<DirectRegisterSubclaim>) {
        let lookup = super::super::direct_lookup::prove_and_verify_native_lookup_blocks(
            &preprocessing(),
            2,
            blocks.to_vec(),
        )
        .unwrap();
        let mut prover_transcript = PoseidonTranscript::new(DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
        let mut verifier_transcript = PoseidonTranscript::new(DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
        let register = blocks
            .iter()
            .map(|block| {
                let subclaim =
                    prove_native_register_subclaim(block, 2, &mut prover_transcript).unwrap();
                verify_native_register_subclaim(&subclaim, &mut verifier_transcript).unwrap();
                subclaim
            })
            .collect();
        (lookup, register)
    }

    #[test]
    fn d3_native_register_sumcheck_round_trip() {
        let blocks = two_blocks();
        let (_, register) = native_subclaims(&blocks);
        assert_eq!(register.len(), 2);
        assert_eq!(register[0].block.end_registers[3], 0x0a);
        assert_eq!(register[1].block.end_registers[3], 0xaf);
        assert_eq!(register[0].proof.compressed_polys.len(), 8);
    }

    #[test]
    fn d3_combined_step_circuit_accepts_honest_block() {
        let blocks = two_blocks();
        let (lookup, register) = native_subclaims(&blocks[..1]);
        let circuit =
            DirectRegisterStepCircuit::for_subclaims(lookup[0].clone(), register[0].clone(), false);
        let z0 = direct_register_initial_z(&preprocessing(), &register[0].block.start_registers);
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z = z0
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("z {index}")), || Ok(value)).unwrap()
            })
            .collect::<Vec<_>>();
        let output = circuit.synthesize(&mut cs, &z).unwrap();
        assert_eq!(output.len(), DIRECT_REGISTER_Z_ARITY);
        assert!(
            cs.is_satisfied(),
            "combined circuit failed at {:?}",
            cs.which_is_unsatisfied()
        );
    }

    #[test]
    fn d3_intra_block_write_is_visible_to_later_read() {
        let block = one_block_two_cycles();
        let (lookup, register) = native_subclaims(std::slice::from_ref(&block));
        assert_eq!(register[0].block.cycles[1].rs1.register, 3);
        assert_eq!(register[0].block.cycles[1].rs1.value, 0x0a);
        assert_eq!(register[0].block.end_registers[4], 0x0f);

        let circuit =
            DirectRegisterStepCircuit::for_subclaims(lookup[0].clone(), register[0].clone(), true);
        let z0 = direct_register_initial_z(&preprocessing(), &register[0].block.start_registers);
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z = z0
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("z {index}")), || Ok(value)).unwrap()
            })
            .collect::<Vec<_>>();
        circuit.synthesize(&mut cs, &z).unwrap();
        assert!(
            cs.is_satisfied(),
            "intra-block dependency failed at {:?}",
            cs.which_is_unsatisfied()
        );
    }

    #[test]
    fn d3_rejects_forged_register_read_or_boundary() {
        let mut blocks = two_blocks();
        if let Cycle::AND(cycle) = &mut blocks[0].cycles[0] {
            cycle.register_state.rs1 ^= 1;
        } else {
            unreachable!();
        }
        assert!(DirectRegisterBlockWitness::from_trace_block(&blocks[0], 2).is_err());

        let mut blocks = two_blocks();
        blocks[1].start_state.registers[3] ^= 1;
        assert!(prove_direct_register_stage(&preprocessing(), 2, blocks).is_err());

        let mut blocks = two_blocks();
        blocks[0].ended_at_tick_boundary = false;
        assert!(prove_direct_register_stage(&preprocessing(), 2, blocks).is_err());
    }

    #[test]
    fn d3_native_verifier_rejects_sumcheck_or_opening_tampering() {
        let blocks = two_blocks();
        let (_, register) = native_subclaims(&blocks[..1]);

        let mut bad_sumcheck = register[0].clone();
        bad_sumcheck.proof.compressed_polys[0].coeffs_except_linear_term[0] += Fr::from(1u64);
        let mut transcript = PoseidonTranscript::new(DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
        assert!(verify_native_register_subclaim(&bad_sumcheck, &mut transcript).is_err());

        let mut bad_opening = register[0].clone();
        bad_opening.opening_claims[0] += Fr::from(1u64);
        let mut transcript = PoseidonTranscript::new(DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
        assert!(verify_native_register_subclaim(&bad_opening, &mut transcript).is_err());
    }

    #[test]
    fn d3_combined_step_shape_is_fixed_across_lookup_tables() {
        let blocks = two_blocks();
        let (lookup, register) = native_subclaims(&blocks);
        let z0 = direct_register_initial_z(&preprocessing(), &register[0].block.start_registers);
        let mut counts = Vec::new();
        for index in 0..2 {
            let circuit = DirectRegisterStepCircuit::for_subclaims(
                lookup[index].clone(),
                register[index].clone(),
                index == 1,
            );
            let mut cs = TestConstraintSystem::<NovaScalar>::new();
            let z = z0
                .iter()
                .copied()
                .enumerate()
                .map(|(slot, value)| {
                    AllocatedNum::alloc(cs.namespace(|| format!("z {slot}")), || Ok(value)).unwrap()
                })
                .collect::<Vec<_>>();
            circuit.synthesize(&mut cs, &z).unwrap();
            counts.push(cs.num_constraints());
        }
        let two_cycle = one_block_two_cycles();
        let (two_cycle_lookup, two_cycle_register) =
            native_subclaims(std::slice::from_ref(&two_cycle));
        let circuit = DirectRegisterStepCircuit::for_subclaims(
            two_cycle_lookup[0].clone(),
            two_cycle_register[0].clone(),
            true,
        );
        let mut cs = TestConstraintSystem::<NovaScalar>::new();
        let z = z0
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("z {slot}")), || Ok(value)).unwrap()
            })
            .collect::<Vec<_>>();
        circuit.synthesize(&mut cs, &z).unwrap();
        counts.push(cs.num_constraints());
        assert!(counts.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn d3_nova_folds_two_blocks_and_spartan_closes_combined_stage() {
        let preprocessing = preprocessing();
        let prover = super::super::DirectChunkedProver::new(
            preprocessing.clone(),
            super::super::DirectChunkedConfig {
                block_capacity: 2,
                compress_final_spartan: true,
            },
        )
        .unwrap();
        let proof = prover.prove_register_stage(two_blocks()).unwrap();
        assert_eq!(proof.statement.block_count, 2);
        assert_eq!(proof.statement.total_cycles, 2);
        assert_eq!(
            proof.statement.relations.get(&DirectRelation::Lookup),
            Some(&DirectRelationState::Proven)
        );
        assert_eq!(
            proof.statement.relations.get(&DirectRelation::Register),
            Some(&DirectRelationState::Proven)
        );
        prover.verify_register_stage(&proof).unwrap();

        let mut replacement_blocks = two_blocks();
        replacement_blocks[0].cycles = vec![and_cycle_with_registers(0, 0x0f, 0xaa, 3, 2, 1)];
        replacement_blocks[1].cycles = vec![or_cycle_with_registers(0x0a, 0x0f, 0xaa, 3, 2, 1)];
        let mut replacement_transcript = PoseidonTranscript::new(DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
        let replacement = replacement_blocks
            .iter()
            .map(|block| {
                prove_native_register_subclaim(block, 2, &mut replacement_transcript).unwrap()
            })
            .collect::<Vec<_>>();
        let mut bad_replacement = proof.clone();
        bad_replacement.register_subclaims = replacement;
        assert!(verify_direct_register_stage(&preprocessing, 2, &bad_replacement).is_err());

        let mut bad = proof.clone();
        bad.register_subclaims[0].block.cycles[0].rs1.value ^= 1;
        assert!(verify_direct_register_stage(&preprocessing, 2, &bad).is_err());

        let mut bad = proof;
        bad.spartan_proof[0] ^= 1;
        assert!(verify_direct_register_stage(&preprocessing, 2, &bad).is_err());
    }
}
