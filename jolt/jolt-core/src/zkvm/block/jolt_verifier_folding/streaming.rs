//! D18 bounded-memory trace-to-fold pipeline.
//!
//! The source trace is captured once and replayed once with one-block CPU
//! lookahead. Each compact block proof is folded into Nova immediately, while
//! its prover-only Dory witness is moved to an integrity-protected temporary
//! spool. Once Nova exposes the final deferred checkpoint, the spool is replayed
//! one record at a time to close every accepted endpoint opening.

use std::time::Instant;

use serde::{Deserialize, Serialize};
use tracer::TraceBlock;

use crate::poly::multilinear_polynomial::PolynomialEvaluation;

use super::super::{
    direct_streaming::DirectTraceSpool, DirectChunkedError, DirectChunkedPreprocessing,
};
use super::{
    close_block_jolt_deferred_pcs_from_spool, prove_pcs_bound_block_jolt_transition,
    ram::{block_ram_polynomials, new_block_ram_transcript, prove_block_ram},
    BlockJoltDeferredPcsProof, BlockJoltDeferredPcsSpool, BlockJoltHostConfig, BlockJoltNovaFolder,
    BlockJoltNovaFoldingProof, BlockJoltNovaSetup, BlockJoltPcsProvingMetrics, BlockJoltProver,
    BlockJoltRelationProvingMetrics, FieldElement,
};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockJoltTraceAuditMetrics {
    pub source_block_count: usize,
    pub block_count: usize,
    pub total_active_cycles: usize,
    pub max_resident_trace_blocks: usize,
    pub max_resident_trace_cycles: usize,
    pub spooled_trace_bytes: usize,
    pub max_serialized_trace_block_bytes: usize,
    pub touched_ram_addresses: usize,
    pub derived_ram_k: usize,
    pub trace_digest: [u8; 32],
    pub trace_capture_micros: u128,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockJoltSetupMetrics {
    pub trace: BlockJoltTraceAuditMetrics,
    pub template_transition_micros: u128,
    pub nova_spartan_setup_micros: u128,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockJoltRamAuditMetrics {
    pub trace: BlockJoltTraceAuditMetrics,
    pub audited_blocks: usize,
    pub audited_polynomials: usize,
    pub audit_micros: u128,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockJoltStreamingMetrics {
    pub trace_passes: usize,
    pub source_block_count: usize,
    pub block_count: usize,
    pub total_active_cycles: usize,
    pub max_resident_trace_blocks: usize,
    pub max_resident_trace_cycles: usize,
    pub spooled_trace_bytes: usize,
    pub max_serialized_trace_block_bytes: usize,
    pub spooled_pcs_witness_bytes: usize,
    pub max_serialized_pcs_witness_bytes: usize,
    pub max_resident_pcs_witnesses: usize,
    pub trace_capture_micros: u128,
    pub block_prove_and_fold_micros: u128,
    pub relation_proving: BlockJoltRelationProvingMetrics,
    pub pcs_proving: BlockJoltPcsProvingMetrics,
    pub nova_fold_micros: u128,
    pub pcs_spool_write_micros: u128,
    pub deferred_pcs_close_micros: u128,
    pub internal_streaming_verify_micros: u128,
    pub peak_observed_physical_memory_bytes: Option<usize>,
}

impl BlockJoltStreamingMetrics {
    pub fn residency_is_bounded(&self, cycle_capacity: usize) -> bool {
        self.trace_passes == 2
            && self.max_resident_trace_blocks <= 2
            && self.max_resident_trace_cycles <= cycle_capacity.saturating_mul(4)
            && self.max_resident_pcs_witnesses <= 1
    }
}

fn validate_cycle_capacity(cycle_capacity: usize) -> Result<(), DirectChunkedError> {
    if cycle_capacity < 2 || !cycle_capacity.is_power_of_two() {
        return Err(DirectChunkedError::InvalidConfiguration(
            "block-Jolt cycle capacity must be a power of two of at least two".to_string(),
        ));
    }
    Ok(())
}

fn derive_minimum_ram_k(
    preprocessing: &DirectChunkedPreprocessing,
    addresses: impl IntoIterator<Item = u64>,
) -> Result<usize, DirectChunkedError> {
    let mut maximum = None;
    for address in addresses {
        let remapped = preprocessing
            .memory_layout
            .remapped_word_address(address)
            .map_err(|error| {
                DirectChunkedError::InvalidConfiguration(format!(
                    "cannot remap touched RAM address {address:#x}: {error}"
                ))
            })?;
        maximum = Some(maximum.map_or(remapped, |current: u64| current.max(remapped)));
    }
    let required = maximum.unwrap_or(0).checked_add(1).ok_or_else(|| {
        DirectChunkedError::InvalidConfiguration("derived RAM size overflowed".to_string())
    })?;
    let required = usize::try_from(required).map_err(|error| {
        DirectChunkedError::InvalidConfiguration(format!(
            "derived RAM size does not fit usize: {error}"
        ))
    })?;
    required.checked_next_power_of_two().ok_or_else(|| {
        DirectChunkedError::InvalidConfiguration("derived RAM K overflowed".to_string())
    })
}

fn trace_audit_metrics(
    spool: &DirectTraceSpool,
    derived_ram_k: usize,
) -> BlockJoltTraceAuditMetrics {
    BlockJoltTraceAuditMetrics {
        source_block_count: spool.metrics.source_block_count,
        block_count: spool.metrics.block_count,
        total_active_cycles: spool.metrics.total_active_cycles,
        max_resident_trace_blocks: spool.metrics.max_resident_trace_blocks,
        max_resident_trace_cycles: spool.metrics.max_resident_trace_cycles,
        spooled_trace_bytes: spool.metrics.spooled_trace_bytes,
        max_serialized_trace_block_bytes: spool.metrics.max_serialized_block_bytes,
        touched_ram_addresses: spool.initial_ram.len(),
        derived_ram_k,
        trace_digest: spool.audit.trace_digest,
        trace_capture_micros: spool.metrics.capture_trace_micros,
    }
}

/// Audits a real lazy trace and derives the smallest power-of-two RAM domain
/// covering every touched verifier-layout address, without running any
/// cryptographic setup or proving.
pub fn audit_block_jolt_trace<I>(
    preprocessing: &DirectChunkedPreprocessing,
    cycle_capacity: usize,
    blocks: I,
) -> Result<BlockJoltTraceAuditMetrics, DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    validate_cycle_capacity(cycle_capacity)?;
    let spool = DirectTraceSpool::capture_production(preprocessing, cycle_capacity, blocks)?;
    let ram_k = derive_minimum_ram_k(preprocessing, spool.initial_ram.keys().copied())?;
    Ok(trace_audit_metrics(&spool, ram_k))
}

/// Replays the production trace and checks every standalone RAM sumcheck
/// endpoint against the polynomial rebuilt from the same block and initial
/// memory registry. It deliberately skips Nova/Dory setup so real-workload
/// RAM mismatches can be localized cheaply.
pub fn audit_block_jolt_ram_endpoints<I>(
    preprocessing: &DirectChunkedPreprocessing,
    cycle_capacity: usize,
    blocks: I,
) -> Result<BlockJoltRamAuditMetrics, DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    validate_cycle_capacity(cycle_capacity)?;
    let mut spool = DirectTraceSpool::capture_production(preprocessing, cycle_capacity, blocks)?;
    let ram_k = derive_minimum_ram_k(preprocessing, spool.initial_ram.keys().copied())?;
    let trace = trace_audit_metrics(&spool, ram_k);
    let mut memory = std::mem::take(&mut spool.initial_ram);
    let mut replay = spool.replay()?;
    let started = Instant::now();
    let mut audited_blocks = 0usize;
    let mut audited_polynomials = 0usize;

    while let Some(block) = replay.read_next()? {
        let before = memory.clone();
        let mut transcript = new_block_ram_transcript();
        let (proof, after, _, _) = prove_block_ram(
            preprocessing,
            &block,
            cycle_capacity,
            ram_k,
            &before,
            &mut transcript,
        )?;
        let polynomials =
            block_ram_polynomials(preprocessing, &block, cycle_capacity, ram_k, &before)?;
        if polynomials.len() != proof.relation_proof.opening_claims.len() {
            return Err(DirectChunkedError::InvalidProofShape(format!(
                "D20 RAM audit block {} has different polynomial and claim counts",
                block.block_index
            )));
        }
        for (index, (polynomial, claim)) in polynomials
            .iter()
            .zip(&proof.relation_proof.opening_claims)
            .enumerate()
        {
            let point = claim
                .opening_point
                .iter()
                .map(|coordinate| coordinate.to_fr())
                .collect::<Vec<_>>();
            let evaluated = (point.len() == polynomial.get_num_vars())
                .then(|| PolynomialEvaluation::evaluate(polynomial, &point));
            if evaluated != Some(claim.claimed_value.to_fr()) {
                return Err(DirectChunkedError::InvalidProofShape(format!(
                    "D20 RAM audit block {} polynomial {index} mismatch (evaluated {:?}, claimed {:?})",
                    block.block_index,
                    evaluated.map(|value| FieldElement::from_fr(&value).0),
                    claim.claimed_value.0,
                )));
            }
        }
        audited_polynomials += polynomials.len();
        audited_blocks += 1;
        memory = after;
    }

    if audited_blocks != trace.block_count {
        return Err(DirectChunkedError::InvalidProofShape(
            "D20 RAM audit did not replay every trace block".to_string(),
        ));
    }
    Ok(BlockJoltRamAuditMetrics {
        trace,
        audited_blocks,
        audited_polynomials,
        audit_micros: started.elapsed().as_micros(),
    })
}

/// Creates the reusable D19 setup from a bounded-memory audit of a real trace.
/// The trace is consumed only to derive the RAM domain and one valid template
/// transition; callers should create a fresh lazy iterator for production
/// proving after this one-time setup step.
pub fn prepare_block_jolt_nova_setup<I>(
    preprocessing: &DirectChunkedPreprocessing,
    cycle_capacity: usize,
    blocks: I,
) -> Result<(BlockJoltNovaSetup, BlockJoltSetupMetrics), DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    validate_cycle_capacity(cycle_capacity)?;
    let mut spool = DirectTraceSpool::capture_production(preprocessing, cycle_capacity, blocks)?;
    let ram_k = derive_minimum_ram_k(preprocessing, spool.initial_ram.keys().copied())?;
    let trace = trace_audit_metrics(&spool, ram_k);
    let config = BlockJoltHostConfig {
        cycle_capacity,
        ram_k,
    };
    config.validate()?;
    let mut replay = spool.replay()?;
    let initial_ram = std::mem::take(&mut spool.initial_ram);
    let current = replay.read_next()?.ok_or(DirectChunkedError::EmptyTrace)?;
    let next = replay.read_next()?;
    let lookahead = next.as_ref().and_then(|block| block.cycles.first());
    let mut prover = BlockJoltProver::new(preprocessing.clone(), config, initial_ram)?;
    let template_started = Instant::now();
    let template = prover.prove_block(&current, lookahead)?;
    let template_transition_micros = template_started.elapsed().as_micros();
    let setup_started = Instant::now();
    let setup = BlockJoltNovaSetup::new(config, &template)?;
    let nova_spartan_setup_micros = setup_started.elapsed().as_micros();
    Ok((
        setup,
        BlockJoltSetupMetrics {
            trace,
            template_transition_micros,
            nova_spartan_setup_micros,
        },
    ))
}

/// D18 proof envelope. Both members are verifier artifacts; the temporary
/// trace and PCS spools have already been dropped when this value is returned.
pub struct BlockJoltStreamingProof {
    folding: BlockJoltNovaFoldingProof,
    deferred_pcs: BlockJoltDeferredPcsProof,
}

impl BlockJoltStreamingProof {
    pub fn block_count(&self) -> usize {
        self.folding.block_count()
    }

    pub fn folding(&self) -> &BlockJoltNovaFoldingProof {
        &self.folding
    }

    pub fn deferred_pcs(&self) -> &BlockJoltDeferredPcsProof {
        &self.deferred_pcs
    }

    pub fn verify(&self) -> Result<(), DirectChunkedError> {
        self.folding.verify()?;
        self.deferred_pcs.verify(&self.folding)
    }
}

/// Executes the production direct block-Jolt path without retaining the full
/// trace, prior compact transitions, or all endpoint polynomials in memory.
pub fn prove_block_jolt_streaming<I>(
    preprocessing: &DirectChunkedPreprocessing,
    config: BlockJoltHostConfig,
    blocks: I,
) -> Result<(BlockJoltStreamingProof, BlockJoltStreamingMetrics), DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    prove_block_jolt_streaming_internal(preprocessing, config, None, blocks)
}

/// D19 production entry point. Nova folding reuses the supplied setup and the
/// resulting recursive artifact is cryptographically labeled with its setup
/// identifier so it can be Spartan-compressed without per-proof key setup.
pub fn prove_block_jolt_streaming_with_setup<I>(
    preprocessing: &DirectChunkedPreprocessing,
    setup: &BlockJoltNovaSetup,
    blocks: I,
) -> Result<(BlockJoltStreamingProof, BlockJoltStreamingMetrics), DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    prove_block_jolt_streaming_internal(preprocessing, setup.config(), Some(setup), blocks)
}

fn prove_block_jolt_streaming_internal<I>(
    preprocessing: &DirectChunkedPreprocessing,
    config: BlockJoltHostConfig,
    setup: Option<&BlockJoltNovaSetup>,
    blocks: I,
) -> Result<(BlockJoltStreamingProof, BlockJoltStreamingMetrics), DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    config.validate()?;
    let capture_start = Instant::now();
    let mut trace_spool =
        DirectTraceSpool::capture_production(preprocessing, config.cycle_capacity, blocks)?;
    let capture_micros = capture_start.elapsed().as_micros();
    let mut replay = trace_spool.replay()?;
    let initial_ram = std::mem::take(&mut trace_spool.initial_ram);
    let trace_metrics = std::mem::take(&mut trace_spool.metrics);

    let mut prover = BlockJoltProver::new(preprocessing.clone(), config, initial_ram)?;
    let mut pcs_spool = BlockJoltDeferredPcsSpool::new()?;
    let mut folder: Option<BlockJoltNovaFolder> = None;
    let mut pcs_proving = BlockJoltPcsProvingMetrics::default();
    let mut nova_fold_micros = 0u128;
    let mut pcs_spool_write_micros = 0u128;
    let prove_start = Instant::now();

    let mut current = replay.read_next()?.ok_or(DirectChunkedError::EmptyTrace)?;
    let mut next = replay.read_next()?;
    loop {
        let lookahead = next.as_ref().and_then(|block| block.cycles.first());
        let bound = prove_pcs_bound_block_jolt_transition(&mut prover, &current, lookahead)?;
        pcs_proving.accumulate(bound.metrics());
        if folder.is_none() {
            folder = Some(match setup {
                Some(setup) => BlockJoltNovaFolder::new_with_setup(setup, bound.transition())?,
                None => BlockJoltNovaFolder::new(config, bound.transition())?,
            });
        }
        let fold_started = Instant::now();
        folder
            .as_mut()
            .expect("D18 folder was initialized above")
            .fold_transition(bound.transition())?;
        nova_fold_micros += fold_started.elapsed().as_micros();
        let spool_started = Instant::now();
        pcs_spool.push(bound)?;
        pcs_spool_write_micros += spool_started.elapsed().as_micros();

        match next.take() {
            Some(next_block) => {
                current = next_block;
                next = replay.read_next()?;
            }
            None => break,
        }
    }
    prover.finish()?;
    let relation_proving = prover.relation_metrics().clone();
    let folding = folder.ok_or(DirectChunkedError::EmptyTrace)?.finish()?;
    let block_prove_and_fold_micros = prove_start.elapsed().as_micros();

    let pcs_close_start = Instant::now();
    let deferred_pcs = close_block_jolt_deferred_pcs_from_spool(&folding, &pcs_spool)?;
    let deferred_pcs_close_micros = pcs_close_start.elapsed().as_micros();
    let proof = BlockJoltStreamingProof {
        folding,
        deferred_pcs,
    };
    let verify_started = Instant::now();
    proof.verify()?;
    let internal_streaming_verify_micros = verify_started.elapsed().as_micros();

    let metrics = BlockJoltStreamingMetrics {
        trace_passes: trace_metrics.trace_passes,
        source_block_count: trace_metrics.source_block_count,
        block_count: proof.block_count(),
        total_active_cycles: trace_metrics.total_active_cycles,
        max_resident_trace_blocks: trace_metrics.max_resident_trace_blocks,
        max_resident_trace_cycles: trace_metrics.max_resident_trace_cycles,
        spooled_trace_bytes: trace_metrics.spooled_trace_bytes,
        max_serialized_trace_block_bytes: trace_metrics.max_serialized_block_bytes,
        spooled_pcs_witness_bytes: pcs_spool.bytes_written(),
        max_serialized_pcs_witness_bytes: pcs_spool.max_record_bytes(),
        max_resident_pcs_witnesses: usize::from(pcs_spool.block_count() > 0),
        trace_capture_micros: capture_micros,
        block_prove_and_fold_micros,
        relation_proving,
        pcs_proving,
        nova_fold_micros,
        pcs_spool_write_micros,
        deferred_pcs_close_micros,
        internal_streaming_verify_micros,
        peak_observed_physical_memory_bytes: memory_stats::memory_stats()
            .map(|stats| stats.physical_mem)
            .into_iter()
            .chain(trace_metrics.peak_observed_physical_memory_bytes)
            .max(),
    };
    if metrics.block_count != trace_metrics.block_count
        || metrics.block_count != pcs_spool.block_count()
        || !metrics.residency_is_bounded(config.cycle_capacity)
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D18 streaming residency or block-count invariant failed".to_string(),
        ));
    }
    Ok((proof, metrics))
}

#[cfg(test)]
mod tests {
    use common::constants::REGISTER_COUNT;
    use tracer::{
        instruction::{
            and::AND,
            format::format_r::{FormatR, RegisterStateFormatR},
            Cycle, RISCVCycle,
        },
        MachineBoundaryState,
    };

    use super::*;

    fn two_terminal_blocks() -> (DirectChunkedPreprocessing, Vec<TraceBlock>) {
        let materialized_instruction: Cycle = RISCVCycle::<AND> {
            instruction: AND {
                address: 0,
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
                rd: (0, 0),
                rs1: 0,
                rs2: 0,
            },
            ram_access: (),
        }
        .into();
        let preprocessing = DirectChunkedPreprocessing::from_trace_cycles(
            b"d18-bounded-spool",
            8,
            &[materialized_instruction],
        )
        .unwrap();
        let registers = [0i64; REGISTER_COUNT as usize];
        let start = MachineBoundaryState {
            global_cycle: 0,
            emulator_trace_len: 0,
            pc: 0,
            registers,
            terminated: false,
        };
        let middle = MachineBoundaryState {
            global_cycle: 1,
            emulator_trace_len: 1,
            pc: 0,
            registers,
            terminated: false,
        };
        let end = MachineBoundaryState {
            global_cycle: 2,
            emulator_trace_len: 2,
            pc: 0,
            registers,
            terminated: true,
        };
        let blocks = vec![
            TraceBlock {
                block_index: 0,
                global_cycle_start: 0,
                active_cycles: 1,
                target_size: 2,
                start_state: start,
                end_state: middle.clone(),
                cycles: vec![Cycle::NoOp],
                ended_at_tick_boundary: true,
            },
            TraceBlock {
                block_index: 1,
                global_cycle_start: 1,
                active_cycles: 1,
                target_size: 2,
                start_state: middle,
                end_state: end,
                cycles: vec![Cycle::NoOp],
                ended_at_tick_boundary: true,
            },
        ];
        (preprocessing, blocks)
    }

    #[test]
    fn d18_streams_trace_folding_and_pcs_witnesses_with_bounded_residency() {
        let (preprocessing, blocks) = two_terminal_blocks();
        let config = BlockJoltHostConfig {
            cycle_capacity: 2,
            ram_k: 2,
        };
        let (proof, metrics) = prove_block_jolt_streaming(&preprocessing, config, blocks).unwrap();
        assert_eq!(proof.block_count(), 2);
        assert_eq!(proof.deferred_pcs().block_count(), 2);
        assert!(proof.deferred_pcs().opening_group_count() > 0);
        assert!(metrics.residency_is_bounded(config.cycle_capacity));
        assert_eq!(metrics.max_resident_pcs_witnesses, 1);
        assert!(metrics.spooled_trace_bytes > 0);
        assert!(metrics.spooled_pcs_witness_bytes > 0);
        proof.verify().unwrap();
    }

    #[test]
    fn d18_rejects_a_nonterminal_stream_before_proving() {
        let (preprocessing, mut blocks) = two_terminal_blocks();
        blocks.last_mut().unwrap().end_state.terminated = false;
        let config = BlockJoltHostConfig {
            cycle_capacity: 2,
            ram_k: 2,
        };
        assert!(prove_block_jolt_streaming(&preprocessing, config, blocks).is_err());
    }
}
