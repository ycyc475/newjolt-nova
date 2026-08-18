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

use super::super::{
    direct_streaming::DirectTraceSpool, DirectChunkedError, DirectChunkedPreprocessing,
};
use super::{
    close_block_jolt_deferred_pcs_from_spool, prove_pcs_bound_block_jolt_transition,
    BlockJoltDeferredPcsProof, BlockJoltDeferredPcsSpool, BlockJoltHostConfig, BlockJoltNovaFolder,
    BlockJoltNovaFoldingProof, BlockJoltNovaSetup, BlockJoltProver,
};

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
    pub deferred_pcs_close_micros: u128,
    pub peak_observed_physical_memory_bytes: Option<usize>,
}

impl BlockJoltStreamingMetrics {
    pub fn residency_is_bounded(&self, cycle_capacity: usize) -> bool {
        self.trace_passes == 2
            && self.max_resident_trace_blocks <= 2
            && self.max_resident_trace_cycles <= cycle_capacity.saturating_mul(3)
            && self.max_resident_pcs_witnesses <= 1
    }
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
    let prove_start = Instant::now();

    let mut current = replay.read_next()?.ok_or(DirectChunkedError::EmptyTrace)?;
    let mut next = replay.read_next()?;
    loop {
        let lookahead = next.as_ref().and_then(|block| block.cycles.first());
        let bound = prove_pcs_bound_block_jolt_transition(&mut prover, &current, lookahead)?;
        if folder.is_none() {
            folder = Some(match setup {
                Some(setup) => BlockJoltNovaFolder::new_with_setup(setup, bound.transition())?,
                None => BlockJoltNovaFolder::new(config, bound.transition())?,
            });
        }
        folder
            .as_mut()
            .expect("D18 folder was initialized above")
            .fold_transition(bound.transition())?;
        pcs_spool.push(bound)?;

        match next.take() {
            Some(next_block) => {
                current = next_block;
                next = replay.read_next()?;
            }
            None => break,
        }
    }
    prover.finish()?;
    let folding = folder.ok_or(DirectChunkedError::EmptyTrace)?.finish()?;
    let block_prove_and_fold_micros = prove_start.elapsed().as_micros();

    let pcs_close_start = Instant::now();
    let deferred_pcs = close_block_jolt_deferred_pcs_from_spool(&folding, &pcs_spool)?;
    let deferred_pcs_close_micros = pcs_close_start.elapsed().as_micros();
    let proof = BlockJoltStreamingProof {
        folding,
        deferred_pcs,
    };
    proof.verify()?;

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
        deferred_pcs_close_micros,
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
