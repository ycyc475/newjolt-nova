//! Bounded-residency trace capture and replay for the direct production path.
//!
//! The RAM address registry must be fixed before authenticated RAM paths can be
//! generated. A single-use tracer therefore cannot be consumed in one pass.
//! D8 solves that dependency without retaining the whole trace: pass one
//! validates and spools each block to a temporary file, while pass two keeps at
//! most the current block and one CPU-lookahead block resident.

use std::{
    collections::BTreeMap,
    io::{BufReader, Read, Write},
    mem::size_of,
    time::Instant,
};

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use tempfile::NamedTempFile;
use tracer::{
    instruction::{Cycle, RAMAccess},
    MachineBoundaryState, TraceBlock,
};

use super::{
    direct::{hash_block_metadata, validate_block},
    direct_ram::validate_ram_address,
    DirectChunkedError, DirectChunkedPreprocessing, DirectTraceAudit,
    DIRECT_CHUNKED_PROTOCOL_VERSION,
};

#[derive(Serialize, Deserialize)]
struct SpoolBoundaryState {
    global_cycle: usize,
    emulator_trace_len: usize,
    pc: u64,
    registers: Vec<i64>,
    terminated: bool,
}

impl From<&MachineBoundaryState> for SpoolBoundaryState {
    fn from(state: &MachineBoundaryState) -> Self {
        Self {
            global_cycle: state.global_cycle,
            emulator_trace_len: state.emulator_trace_len,
            pc: state.pc,
            registers: state.registers.to_vec(),
            terminated: state.terminated,
        }
    }
}

impl TryFrom<SpoolBoundaryState> for MachineBoundaryState {
    type Error = DirectChunkedError;

    fn try_from(state: SpoolBoundaryState) -> Result<Self, Self::Error> {
        let registers = state.registers.try_into().map_err(|registers: Vec<i64>| {
            DirectChunkedError::InvalidProofShape(format!(
                "D8 trace spool boundary contains {} registers instead of {}",
                registers.len(),
                common::constants::REGISTER_COUNT
            ))
        })?;
        Ok(Self {
            global_cycle: state.global_cycle,
            emulator_trace_len: state.emulator_trace_len,
            pc: state.pc,
            registers,
            terminated: state.terminated,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct SpoolTraceBlock {
    block_index: usize,
    global_cycle_start: usize,
    active_cycles: usize,
    target_size: usize,
    start_state: SpoolBoundaryState,
    end_state: SpoolBoundaryState,
    cycles: Vec<Cycle>,
    ended_at_tick_boundary: bool,
}

impl From<&TraceBlock> for SpoolTraceBlock {
    fn from(block: &TraceBlock) -> Self {
        Self {
            block_index: block.block_index,
            global_cycle_start: block.global_cycle_start,
            active_cycles: block.active_cycles,
            target_size: block.target_size,
            start_state: (&block.start_state).into(),
            end_state: (&block.end_state).into(),
            cycles: block.cycles.clone(),
            ended_at_tick_boundary: block.ended_at_tick_boundary,
        }
    }
}

impl TryFrom<SpoolTraceBlock> for TraceBlock {
    type Error = DirectChunkedError;

    fn try_from(block: SpoolTraceBlock) -> Result<Self, Self::Error> {
        Ok(Self {
            block_index: block.block_index,
            global_cycle_start: block.global_cycle_start,
            active_cycles: block.active_cycles,
            target_size: block.target_size,
            start_state: block.start_state.try_into()?,
            end_state: block.end_state.try_into()?,
            cycles: block.cycles,
            ended_at_tick_boundary: block.ended_at_tick_boundary,
        })
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DirectProvingMetrics {
    pub source_block_count: usize,
    pub block_count: usize,
    pub total_active_cycles: usize,
    pub trace_passes: usize,
    pub max_source_trace_cycles: usize,
    pub max_resident_trace_blocks: usize,
    pub max_resident_trace_cycles: usize,
    pub estimated_peak_resident_trace_bytes: usize,
    pub spooled_trace_bytes: usize,
    pub max_serialized_block_bytes: usize,
    pub peak_tracked_ram_addresses: usize,
    pub initial_ram_registry_bytes: usize,
    pub retained_relation_subclaims: usize,
    pub pcs_polynomial_coefficients: usize,
    pub pcs_polynomial_bytes: usize,
    pub nova_recursive_debug_bytes: usize,
    pub spartan_proof_bytes: usize,
    pub capture_trace_micros: u128,
    pub relation_preparation_micros: u128,
    pub pcs_commit_micros: u128,
    pub nova_fold_micros: u128,
    pub spartan_compress_micros: u128,
    pub pcs_open_micros: u128,
    pub self_verify_micros: u128,
    pub start_physical_memory_bytes: Option<usize>,
    pub after_capture_physical_memory_bytes: Option<usize>,
    pub after_relations_physical_memory_bytes: Option<usize>,
    pub after_pcs_physical_memory_bytes: Option<usize>,
    pub peak_observed_physical_memory_bytes: Option<usize>,
}

impl DirectProvingMetrics {
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn trace_residency_is_bounded(&self, block_capacity: usize) -> bool {
        self.trace_passes == 2
            && self.max_resident_trace_blocks <= 2
            && self.max_source_trace_cycles <= block_capacity.saturating_mul(2)
            && self.max_resident_trace_cycles <= block_capacity.saturating_mul(3)
    }

    pub fn peak_physical_memory_delta_bytes(&self) -> Option<isize> {
        Some(
            self.peak_observed_physical_memory_bytes? as isize
                - self.start_physical_memory_bytes? as isize,
        )
    }

    pub(super) fn observe_memory(&mut self) -> Option<usize> {
        let current = memory_stats::memory_stats().map(|stats| stats.physical_mem);
        if let Some(current) = current {
            self.peak_observed_physical_memory_bytes = Some(
                self.peak_observed_physical_memory_bytes
                    .map_or(current, |peak| peak.max(current)),
            );
        }
        current
    }

    pub(super) fn observe_resident_blocks(
        &mut self,
        current: &TraceBlock,
        next: Option<&TraceBlock>,
    ) {
        let resident_blocks = 1 + usize::from(next.is_some());
        let resident_cycles = current.cycles.len() + next.map_or(0, |block| block.cycles.len());
        self.max_resident_trace_blocks = self.max_resident_trace_blocks.max(resident_blocks);
        self.max_resident_trace_cycles = self.max_resident_trace_cycles.max(resident_cycles);
        self.estimated_peak_resident_trace_bytes = self.estimated_peak_resident_trace_bytes.max(
            resident_blocks
                .saturating_mul(size_of::<TraceBlock>())
                .saturating_add(resident_cycles.saturating_mul(size_of::<Cycle>())),
        );
        let _ = self.observe_memory();
    }

    fn observe_source_rechunk(&mut self, source_cycles: usize, capacity: usize) {
        self.max_source_trace_cycles = self.max_source_trace_cycles.max(source_cycles);
        let resident_cycles = source_cycles.saturating_add(source_cycles.min(capacity));
        self.max_resident_trace_blocks = self.max_resident_trace_blocks.max(2);
        self.max_resident_trace_cycles = self.max_resident_trace_cycles.max(resident_cycles);
        self.estimated_peak_resident_trace_bytes = self.estimated_peak_resident_trace_bytes.max(
            2usize
                .saturating_mul(size_of::<TraceBlock>())
                .saturating_add(resident_cycles.saturating_mul(size_of::<Cycle>())),
        );
        let _ = self.observe_memory();
    }
}

pub(super) struct DirectTraceSpool {
    file: NamedTempFile,
    pub(super) audit: DirectTraceAudit,
    pub(super) initial_ram: BTreeMap<u64, u64>,
    pub(super) metrics: DirectProvingMetrics,
}

impl DirectTraceSpool {
    pub(super) fn capture_production<I>(
        preprocessing: &DirectChunkedPreprocessing,
        capacity: usize,
        blocks: I,
    ) -> Result<Self, DirectChunkedError>
    where
        I: IntoIterator<Item = TraceBlock>,
    {
        let capture_start = Instant::now();
        let mut file = NamedTempFile::new().map_err(spool_error)?;
        let mut metrics = DirectProvingMetrics {
            trace_passes: 2,
            start_physical_memory_bytes: memory_stats::memory_stats()
                .map(|stats| stats.physical_mem),
            ..DirectProvingMetrics::default()
        };
        metrics.peak_observed_physical_memory_bytes = metrics.start_physical_memory_bytes;

        let mut hasher = Sha3_256::new();
        hasher.update(DIRECT_CHUNKED_PROTOCOL_VERSION.as_bytes());
        hasher.update(preprocessing.program_digest);
        hasher.update(preprocessing.lookup_table_commitment);
        let mut first_state: Option<MachineBoundaryState> = None;
        let mut previous_end: Option<(usize, MachineBoundaryState)> = None;
        let mut initial_ram = BTreeMap::new();

        for source_block in blocks {
            validate_source_block(&source_block)?;
            let expected_source_index = metrics.source_block_count;
            if source_block.block_index != expected_source_index {
                return Err(DirectChunkedError::InvalidBlock {
                    block_index: source_block.block_index,
                    reason: format!(
                        "expected sequential source block index {expected_source_index}"
                    ),
                });
            }
            if let Some((previous_index, previous_state)) = &previous_end {
                if previous_state != &source_block.start_state {
                    return Err(DirectChunkedError::DiscontinuousBlocks {
                        previous_block: *previous_index,
                        next_block: source_block.block_index,
                        reason: "end_state does not equal the next start_state".to_string(),
                    });
                }
            } else {
                if source_block.block_index != 0 || source_block.global_cycle_start != 0 {
                    return Err(DirectChunkedError::InvalidBlock {
                        block_index: source_block.block_index,
                        reason: "the direct trace must begin at block 0, global cycle 0"
                            .to_string(),
                    });
                }
                if source_block.start_state.pc != preprocessing.bytecode.entry_address
                    || source_block
                        .start_state
                        .registers
                        .iter()
                        .any(|value| *value != 0)
                {
                    return Err(DirectChunkedError::InvalidBlock {
                        block_index: 0,
                        reason: "D8 production trace does not start from the verifier-known entry PC and zero register state"
                            .to_string(),
                    });
                }
                first_state = Some(source_block.start_state.clone());
            }
            metrics.max_source_trace_cycles = metrics
                .max_source_trace_cycles
                .max(source_block.cycles.len());
            let maximum_source_cycles = capacity.saturating_mul(2);
            if source_block.cycles.len() > maximum_source_cycles {
                return Err(DirectChunkedError::InvalidBlock {
                    block_index: source_block.block_index,
                    reason: format!(
                        "soft source block has {} cycles, exceeding the bounded hard-rechunk limit {maximum_source_cycles}",
                        source_block.cycles.len()
                    ),
                });
            }
            if source_block.active_cycles > capacity {
                metrics.observe_source_rechunk(source_block.cycles.len(), capacity);
            } else {
                metrics.observe_resident_blocks(&source_block, None);
            }
            metrics.source_block_count += 1;
            for block in rechunk_source_block(source_block, capacity, metrics.block_count) {
                let block = block?;
                validate_block(&block, capacity)?;
                for (row, cycle) in block.cycles.iter().enumerate() {
                    match cycle.ram_access() {
                        RAMAccess::Read(read) => {
                            validate_ram_address(
                                preprocessing,
                                block.block_index,
                                row,
                                read.address,
                            )?;
                            initial_ram.entry(read.address).or_insert(read.value);
                        }
                        RAMAccess::Write(write) => {
                            validate_ram_address(
                                preprocessing,
                                block.block_index,
                                row,
                                write.address,
                            )?;
                            initial_ram.entry(write.address).or_insert(write.pre_value);
                        }
                        RAMAccess::NoOp => {}
                    }
                }

                hash_block_metadata(&mut hasher, &block);
                metrics.total_active_cycles = metrics
                    .total_active_cycles
                    .checked_add(block.active_cycles)
                    .ok_or_else(|| DirectChunkedError::InvalidBlock {
                        block_index: block.block_index,
                        reason: "cycle counter overflow".to_string(),
                    })?;
                if metrics.total_active_cycles > preprocessing.max_padded_trace_length {
                    return Err(DirectChunkedError::TraceTooLong {
                        observed: metrics.total_active_cycles,
                        maximum: preprocessing.max_padded_trace_length,
                    });
                }

                let encoded =
                    postcard::to_stdvec(&SpoolTraceBlock::from(&block)).map_err(spool_error)?;
                let encoded_len = encoded.len();
                let encoded_len_u64 = u64::try_from(encoded_len).map_err(spool_error)?;
                file.as_file_mut()
                    .write_all(&encoded_len_u64.to_le_bytes())
                    .and_then(|_| file.as_file_mut().write_all(&encoded))
                    .map_err(spool_error)?;
                metrics.spooled_trace_bytes = metrics
                    .spooled_trace_bytes
                    .saturating_add(8)
                    .saturating_add(encoded_len);
                metrics.max_serialized_block_bytes =
                    metrics.max_serialized_block_bytes.max(encoded_len);
                previous_end = Some((block.block_index, block.end_state.clone()));
                metrics.block_count += 1;
            }
        }

        let first_state = first_state.ok_or(DirectChunkedError::EmptyTrace)?;
        let final_state = previous_end.ok_or(DirectChunkedError::EmptyTrace)?.1;
        if !final_state.terminated {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: metrics.block_count - 1,
                reason: "D8 production trace is not terminated".to_string(),
            });
        }
        file.as_file_mut().flush().map_err(spool_error)?;
        metrics.peak_tracked_ram_addresses = initial_ram.len();
        metrics.initial_ram_registry_bytes = initial_ram
            .len()
            .saturating_mul(size_of::<u64>().saturating_mul(2));
        metrics.capture_trace_micros = capture_start.elapsed().as_micros();
        metrics.after_capture_physical_memory_bytes = metrics.observe_memory();
        let audit = DirectTraceAudit {
            block_count: metrics.block_count,
            total_cycles: metrics.total_active_cycles,
            first_state,
            final_state,
            trace_digest: hasher.finalize().into(),
        };
        Ok(Self {
            file,
            audit,
            initial_ram,
            metrics,
        })
    }

    pub(super) fn replay(&self) -> Result<DirectTraceReplay, DirectChunkedError> {
        Ok(DirectTraceReplay {
            reader: BufReader::new(self.file.reopen().map_err(spool_error)?),
            max_serialized_block_bytes: self.metrics.max_serialized_block_bytes,
        })
    }
}

pub(super) fn audit_direct_trace_streaming<I>(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    blocks: I,
) -> Result<(DirectTraceAudit, DirectProvingMetrics), DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    let spool = DirectTraceSpool::capture_production(preprocessing, capacity, blocks)?;
    Ok((spool.audit, spool.metrics))
}

pub(super) struct DirectTraceReplay {
    reader: BufReader<std::fs::File>,
    max_serialized_block_bytes: usize,
}

impl DirectTraceReplay {
    pub(super) fn read_next(&mut self) -> Result<Option<TraceBlock>, DirectChunkedError> {
        let mut length = [0u8; 8];
        match self.reader.read(&mut length[..1]) {
            Ok(0) => return Ok(None),
            Ok(1) => {}
            Ok(_) => unreachable!("one-byte read returned more than one byte"),
            Err(error) => return Err(spool_error(error)),
        }
        self.reader
            .read_exact(&mut length[1..])
            .map_err(spool_error)?;
        let length = usize::try_from(u64::from_le_bytes(length)).map_err(spool_error)?;
        if length > self.max_serialized_block_bytes {
            return Err(DirectChunkedError::InvalidProofShape(format!(
                "D8 trace spool record length {length} exceeds captured maximum {}",
                self.max_serialized_block_bytes
            )));
        }
        let mut encoded = vec![0u8; length];
        self.reader.read_exact(&mut encoded).map_err(spool_error)?;
        let block = postcard::from_bytes::<SpoolTraceBlock>(&encoded).map_err(spool_error)?;
        block.try_into().map(Some)
    }
}

fn validate_source_block(block: &TraceBlock) -> Result<(), DirectChunkedError> {
    let invalid = |reason: &str| DirectChunkedError::InvalidBlock {
        block_index: block.block_index,
        reason: reason.to_string(),
    };
    if block.active_cycles == 0 || block.active_cycles != block.cycles.len() {
        return Err(invalid(
            "source block must contain one non-empty active cycle prefix",
        ));
    }
    if !block.ended_at_tick_boundary {
        return Err(invalid(
            "source block is not cut at an emulator tick boundary",
        ));
    }
    if block.start_state.terminated {
        return Err(invalid("an active source block cannot start terminated"));
    }
    if block.global_cycle_start != block.start_state.global_cycle {
        return Err(invalid("source start_state global cycle mismatch"));
    }
    let expected_end = block
        .global_cycle_start
        .checked_add(block.active_cycles)
        .ok_or_else(|| invalid("source global cycle overflow"))?;
    if expected_end != block.end_state.global_cycle {
        return Err(invalid("source end_state global cycle mismatch"));
    }
    Ok(())
}

struct HardRechunkIterator {
    original_block_index: usize,
    original_end: MachineBoundaryState,
    capacity: usize,
    next_block_index: usize,
    next_global_cycle: usize,
    source_row_offset: usize,
    next_start_state: MachineBoundaryState,
    cycles: std::vec::IntoIter<Cycle>,
}

impl Iterator for HardRechunkIterator {
    type Item = Result<TraceBlock, DirectChunkedError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cycles.len() == 0 {
            return None;
        }

        let cycles = self.cycles.by_ref().take(self.capacity).collect::<Vec<_>>();
        let active_cycles = cycles.len();
        let mut registers = self.next_start_state.registers;
        for (row, cycle) in cycles.iter().enumerate() {
            if let Some((register, pre_value, post_value)) = cycle.rd_write() {
                let register = register as usize;
                if register >= registers.len() || registers[register] as u64 != pre_value {
                    return Some(Err(DirectChunkedError::InvalidBlock {
                        block_index: self.original_block_index,
                        reason: format!(
                            "hard rechunk register pre-value mismatch at source row {}",
                            self.source_row_offset + row
                        ),
                    }));
                }
                registers[register] = post_value as i64;
            }
        }

        let is_last = self.cycles.len() == 0;
        if is_last && self.original_end.registers != registers {
            return Some(Err(DirectChunkedError::InvalidBlock {
                block_index: self.original_block_index,
                reason: "hard rechunk register transitions do not reconstruct the source end state"
                    .to_string(),
            }));
        }
        let end_state = if is_last {
            self.original_end.clone()
        } else {
            MachineBoundaryState {
                global_cycle: self.next_global_cycle + active_cycles,
                // These two emulator-level fields have no meaning inside one
                // expanded instruction. They remain audit metadata; the
                // recursive boundary is enforced by exact register state and
                // D5's next-row CPU witness.
                emulator_trace_len: self.next_start_state.emulator_trace_len,
                pc: self.next_start_state.pc,
                registers,
                terminated: false,
            }
        };
        let block = TraceBlock {
            block_index: self.next_block_index,
            global_cycle_start: self.next_global_cycle,
            active_cycles,
            target_size: self.capacity,
            start_state: self.next_start_state.clone(),
            end_state: end_state.clone(),
            cycles,
            // In the direct protocol this compatibility bit means that the
            // boundary is safe for folding. D5 carries the exact next Jolt row,
            // so a hard boundary inside one emulator tick is also safe.
            ended_at_tick_boundary: true,
        };
        self.next_block_index += 1;
        self.next_global_cycle += active_cycles;
        self.source_row_offset += active_cycles;
        self.next_start_state = end_state;
        Some(Ok(block))
    }
}

fn rechunk_source_block(
    block: TraceBlock,
    capacity: usize,
    first_block_index: usize,
) -> HardRechunkIterator {
    HardRechunkIterator {
        original_block_index: block.block_index,
        original_end: block.end_state,
        capacity,
        next_block_index: first_block_index,
        next_global_cycle: block.global_cycle_start,
        source_row_offset: 0,
        next_start_state: block.start_state,
        cycles: block.cycles.into_iter(),
    }
}

fn spool_error(error: impl std::fmt::Display) -> DirectChunkedError {
    DirectChunkedError::InvalidProofShape(format!("D8 bounded trace spool failure: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary(cycle: usize, terminated: bool) -> MachineBoundaryState {
        MachineBoundaryState {
            global_cycle: cycle,
            emulator_trace_len: cycle,
            pc: 0,
            registers: [0; common::constants::REGISTER_COUNT as usize],
            terminated,
        }
    }

    fn blocks(count: usize) -> Vec<TraceBlock> {
        (0..count)
            .map(|index| TraceBlock {
                block_index: index,
                global_cycle_start: index,
                active_cycles: 1,
                target_size: 2,
                start_state: boundary(index, false),
                end_state: boundary(index + 1, index + 1 == count),
                cycles: vec![Cycle::NoOp],
                ended_at_tick_boundary: true,
            })
            .collect()
    }

    fn preprocessing() -> DirectChunkedPreprocessing {
        DirectChunkedPreprocessing::from_program_bytes(b"d8-spool-test", 64)
    }

    #[test]
    fn d8_trace_spool_replay_keeps_only_current_and_lookahead_blocks() {
        let spool = DirectTraceSpool::capture_production(&preprocessing(), 2, blocks(4)).unwrap();
        assert_eq!(spool.audit.block_count, 4);
        assert_eq!(spool.metrics.max_resident_trace_blocks, 1);
        let mut replay = spool.replay().unwrap();
        let mut metrics = spool.metrics.clone();
        let mut current = replay.read_next().unwrap().unwrap();
        let mut next = replay.read_next().unwrap();
        loop {
            metrics.observe_resident_blocks(&current, next.as_ref());
            match next.take() {
                Some(next_block) => {
                    current = next_block;
                    next = replay.read_next().unwrap();
                }
                None => break,
            }
        }
        assert!(replay.read_next().unwrap().is_none());
        assert_eq!(metrics.max_resident_trace_blocks, 2);
        assert_eq!(metrics.max_resident_trace_cycles, 2);
        assert!(metrics.trace_residency_is_bounded(2));
        assert!(metrics.spooled_trace_bytes > 0);
    }

    #[test]
    fn d8_trace_residency_does_not_grow_with_block_count() {
        let replay_metrics = |count| {
            let spool =
                DirectTraceSpool::capture_production(&preprocessing(), 2, blocks(count)).unwrap();
            let mut replay = spool.replay().unwrap();
            let mut metrics = spool.metrics.clone();
            let mut current = replay.read_next().unwrap().unwrap();
            let mut next = replay.read_next().unwrap();
            loop {
                metrics.observe_resident_blocks(&current, next.as_ref());
                match next.take() {
                    Some(next_block) => {
                        current = next_block;
                        next = replay.read_next().unwrap();
                    }
                    None => break,
                }
            }
            metrics
        };

        let four_blocks = replay_metrics(4);
        let sixty_four_blocks = replay_metrics(64);
        assert_eq!(four_blocks.max_resident_trace_blocks, 2);
        assert_eq!(sixty_four_blocks.max_resident_trace_blocks, 2);
        assert_eq!(four_blocks.max_resident_trace_cycles, 2);
        assert_eq!(sixty_four_blocks.max_resident_trace_cycles, 2);
        assert_eq!(
            four_blocks.estimated_peak_resident_trace_bytes,
            sixty_four_blocks.estimated_peak_resident_trace_bytes
        );
        assert!(sixty_four_blocks.trace_residency_is_bounded(2));
    }

    #[test]
    fn d8_trace_spool_rejects_dropped_reordered_duplicated_and_forged_blocks() {
        let pp = preprocessing();

        let mut dropped = blocks(3);
        dropped.remove(1);
        assert!(DirectTraceSpool::capture_production(&pp, 2, dropped).is_err());

        let mut reordered = blocks(3);
        reordered.swap(0, 1);
        assert!(DirectTraceSpool::capture_production(&pp, 2, reordered).is_err());

        let mut duplicated = blocks(3);
        duplicated.insert(2, duplicated[1].clone());
        assert!(DirectTraceSpool::capture_production(&pp, 2, duplicated).is_err());

        let mut forged = blocks(3);
        forged[1].start_state.registers[7] = 1;
        assert!(DirectTraceSpool::capture_production(&pp, 2, forged).is_err());
    }

    #[test]
    fn d8_soft_source_block_is_hard_rechunked_without_losing_boundaries() {
        let spool = DirectTraceSpool::capture_production(
            &preprocessing(),
            2,
            blocks(1).into_iter().map(|mut block| {
                block.active_cycles = 3;
                block.cycles = vec![Cycle::NoOp; 3];
                block.end_state.global_cycle = 3;
                block.end_state.emulator_trace_len = 3;
                block
            }),
        )
        .unwrap();
        assert_eq!(spool.metrics.source_block_count, 1);
        assert_eq!(spool.audit.block_count, 2);
        assert_eq!(spool.audit.total_cycles, 3);
        assert_eq!(spool.metrics.max_source_trace_cycles, 3);
        assert!(spool.metrics.trace_residency_is_bounded(2));

        let mut replay = spool.replay().unwrap();
        let first = replay.read_next().unwrap().unwrap();
        let second = replay.read_next().unwrap().unwrap();
        assert_eq!((first.block_index, first.active_cycles), (0, 2));
        assert_eq!((second.block_index, second.active_cycles), (1, 1));
        assert_eq!(first.end_state, second.start_state);
        assert!(second.end_state.terminated);
        assert!(replay.read_next().unwrap().is_none());
    }

    #[test]
    fn d8_rejects_unbounded_soft_source_expansion() {
        let mut oversized = blocks(1).remove(0);
        oversized.active_cycles = 5;
        oversized.cycles = vec![Cycle::NoOp; 5];
        oversized.end_state.global_cycle = 5;
        oversized.end_state.emulator_trace_len = 5;
        let error = match DirectTraceSpool::capture_production(&preprocessing(), 2, [oversized]) {
            Ok(_) => panic!("oversized soft source block was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("bounded hard-rechunk limit 4"));
    }
}
