//! Direct, trace-first Jolt-Nova proving seam.
//!
//! This module deliberately starts at [`TraceBlock`].  It does not accept an
//! already-produced monolithic proof artifact and it never treats a host-side
//! check as a recursive proof.  Relations that have not been internalized are
//! represented by [`DirectRelationState::Unsupported`] and cause the production
//! entry point to fail closed.

use std::{collections::BTreeMap, error::Error, fmt};

use ark_serialize::CanonicalSerialize;
use common::jolt_device::MemoryLayout;
use sha3::{Digest as ShaDigest, Sha3_256};
use tracer::{MachineBoundaryState, TraceBlock};

use crate::{
    poly::commitment::commitment_scheme::CommitmentScheme, zkvm::verifier::JoltSharedPreprocessing,
};

/// Domain/version tag included in every direct statement.
pub const DIRECT_CHUNKED_PROTOCOL_VERSION: &str = "jolt-nova/direct-chunked/v1";

/// A relation which must eventually be verified by the direct recursive path.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DirectRelation {
    Lookup,
    Register,
    Ram,
    Cpu,
    Pcs,
}

impl DirectRelation {
    pub const ALL: [Self; 5] = [
        Self::Lookup,
        Self::Register,
        Self::Ram,
        Self::Cpu,
        Self::Pcs,
    ];
}

impl fmt::Display for DirectRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Lookup => "lookup",
            Self::Register => "register",
            Self::Ram => "RAM",
            Self::Cpu => "CPU/R1CS",
            Self::Pcs => "PCS",
        };
        f.write_str(name)
    }
}

/// Security status carried explicitly by a partial direct statement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectRelationState {
    /// The relation is constrained and verified by the recursive path.
    Proven,
    /// The relation produced a folded obligation which must be closed later.
    Deferred,
    /// No cryptographic relation exists yet. Production proving must fail.
    Unsupported,
}

/// Immutable identity/configuration data supplied before trace streaming starts.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectChunkedPreprocessing {
    pub program_digest: [u8; 32],
    pub lookup_table_commitment: [u8; 32],
    pub max_padded_trace_length: usize,
    /// Materialized bytecode used by D5. This is verifier-known preprocessing,
    /// not a native Jolt proof or a host-verified receipt.
    pub bytecode: crate::zkvm::bytecode::BytecodePreprocessing,
    /// Verifier-known memory map used to bind D6 public I/O and advice regions.
    pub memory_layout: MemoryLayout,
    /// First byte address of the verifier-known ELF memory image.
    pub program_image_start: u64,
    /// Verifier-known ELF image packed exactly as Jolt's RAM preprocessing.
    pub program_image_words: Vec<u64>,
}

impl DirectChunkedPreprocessing {
    /// Builds the direct identity from the canonical shared preprocessing.
    pub fn from_shared<PCS>(shared: &JoltSharedPreprocessing<PCS>) -> Self
    where
        PCS: CommitmentScheme,
        PCS::Commitment: CanonicalSerialize,
    {
        let full = shared.program.as_full().ok();
        Self {
            program_digest: shared.digest(),
            lookup_table_commitment: fixed_lookup_registry_commitment(),
            max_padded_trace_length: shared.max_padded_trace_length,
            bytecode: full
                .map(|program| (*program.bytecode).clone())
                .unwrap_or_default(),
            memory_layout: shared.memory_layout.clone(),
            program_image_start: full
                .map(|program| program.ram.min_bytecode_address)
                .unwrap_or_default(),
            program_image_words: full
                .map(|program| program.ram.bytecode_words.clone())
                .unwrap_or_default(),
        }
    }

    /// Convenience constructor for architecture tests and trace-only runners.
    /// Production callers should prefer [`Self::from_shared`].
    pub fn from_program_bytes(program: &[u8], max_padded_trace_length: usize) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(DIRECT_CHUNKED_PROTOCOL_VERSION.as_bytes());
        hasher.update((program.len() as u64).to_le_bytes());
        hasher.update(program);
        Self {
            program_digest: hasher.finalize().into(),
            lookup_table_commitment: fixed_lookup_registry_commitment(),
            max_padded_trace_length,
            bytecode: crate::zkvm::bytecode::BytecodePreprocessing::default(),
            memory_layout: MemoryLayout::default(),
            program_image_start: 0,
            program_image_words: Vec::new(),
        }
    }

    /// Test/trace constructor that derives the exact bytecode table from final
    /// Jolt cycles. Production callers should use [`Self::from_shared`].
    pub fn from_trace_cycles(
        program: &[u8],
        max_padded_trace_length: usize,
        cycles: &[tracer::instruction::Cycle],
    ) -> Result<Self, DirectChunkedError> {
        use jolt_riscv::RV64IMAC_JOLT;

        let rows = cycles
            .iter()
            .map(|cycle| {
                cycle
                    .instruction()
                    .try_jolt_instruction_row()
                    .map_err(|kind| {
                        DirectChunkedError::InvalidConfiguration(format!(
                            "direct CPU preprocessing cannot materialize instruction kind {kind:?}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let entry = rows
            .first()
            .map(|row| row.address as u64)
            .unwrap_or_default();
        let bytecode =
            crate::zkvm::bytecode::BytecodePreprocessing::preprocess(rows, entry, RV64IMAC_JOLT)
                .map_err(|error| {
                    DirectChunkedError::InvalidConfiguration(format!(
                        "direct CPU bytecode preprocessing failed: {error}"
                    ))
                })?;
        let mut preprocessing = Self::from_program_bytes(program, max_padded_trace_length);
        preprocessing.bytecode = bytecode;
        Ok(preprocessing)
    }

    /// Test-only helper for a verifier-known initial RAM image. Production
    /// callers obtain these fields from [`Self::from_shared`].
    #[cfg(test)]
    pub(crate) fn with_initial_memory(
        mut self,
        memory_layout: MemoryLayout,
        program_image_start: u64,
        program_image_words: Vec<u64>,
    ) -> Self {
        self.memory_layout = memory_layout;
        self.program_image_start = program_image_start;
        self.program_image_words = program_image_words;
        self
    }
}

/// Direct prover configuration. No field can inject a pre-verified artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectChunkedConfig {
    /// Fixed circuit capacity. Trace blocks may contain fewer active rows.
    pub block_capacity: usize,
    /// Compress the final recursive accumulator after all relations are closed.
    pub compress_final_spartan: bool,
}

impl DirectChunkedConfig {
    pub fn validate(&self) -> Result<(), DirectChunkedError> {
        if self.block_capacity == 0 || !self.block_capacity.is_power_of_two() {
            return Err(DirectChunkedError::InvalidConfiguration(
                "block_capacity must be a non-zero power of two".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for DirectChunkedConfig {
    fn default() -> Self {
        Self {
            block_capacity: 1 << 12,
            compress_final_spartan: true,
        }
    }
}

/// Bounded metadata obtained while consuming a trace-block iterator.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectTraceAudit {
    pub block_count: usize,
    pub total_cycles: usize,
    pub first_state: MachineBoundaryState,
    pub final_state: MachineBoundaryState,
    pub trace_digest: [u8; 32],
}

/// Public statement eventually bound to the Nova and Spartan artifacts.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectChunkedStatement {
    pub protocol_version: &'static str,
    pub program_digest: [u8; 32],
    pub lookup_table_commitment: [u8; 32],
    pub block_count: usize,
    pub total_cycles: usize,
    pub first_state: MachineBoundaryState,
    pub final_state: MachineBoundaryState,
    pub trace_digest: [u8; 32],
    pub relations: BTreeMap<DirectRelation, DirectRelationState>,
}

/// Serialization boundary for the new direct flow.
///
/// The byte vectors are generated only by the recursive and compressed proving
/// APIs. There is intentionally no field for importing a result from another
/// proving flow.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectChunkedProof {
    pub statement: DirectChunkedStatement,
    pub folded_instance: Vec<u8>,
    pub final_spartan: Option<Vec<u8>>,
    pub deferred_opening_digest: Option<[u8; 32]>,
}

impl DirectChunkedProof {
    fn ensure_closed(statement: &DirectChunkedStatement) -> Result<(), DirectChunkedError> {
        for relation in DirectRelation::ALL {
            let state = statement
                .relations
                .get(&relation)
                .copied()
                .unwrap_or(DirectRelationState::Unsupported);
            if state != DirectRelationState::Proven {
                return Err(DirectChunkedError::UnclosedRelation { relation, state });
            }
        }
        Ok(())
    }

    /// Returns success only for a fully closed production proof.
    pub fn validate_production_shape(&self) -> Result<(), DirectChunkedError> {
        Self::ensure_closed(&self.statement)?;
        if self.folded_instance.is_empty() {
            return Err(DirectChunkedError::InvalidProofShape(
                "folded_instance must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Fail-closed errors for the direct architecture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirectChunkedError {
    InvalidConfiguration(String),
    EmptyTrace,
    TraceTooLong {
        observed: usize,
        maximum: usize,
    },
    InvalidBlock {
        block_index: usize,
        reason: String,
    },
    DiscontinuousBlocks {
        previous_block: usize,
        next_block: usize,
        reason: String,
    },
    UnsupportedRelation {
        relation: DirectRelation,
        audited_blocks: usize,
    },
    UnclosedRelation {
        relation: DirectRelation,
        state: DirectRelationState,
    },
    InvalidProofShape(String),
}

impl fmt::Display for DirectChunkedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(reason) => write!(f, "invalid direct configuration: {reason}"),
            Self::EmptyTrace => f.write_str("direct proving requires at least one trace block"),
            Self::TraceTooLong { observed, maximum } => write!(
                f,
                "direct trace has {observed} cycles, exceeding configured preprocessing maximum {maximum}"
            ),
            Self::InvalidBlock { block_index, reason } => {
                write!(f, "invalid direct block {block_index}: {reason}")
            }
            Self::DiscontinuousBlocks {
                previous_block,
                next_block,
                reason,
            } => write!(
                f,
                "direct blocks {previous_block} and {next_block} are discontinuous: {reason}"
            ),
            Self::UnsupportedRelation {
                relation,
                audited_blocks,
            } => write!(
                f,
                "direct relation {relation} is not internalized after auditing {audited_blocks} blocks"
            ),
            Self::UnclosedRelation { relation, state } => {
                write!(f, "direct relation {relation} is not closed: {state:?}")
            }
            Self::InvalidProofShape(reason) => write!(f, "invalid direct proof shape: {reason}"),
        }
    }
}

impl Error for DirectChunkedError {}

/// Entry point for the isolated direct architecture.
#[derive(Clone, Debug)]
pub struct DirectChunkedProver {
    preprocessing: DirectChunkedPreprocessing,
    config: DirectChunkedConfig,
}

impl DirectChunkedProver {
    pub fn new(
        preprocessing: DirectChunkedPreprocessing,
        config: DirectChunkedConfig,
    ) -> Result<Self, DirectChunkedError> {
        config.validate()?;
        Ok(Self {
            preprocessing,
            config,
        })
    }

    /// Consumes the iterator one block at a time and retains only bounded audit
    /// state. This is the architecture seam used by all subsequent D-stages.
    pub fn audit_trace_blocks<I>(&self, blocks: I) -> Result<DirectTraceAudit, DirectChunkedError>
    where
        I: IntoIterator<Item = TraceBlock>,
    {
        let mut hasher = Sha3_256::new();
        hasher.update(DIRECT_CHUNKED_PROTOCOL_VERSION.as_bytes());
        hasher.update(self.preprocessing.program_digest);
        hasher.update(self.preprocessing.lookup_table_commitment);

        let mut block_count = 0usize;
        let mut total_cycles = 0usize;
        let mut first_state = None;
        let mut previous_end: Option<(usize, MachineBoundaryState)> = None;

        for block in blocks {
            validate_block(&block, self.config.block_capacity)?;
            if block.block_index != block_count {
                return Err(DirectChunkedError::InvalidBlock {
                    block_index: block.block_index,
                    reason: format!("expected sequential block index {block_count}"),
                });
            }
            if let Some((previous_index, previous_state)) = &previous_end {
                if previous_state != &block.start_state {
                    return Err(DirectChunkedError::DiscontinuousBlocks {
                        previous_block: *previous_index,
                        next_block: block.block_index,
                        reason: "end_state does not equal the next start_state".to_string(),
                    });
                }
            } else {
                if block.block_index != 0 || block.global_cycle_start != 0 {
                    return Err(DirectChunkedError::InvalidBlock {
                        block_index: block.block_index,
                        reason: "the direct trace must begin at block 0, global cycle 0"
                            .to_string(),
                    });
                }
                first_state = Some(block.start_state.clone());
            }

            hash_block_metadata(&mut hasher, &block);
            total_cycles = total_cycles
                .checked_add(block.active_cycles)
                .ok_or_else(|| DirectChunkedError::InvalidBlock {
                    block_index: block.block_index,
                    reason: "cycle counter overflow".to_string(),
                })?;
            if total_cycles > self.preprocessing.max_padded_trace_length {
                return Err(DirectChunkedError::TraceTooLong {
                    observed: total_cycles,
                    maximum: self.preprocessing.max_padded_trace_length,
                });
            }
            previous_end = Some((block.block_index, block.end_state.clone()));
            block_count += 1;
        }

        let first_state = first_state.ok_or(DirectChunkedError::EmptyTrace)?;
        let final_state = previous_end.expect("non-empty trace has an end state").1;
        Ok(DirectTraceAudit {
            block_count,
            total_cycles,
            first_state,
            final_state,
            trace_digest: hasher.finalize().into(),
        })
    }

    /// Production entry point. D1 intentionally fails at the first relation;
    /// later stages replace each explicit failure with a recursive proof.
    pub fn prove<I>(&self, blocks: I) -> Result<DirectChunkedProof, DirectChunkedError>
    where
        I: IntoIterator<Item = TraceBlock>,
    {
        let audit = self.audit_trace_blocks(blocks)?;
        Err(DirectChunkedError::UnsupportedRelation {
            relation: DirectRelation::Lookup,
            audited_blocks: audit.block_count,
        })
    }

    pub fn architecture_statement(&self, audit: DirectTraceAudit) -> DirectChunkedStatement {
        DirectChunkedStatement {
            protocol_version: DIRECT_CHUNKED_PROTOCOL_VERSION,
            program_digest: self.preprocessing.program_digest,
            lookup_table_commitment: self.preprocessing.lookup_table_commitment,
            block_count: audit.block_count,
            total_cycles: audit.total_cycles,
            first_state: audit.first_state,
            final_state: audit.final_state,
            trace_digest: audit.trace_digest,
            relations: DirectRelation::ALL
                .into_iter()
                .map(|relation| (relation, DirectRelationState::Unsupported))
                .collect(),
        }
    }

    /// D2 proving entry point. It consumes raw blocks, derives Jolt's native
    /// lookup witnesses internally, folds one verified lookup relation per
    /// block with Nova, and closes the stage with Spartan.
    pub fn prove_lookup_stage<I>(
        &self,
        blocks: I,
    ) -> Result<super::DirectLookupStageProof, DirectChunkedError>
    where
        I: IntoIterator<Item = TraceBlock>,
    {
        super::prove_direct_lookup_stage(&self.preprocessing, self.config.block_capacity, blocks)
    }

    pub fn verify_lookup_stage(
        &self,
        proof: &super::DirectLookupStageProof,
    ) -> Result<(), DirectChunkedError> {
        super::verify_direct_lookup_stage(&self.preprocessing, self.config.block_capacity, proof)
    }

    /// D3 proving entry point. The lookup and register relations are verified
    /// together in every Nova step and closed by one Spartan proof.
    pub fn prove_register_stage<I>(
        &self,
        blocks: I,
    ) -> Result<super::DirectRegisterStageProof, DirectChunkedError>
    where
        I: IntoIterator<Item = TraceBlock>,
    {
        super::prove_direct_register_stage(&self.preprocessing, self.config.block_capacity, blocks)
    }

    pub fn verify_register_stage(
        &self,
        proof: &super::DirectRegisterStageProof,
    ) -> Result<(), DirectChunkedError> {
        super::verify_direct_register_stage(&self.preprocessing, self.config.block_capacity, proof)
    }

    /// D4 proving entry point. Lookup, register, and authenticated sparse RAM
    /// transitions are constrained in the same recursive step.
    pub fn prove_ram_stage<I>(
        &self,
        blocks: I,
    ) -> Result<super::DirectRamStageProof, DirectChunkedError>
    where
        I: IntoIterator<Item = TraceBlock>,
    {
        super::prove_direct_ram_stage(&self.preprocessing, self.config.block_capacity, blocks)
    }

    pub fn verify_ram_stage(
        &self,
        proof: &super::DirectRamStageProof,
    ) -> Result<(), DirectChunkedError> {
        super::verify_direct_ram_stage(&self.preprocessing, self.config.block_capacity, proof)
    }

    /// D5 proving entry point. CPU/R1CS rows are folded in the same recursive
    /// step as the D2--D4 relations.
    pub fn prove_cpu_stage<I>(
        &self,
        blocks: I,
    ) -> Result<super::DirectCpuStageProof, DirectChunkedError>
    where
        I: IntoIterator<Item = TraceBlock>,
    {
        super::prove_direct_cpu_stage(&self.preprocessing, self.config.block_capacity, blocks)
    }

    pub fn verify_cpu_stage(
        &self,
        proof: &super::DirectCpuStageProof,
    ) -> Result<(), DirectChunkedError> {
        super::verify_direct_cpu_stage(&self.preprocessing, self.config.block_capacity, proof)
    }

    /// D6 proving entry point. The fixed-shape global CPU witness is committed
    /// with Dory and evaluated from the same allocated rows folded by Nova.
    pub fn prove_pcs_stage<I>(
        &self,
        execution: super::DirectExecutionInputs,
        blocks: I,
    ) -> Result<super::DirectPcsStageProof, DirectChunkedError>
    where
        I: IntoIterator<Item = TraceBlock>,
    {
        super::prove_direct_pcs_stage(
            &self.preprocessing,
            self.config.block_capacity,
            execution,
            blocks,
        )
    }

    pub fn verify_pcs_stage(
        &self,
        proof: &super::DirectPcsStageProof,
    ) -> Result<(), DirectChunkedError> {
        super::verify_direct_pcs_stage(&self.preprocessing, self.config.block_capacity, proof)
    }
}

pub(super) fn validate_block(
    block: &TraceBlock,
    capacity: usize,
) -> Result<(), DirectChunkedError> {
    let invalid = |reason: String| DirectChunkedError::InvalidBlock {
        block_index: block.block_index,
        reason,
    };
    if block.active_cycles == 0 {
        return Err(invalid("active_cycles must be non-zero".to_string()));
    }
    if block.active_cycles != block.cycles.len() {
        return Err(invalid(format!(
            "active_cycles {} does not match cycle vector length {}",
            block.active_cycles,
            block.cycles.len()
        )));
    }
    if block.active_cycles > capacity {
        return Err(invalid(format!(
            "active_cycles {} exceeds fixed circuit capacity {capacity}",
            block.active_cycles
        )));
    }
    if !block.ended_at_tick_boundary {
        return Err(invalid(
            "block is not cut at an emulator tick boundary".to_string(),
        ));
    }
    if block.start_state.terminated {
        return Err(invalid(
            "an active block cannot start from a terminated machine state".to_string(),
        ));
    }
    if block.global_cycle_start != block.start_state.global_cycle {
        return Err(invalid("start_state global cycle mismatch".to_string()));
    }
    let expected_end = block
        .global_cycle_start
        .checked_add(block.active_cycles)
        .ok_or_else(|| invalid("global cycle overflow".to_string()))?;
    if expected_end != block.end_state.global_cycle {
        return Err(invalid("end_state global cycle mismatch".to_string()));
    }
    Ok(())
}

fn hash_block_metadata(hasher: &mut Sha3_256, block: &TraceBlock) {
    hasher.update((block.block_index as u64).to_le_bytes());
    hasher.update((block.global_cycle_start as u64).to_le_bytes());
    hasher.update((block.active_cycles as u64).to_le_bytes());
    hash_boundary_state(hasher, &block.start_state);
    hash_boundary_state(hasher, &block.end_state);
    // At D1 this digest is only a streaming identity. Subsequent relations bind
    // each cycle cryptographically inside the recursive circuit.
    for cycle in &block.cycles {
        hasher.update(format!("{cycle:?}").as_bytes());
    }
}

fn hash_boundary_state(hasher: &mut Sha3_256, state: &MachineBoundaryState) {
    hasher.update((state.global_cycle as u64).to_le_bytes());
    hasher.update((state.emulator_trace_len as u64).to_le_bytes());
    hasher.update(state.pc.to_le_bytes());
    for register in state.registers {
        hasher.update(register.to_le_bytes());
    }
    hasher.update([u8::from(state.terminated)]);
}

pub(super) fn fixed_lookup_registry_commitment() -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(DIRECT_CHUNKED_PROTOCOL_VERSION.as_bytes());
    hasher.update(b"fixed-jolt-lookup-registry");
    hasher.update((common::constants::XLEN as u64).to_le_bytes());
    // The ordered IDs and evaluator code are part of the direct verification
    // key. Bumping the protocol version is mandatory if this registry changes.
    for table_id in 0u64..40 {
        hasher.update(table_id.to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::constants::REGISTER_COUNT;
    use tracer::instruction::Cycle;

    fn boundary(global_cycle: usize, terminated: bool) -> MachineBoundaryState {
        MachineBoundaryState {
            global_cycle,
            emulator_trace_len: global_cycle,
            pc: (global_cycle * 4) as u64,
            registers: [0; REGISTER_COUNT as usize],
            terminated,
        }
    }

    fn block(index: usize, start: usize, active: usize, terminated: bool) -> TraceBlock {
        TraceBlock {
            block_index: index,
            global_cycle_start: start,
            active_cycles: active,
            target_size: active,
            start_state: boundary(start, false),
            end_state: boundary(start + active, terminated),
            cycles: vec![Cycle::NoOp; active],
            ended_at_tick_boundary: true,
        }
    }

    fn prover(capacity: usize) -> DirectChunkedProver {
        DirectChunkedProver::new(
            DirectChunkedPreprocessing::from_program_bytes(b"direct-test", 64),
            DirectChunkedConfig {
                block_capacity: capacity,
                compress_final_spartan: false,
            },
        )
        .unwrap()
    }

    #[test]
    fn d1_streams_contiguous_blocks() {
        let audit = prover(4)
            .audit_trace_blocks(vec![block(0, 0, 2, false), block(1, 2, 2, true)])
            .unwrap();
        assert_eq!(audit.block_count, 2);
        assert_eq!(audit.total_cycles, 4);
        assert!(audit.final_state.terminated);
    }

    #[test]
    fn d1_production_entry_fails_closed() {
        let error = prover(4).prove(vec![block(0, 0, 1, true)]).unwrap_err();
        assert_eq!(
            error,
            DirectChunkedError::UnsupportedRelation {
                relation: DirectRelation::Lookup,
                audited_blocks: 1,
            }
        );
    }

    #[test]
    fn d1_rejects_discontinuous_or_oversized_blocks() {
        let mut second = block(1, 2, 1, true);
        second.start_state.pc ^= 4;
        assert!(matches!(
            prover(4).audit_trace_blocks(vec![block(0, 0, 2, false), second]),
            Err(DirectChunkedError::DiscontinuousBlocks { .. })
        ));
        assert!(matches!(
            prover(2).audit_trace_blocks(vec![block(0, 0, 3, true)]),
            Err(DirectChunkedError::InvalidBlock { .. })
        ));
    }

    #[test]
    fn d1_partial_statement_cannot_be_accepted_as_production_proof() {
        let prover = prover(4);
        let audit = prover
            .audit_trace_blocks(vec![block(0, 0, 1, true)])
            .unwrap();
        let proof = DirectChunkedProof {
            statement: prover.architecture_statement(audit),
            folded_instance: vec![1],
            final_spartan: None,
            deferred_opening_digest: None,
        };
        assert!(matches!(
            proof.validate_production_shape(),
            Err(DirectChunkedError::UnclosedRelation {
                relation: DirectRelation::Lookup,
                ..
            })
        ));
    }
}
