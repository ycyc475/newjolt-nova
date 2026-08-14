//! Direct V2 block-Jolt proof boundary.
//!
//! D9-D14 build a compact host-side proof before the verifier is circuitized.
//! The D8 row-level proof remains available as a regression oracle but is not
//! embedded in any type exported by this module.

mod compat;
mod lookup;
mod types;

pub use compat::d8_statement_adapter;
pub use lookup::{
    new_block_lookup_transcript, prove_block_lookup_lasso, verify_block_lookup_lasso,
};
pub use types::{
    BlockBoundaryState, BlockJoltProof, BlockJoltStatement, BlockRelation, CompactSumcheckProof,
    CpuBlockRelationProof, DeferredPcsClaim, FieldElement, LookupBlockProof, RamBlockProof,
    RegisterBlockProof, StreamingRecursiveState, TranscriptCheckpoint, BLOCK_JOLT_PROTOCOL_VERSION,
};
