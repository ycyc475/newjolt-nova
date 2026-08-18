//! Direct V2 block-Jolt proof boundary.
//!
//! D9-D14 build a compact host-side proof before the verifier is circuitized.
//! The D8 row-level proof remains available as a regression oracle but is not
//! embedded in any type exported by this module.

mod compact_verifier_circuit;
mod compat;
mod cpu;
mod engine;
mod folding;
mod lookup;
mod ram;
mod register;
mod types;

pub use compact_verifier_circuit::{BlockJoltVerifierStepCircuit, BLOCK_JOLT_VERIFIER_Z_ARITY};
pub use compat::d8_statement_adapter;
pub use cpu::{new_block_cpu_transcript, prove_block_cpu_r1cs, verify_block_cpu_r1cs};
pub use engine::{
    BlockJoltHostConfig, BlockJoltProver, BlockJoltVerifier, VerifiedBlockJoltTransition,
};
pub use folding::{fold_verified_block_jolt_transitions, BlockJoltNovaFoldingProof};
pub use lookup::{
    new_block_lookup_transcript, prove_block_lookup_lasso, verify_block_lookup_lasso,
};
pub use ram::{new_block_ram_transcript, prove_block_ram, verify_block_ram};
pub use register::{new_block_register_transcript, prove_block_register, verify_block_register};
pub use types::{
    deferred_claim_root, BlockBoundaryState, BlockJoltProof, BlockJoltStatement, BlockRelation,
    CompactSumcheckProof, CpuBlockRelationProof, DeferredPcsClaim, FieldElement, LookupBlockProof,
    RamBlockProof, RegisterBlockProof, StreamingRecursiveState, TranscriptCheckpoint,
    BLOCK_JOLT_PROTOCOL_VERSION, BLOCK_JOLT_WIRE_VERSION,
};
