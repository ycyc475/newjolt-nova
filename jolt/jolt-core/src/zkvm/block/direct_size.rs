//! Complete deterministic size accounting for the D8 direct proof artifact.
//!
//! The current Rust proof type intentionally retains native subclaims for
//! standalone verification and adversarial auditing. It does not yet expose a
//! stable transport serializer, so D8 defines one unambiguous accounting wire
//! model: integers are fixed-width little-endian, vectors have an eight-byte
//! length prefix, field/group/proof values use compressed Ark serialization,
//! and every verifier-consumed field is counted exactly once.

use ark_serialize::{CanonicalSerialize, Compress};
use serde::{Deserialize, Serialize};

use super::{
    DirectLookupSubclaim, DirectPcsStageProof, DirectRamSubclaim, DirectRegisterSubclaim,
    DirectRelation, DirectRelationState,
};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DirectProofSizeBreakdown {
    pub statement_and_execution_bytes: usize,
    pub lookup_subclaims_bytes: usize,
    pub register_subclaims_bytes: usize,
    pub ram_subclaims_bytes: usize,
    pub cpu_subclaims_bytes: usize,
    pub dory_pcs_bytes: usize,
    pub nova_recursive_debug_bytes: usize,
    pub spartan_compressed_bytes: usize,
    pub recursive_public_state_bytes: usize,
    pub total_bytes: usize,
}

impl DirectPcsStageProof {
    pub fn benchmark_size_breakdown(&self) -> DirectProofSizeBreakdown {
        let statement_and_execution_bytes = statement_size(self);
        let lookup_subclaims_bytes = vec_prefix()
            + self
                .lookup_subclaims
                .iter()
                .map(lookup_subclaim_size)
                .sum::<usize>();
        let register_subclaims_bytes = vec_prefix()
            + self
                .register_subclaims
                .iter()
                .map(register_subclaim_size)
                .sum::<usize>();
        let ram_subclaims_bytes = vec_prefix()
            + self
                .ram_subclaims
                .iter()
                .map(ram_subclaim_size)
                .sum::<usize>();
        let cpu_subclaims_bytes = vec_prefix()
            + self
                .cpu_subclaims
                .iter()
                .map(cpu_subclaim_size)
                .sum::<usize>();
        let dory_pcs_bytes = canonical_size(&self.pcs_commitment)
            + field_vec_size(&self.pcs_opening_point)
            + canonical_size(&self.pcs_opening_proof);
        let nova_recursive_debug_bytes = byte_vec_size(&self.nova_recursive_snark);
        let spartan_compressed_bytes = byte_vec_size(&self.spartan_proof);
        let recursive_public_state_bytes =
            field_storage_vec_size(&self.initial_z) + field_storage_vec_size(&self.final_z);
        let total_bytes = [
            statement_and_execution_bytes,
            lookup_subclaims_bytes,
            register_subclaims_bytes,
            ram_subclaims_bytes,
            cpu_subclaims_bytes,
            dory_pcs_bytes,
            nova_recursive_debug_bytes,
            spartan_compressed_bytes,
            recursive_public_state_bytes,
        ]
        .into_iter()
        .sum();
        DirectProofSizeBreakdown {
            statement_and_execution_bytes,
            lookup_subclaims_bytes,
            register_subclaims_bytes,
            ram_subclaims_bytes,
            cpu_subclaims_bytes,
            dory_pcs_bytes,
            nova_recursive_debug_bytes,
            spartan_compressed_bytes,
            recursive_public_state_bytes,
            total_bytes,
        }
    }
}

fn statement_size(proof: &DirectPcsStageProof) -> usize {
    let statement = &proof.statement;
    let cpu = &statement.cpu;
    let cpu_statement = 32
        + 32
        + field_size()
        + 2 * usize_size()
        + 2 * cpu.initial_registers.len() * u64_size()
        + u64_vec_size(&cpu.ram_addresses)
        + 4 * field_size()
        + bool_size()
        + relation_map_size(&cpu.relations);
    let outer_statement = cpu_statement
        + byte_vec_size(&statement.public_inputs)
        + byte_vec_size(&statement.public_outputs)
        + 32
        + bool_size()
        + usize_size()
        + field_size()
        + 32
        + u32_size()
        + relation_map_size(&statement.relations);
    let execution = byte_vec_size(&proof.execution.public_inputs)
        + byte_vec_size(&proof.execution.public_outputs)
        + byte_vec_size(&proof.execution.trusted_advice)
        + byte_vec_size(&proof.execution.untrusted_advice)
        + bool_size();
    outer_statement + execution
}

fn lookup_subclaim_size(subclaim: &DirectLookupSubclaim) -> usize {
    let block = &subclaim.block;
    let block_size = 4 * usize_size()
        + bool_size()
        + vec_prefix()
        + block
            .cycles
            .iter()
            .map(|_| {
                bool_size()
                    + u8_size()
                    + u128_size()
                    + u64_size()
                    + u128_size()
                    + u64_size()
                    + bool_size()
            })
            .sum::<usize>();
    block_size
        + field_size()
        + field_vec_size(&subclaim.r_reduction)
        + subclaim.input_claims.len() * field_size()
        + 4 * field_size()
        + canonical_size(&subclaim.proof)
        + field_vec_size(&subclaim.sumcheck_challenges)
        + usize_size()
        + field_vec_size(&subclaim.opening_claims)
        + 32
        + u32_size()
}

fn register_subclaim_size(subclaim: &DirectRegisterSubclaim) -> usize {
    let block = &subclaim.block;
    let cycle_size = bool_size()
        + 2 * (bool_size() + u8_size() + u64_size())
        + bool_size()
        + u8_size()
        + 2 * u64_size();
    let block_size = 4 * usize_size()
        + bool_size()
        + 2 * block.start_registers.len() * u64_size()
        + vec_prefix()
        + block.cycles.len() * cycle_size;
    block_size
        + field_size()
        + field_vec_size(&subclaim.r_reduction)
        + subclaim.input_claims.len() * field_size()
        + 4 * field_size()
        + canonical_size(&subclaim.proof)
        + field_vec_size(&subclaim.sumcheck_challenges)
        + usize_size()
        + subclaim.opening_claims.len() * field_size()
        + 32
        + u32_size()
}

fn ram_subclaim_size(subclaim: &DirectRamSubclaim) -> usize {
    let block = &subclaim.block;
    let cycle_bytes = block
        .cycles
        .iter()
        .map(|cycle| {
            bool_size()
                + u8_size()
                + 4 * u64_size()
                + field_vec_size(&cycle.registry_path)
                + field_vec_size(&cycle.memory_path)
                + 2 * field_size()
        })
        .sum::<usize>();
    let block_size = 6 * usize_size() + bool_size() + 4 * field_size() + vec_prefix() + cycle_bytes;
    block_size + 2 * (32 + u32_size())
}

fn cpu_subclaim_size(subclaim: &super::DirectCpuSubclaim) -> usize {
    let block = &subclaim.block;
    let cycle_bytes = block
        .cycles
        .iter()
        .map(|cycle| {
            bool_size()
                + cycle.inputs.len() * i128_size()
                + u16_size()
                + 7 * u8_size()
                + cycle.instruction_flags.len() * bool_size()
                + bool_size()
                + field_vec_size(&cycle.bytecode_path)
        })
        .sum::<usize>();
    let block_size = 5 * usize_size() + bool_size() + 3 * field_size() + vec_prefix() + cycle_bytes;
    block_size + 2 * (32 + u32_size())
}

fn relation_map_size(
    map: &std::collections::BTreeMap<DirectRelation, DirectRelationState>,
) -> usize {
    vec_prefix() + map.len() * 2
}

fn canonical_size(value: &impl CanonicalSerialize) -> usize {
    value.serialized_size(Compress::Yes)
}

fn field_size() -> usize {
    canonical_size(&ark_bn254::Fr::default())
}

fn field_vec_size(values: &[ark_bn254::Fr]) -> usize {
    vec_prefix() + values.len() * field_size()
}

fn field_storage_vec_size(values: &[[u8; 32]]) -> usize {
    vec_prefix() + values.len() * 32
}

fn u64_vec_size(values: &[u64]) -> usize {
    vec_prefix() + values.len() * u64_size()
}

fn byte_vec_size(values: &[u8]) -> usize {
    vec_prefix() + values.len()
}

const fn vec_prefix() -> usize {
    8
}

const fn usize_size() -> usize {
    8
}

const fn bool_size() -> usize {
    1
}

const fn u8_size() -> usize {
    1
}

const fn u16_size() -> usize {
    2
}

const fn u32_size() -> usize {
    4
}

const fn u64_size() -> usize {
    8
}

const fn u128_size() -> usize {
    16
}

const fn i128_size() -> usize {
    16
}
