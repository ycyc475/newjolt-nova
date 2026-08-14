//! Block-native authenticated RAM relation for the direct Jolt-Nova path.
//!
//! The recursive state carries two Merkle roots over the canonically sorted set
//! of touched RAM addresses: an immutable address-registry root and a mutable
//! value root. Every read proves membership of the current `(index, address,
//! value)` leaf. Every write verifies the old leaf and recomputes the new root
//! along the same path. This removes the Stage-19 `latest_ram_values` host trust
//! boundary while retaining `O(touched RAM)` prover memory and logarithmic
//! authentication paths. D5 binds the same address/read/write tuple to Jolt's
//! CPU/R1CS relation; D6 binds the initial value root to program/input PCS data.

use std::collections::BTreeMap;

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::Zero;
use nova_snark::{
    frontend::{
        gadgets::{boolean::AllocatedBit, num::AllocatedNum},
        ConstraintSystem, SynthesisError,
    },
    traits::circuit::StepCircuit,
};
use tracer::{instruction::RAMAccess, TraceBlock};

use crate::transcripts::{PoseidonTranscript, Transcript};

use super::{
    direct_lookup::{
        alloc_u64_bits, alloc_witness_num, allocated_poseidon_initial, bit_as_num,
        enforce_num_equal, field_from_digest, mul_nums, nova_from_fr, nova_to_storage,
        poseidon_absorb, scale_num, sub_nums,
    },
    direct_register::{
        direct_register_initial_z, register_stage_error, validate_combined_sequence,
        DirectRegisterStepCircuit, DIRECT_REGISTER_Z_ARITY,
    },
    DirectChunkedError, DirectChunkedPreprocessing, DirectLookupSubclaim, DirectRegisterSubclaim,
    DirectRelation, DirectRelationState, NovaPrimaryEngine, NovaPrimarySpartanSnark, NovaScalar,
    NovaSecondaryEngine, NovaSecondarySpartanSnark,
};

const DIRECT_RAM_REGISTRY_LEAF_DOMAIN: &[u8] = b"direct-ram-registry-leaf-v1";
const DIRECT_RAM_MEMORY_LEAF_DOMAIN: &[u8] = b"direct-ram-memory-leaf-v1";
const DIRECT_RAM_REGISTRY_PAD_DOMAIN: &[u8] = b"direct-ram-registry-pad-v1";
const DIRECT_RAM_MEMORY_PAD_DOMAIN: &[u8] = b"direct-ram-memory-pad-v1";
const DIRECT_RAM_NODE_DOMAIN: &[u8] = b"direct-ram-node-v1";
pub(super) const DIRECT_RAM_QUERY_DOMAIN: &[u8] = b"direct-ram-query-v1";
pub(super) const DIRECT_RAM_Z_ARITY: usize = DIRECT_REGISTER_Z_ARITY + 4;
pub(super) const RAM_REGISTRY_ROOT_SLOT: usize = DIRECT_REGISTER_Z_ARITY;
pub(super) const RAM_ROOT_SLOT: usize = RAM_REGISTRY_ROOT_SLOT + 1;
pub(super) const RAM_TRANSCRIPT_STATE_SLOT: usize = RAM_ROOT_SLOT + 1;
pub(super) const RAM_TRANSCRIPT_ROUND_SLOT: usize = RAM_ROOT_SLOT + 2;

type DirectRamNovaSnark =
    nova_snark::nova::RecursiveSNARK<NovaPrimaryEngine, NovaSecondaryEngine, DirectRamStepCircuit>;
type DirectRamCompressedSnark = nova_snark::nova::CompressedSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    DirectRamStepCircuit,
    NovaPrimarySpartanSnark,
    NovaSecondarySpartanSnark,
>;
type DirectRamPublicParams =
    nova_snark::nova::PublicParams<NovaPrimaryEngine, NovaSecondaryEngine, DirectRamStepCircuit>;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DirectRamCycleWitness {
    pub active: bool,
    /// 0 = no-op, 1 = read, 2 = write.
    pub kind: u8,
    pub address: u64,
    pub read_value: u64,
    pub write_value: u64,
    pub leaf_index: u64,
    /// Siblings from leaf level to root for the immutable address registry.
    pub registry_path: Vec<Fr>,
    /// Siblings from leaf level to root for the mutable memory tree.
    pub memory_path: Vec<Fr>,
    pub root_before: Fr,
    pub root_after: Fr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectRamBlockWitness {
    pub block_index: usize,
    pub global_cycle_start: usize,
    pub active_cycles: usize,
    pub cycle_capacity: usize,
    pub terminated: bool,
    pub address_count: usize,
    pub tree_depth: usize,
    pub registry_root: Fr,
    pub start_root: Fr,
    pub end_root: Fr,
    pub query_root: Fr,
    pub cycles: Vec<DirectRamCycleWitness>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectRamSubclaim {
    pub block: DirectRamBlockWitness,
    pub transcript_state_before: [u8; 32],
    pub transcript_round_before: u32,
    pub transcript_state_after: [u8; 32],
    pub transcript_round_after: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectRamStageStatement {
    pub program_digest: [u8; 32],
    pub lookup_table_commitment: [u8; 32],
    pub block_count: usize,
    pub total_cycles: usize,
    pub initial_registers: [u64; common::constants::REGISTER_COUNT as usize],
    pub final_registers: [u64; common::constants::REGISTER_COUNT as usize],
    pub ram_addresses: Vec<u64>,
    pub ram_registry_root: Fr,
    pub initial_ram_root: Fr,
    pub final_ram_root: Fr,
    pub terminated: bool,
    pub relations: BTreeMap<DirectRelation, DirectRelationState>,
}

#[derive(Clone)]
pub struct DirectRamStageProof {
    pub statement: DirectRamStageStatement,
    pub lookup_subclaims: Vec<DirectLookupSubclaim>,
    pub register_subclaims: Vec<DirectRegisterSubclaim>,
    pub ram_subclaims: Vec<DirectRamSubclaim>,
    pub nova_recursive_snark: Vec<u8>,
    pub spartan_proof: Vec<u8>,
    pub initial_z: Vec<[u8; 32]>,
    pub final_z: Vec<[u8; 32]>,
}

fn ram_hash(domain: &'static [u8], values: &[Fr]) -> Fr {
    let mut transcript = PoseidonTranscript::new(domain);
    for value in values {
        transcript.append_scalar(b"ram_hash_word", value);
    }
    Fr::from_le_bytes_mod_order(&transcript.state)
}

fn registry_leaf(index: usize, address: u64) -> Fr {
    ram_hash(
        DIRECT_RAM_REGISTRY_LEAF_DOMAIN,
        &[Fr::from(index as u64), Fr::from(address)],
    )
}

fn memory_leaf(index: usize, address: u64, value: u64) -> Fr {
    ram_hash(
        DIRECT_RAM_MEMORY_LEAF_DOMAIN,
        &[Fr::from(index as u64), Fr::from(address), Fr::from(value)],
    )
}

fn registry_pad_leaf(index: usize) -> Fr {
    ram_hash(DIRECT_RAM_REGISTRY_PAD_DOMAIN, &[Fr::from(index as u64)])
}

fn memory_pad_leaf(index: usize) -> Fr {
    ram_hash(DIRECT_RAM_MEMORY_PAD_DOMAIN, &[Fr::from(index as u64)])
}

fn ram_node(left: Fr, right: Fr) -> Fr {
    ram_hash(DIRECT_RAM_NODE_DOMAIN, &[left, right])
}

#[derive(Clone)]
struct RamMerkleTree {
    levels: Vec<Vec<Fr>>,
}

impl RamMerkleTree {
    fn new(leaves: Vec<Fr>) -> Self {
        debug_assert!(!leaves.is_empty() && leaves.len().is_power_of_two());
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let next = levels
                .last()
                .unwrap()
                .chunks_exact(2)
                .map(|pair| ram_node(pair[0], pair[1]))
                .collect();
            levels.push(next);
        }
        Self { levels }
    }

    fn root(&self) -> Fr {
        self.levels.last().unwrap()[0]
    }

    fn depth(&self) -> usize {
        self.levels.len() - 1
    }

    fn authentication_path(&self, mut index: usize) -> Vec<Fr> {
        let mut path = Vec::with_capacity(self.depth());
        for level in &self.levels[..self.depth()] {
            path.push(level[index ^ 1]);
            index >>= 1;
        }
        path
    }

    fn update(&mut self, mut index: usize, leaf: Fr) {
        self.levels[0][index] = leaf;
        for level in 0..self.depth() {
            let parent = index >> 1;
            self.levels[level + 1][parent] = ram_node(
                self.levels[level][parent << 1],
                self.levels[level][(parent << 1) | 1],
            );
            index = parent;
        }
    }
}

fn tree_size(address_count: usize) -> usize {
    address_count.max(1).next_power_of_two()
}

fn build_ram_trees(addresses: &[u64], values: &[u64]) -> (RamMerkleTree, RamMerkleTree) {
    debug_assert_eq!(addresses.len(), values.len());
    let size = tree_size(addresses.len());
    let registry_leaves = (0..size)
        .map(|index| {
            addresses.get(index).map_or_else(
                || registry_pad_leaf(index),
                |address| registry_leaf(index, *address),
            )
        })
        .collect();
    let memory_leaves = (0..size)
        .map(|index| {
            addresses.get(index).map_or_else(
                || memory_pad_leaf(index),
                |address| memory_leaf(index, *address, values[index]),
            )
        })
        .collect();
    (
        RamMerkleTree::new(registry_leaves),
        RamMerkleTree::new(memory_leaves),
    )
}

pub(super) fn ram_memory_root(addresses: &[u64], values: &[u64]) -> Fr {
    build_ram_trees(addresses, values).1.root()
}

pub(super) fn ram_registry_root(addresses: &[u64]) -> Fr {
    build_ram_trees(addresses, &vec![0; addresses.len()])
        .0
        .root()
}

fn initial_ram_table(blocks: &[TraceBlock]) -> (Vec<u64>, Vec<u64>) {
    let mut values = BTreeMap::new();
    for block in blocks {
        for cycle in &block.cycles {
            match cycle.ram_access() {
                RAMAccess::Read(read) => {
                    values.entry(read.address).or_insert(read.value);
                }
                RAMAccess::Write(write) => {
                    values.entry(write.address).or_insert(write.pre_value);
                }
                RAMAccess::NoOp => {}
            }
        }
    }
    values.into_iter().unzip()
}

pub(super) fn validate_ram_address(
    preprocessing: &DirectChunkedPreprocessing,
    block_index: usize,
    row: usize,
    address: u64,
) -> Result<(), DirectChunkedError> {
    let lowest = preprocessing.memory_layout.get_lowest_address();
    let highest = preprocessing.memory_layout.heap_end;
    if address < lowest || address >= highest {
        return Err(DirectChunkedError::InvalidBlock {
            block_index,
            reason: format!(
                "RAM access at row {row} uses address {address:#x} outside the canonical Jolt memory interval [{lowest:#x}, {highest:#x})"
            ),
        });
    }
    if (address - lowest) % 8 != 0 {
        return Err(DirectChunkedError::InvalidBlock {
            block_index,
            reason: format!(
                "RAM access at row {row} uses non-canonical address {address:#x}; final Jolt RAM rows must be 8-byte aligned relative to {lowest:#x}"
            ),
        });
    }
    Ok(())
}

fn validate_trace_ram_addresses(
    preprocessing: &DirectChunkedPreprocessing,
    blocks: &[TraceBlock],
) -> Result<(), DirectChunkedError> {
    for block in blocks {
        for (row, cycle) in block.cycles.iter().enumerate() {
            let address = match cycle.ram_access() {
                RAMAccess::Read(read) => read.address,
                RAMAccess::Write(write) => write.address,
                RAMAccess::NoOp => continue,
            };
            validate_ram_address(preprocessing, block.block_index, row, address)?;
        }
    }
    Ok(())
}

pub(super) struct DirectRamStreamBuilder {
    addresses: Vec<u64>,
    address_indices: BTreeMap<u64, usize>,
    memory: Vec<u64>,
    registry_tree: RamMerkleTree,
    memory_tree: RamMerkleTree,
    initial_root: Fr,
    prover_transcript: PoseidonTranscript,
    verifier_transcript: PoseidonTranscript,
}

impl DirectRamStreamBuilder {
    pub(super) fn new(initial_ram: BTreeMap<u64, u64>) -> Self {
        let (addresses, memory): (Vec<_>, Vec<_>) = initial_ram.into_iter().unzip();
        let address_indices = addresses
            .iter()
            .enumerate()
            .map(|(index, address)| (*address, index))
            .collect();
        let (registry_tree, memory_tree) = build_ram_trees(&addresses, &memory);
        let initial_root = memory_tree.root();
        Self {
            addresses,
            address_indices,
            memory,
            registry_tree,
            memory_tree,
            initial_root,
            prover_transcript: PoseidonTranscript::new(DIRECT_RAM_QUERY_DOMAIN),
            verifier_transcript: PoseidonTranscript::new(DIRECT_RAM_QUERY_DOMAIN),
        }
    }

    pub(super) fn derive_block(
        &mut self,
        block: &TraceBlock,
        capacity: usize,
    ) -> Result<DirectRamSubclaim, DirectChunkedError> {
        let subclaim = derive_ram_subclaim(
            block,
            capacity,
            &self.addresses,
            &self.address_indices,
            &mut self.memory,
            &self.registry_tree,
            &mut self.memory_tree,
            &mut self.prover_transcript,
        )?;
        verify_ram_subclaim(&subclaim, &mut self.verifier_transcript)?;
        Ok(subclaim)
    }

    pub(super) fn into_summary(self) -> (Vec<u64>, Fr, Fr, Fr) {
        (
            self.addresses,
            self.registry_tree.root(),
            self.initial_root,
            self.memory_tree.root(),
        )
    }
}

fn merkle_root_from_path(mut leaf: Fr, mut index: usize, path: &[Fr]) -> Fr {
    for sibling in path {
        leaf = if index & 1 == 0 {
            ram_node(leaf, *sibling)
        } else {
            ram_node(*sibling, leaf)
        };
        index >>= 1;
    }
    leaf
}

fn append_ram_cycle(transcript: &mut PoseidonTranscript, cycle: &DirectRamCycleWitness) {
    for value in [
        Fr::from(u64::from(cycle.active)),
        Fr::from(cycle.kind as u64),
        Fr::from(cycle.address),
        Fr::from(cycle.read_value),
        Fr::from(cycle.write_value),
        Fr::from(cycle.leaf_index),
        cycle.root_before,
        cycle.root_after,
    ] {
        transcript.append_scalar(b"ram_word", &value);
    }
    for sibling in &cycle.registry_path {
        transcript.append_scalar(b"ram_registry_sibling", sibling);
    }
    for sibling in &cycle.memory_path {
        transcript.append_scalar(b"ram_memory_sibling", sibling);
    }
}

fn ram_query_root(block: &DirectRamBlockWitness) -> Fr {
    let mut transcript = PoseidonTranscript::new(DIRECT_RAM_QUERY_DOMAIN);
    transcript.append_scalar(b"block_index", &Fr::from(block.block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(block.global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(block.active_cycles as u64));
    transcript.append_scalar(b"address_count", &Fr::from(block.address_count as u64));
    transcript.append_scalar(b"tree_depth", &Fr::from(block.tree_depth as u64));
    transcript.append_scalar(b"registry_root", &block.registry_root);
    for cycle in &block.cycles {
        append_ram_cycle(&mut transcript, cycle);
    }
    Fr::from_le_bytes_mod_order(&transcript.state)
}

fn derive_ram_subclaim(
    block: &TraceBlock,
    capacity: usize,
    addresses: &[u64],
    address_indices: &BTreeMap<u64, usize>,
    memory: &mut [u64],
    registry_tree: &RamMerkleTree,
    memory_tree: &mut RamMerkleTree,
    transcript: &mut PoseidonTranscript,
) -> Result<DirectRamSubclaim, DirectChunkedError> {
    super::direct::validate_block(block, capacity)?;
    let state_before = transcript.state;
    let round_before = transcript.n_rounds;
    let start_root = memory_tree.root();
    let depth = memory_tree.depth();
    let mut cycles = Vec::with_capacity(capacity);
    for (row, cycle) in block.cycles.iter().enumerate() {
        let before = memory_tree.root();
        let witness = match cycle.ram_access() {
            RAMAccess::NoOp => DirectRamCycleWitness {
                active: true,
                registry_path: vec![Fr::zero(); depth],
                memory_path: vec![Fr::zero(); depth],
                root_before: before,
                root_after: before,
                ..Default::default()
            },
            RAMAccess::Read(read) => {
                let index = address_indices[&read.address];
                if memory[index] != read.value {
                    return Err(DirectChunkedError::InvalidBlock {
                        block_index: block.block_index,
                        reason: format!(
                            "RAM read at row {row}, address {:#x} has value {}, expected {}",
                            read.address, read.value, memory[index]
                        ),
                    });
                }
                DirectRamCycleWitness {
                    active: true,
                    kind: 1,
                    address: read.address,
                    read_value: read.value,
                    write_value: read.value,
                    leaf_index: index as u64,
                    registry_path: registry_tree.authentication_path(index),
                    memory_path: memory_tree.authentication_path(index),
                    root_before: before,
                    root_after: before,
                }
            }
            RAMAccess::Write(write) => {
                let index = address_indices[&write.address];
                if memory[index] != write.pre_value {
                    return Err(DirectChunkedError::InvalidBlock {
                        block_index: block.block_index,
                        reason: format!(
                            "RAM write at row {row}, address {:#x} has pre-value {}, expected {}",
                            write.address, write.pre_value, memory[index]
                        ),
                    });
                }
                let registry_path = registry_tree.authentication_path(index);
                let memory_path = memory_tree.authentication_path(index);
                memory_tree.update(index, memory_leaf(index, write.address, write.post_value));
                memory[index] = write.post_value;
                DirectRamCycleWitness {
                    active: true,
                    kind: 2,
                    address: write.address,
                    read_value: write.pre_value,
                    write_value: write.post_value,
                    leaf_index: index as u64,
                    registry_path,
                    memory_path,
                    root_before: before,
                    root_after: memory_tree.root(),
                }
            }
        };
        cycles.push(witness);
    }
    cycles.resize_with(capacity, || DirectRamCycleWitness {
        registry_path: vec![Fr::zero(); depth],
        memory_path: vec![Fr::zero(); depth],
        root_before: memory_tree.root(),
        root_after: memory_tree.root(),
        ..Default::default()
    });
    let mut block_witness = DirectRamBlockWitness {
        block_index: block.block_index,
        global_cycle_start: block.global_cycle_start,
        active_cycles: block.active_cycles,
        cycle_capacity: capacity,
        terminated: block.end_state.terminated,
        address_count: addresses.len(),
        tree_depth: depth,
        registry_root: registry_tree.root(),
        start_root,
        end_root: memory_tree.root(),
        query_root: Fr::zero(),
        cycles,
    };
    block_witness.query_root = ram_query_root(&block_witness);
    transcript.append_scalar(b"block_index", &Fr::from(block.block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(block.global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(block.active_cycles as u64));
    transcript.append_scalar(b"address_count", &Fr::from(addresses.len() as u64));
    transcript.append_scalar(b"tree_depth", &Fr::from(depth as u64));
    transcript.append_scalar(b"registry_root", &registry_tree.root());
    transcript.append_scalar(b"ram_query_root", &block_witness.query_root);
    transcript.append_scalar(b"ram_start_root", &block_witness.start_root);
    transcript.append_scalar(b"ram_end_root", &block_witness.end_root);
    Ok(DirectRamSubclaim {
        block: block_witness,
        transcript_state_before: state_before,
        transcript_round_before: round_before,
        transcript_state_after: transcript.state,
        transcript_round_after: transcript.n_rounds,
    })
}

pub(super) fn verify_ram_subclaim(
    subclaim: &DirectRamSubclaim,
    transcript: &mut PoseidonTranscript,
) -> Result<(), DirectChunkedError> {
    let block = &subclaim.block;
    if subclaim.transcript_state_before != transcript.state
        || subclaim.transcript_round_before != transcript.n_rounds
        || block.cycles.len() != block.cycle_capacity
        || block.active_cycles == 0
        || block.active_cycles > block.cycle_capacity
        || block.tree_depth != tree_size(block.address_count).ilog2() as usize
        || block.query_root != ram_query_root(block)
    {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: block.block_index,
            reason: "RAM subclaim header, shape, or query commitment is invalid".to_string(),
        });
    }
    let mut root = block.start_root;
    for (row, cycle) in block.cycles.iter().enumerate() {
        if cycle.active != (row < block.active_cycles) || cycle.root_before != root {
            return Err(DirectChunkedError::InvalidBlock {
                block_index: block.block_index,
                reason: format!("RAM active prefix or root chain fails at row {row}"),
            });
        }
        match cycle.kind {
            0 if cycle.address == 0
                && cycle.read_value == 0
                && cycle.write_value == 0
                && cycle.leaf_index == 0
                && cycle.registry_path.len() == block.tree_depth
                && cycle.memory_path.len() == block.tree_depth
                && cycle.registry_path.iter().all(Fr::is_zero)
                && cycle.memory_path.iter().all(Fr::is_zero)
                && cycle.root_after == root => {}
            1 if cycle.active
                && (cycle.leaf_index as usize) < block.address_count
                && cycle.read_value == cycle.write_value
                && cycle.registry_path.len() == block.tree_depth
                && cycle.memory_path.len() == block.tree_depth
                && merkle_root_from_path(
                    registry_leaf(cycle.leaf_index as usize, cycle.address),
                    cycle.leaf_index as usize,
                    &cycle.registry_path,
                ) == block.registry_root
                && merkle_root_from_path(
                    memory_leaf(cycle.leaf_index as usize, cycle.address, cycle.read_value),
                    cycle.leaf_index as usize,
                    &cycle.memory_path,
                ) == root
                && cycle.root_after == root => {}
            2 if cycle.active
                && (cycle.leaf_index as usize) < block.address_count
                && cycle.registry_path.len() == block.tree_depth
                && cycle.memory_path.len() == block.tree_depth
                && merkle_root_from_path(
                    registry_leaf(cycle.leaf_index as usize, cycle.address),
                    cycle.leaf_index as usize,
                    &cycle.registry_path,
                ) == block.registry_root
                && merkle_root_from_path(
                    memory_leaf(cycle.leaf_index as usize, cycle.address, cycle.read_value),
                    cycle.leaf_index as usize,
                    &cycle.memory_path,
                ) == root =>
            {
                root = merkle_root_from_path(
                    memory_leaf(cycle.leaf_index as usize, cycle.address, cycle.write_value),
                    cycle.leaf_index as usize,
                    &cycle.memory_path,
                );
                if cycle.root_after != root {
                    return Err(DirectChunkedError::InvalidBlock {
                        block_index: block.block_index,
                        reason: format!("RAM write root transition fails at row {row}"),
                    });
                }
            }
            _ => {
                return Err(DirectChunkedError::InvalidBlock {
                    block_index: block.block_index,
                    reason: format!("RAM cycle encoding is invalid at row {row}"),
                });
            }
        }
        root = cycle.root_after;
    }
    if root != block.end_root {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: block.block_index,
            reason: "RAM final root does not close".to_string(),
        });
    }
    transcript.append_scalar(b"block_index", &Fr::from(block.block_index as u64));
    transcript.append_scalar(b"cycle_start", &Fr::from(block.global_cycle_start as u64));
    transcript.append_scalar(b"active_cycles", &Fr::from(block.active_cycles as u64));
    transcript.append_scalar(b"address_count", &Fr::from(block.address_count as u64));
    transcript.append_scalar(b"tree_depth", &Fr::from(block.tree_depth as u64));
    transcript.append_scalar(b"registry_root", &block.registry_root);
    transcript.append_scalar(b"ram_query_root", &block.query_root);
    transcript.append_scalar(b"ram_start_root", &block.start_root);
    transcript.append_scalar(b"ram_end_root", &block.end_root);
    if subclaim.transcript_state_after != transcript.state
        || subclaim.transcript_round_after != transcript.n_rounds
    {
        return Err(DirectChunkedError::InvalidBlock {
            block_index: block.block_index,
            reason: "RAM transcript binding is invalid".to_string(),
        });
    }
    Ok(())
}

pub(super) struct AllocatedRamCycle {
    pub(super) active: AllocatedBit,
    pub(super) is_read: AllocatedBit,
    pub(super) is_write: AllocatedBit,
    pub(super) address: AllocatedNum<NovaScalar>,
    pub(super) read_value: AllocatedNum<NovaScalar>,
    pub(super) write_value: AllocatedNum<NovaScalar>,
    pub(super) leaf_index: AllocatedNum<NovaScalar>,
    pub(super) leaf_index_bits: Vec<AllocatedBit>,
    pub(super) registry_path: Vec<AllocatedNum<NovaScalar>>,
    pub(super) memory_path: Vec<AllocatedNum<NovaScalar>>,
    pub(super) root_before: AllocatedNum<NovaScalar>,
    pub(super) root_after: AllocatedNum<NovaScalar>,
}

fn synthesize_hash<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    domain: &'static [u8],
    values: &[&AllocatedNum<NovaScalar>],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let mut state = allocated_poseidon_initial(cs.namespace(|| "RAM hash initial"), domain)?;
    for (index, value) in values.iter().enumerate() {
        state = poseidon_absorb(
            cs.namespace(|| format!("RAM hash word {index}")),
            &state,
            value,
            "ram_hash_word",
        )?;
    }
    Ok(state.state)
}

fn select_num<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    condition: &AllocatedBit,
    when_true: &AllocatedNum<NovaScalar>,
    when_false: &AllocatedNum<NovaScalar>,
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let delta = sub_nums(
        cs.namespace(|| format!("{label} delta")),
        when_true,
        when_false,
        label,
    )?;
    let selected_delta = mul_nums(
        cs.namespace(|| format!("{label} selected delta")),
        &bit_as_num(condition),
        &delta,
        label,
    )?;
    super::direct_lookup::add_nums(
        cs.namespace(|| format!("{label} output")),
        when_false,
        &selected_delta,
        label,
    )
}

fn synthesize_merkle_root<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    leaf: &AllocatedNum<NovaScalar>,
    index_bits: &[AllocatedBit],
    path: &[AllocatedNum<NovaScalar>],
    label: &'static str,
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    if path.len() > index_bits.len() {
        return Err(SynthesisError::Unsatisfiable(format!(
            "{label} path exceeds index width"
        )));
    }
    let mut node = leaf.clone();
    for (level, (bit, sibling)) in index_bits.iter().zip(path).enumerate() {
        let left = select_num(
            cs.namespace(|| format!("{label} left {level}")),
            bit,
            sibling,
            &node,
            "RAM Merkle left selection",
        )?;
        let right = select_num(
            cs.namespace(|| format!("{label} right {level}")),
            bit,
            &node,
            sibling,
            "RAM Merkle right selection",
        )?;
        node = synthesize_hash(
            cs.namespace(|| format!("{label} node {level}")),
            DIRECT_RAM_NODE_DOMAIN,
            &[&left, &right],
        )?;
    }
    Ok(node)
}

fn enforce_conditional_equal<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    enabled: &AllocatedNum<NovaScalar>,
    left: &AllocatedNum<NovaScalar>,
    right: &AllocatedNum<NovaScalar>,
    label: &'static str,
) {
    cs.enforce(
        || label,
        |lc| lc + enabled.get_variable(),
        |lc| lc + left.get_variable() - right.get_variable(),
        |lc| lc,
    );
}

fn allocate_ram_cycle<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    cycle: &DirectRamCycleWitness,
) -> Result<AllocatedRamCycle, SynthesisError> {
    let active = AllocatedBit::alloc(cs.namespace(|| "active"), Some(cycle.active))?;
    let is_read = AllocatedBit::alloc(cs.namespace(|| "read"), Some(cycle.kind == 1))?;
    let is_write = AllocatedBit::alloc(cs.namespace(|| "write"), Some(cycle.kind == 2))?;
    let (address, _) = alloc_u64_bits(cs.namespace(|| "address"), cycle.address, "RAM address")?;
    let (read_value, _) = alloc_u64_bits(
        cs.namespace(|| "read value"),
        cycle.read_value,
        "RAM read value",
    )?;
    let (write_value, _) = alloc_u64_bits(
        cs.namespace(|| "write value"),
        cycle.write_value,
        "RAM write value",
    )?;
    let (leaf_index, leaf_index_bits) = alloc_u64_bits(
        cs.namespace(|| "leaf index"),
        cycle.leaf_index,
        "RAM leaf index",
    )?;
    let registry_path = cycle
        .registry_path
        .iter()
        .enumerate()
        .map(|(level, sibling)| {
            alloc_witness_num(
                cs.namespace(|| format!("registry sibling {level}")),
                nova_from_fr(sibling)?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let memory_path = cycle
        .memory_path
        .iter()
        .enumerate()
        .map(|(level, sibling)| {
            alloc_witness_num(
                cs.namespace(|| format!("memory sibling {level}")),
                nova_from_fr(sibling)?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let root_before = alloc_witness_num(
        cs.namespace(|| "root before"),
        nova_from_fr(&cycle.root_before)?,
    )?;
    let root_after = alloc_witness_num(
        cs.namespace(|| "root after"),
        nova_from_fr(&cycle.root_after)?,
    )?;
    cs.enforce(
        || "RAM kind is at most one access",
        |lc| lc + is_read.get_variable(),
        |lc| lc + is_write.get_variable(),
        |lc| lc,
    );
    cs.enforce(
        || "RAM access requires active row",
        |lc| lc + is_read.get_variable() + is_write.get_variable(),
        |lc| lc + CS::one() - active.get_variable(),
        |lc| lc,
    );
    cs.enforce(
        || "RAM disabled address zero",
        |lc| lc + CS::one() - is_read.get_variable() - is_write.get_variable(),
        |lc| lc + address.get_variable(),
        |lc| lc,
    );
    cs.enforce(
        || "RAM disabled read zero",
        |lc| lc + CS::one() - is_read.get_variable() - is_write.get_variable(),
        |lc| lc + read_value.get_variable(),
        |lc| lc,
    );
    cs.enforce(
        || "RAM disabled write zero",
        |lc| lc + CS::one() - is_read.get_variable() - is_write.get_variable(),
        |lc| lc + write_value.get_variable(),
        |lc| lc,
    );
    cs.enforce(
        || "RAM disabled leaf index zero",
        |lc| lc + CS::one() - is_read.get_variable() - is_write.get_variable(),
        |lc| lc + leaf_index.get_variable(),
        |lc| lc,
    );
    let enabled = super::direct_lookup::add_nums(
        cs.namespace(|| "RAM access enabled"),
        &bit_as_num(&is_read),
        &bit_as_num(&is_write),
        "RAM access enabled",
    )?;
    for (level, sibling) in registry_path.iter().enumerate() {
        cs.enforce(
            || format!("RAM disabled registry sibling zero {level}"),
            |lc| lc + CS::one() - enabled.get_variable(),
            |lc| lc + sibling.get_variable(),
            |lc| lc,
        );
    }
    for (level, sibling) in memory_path.iter().enumerate() {
        cs.enforce(
            || format!("RAM disabled memory sibling zero {level}"),
            |lc| lc + CS::one() - enabled.get_variable(),
            |lc| lc + sibling.get_variable(),
            |lc| lc,
        );
    }
    cs.enforce(
        || "RAM read preserves value",
        |lc| lc + is_read.get_variable(),
        |lc| lc + read_value.get_variable() - write_value.get_variable(),
        |lc| lc,
    );
    Ok(AllocatedRamCycle {
        active,
        is_read,
        is_write,
        address,
        read_value,
        write_value,
        leaf_index,
        leaf_index_bits,
        registry_path,
        memory_path,
        root_before,
        root_after,
    })
}

fn synthesize_ram_query_root<CS: ConstraintSystem<NovaScalar>>(
    mut cs: CS,
    block_index: &AllocatedNum<NovaScalar>,
    cycle_start: &AllocatedNum<NovaScalar>,
    active_cycles: &AllocatedNum<NovaScalar>,
    address_count: &AllocatedNum<NovaScalar>,
    tree_depth: &AllocatedNum<NovaScalar>,
    registry_root: &AllocatedNum<NovaScalar>,
    cycles: &[AllocatedRamCycle],
) -> Result<AllocatedNum<NovaScalar>, SynthesisError> {
    let mut state = allocated_poseidon_initial(
        cs.namespace(|| "RAM query initial"),
        DIRECT_RAM_QUERY_DOMAIN,
    )?;
    for (label, value) in [
        ("block_index", block_index),
        ("cycle_start", cycle_start),
        ("active_cycles", active_cycles),
        ("address_count", address_count),
        ("tree_depth", tree_depth),
        ("registry_root", registry_root),
    ] {
        state = poseidon_absorb(cs.namespace(|| label), &state, value, label)?;
    }
    for (row, cycle) in cycles.iter().enumerate() {
        let active = bit_as_num(&cycle.active);
        let is_read = bit_as_num(&cycle.is_read);
        let is_write = bit_as_num(&cycle.is_write);
        let kind = scale_num(
            cs.namespace(|| format!("RAM kind write {row}")),
            &is_write,
            NovaScalar::from(2u64),
            "RAM kind write",
        )?;
        let kind = super::direct_lookup::add_nums(
            cs.namespace(|| format!("RAM kind {row}")),
            &is_read,
            &kind,
            "RAM kind",
        )?;
        for (word, value) in [
            &active,
            &kind,
            &cycle.address,
            &cycle.read_value,
            &cycle.write_value,
            &cycle.leaf_index,
            &cycle.root_before,
            &cycle.root_after,
        ]
        .into_iter()
        .enumerate()
        {
            state = poseidon_absorb(
                cs.namespace(|| format!("RAM row {row} word {word}")),
                &state,
                value,
                "ram_word",
            )?;
        }
        for (level, sibling) in cycle.registry_path.iter().enumerate() {
            state = poseidon_absorb(
                cs.namespace(|| format!("RAM registry sibling {row} {level}")),
                &state,
                sibling,
                "ram_registry_sibling",
            )?;
        }
        for (level, sibling) in cycle.memory_path.iter().enumerate() {
            state = poseidon_absorb(
                cs.namespace(|| format!("RAM memory sibling {row} {level}")),
                &state,
                sibling,
                "ram_memory_sibling",
            )?;
        }
    }
    Ok(state.state)
}

#[derive(Clone, Default)]
pub(super) struct DirectRamStepCircuit {
    lookup: Option<DirectLookupSubclaim>,
    register: Option<DirectRegisterSubclaim>,
    ram: Option<DirectRamSubclaim>,
    final_step: bool,
}

impl DirectRamStepCircuit {
    pub(super) fn for_subclaims(
        lookup: DirectLookupSubclaim,
        register: DirectRegisterSubclaim,
        ram: DirectRamSubclaim,
        final_step: bool,
    ) -> Self {
        Self {
            lookup: Some(lookup),
            register: Some(register),
            ram: Some(ram),
            final_step,
        }
    }
}

impl DirectRamStepCircuit {
    pub(super) fn synthesize_with_observations<CS: ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<NovaScalar>],
    ) -> Result<
        (
            Vec<AllocatedNum<NovaScalar>>,
            Vec<super::direct_lookup::AllocatedDirectLookupCycle>,
            Vec<super::direct_register::AllocatedRegisterCycle>,
            Vec<AllocatedRamCycle>,
        ),
        SynthesisError,
    > {
        if z.len() != DIRECT_RAM_Z_ARITY {
            return Err(SynthesisError::Unsatisfiable(
                "direct RAM state arity".to_string(),
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
        let ram = self.ram.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
        let block = &ram.block;
        if lookup.block.block_index != block.block_index
            || register.block.block_index != block.block_index
            || lookup.block.global_cycle_start != block.global_cycle_start
            || register.block.global_cycle_start != block.global_cycle_start
            || lookup.block.active_cycles != block.active_cycles
            || register.block.active_cycles != block.active_cycles
            || lookup.block.cycle_capacity != block.cycle_capacity
            || register.block.cycle_capacity != block.cycle_capacity
            || lookup.block.terminated != block.terminated
            || register.block.terminated != block.terminated
        {
            return Err(SynthesisError::Unsatisfiable(
                "D2-D3-D4 block metadata mismatch".to_string(),
            ));
        }
        let register_circuit = DirectRegisterStepCircuit::for_subclaims(
            lookup.clone(),
            register.clone(),
            self.final_step,
        );
        let (register_output, lookup_cycles, register_cycles) = {
            let mut namespace = cs.namespace(|| "D2+D3 lookup-register relation");
            register_circuit
                .synthesize_with_observations(&mut namespace, &z[..DIRECT_REGISTER_Z_ARITY])?
        };
        let block_index = alloc_witness_num(
            cs.namespace(|| "RAM block index"),
            NovaScalar::from(block.block_index as u64),
        )?;
        let cycle_start = alloc_witness_num(
            cs.namespace(|| "RAM cycle start"),
            NovaScalar::from(block.global_cycle_start as u64),
        )?;
        let active_cycles = alloc_witness_num(
            cs.namespace(|| "RAM active cycles"),
            NovaScalar::from(block.active_cycles as u64),
        )?;
        let address_count = alloc_witness_num(
            cs.namespace(|| "RAM address count"),
            NovaScalar::from(block.address_count as u64),
        )?;
        let tree_depth = alloc_witness_num(
            cs.namespace(|| "RAM tree depth"),
            NovaScalar::from(block.tree_depth as u64),
        )?;
        let registry_root = alloc_witness_num(
            cs.namespace(|| "RAM registry root"),
            nova_from_fr(&block.registry_root)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "RAM block index state"),
            &block_index,
            &z[0],
            "RAM block index state",
        );
        enforce_num_equal(
            cs.namespace(|| "RAM cycle start state"),
            &cycle_start,
            &z[2],
            "RAM cycle start state",
        );
        let cycles = block
            .cycles
            .iter()
            .enumerate()
            .map(|(row, cycle)| {
                allocate_ram_cycle(cs.namespace(|| format!("RAM cycle {row}")), cycle)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if cycles.is_empty()
            || cycles.iter().any(|cycle| {
                cycle.registry_path.len() != block.tree_depth
                    || cycle.memory_path.len() != block.tree_depth
            })
        {
            return Err(SynthesisError::Unsatisfiable(
                "RAM Merkle paths have non-canonical depth".to_string(),
            ));
        }
        cs.enforce(
            || "first RAM row is active",
            |lc| lc + cycles[0].active.get_variable() - CS::one(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        for row in 1..cycles.len() {
            cs.enforce(
                || format!("RAM active rows form prefix {row}"),
                |lc| lc + cycles[row].active.get_variable(),
                |lc| lc + CS::one() - cycles[row - 1].active.get_variable(),
                |lc| lc,
            );
        }
        let active_sum = cycles.iter().fold(
            nova_snark::frontend::LinearCombination::<NovaScalar>::zero(),
            |lc, cycle| lc + cycle.active.get_variable(),
        );
        cs.enforce(
            || "RAM active row count",
            |_| active_sum - active_cycles.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
        enforce_num_equal(
            cs.namespace(|| "RAM registry public state"),
            &registry_root,
            &z[RAM_REGISTRY_ROOT_SLOT],
            "RAM registry public state",
        );
        let mut running_root = z[RAM_ROOT_SLOT].clone();
        for (row, cycle) in cycles.iter().enumerate() {
            enforce_num_equal(
                cs.namespace(|| format!("RAM root before {row}")),
                &cycle.root_before,
                &running_root,
                "RAM root before",
            );
            let registry_leaf = synthesize_hash(
                cs.namespace(|| format!("RAM registry leaf {row}")),
                DIRECT_RAM_REGISTRY_LEAF_DOMAIN,
                &[&cycle.leaf_index, &cycle.address],
            )?;
            let old_leaf = synthesize_hash(
                cs.namespace(|| format!("RAM old leaf {row}")),
                DIRECT_RAM_MEMORY_LEAF_DOMAIN,
                &[&cycle.leaf_index, &cycle.address, &cycle.read_value],
            )?;
            let new_leaf = synthesize_hash(
                cs.namespace(|| format!("RAM new leaf {row}")),
                DIRECT_RAM_MEMORY_LEAF_DOMAIN,
                &[&cycle.leaf_index, &cycle.address, &cycle.write_value],
            )?;
            let computed_registry_root = synthesize_merkle_root(
                cs.namespace(|| format!("RAM registry path {row}")),
                &registry_leaf,
                &cycle.leaf_index_bits,
                &cycle.registry_path,
                "RAM registry path",
            )?;
            let computed_old_root = synthesize_merkle_root(
                cs.namespace(|| format!("RAM old memory path {row}")),
                &old_leaf,
                &cycle.leaf_index_bits,
                &cycle.memory_path,
                "RAM old memory path",
            )?;
            let computed_new_root = synthesize_merkle_root(
                cs.namespace(|| format!("RAM new memory path {row}")),
                &new_leaf,
                &cycle.leaf_index_bits,
                &cycle.memory_path,
                "RAM new memory path",
            )?;
            let read = bit_as_num(&cycle.is_read);
            let write = bit_as_num(&cycle.is_write);
            let enabled = super::direct_lookup::add_nums(
                cs.namespace(|| format!("RAM access enabled {row}")),
                &read,
                &write,
                "RAM access enabled",
            )?;
            enforce_conditional_equal(
                cs.namespace(|| format!("RAM registry membership {row}")),
                &enabled,
                &computed_registry_root,
                &registry_root,
                "RAM registry membership",
            );
            enforce_conditional_equal(
                cs.namespace(|| format!("RAM old-value membership {row}")),
                &enabled,
                &computed_old_root,
                &running_root,
                "RAM old-value membership",
            );
            let expected_after = select_num(
                cs.namespace(|| format!("RAM write root selection {row}")),
                &cycle.is_write,
                &computed_new_root,
                &running_root,
                "RAM write root selection",
            )?;
            enforce_num_equal(
                cs.namespace(|| format!("RAM root after {row}")),
                &cycle.root_after,
                &expected_after,
                "RAM root after",
            );
            running_root = cycle.root_after.clone();
        }
        let claimed_start = alloc_witness_num(
            cs.namespace(|| "claimed RAM start root"),
            nova_from_fr(&block.start_root)?,
        )?;
        let claimed_end = alloc_witness_num(
            cs.namespace(|| "claimed RAM end root"),
            nova_from_fr(&block.end_root)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "RAM start public state"),
            &claimed_start,
            &z[RAM_ROOT_SLOT],
            "RAM start public state",
        );
        enforce_num_equal(
            cs.namespace(|| "RAM end derived"),
            &claimed_end,
            &running_root,
            "RAM end derived",
        );
        let query_root = synthesize_ram_query_root(
            cs.namespace(|| "RAM query commitment"),
            &block_index,
            &cycle_start,
            &active_cycles,
            &address_count,
            &tree_depth,
            &registry_root,
            &cycles,
        )?;
        let claimed_query = alloc_witness_num(
            cs.namespace(|| "claimed RAM query"),
            nova_from_fr(&block.query_root)?,
        )?;
        enforce_num_equal(
            cs.namespace(|| "RAM query trace binding"),
            &query_root,
            &claimed_query,
            "RAM query trace binding",
        );
        let mut transcript =
            super::recursive_relations::AllocatedRecursivePoseidonTranscriptState {
                state: z[RAM_TRANSCRIPT_STATE_SLOT].clone(),
                n_rounds: z[RAM_TRANSCRIPT_ROUND_SLOT].clone(),
            };
        for (label, value) in [
            ("block_index", &block_index),
            ("cycle_start", &cycle_start),
            ("active_cycles", &active_cycles),
            ("address_count", &address_count),
            ("tree_depth", &tree_depth),
            ("registry_root", &registry_root),
            ("ram_query_root", &query_root),
            ("ram_start_root", &claimed_start),
            ("ram_end_root", &claimed_end),
        ] {
            transcript = poseidon_absorb(
                cs.namespace(|| format!("RAM transcript {label}")),
                &transcript,
                value,
                label,
            )?;
        }
        let claimed_state = alloc_witness_num(
            cs.namespace(|| "claimed RAM transcript state"),
            nova_from_fr(&field_from_digest(&ram.transcript_state_after))?,
        )?;
        let claimed_round = alloc_witness_num(
            cs.namespace(|| "claimed RAM transcript round"),
            NovaScalar::from(ram.transcript_round_after as u64),
        )?;
        enforce_num_equal(
            cs.namespace(|| "RAM transcript state binding"),
            &transcript.state,
            &claimed_state,
            "RAM transcript state binding",
        );
        enforce_num_equal(
            cs.namespace(|| "RAM transcript round binding"),
            &transcript.n_rounds,
            &claimed_round,
            "RAM transcript round binding",
        );
        let mut output = register_output;
        output.push(registry_root);
        output.push(running_root);
        output.push(transcript.state);
        output.push(transcript.n_rounds);
        Ok((output, lookup_cycles, register_cycles, cycles))
    }
}

impl StepCircuit<NovaScalar> for DirectRamStepCircuit {
    fn arity(&self) -> usize {
        DIRECT_RAM_Z_ARITY
    }

    fn synthesize<CS: ConstraintSystem<NovaScalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<NovaScalar>],
    ) -> Result<Vec<AllocatedNum<NovaScalar>>, SynthesisError> {
        self.synthesize_with_observations(cs, z)
            .map(|(output, _, _, _)| output)
    }
}

pub(super) fn direct_ram_initial_z(
    preprocessing: &DirectChunkedPreprocessing,
    registers: &[u64; common::constants::REGISTER_COUNT as usize],
    registry_root: Fr,
    initial_root: Fr,
) -> Vec<NovaScalar> {
    let mut z = direct_register_initial_z(preprocessing, registers);
    z.push(nova_from_fr(&registry_root).expect("RAM registry root is canonical"));
    z.push(nova_from_fr(&initial_root).expect("RAM root is canonical"));
    let transcript = PoseidonTranscript::new(DIRECT_RAM_QUERY_DOMAIN);
    z.push(
        nova_from_fr(&field_from_digest(&transcript.state)).expect("RAM transcript is canonical"),
    );
    z.push(NovaScalar::zero());
    z
}

fn setup_ram_pp(
    circuit: &DirectRamStepCircuit,
) -> Result<DirectRamPublicParams, DirectChunkedError> {
    DirectRamPublicParams::setup(
        circuit,
        &*nova_snark::traits::snark::default_ck_hint::<NovaPrimaryEngine>(),
        &*nova_snark::traits::snark::default_ck_hint::<NovaSecondaryEngine>(),
    )
    .map_err(|error| register_stage_error("direct RAM Nova setup failed", error))
}

pub(super) struct PreparedDirectRamStage {
    pub(super) block_count: usize,
    pub(super) total_cycles: usize,
    pub(super) initial_registers: [u64; common::constants::REGISTER_COUNT as usize],
    pub(super) final_registers: [u64; common::constants::REGISTER_COUNT as usize],
    pub(super) addresses: Vec<u64>,
    pub(super) registry_root: Fr,
    pub(super) initial_root: Fr,
    pub(super) final_root: Fr,
    pub(super) lookup: Vec<DirectLookupSubclaim>,
    pub(super) registers: Vec<DirectRegisterSubclaim>,
    pub(super) rams: Vec<DirectRamSubclaim>,
}

pub(super) fn prepare_direct_ram_stage(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    blocks: &[TraceBlock],
) -> Result<PreparedDirectRamStage, DirectChunkedError> {
    if blocks.is_empty() {
        return Err(DirectChunkedError::EmptyTrace);
    }
    validate_trace_ram_addresses(preprocessing, blocks)?;
    let lookup = super::direct_lookup::prove_and_verify_native_lookup_blocks(
        preprocessing,
        capacity,
        blocks.to_vec(),
    )?;
    let mut register_prover =
        PoseidonTranscript::new(super::direct_register::DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
    let mut register_verifier =
        PoseidonTranscript::new(super::direct_register::DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
    let mut registers = Vec::with_capacity(blocks.len());
    for block in blocks {
        let subclaim = super::direct_register::prove_native_register_subclaim(
            block,
            capacity,
            &mut register_prover,
        )?;
        super::direct_register::verify_native_register_subclaim(&subclaim, &mut register_verifier)?;
        registers.push(subclaim);
    }
    let (block_count, total_cycles) =
        validate_combined_sequence(preprocessing, capacity, &lookup, &registers)?;
    let (addresses, mut memory) = initial_ram_table(blocks);
    let address_indices = addresses
        .iter()
        .enumerate()
        .map(|(index, address)| (*address, index))
        .collect::<BTreeMap<_, _>>();
    let (registry_tree, mut memory_tree) = build_ram_trees(&addresses, &memory);
    let registry_root = registry_tree.root();
    let initial_root = memory_tree.root();
    let mut ram_prover = PoseidonTranscript::new(DIRECT_RAM_QUERY_DOMAIN);
    let mut rams = Vec::with_capacity(block_count);
    for block in blocks {
        rams.push(derive_ram_subclaim(
            block,
            capacity,
            &addresses,
            &address_indices,
            &mut memory,
            &registry_tree,
            &mut memory_tree,
            &mut ram_prover,
        )?);
    }
    let mut ram_verifier = PoseidonTranscript::new(DIRECT_RAM_QUERY_DOMAIN);
    for ram in &rams {
        verify_ram_subclaim(ram, &mut ram_verifier)?;
    }
    Ok(PreparedDirectRamStage {
        block_count,
        total_cycles,
        initial_registers: registers[0].block.start_registers,
        final_registers: registers[block_count - 1].block.end_registers,
        addresses,
        registry_root,
        initial_root,
        final_root: memory_tree.root(),
        lookup,
        registers,
        rams,
    })
}

pub fn prove_direct_ram_stage<I>(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    blocks: I,
) -> Result<DirectRamStageProof, DirectChunkedError>
where
    I: IntoIterator<Item = TraceBlock>,
{
    let blocks = blocks.into_iter().collect::<Vec<_>>();
    let prepared = prepare_direct_ram_stage(preprocessing, capacity, &blocks)?;
    let PreparedDirectRamStage {
        block_count,
        total_cycles,
        initial_registers,
        final_registers,
        addresses,
        registry_root,
        initial_root,
        final_root,
        lookup,
        registers,
        rams,
    } = prepared;
    let z0 = direct_ram_initial_z(
        preprocessing,
        &initial_registers,
        registry_root,
        initial_root,
    );
    let first = DirectRamStepCircuit::for_subclaims(
        lookup[0].clone(),
        registers[0].clone(),
        rams[0].clone(),
        block_count == 1,
    );
    let pp = setup_ram_pp(&first)?;
    let mut recursive = DirectRamNovaSnark::new(&pp, &first, &z0)
        .map_err(|e| register_stage_error("RAM Nova initialization failed", e))?;
    recursive
        .prove_step(&pp, &first)
        .map_err(|e| register_stage_error("RAM Nova first step failed", e))?;
    for position in 1..block_count {
        let circuit = DirectRamStepCircuit::for_subclaims(
            lookup[position].clone(),
            registers[position].clone(),
            rams[position].clone(),
            position + 1 == block_count,
        );
        recursive
            .prove_step(&pp, &circuit)
            .map_err(|e| register_stage_error("RAM Nova step failed", e))?;
    }
    let output = recursive
        .verify(&pp, block_count, &z0)
        .map_err(|e| register_stage_error("RAM Nova self verification failed", e))?;
    let (pk, vk) = DirectRamCompressedSnark::setup(&pp)
        .map_err(|e| register_stage_error("RAM Spartan setup failed", e))?;
    let compressed = DirectRamCompressedSnark::prove(&pp, &pk, &recursive)
        .map_err(|e| register_stage_error("RAM Spartan proving failed", e))?;
    let compressed_output = compressed
        .verify(&vk, block_count, &z0)
        .map_err(|e| register_stage_error("RAM Spartan self verification failed", e))?;
    if output != compressed_output
        || output[RAM_ROOT_SLOT]
            != nova_from_fr(&final_root)
                .map_err(|e| register_stage_error("RAM root conversion failed", e))?
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D4 recursive output does not close RAM".to_string(),
        ));
    }
    let relations = DirectRelation::ALL
        .into_iter()
        .map(|relation| {
            (
                relation,
                match relation {
                    DirectRelation::Lookup | DirectRelation::Register | DirectRelation::Ram => {
                        DirectRelationState::Proven
                    }
                    DirectRelation::Cpu => DirectRelationState::Unsupported,
                    DirectRelation::Pcs => DirectRelationState::Deferred,
                },
            )
        })
        .collect();
    let proof = DirectRamStageProof {
        statement: DirectRamStageStatement {
            program_digest: preprocessing.program_digest,
            lookup_table_commitment: preprocessing.lookup_table_commitment,
            block_count,
            total_cycles,
            initial_registers,
            final_registers,
            ram_addresses: addresses,
            ram_registry_root: registry_root,
            initial_ram_root: initial_root,
            final_ram_root: final_root,
            terminated: true,
            relations,
        },
        lookup_subclaims: lookup,
        register_subclaims: registers,
        ram_subclaims: rams,
        nova_recursive_snark: postcard::to_stdvec(&recursive)
            .map_err(|e| register_stage_error("RAM Nova serialization failed", e))?,
        spartan_proof: postcard::to_stdvec(&compressed)
            .map_err(|e| register_stage_error("RAM Spartan serialization failed", e))?,
        initial_z: z0.iter().copied().map(nova_to_storage).collect(),
        final_z: output.iter().copied().map(nova_to_storage).collect(),
    };
    verify_direct_ram_stage(preprocessing, capacity, &proof)?;
    Ok(proof)
}

pub fn verify_direct_ram_stage(
    preprocessing: &DirectChunkedPreprocessing,
    capacity: usize,
    proof: &DirectRamStageProof,
) -> Result<(), DirectChunkedError> {
    let expected_relations = DirectRelation::ALL
        .into_iter()
        .map(|relation| {
            (
                relation,
                match relation {
                    DirectRelation::Lookup | DirectRelation::Register | DirectRelation::Ram => {
                        DirectRelationState::Proven
                    }
                    DirectRelation::Cpu => DirectRelationState::Unsupported,
                    DirectRelation::Pcs => DirectRelationState::Deferred,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let n = proof.statement.block_count;
    if n == 0
        || proof.statement.program_digest != preprocessing.program_digest
        || proof.statement.lookup_table_commitment != preprocessing.lookup_table_commitment
        || proof.statement.relations != expected_relations
        || proof
            .statement
            .ram_addresses
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || proof.statement.ram_registry_root
            != build_ram_trees(
                &proof.statement.ram_addresses,
                &vec![0; proof.statement.ram_addresses.len()],
            )
            .0
            .root()
        || proof.lookup_subclaims.len() != n
        || proof.register_subclaims.len() != n
        || proof.ram_subclaims.len() != n
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D4 statement/envelope mismatch".to_string(),
        ));
    }
    let (block_count, total_cycles) = validate_combined_sequence(
        preprocessing,
        capacity,
        &proof.lookup_subclaims,
        &proof.register_subclaims,
    )?;
    if block_count != n || total_cycles != proof.statement.total_cycles {
        return Err(DirectChunkedError::InvalidProofShape(
            "D4 counters mismatch".to_string(),
        ));
    }
    let mut lookup_transcript =
        PoseidonTranscript::new(super::direct_lookup::DIRECT_LOOKUP_TRANSCRIPT_DOMAIN);
    for subclaim in &proof.lookup_subclaims {
        super::direct_lookup::verify_native_subclaim(subclaim, &mut lookup_transcript)?;
    }
    let mut register_transcript =
        PoseidonTranscript::new(super::direct_register::DIRECT_REGISTER_TRANSCRIPT_DOMAIN);
    for subclaim in &proof.register_subclaims {
        super::direct_register::verify_native_register_subclaim(
            subclaim,
            &mut register_transcript,
        )?;
    }
    let mut ram_transcript = PoseidonTranscript::new(DIRECT_RAM_QUERY_DOMAIN);
    for (position, ram) in proof.ram_subclaims.iter().enumerate() {
        verify_ram_subclaim(ram, &mut ram_transcript)?;
        if ram.block.block_index != position
            || ram.block.block_index != proof.lookup_subclaims[position].block.block_index
            || ram.block.address_count != proof.statement.ram_addresses.len()
            || ram.block.registry_root != proof.statement.ram_registry_root
            || (position > 0
                && proof.ram_subclaims[position - 1].block.end_root != ram.block.start_root)
        {
            return Err(DirectChunkedError::InvalidProofShape(
                "D4 block or RAM-root continuity mismatch".to_string(),
            ));
        }
    }
    if proof.ram_subclaims[0].block.start_root != proof.statement.initial_ram_root
        || proof.ram_subclaims[n - 1].block.end_root != proof.statement.final_ram_root
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D4 RAM boundaries mismatch".to_string(),
        ));
    }
    let z0 = direct_ram_initial_z(
        preprocessing,
        &proof.statement.initial_registers,
        proof.statement.ram_registry_root,
        proof.statement.initial_ram_root,
    );
    let setup = DirectRamStepCircuit::for_subclaims(
        proof.lookup_subclaims[0].clone(),
        proof.register_subclaims[0].clone(),
        proof.ram_subclaims[0].clone(),
        n == 1,
    );
    let pp = setup_ram_pp(&setup)?;
    let recursive = postcard::from_bytes::<DirectRamNovaSnark>(&proof.nova_recursive_snark)
        .map_err(|e| register_stage_error("RAM Nova deserialization failed", e))?;
    let output = recursive
        .verify(&pp, n, &z0)
        .map_err(|e| register_stage_error("RAM Nova verification failed", e))?;
    let (_, vk) = DirectRamCompressedSnark::setup(&pp)
        .map_err(|e| register_stage_error("RAM Spartan setup failed", e))?;
    let compressed = postcard::from_bytes::<DirectRamCompressedSnark>(&proof.spartan_proof)
        .map_err(|e| register_stage_error("RAM Spartan deserialization failed", e))?;
    let compressed_output = compressed
        .verify(&vk, n, &z0)
        .map_err(|e| register_stage_error("RAM Spartan verification failed", e))?;
    let expected_ram_state = nova_from_fr(&field_from_digest(&ram_transcript.state))
        .map_err(|e| register_stage_error("RAM transcript conversion failed", e))?;
    if output != compressed_output
        || output.len() != DIRECT_RAM_Z_ARITY
        || output[RAM_REGISTRY_ROOT_SLOT]
            != nova_from_fr(&proof.statement.ram_registry_root)
                .map_err(|e| register_stage_error("RAM registry conversion failed", e))?
        || output[RAM_ROOT_SLOT]
            != nova_from_fr(&proof.statement.final_ram_root)
                .map_err(|e| register_stage_error("RAM root conversion failed", e))?
        || output[RAM_TRANSCRIPT_STATE_SLOT] != expected_ram_state
        || output[RAM_TRANSCRIPT_ROUND_SLOT] != NovaScalar::from(ram_transcript.n_rounds as u64)
        || proof.initial_z != z0.iter().copied().map(nova_to_storage).collect::<Vec<_>>()
        || proof.final_z
            != output
                .iter()
                .copied()
                .map(nova_to_storage)
                .collect::<Vec<_>>()
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D4 Nova/Spartan closure mismatch".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::constants::REGISTER_COUNT;
    use tracer::instruction::{
        format::{
            format_load::{FormatLoad, RegisterStateFormatLoad},
            format_s::{FormatS, RegisterStateFormatS},
        },
        ld::LD,
        sd::SD,
        RAMRead, RAMWrite, RISCVCycle,
    };

    fn boundary(
        cycle: usize,
        regs: [u64; REGISTER_COUNT as usize],
        terminated: bool,
    ) -> tracer::MachineBoundaryState {
        tracer::MachineBoundaryState {
            global_cycle: cycle,
            emulator_trace_len: cycle,
            pc: 0x8000_0000 + 4 * cycle as u64,
            registers: regs.map(|x| x as i64),
            terminated,
        }
    }
    fn store(address: u64, pre: u64, post: u64, base: u64) -> tracer::instruction::Cycle {
        RISCVCycle::<SD> {
            instruction: SD {
                address: 0x8000_0000,
                operands: FormatS {
                    rs1: 1,
                    rs2: 2,
                    imm: (address - base) as i64,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            register_state: RegisterStateFormatS {
                rs1: base,
                rs2: post,
            },
            ram_access: RAMWrite {
                address,
                pre_value: pre,
                post_value: post,
            },
        }
        .into()
    }
    fn load(address: u64, pre: u64, value: u64, base: u64) -> tracer::instruction::Cycle {
        RISCVCycle::<LD> {
            instruction: LD {
                address: 0x8000_0004,
                operands: FormatLoad {
                    rd: 3,
                    rs1: 1,
                    imm: (address - base) as i64,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            register_state: RegisterStateFormatLoad {
                rd: (pre, value),
                rs1: base,
            },
            ram_access: RAMRead { address, value },
        }
        .into()
    }
    fn blocks() -> Vec<TraceBlock> {
        let address = common::constants::RAM_START_ADDRESS + 0x100;
        let second_address = address + 8;
        let mut a = [0u64; REGISTER_COUNT as usize];
        a[1] = address;
        a[2] = 9;
        let b = a;
        let mut c = b;
        c[3] = 9;
        vec![
            TraceBlock {
                block_index: 0,
                global_cycle_start: 0,
                active_cycles: 2,
                target_size: 2,
                start_state: boundary(0, a, false),
                end_state: boundary(2, b, false),
                cycles: vec![
                    store(address, 0, 9, address),
                    store(second_address, 0, 9, address),
                ],
                ended_at_tick_boundary: true,
            },
            TraceBlock {
                block_index: 1,
                global_cycle_start: 2,
                active_cycles: 2,
                target_size: 2,
                start_state: boundary(2, b, false),
                end_state: boundary(4, c, true),
                cycles: vec![
                    load(address, 0, 9, address),
                    load(second_address, 9, 9, address),
                ],
                ended_at_tick_boundary: true,
            },
        ]
    }
    fn preprocessing() -> DirectChunkedPreprocessing {
        DirectChunkedPreprocessing::from_program_bytes(b"ram-test", 64)
    }

    #[test]
    fn d4_rejects_cross_block_stale_read() {
        let mut bad = blocks();
        if let tracer::instruction::Cycle::LD(cycle) = &mut bad[1].cycles[0] {
            cycle.ram_access.value = 7;
            cycle.register_state.rd.1 = 7;
        }
        assert!(prove_direct_ram_stage(&preprocessing(), 2, bad).is_err());
    }

    #[test]
    fn d8_rejects_unaligned_cross_block_word_alias() {
        let mut bad = blocks();
        let address = common::constants::RAM_START_ADDRESS + 0x100;
        bad[1].cycles[0] = load(address + 4, 0, 9, address);
        let error = prepare_direct_ram_stage(&preprocessing(), 2, &bad)
            .err()
            .expect("unaligned RAM address must fail");
        assert!(error.to_string().contains("8-byte aligned"));
    }

    #[test]
    fn d8_rejects_ram_address_outside_the_verifier_memory_layout() {
        let mut bad = blocks();
        let address = preprocessing().memory_layout.heap_end;
        bad[0].cycles[0] = store(address, 0, 9, address);
        let error = prepare_direct_ram_stage(&preprocessing(), 2, &bad)
            .err()
            .expect("out-of-range RAM address must fail");
        assert!(error
            .to_string()
            .contains("outside the canonical Jolt memory"));
    }

    #[test]
    fn d4_nova_and_spartan_close_authenticated_ram() {
        let proof = prove_direct_ram_stage(&preprocessing(), 2, blocks()).unwrap();
        assert_ne!(
            proof.statement.final_ram_root,
            proof.statement.initial_ram_root
        );
        assert_eq!(
            proof.statement.relations[&DirectRelation::Ram],
            DirectRelationState::Proven
        );
        verify_direct_ram_stage(&preprocessing(), 2, &proof).unwrap();
        let mut tampered = proof;
        tampered.ram_subclaims[1].block.cycles[0].read_value ^= 1;
        assert!(verify_direct_ram_stage(&preprocessing(), 2, &tampered).is_err());
    }
}
