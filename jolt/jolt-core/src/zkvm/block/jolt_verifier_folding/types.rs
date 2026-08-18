use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};

/// Direct V2 protocol tag. It is absorbed before every statement and proof.
pub const BLOCK_JOLT_PROTOCOL_VERSION: &str = "jolt-nova/direct-block-jolt/v3";
pub const BLOCK_JOLT_WIRE_VERSION: u16 = 3;

/// Canonical little-endian BN254 scalar bytes or a versioned 32-byte digest.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct FieldElement(pub [u8; 32]);

impl FieldElement {
    pub const ZERO: Self = Self([0; 32]);

    pub(crate) fn from_fr(value: &Fr) -> Self {
        let mut bytes = [0u8; 32];
        let encoded = value.into_bigint().to_bytes_le();
        bytes[..encoded.len()].copy_from_slice(&encoded);
        Self(bytes)
    }

    pub(crate) fn to_fr(self) -> Fr {
        Fr::from_le_bytes_mod_order(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BlockRelation {
    LookupLasso,
    Register,
    Ram,
    CpuR1cs,
}

impl BlockRelation {
    pub const ORDERED: [Self; 4] = [Self::LookupLasso, Self::Register, Self::Ram, Self::CpuR1cs];

    pub const fn tag(self) -> u8 {
        match self {
            Self::LookupLasso => 1,
            Self::Register => 2,
            Self::Ram => 3,
            Self::CpuR1cs => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TranscriptCheckpoint {
    pub state: [u8; 32],
    pub round: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockBoundaryState {
    pub machine_state: [u8; 32],
    pub register_state: [u8; 32],
    pub ram_root: FieldElement,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeferredPcsClaim {
    pub relation: BlockRelation,
    pub polynomial_id: [u8; 32],
    pub commitment_id: [u8; 32],
    pub opening_point: Vec<FieldElement>,
    pub claimed_value: FieldElement,
}

impl DeferredPcsClaim {
    pub fn digest(&self) -> [u8; 32] {
        wire_digest(b"deferred-pcs-claim", self)
    }
}

/// Compact clear-sumcheck transcript. Polynomial openings are deferred to D17.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompactSumcheckProof {
    pub rounds: u32,
    pub degree_bound: u32,
    pub initial_claim: FieldElement,
    pub final_claim: FieldElement,
    pub compressed_proof: Vec<u8>,
    pub challenges: Vec<FieldElement>,
    pub opening_claims: Vec<DeferredPcsClaim>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LookupBlockProof {
    pub query_commitment: FieldElement,
    pub table_commitment: [u8; 32],
    pub accumulator_before: FieldElement,
    /// Poseidon round counter paired with `accumulator_before`.
    pub accumulator_round_before: u64,
    pub accumulator_after: FieldElement,
    /// Poseidon round counter paired with `accumulator_after`.
    pub accumulator_round_after: u64,
    pub reduction_point: Vec<FieldElement>,
    pub input_claims: [FieldElement; 3],
    pub gamma: FieldElement,
    pub batching_coefficient: FieldElement,
    pub output_claims: Vec<FieldElement>,
    pub sumcheck: CompactSumcheckProof,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RegisterBlockProof {
    pub access_commitment: FieldElement,
    pub state_before: [u8; 32],
    pub state_after: [u8; 32],
    pub reduction_point: Vec<FieldElement>,
    pub input_claims: [FieldElement; 3],
    pub gamma: FieldElement,
    pub batching_coefficient: FieldElement,
    pub output_claims: Vec<FieldElement>,
    pub sumcheck: CompactSumcheckProof,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RamBlockProof {
    pub access_commitment: FieldElement,
    pub ram_k: u64,
    pub registry_root: FieldElement,
    pub root_before: FieldElement,
    pub root_after: FieldElement,
    pub reduction_point: Vec<FieldElement>,
    pub input_claims: [FieldElement; 2],
    pub gamma: FieldElement,
    pub batching_coefficient: FieldElement,
    pub output_claims: Vec<FieldElement>,
    pub relation_proof: CompactSumcheckProof,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CpuBlockRelationProof {
    pub row_commitment: [u8; 32],
    pub bytecode_root: FieldElement,
    pub start_pc: u64,
    pub end_pc: u64,
    pub terminal: bool,
    /// Original Jolt Spartan outer univariate-skip first-round proof.
    pub uniskip_proof: Vec<u8>,
    pub uniskip_challenge: FieldElement,
    pub uniskip_claim: FieldElement,
    pub batching_coefficient: FieldElement,
    pub relation_proof: CompactSumcheckProof,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockJoltStatement {
    pub wire_version: u16,
    pub preprocessing_id: [u8; 32],
    pub program_digest: [u8; 32],
    pub lookup_table_commitment: [u8; 32],
    pub bytecode_commitment: FieldElement,
    pub block_index: u64,
    pub global_cycle_start: u64,
    pub global_cycle_end: u64,
    pub active_cycles: u64,
    pub cycle_capacity: u64,
    pub terminal: bool,
    pub start_pc: u64,
    pub end_pc: u64,
    pub start: BlockBoundaryState,
    pub end: BlockBoundaryState,
    pub lookup_accumulator_before: FieldElement,
    pub lookup_transcript_round_before: u64,
    pub lookup_accumulator_after: FieldElement,
    pub lookup_transcript_round_after: u64,
    pub transcript_before: TranscriptCheckpoint,
    pub transcript_after: TranscriptCheckpoint,
    pub deferred_pcs_claim_root: [u8; 32],
}

impl BlockJoltStatement {
    pub fn validate_shape(&self) -> Result<(), String> {
        if self.wire_version != BLOCK_JOLT_WIRE_VERSION {
            return Err("unsupported block-Jolt wire version".to_string());
        }
        if self.cycle_capacity == 0 || !self.cycle_capacity.is_power_of_two() {
            return Err("block cycle capacity must be a non-zero power of two".to_string());
        }
        if self.active_cycles == 0 || self.active_cycles > self.cycle_capacity {
            return Err("active cycles are outside the fixed block shape".to_string());
        }
        if self.global_cycle_end != self.global_cycle_start + self.active_cycles {
            return Err("block global cycle interval is inconsistent".to_string());
        }
        if self.transcript_after.round < self.transcript_before.round {
            return Err("block transcript round moved backwards".to_string());
        }
        if self.lookup_transcript_round_after < self.lookup_transcript_round_before {
            return Err("lookup transcript round moved backwards".to_string());
        }
        Ok(())
    }

    pub fn digest(&self) -> [u8; 32] {
        wire_digest(b"block-jolt-statement", self)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        postcard::to_stdvec(self).map_err(|error| error.to_string())
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let statement: Self = postcard::from_bytes(bytes).map_err(|error| error.to_string())?;
        statement.validate_shape()?;
        Ok(statement)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockJoltProof {
    pub wire_version: u16,
    pub statement_digest: [u8; 32],
    pub transcript_before: TranscriptCheckpoint,
    pub lookup: LookupBlockProof,
    pub register: RegisterBlockProof,
    pub ram: RamBlockProof,
    pub cpu: CpuBlockRelationProof,
    pub transcript_after: TranscriptCheckpoint,
    pub deferred_pcs_claims: Vec<DeferredPcsClaim>,
    pub proof_digest: [u8; 32],
}

impl BlockJoltProof {
    pub fn validate_structure(&self, statement: &BlockJoltStatement) -> Result<(), String> {
        statement.validate_shape()?;
        if self.wire_version != BLOCK_JOLT_WIRE_VERSION {
            return Err("unsupported block-Jolt proof wire version".to_string());
        }
        if self.statement_digest != statement.digest() {
            return Err("block proof is bound to another statement".to_string());
        }
        if self.transcript_before != statement.transcript_before
            || self.transcript_after != statement.transcript_after
        {
            return Err("block proof transcript endpoints do not match the statement".to_string());
        }
        if self.lookup.table_commitment != statement.lookup_table_commitment
            || self.lookup.accumulator_before != statement.lookup_accumulator_before
            || self.lookup.accumulator_round_before != statement.lookup_transcript_round_before
            || self.lookup.accumulator_after != statement.lookup_accumulator_after
            || self.lookup.accumulator_round_after != statement.lookup_transcript_round_after
        {
            return Err("lookup proof does not match the block statement".to_string());
        }
        if self.register.state_before != statement.start.register_state
            || self.register.state_after != statement.end.register_state
            || self.ram.root_before != statement.start.ram_root
            || self.ram.root_after != statement.end.ram_root
            || self.cpu.bytecode_root != statement.bytecode_commitment
            || self.cpu.start_pc != statement.start_pc
            || self.cpu.end_pc != statement.end_pc
            || self.cpu.terminal != statement.terminal
        {
            return Err("relation proof boundary does not match the block statement".to_string());
        }
        if self
            .deferred_pcs_claims
            .iter()
            .any(|claim| !BlockRelation::ORDERED.contains(&claim.relation))
        {
            return Err("block proof contains an unknown deferred PCS relation".to_string());
        }
        if deferred_claim_root(&self.deferred_pcs_claims) != statement.deferred_pcs_claim_root {
            return Err("deferred PCS claims do not match the statement root".to_string());
        }
        if self.proof_digest != self.recompute_digest() {
            return Err("block proof digest mismatch".to_string());
        }
        Ok(())
    }

    pub fn recompute_digest(&self) -> [u8; 32] {
        let mut canonical = self.clone();
        canonical.proof_digest = [0; 32];
        wire_digest(b"block-jolt-proof", &canonical)
    }

    pub fn seal(&mut self) {
        self.proof_digest = self.recompute_digest();
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        postcard::to_stdvec(self).map_err(|error| error.to_string())
    }

    pub fn from_bytes(bytes: &[u8], statement: &BlockJoltStatement) -> Result<Self, String> {
        let proof: Self = postcard::from_bytes(bytes).map_err(|error| error.to_string())?;
        proof.validate_structure(statement)?;
        Ok(proof)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StreamingRecursiveState {
    pub wire_version: u16,
    pub preprocessing_id: [u8; 32],
    pub program_digest: [u8; 32],
    pub lookup_table_commitment: [u8; 32],
    pub next_block_index: u64,
    pub next_global_cycle: u64,
    pub boundary: BlockBoundaryState,
    pub lookup_accumulator: FieldElement,
    pub lookup_transcript_round: u64,
    pub transcript: TranscriptCheckpoint,
    pub deferred_pcs_accumulator: [u8; 32],
    pub total_active_cycles: u64,
    pub terminated: bool,
}

impl StreamingRecursiveState {
    pub fn validate_next(&self, statement: &BlockJoltStatement) -> Result<(), String> {
        statement.validate_shape()?;
        if self.wire_version != statement.wire_version
            || self.preprocessing_id != statement.preprocessing_id
            || self.program_digest != statement.program_digest
            || self.lookup_table_commitment != statement.lookup_table_commitment
        {
            return Err("block preprocessing identity changed".to_string());
        }
        if self.terminated {
            return Err("a block followed a terminal recursive state".to_string());
        }
        if self.next_block_index != statement.block_index
            || self.next_global_cycle != statement.global_cycle_start
            || self.boundary != statement.start
            || self.lookup_accumulator != statement.lookup_accumulator_before
            || self.lookup_transcript_round != statement.lookup_transcript_round_before
            || self.transcript != statement.transcript_before
        {
            return Err("block is not contiguous with the recursive state".to_string());
        }
        Ok(())
    }
}

pub fn deferred_claim_root(claims: &[DeferredPcsClaim]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(b"deferred-pcs-root");
    hasher.update((claims.len() as u64).to_le_bytes());
    for claim in claims {
        hasher.update([claim.relation.tag()]);
        hasher.update(claim.digest());
    }
    hasher.finalize().into()
}

fn wire_digest<T: Serialize>(domain: &[u8], value: &T) -> [u8; 32] {
    let bytes = postcard::to_stdvec(value).expect("serializing an in-memory V2 protocol value");
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim(relation: BlockRelation, byte: u8) -> DeferredPcsClaim {
        DeferredPcsClaim {
            relation,
            polynomial_id: [byte; 32],
            commitment_id: [byte.wrapping_add(1); 32],
            opening_point: vec![FieldElement([byte.wrapping_add(2); 32])],
            claimed_value: FieldElement([byte.wrapping_add(3); 32]),
        }
    }

    fn sumcheck(relation: BlockRelation, byte: u8) -> CompactSumcheckProof {
        CompactSumcheckProof {
            rounds: 1,
            degree_bound: 2,
            initial_claim: FieldElement([byte; 32]),
            final_claim: FieldElement([byte.wrapping_add(1); 32]),
            compressed_proof: vec![byte, byte.wrapping_add(1)],
            challenges: vec![FieldElement([byte.wrapping_add(2); 32])],
            opening_claims: vec![claim(relation, byte)],
        }
    }

    fn fixture() -> (BlockJoltStatement, BlockJoltProof) {
        let claims = vec![
            claim(BlockRelation::LookupLasso, 1),
            claim(BlockRelation::Register, 2),
            claim(BlockRelation::Ram, 3),
            claim(BlockRelation::CpuR1cs, 4),
        ];
        let start = BlockBoundaryState {
            machine_state: [10; 32],
            register_state: [11; 32],
            ram_root: FieldElement([12; 32]),
        };
        let end = BlockBoundaryState {
            machine_state: [20; 32],
            register_state: [21; 32],
            ram_root: FieldElement([22; 32]),
        };
        let before = TranscriptCheckpoint {
            state: [30; 32],
            round: 7,
        };
        let after = TranscriptCheckpoint {
            state: [31; 32],
            round: 11,
        };
        let statement = BlockJoltStatement {
            wire_version: BLOCK_JOLT_WIRE_VERSION,
            preprocessing_id: [1; 32],
            program_digest: [2; 32],
            lookup_table_commitment: [3; 32],
            bytecode_commitment: FieldElement([4; 32]),
            block_index: 0,
            global_cycle_start: 0,
            global_cycle_end: 8,
            active_cycles: 8,
            cycle_capacity: 8,
            terminal: false,
            start_pc: 0,
            end_pc: 4,
            start: start.clone(),
            end: end.clone(),
            lookup_accumulator_before: FieldElement([5; 32]),
            lookup_transcript_round_before: 3,
            lookup_accumulator_after: FieldElement([6; 32]),
            lookup_transcript_round_after: 9,
            transcript_before: before,
            transcript_after: after,
            deferred_pcs_claim_root: deferred_claim_root(&claims),
        };
        let mut proof = BlockJoltProof {
            wire_version: BLOCK_JOLT_WIRE_VERSION,
            statement_digest: statement.digest(),
            transcript_before: before,
            lookup: LookupBlockProof {
                query_commitment: FieldElement([40; 32]),
                table_commitment: statement.lookup_table_commitment,
                accumulator_before: statement.lookup_accumulator_before,
                accumulator_round_before: statement.lookup_transcript_round_before,
                accumulator_after: statement.lookup_accumulator_after,
                accumulator_round_after: statement.lookup_transcript_round_after,
                reduction_point: vec![FieldElement([45; 32])],
                input_claims: [
                    FieldElement([46; 32]),
                    FieldElement([47; 32]),
                    FieldElement([48; 32]),
                ],
                gamma: FieldElement([49; 32]),
                batching_coefficient: FieldElement([50; 32]),
                output_claims: vec![FieldElement([51; 32])],
                sumcheck: sumcheck(BlockRelation::LookupLasso, 1),
            },
            register: RegisterBlockProof {
                access_commitment: FieldElement([41; 32]),
                state_before: start.register_state,
                state_after: end.register_state,
                reduction_point: vec![FieldElement([52; 32])],
                input_claims: [
                    FieldElement([53; 32]),
                    FieldElement([54; 32]),
                    FieldElement([55; 32]),
                ],
                gamma: FieldElement([56; 32]),
                batching_coefficient: FieldElement([57; 32]),
                output_claims: vec![FieldElement([58; 32])],
                sumcheck: sumcheck(BlockRelation::Register, 2),
            },
            ram: RamBlockProof {
                access_commitment: FieldElement([42; 32]),
                ram_k: 8,
                registry_root: FieldElement([43; 32]),
                root_before: start.ram_root,
                root_after: end.ram_root,
                reduction_point: vec![FieldElement([59; 32])],
                input_claims: [FieldElement([60; 32]), FieldElement([61; 32])],
                gamma: FieldElement([62; 32]),
                batching_coefficient: FieldElement([63; 32]),
                output_claims: vec![FieldElement([64; 32])],
                relation_proof: sumcheck(BlockRelation::Ram, 3),
            },
            cpu: CpuBlockRelationProof {
                row_commitment: [44; 32],
                bytecode_root: statement.bytecode_commitment,
                start_pc: 0,
                end_pc: 4,
                terminal: false,
                uniskip_proof: vec![65, 66],
                uniskip_challenge: FieldElement([67; 32]),
                uniskip_claim: FieldElement([68; 32]),
                batching_coefficient: FieldElement([69; 32]),
                relation_proof: sumcheck(BlockRelation::CpuR1cs, 4),
            },
            transcript_after: after,
            deferred_pcs_claims: claims,
            proof_digest: [0; 32],
        };
        proof.seal();
        (statement, proof)
    }

    #[test]
    fn d10_statement_and_proof_round_trip() {
        let (statement, proof) = fixture();
        let decoded_statement =
            BlockJoltStatement::from_bytes(&statement.to_bytes().unwrap()).unwrap();
        let decoded_proof =
            BlockJoltProof::from_bytes(&proof.to_bytes().unwrap(), &decoded_statement).unwrap();
        assert_eq!(statement, decoded_statement);
        assert_eq!(proof, decoded_proof);
    }

    #[test]
    fn d10_statement_and_proof_tampering_is_rejected() {
        let (statement, mut proof) = fixture();
        proof.lookup.table_commitment[0] ^= 1;
        proof.seal();
        assert!(proof.validate_structure(&statement).is_err());

        let (_, mut proof) = fixture();
        proof.deferred_pcs_claims[0].claimed_value.0[0] ^= 1;
        proof.seal();
        assert!(proof.validate_structure(&statement).is_err());

        let (mut malformed, _) = fixture();
        malformed.global_cycle_end += 1;
        assert!(malformed.validate_shape().is_err());
    }
}
