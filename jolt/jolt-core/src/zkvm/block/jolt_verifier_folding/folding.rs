//! D16 real Nova folding for the compact D15 block verifier.
//!
//! Exactly one `RecursiveSNARK::prove_step` is executed for every compact
//! block proof. The folded artifact retains neither trace rows nor D8 relation
//! witnesses. Spartan compression and the final Dory decider remain D17/D19
//! responsibilities.

use std::{
    io::{self, Write},
    sync::Arc,
};

use nova_snark::nova::{CompressedSNARK, ProverKey, RecursiveSNARK, VerifierKey};
use serde::Serialize;
use sha3::{Digest, Sha3_256};

use super::super::{
    direct_lookup::nova_from_fr, DirectChunkedError, NovaPrimaryEngine, NovaScalar,
    NovaSecondaryEngine,
};
use super::{
    compact_verifier_circuit::{
        BLOCK_SLOT, CYCLE_SLOT, DEFERRED_ROUND_SLOT, DEFERRED_STATE_SLOT, LOOKUP_ROUND_SLOT,
        LOOKUP_STATE_SLOT, MACHINE_OFFSET, RAM_ROOT_SLOT, REGISTER_OFFSET, TERMINATED_SLOT,
        TOTAL_CYCLES_SLOT,
    },
    BlockJoltHostConfig, BlockJoltVerifierStepCircuit, VerifiedBlockJoltTransition,
    BLOCK_JOLT_PROTOCOL_VERSION,
};

pub(super) type BlockJoltNovaSnark =
    RecursiveSNARK<NovaPrimaryEngine, NovaSecondaryEngine, BlockJoltVerifierStepCircuit>;
pub(super) type BlockJoltNovaPublicParams = nova_snark::nova::PublicParams<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    BlockJoltVerifierStepCircuit,
>;
pub(super) type BlockJoltNovaEvaluationEngine<E> =
    nova_snark::provider::ipa_pc::EvaluationEngine<E>;
pub(super) type BlockJoltNovaPrimarySpartan = nova_snark::spartan::snark::RelaxedR1CSSNARK<
    NovaPrimaryEngine,
    BlockJoltNovaEvaluationEngine<NovaPrimaryEngine>,
>;
pub(super) type BlockJoltNovaSecondarySpartan = nova_snark::spartan::snark::RelaxedR1CSSNARK<
    NovaSecondaryEngine,
    BlockJoltNovaEvaluationEngine<NovaSecondaryEngine>,
>;
pub(super) type BlockJoltNovaCompressedSnark = CompressedSNARK<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    BlockJoltVerifierStepCircuit,
    BlockJoltNovaPrimarySpartan,
    BlockJoltNovaSecondarySpartan,
>;
pub(super) type BlockJoltNovaCompressedProverKey = ProverKey<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    BlockJoltVerifierStepCircuit,
    BlockJoltNovaPrimarySpartan,
    BlockJoltNovaSecondarySpartan,
>;
pub(super) type BlockJoltNovaCompressedVerifierKey = VerifierKey<
    NovaPrimaryEngine,
    NovaSecondaryEngine,
    BlockJoltVerifierStepCircuit,
    BlockJoltNovaPrimarySpartan,
    BlockJoltNovaSecondarySpartan,
>;

const D19_SETUP_DOMAIN: &[u8] = b"block-jolt-nova-spartan-setup-v1";

fn folding_error(context: &str, error: impl core::fmt::Debug) -> DirectChunkedError {
    DirectChunkedError::InvalidProofShape(format!("{context}: {error:?}"))
}

/// Streams postcard bytes directly into the setup digest. This avoids holding
/// a second, potentially very large, copy of Nova's public parameters or
/// Spartan verifier key while preserving the exact D19 hash preimage.
struct SetupDigestWriter<'a> {
    hasher: &'a mut Sha3_256,
}

impl Write for SetupDigestWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.hasher.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn update_setup_digest_with_postcard<T: Serialize + ?Sized>(
    hasher: &mut Sha3_256,
    value: &T,
    context: &str,
) -> Result<(), DirectChunkedError> {
    let serialized_len = postcard::experimental::serialized_size(value)
        .map_err(|error| folding_error(context, error))?;
    hasher.update((serialized_len as u64).to_le_bytes());
    postcard::to_io(value, SetupDigestWriter { hasher })
        .map_err(|error| folding_error(context, error))?;
    Ok(())
}

fn setup_identifier(
    config: BlockJoltHostConfig,
    public_params: &BlockJoltNovaPublicParams,
    verifier_key: &BlockJoltNovaCompressedVerifierKey,
) -> Result<[u8; 32], DirectChunkedError> {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(D19_SETUP_DOMAIN);
    hasher.update((config.cycle_capacity as u64).to_le_bytes());
    hasher.update((config.ram_k as u64).to_le_bytes());
    update_setup_digest_with_postcard(
        &mut hasher,
        public_params,
        "D19 public-parameter serialization failed",
    )?;
    update_setup_digest_with_postcard(
        &mut hasher,
        verifier_key,
        "D19 verifier-key serialization failed",
    )?;
    Ok(hasher.finalize().into())
}

/// Reusable, shape-specific D19 preprocessing. It is created independently of
/// a production proof and identified by a digest of the Nova public parameters,
/// Spartan verifier key, protocol version, and block shape.
pub struct BlockJoltNovaSetup {
    config: BlockJoltHostConfig,
    setup_id: [u8; 32],
    public_params: Arc<BlockJoltNovaPublicParams>,
    prover_key: BlockJoltNovaCompressedProverKey,
    verifier_key: BlockJoltNovaCompressedVerifierKey,
}

impl BlockJoltNovaSetup {
    pub fn new(
        config: BlockJoltHostConfig,
        template: &VerifiedBlockJoltTransition,
    ) -> Result<Self, DirectChunkedError> {
        config.validate()?;
        let circuit = BlockJoltVerifierStepCircuit::new(
            config,
            template.statement.clone(),
            template.proof.clone(),
        )
        .map_err(DirectChunkedError::InvalidProofShape)?;
        let public_params = Arc::new(
            BlockJoltNovaPublicParams::setup(
                &circuit,
                &*nova_snark::traits::snark::default_ck_hint::<NovaPrimaryEngine>(),
                &*nova_snark::traits::snark::default_ck_hint::<NovaSecondaryEngine>(),
            )
            .map_err(|error| folding_error("D19 Nova setup failed", error))?,
        );
        let (prover_key, verifier_key) = BlockJoltNovaCompressedSnark::setup(&public_params)
            .map_err(|error| folding_error("D19 Spartan key setup failed", error))?;
        let setup_id = setup_identifier(config, &public_params, &verifier_key)?;
        Ok(Self {
            config,
            setup_id,
            public_params,
            prover_key,
            verifier_key,
        })
    }

    pub fn setup_id(&self) -> [u8; 32] {
        self.setup_id
    }

    pub fn config(&self) -> BlockJoltHostConfig {
        self.config
    }

    pub(super) fn public_params(&self) -> &BlockJoltNovaPublicParams {
        &self.public_params
    }

    pub(super) fn public_params_arc(&self) -> Arc<BlockJoltNovaPublicParams> {
        Arc::clone(&self.public_params)
    }

    pub(super) fn prover_key(&self) -> &BlockJoltNovaCompressedProverKey {
        &self.prover_key
    }

    pub(super) fn verifier_key(&self) -> &BlockJoltNovaCompressedVerifierKey {
        &self.verifier_key
    }
}

fn digest_words(value: &[u8; 32]) -> [NovaScalar; 4] {
    std::array::from_fn(|index| {
        NovaScalar::from(u64::from_le_bytes(
            value[index * 8..(index + 1) * 8].try_into().unwrap(),
        ))
    })
}

fn validate_final_output(
    block_count: usize,
    total_active_cycles: u64,
    final_transition: &VerifiedBlockJoltTransition,
    output: &[NovaScalar],
) -> Result<(), DirectChunkedError> {
    let statement = &final_transition.statement;
    if output.len() != super::BLOCK_JOLT_VERIFIER_Z_ARITY
        || output[BLOCK_SLOT] != NovaScalar::from(block_count as u64)
        || output[CYCLE_SLOT] != NovaScalar::from(statement.global_cycle_end)
        || output[MACHINE_OFFSET..MACHINE_OFFSET + 4] != digest_words(&statement.end.machine_state)
        || output[REGISTER_OFFSET..REGISTER_OFFSET + 4]
            != digest_words(&statement.end.register_state)
        || output[RAM_ROOT_SLOT]
            != nova_from_fr(&statement.end.ram_root.to_fr())
                .map_err(|error| folding_error("D16 RAM-root conversion failed", error))?
        || output[LOOKUP_STATE_SLOT]
            != nova_from_fr(&statement.lookup_accumulator_after.to_fr())
                .map_err(|error| folding_error("D16 lookup-state conversion failed", error))?
        || output[LOOKUP_ROUND_SLOT] != NovaScalar::from(statement.lookup_transcript_round_after)
        || output[TOTAL_CYCLES_SLOT] != NovaScalar::from(total_active_cycles)
        || output[TERMINATED_SLOT] != NovaScalar::from(u64::from(statement.terminal))
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D16 Nova output does not match the final compact block statement".to_string(),
        ));
    }
    Ok(())
}

/// Incremental D18 folder. It owns only Nova's constant-size recursive
/// accumulator plus the final compact statement; prior block proofs and trace
/// witnesses can be discarded immediately after `fold_transition` returns.
pub struct BlockJoltNovaFolder {
    config: BlockJoltHostConfig,
    public_params: Arc<BlockJoltNovaPublicParams>,
    setup_id: Option<[u8; 32]>,
    recursive_snark: BlockJoltNovaSnark,
    initial_z: Vec<NovaScalar>,
    block_count: usize,
    total_active_cycles: u64,
    final_transition: Option<VerifiedBlockJoltTransition>,
    terminated: bool,
}

impl BlockJoltNovaFolder {
    pub fn new(
        config: BlockJoltHostConfig,
        first_transition: &VerifiedBlockJoltTransition,
    ) -> Result<Self, DirectChunkedError> {
        let first = BlockJoltVerifierStepCircuit::new(
            config,
            first_transition.statement.clone(),
            first_transition.proof.clone(),
        )
        .map_err(DirectChunkedError::InvalidProofShape)?;
        let initial_z = first
            .initial_z()
            .map_err(DirectChunkedError::InvalidProofShape)?;
        let public_params = Arc::new(
            BlockJoltNovaPublicParams::setup(
                &first,
                &*nova_snark::traits::snark::default_ck_hint::<NovaPrimaryEngine>(),
                &*nova_snark::traits::snark::default_ck_hint::<NovaSecondaryEngine>(),
            )
            .map_err(|error| folding_error("D18 Nova public-parameter setup failed", error))?,
        );
        Self::initialize(config, first, initial_z, public_params, None)
    }

    pub fn new_with_setup(
        setup: &BlockJoltNovaSetup,
        first_transition: &VerifiedBlockJoltTransition,
    ) -> Result<Self, DirectChunkedError> {
        let first = BlockJoltVerifierStepCircuit::new(
            setup.config,
            first_transition.statement.clone(),
            first_transition.proof.clone(),
        )
        .map_err(DirectChunkedError::InvalidProofShape)?;
        let initial_z = first
            .initial_z()
            .map_err(DirectChunkedError::InvalidProofShape)?;
        Self::initialize(
            setup.config,
            first,
            initial_z,
            setup.public_params_arc(),
            Some(setup.setup_id),
        )
    }

    fn initialize(
        config: BlockJoltHostConfig,
        first: BlockJoltVerifierStepCircuit,
        initial_z: Vec<NovaScalar>,
        public_params: Arc<BlockJoltNovaPublicParams>,
        setup_id: Option<[u8; 32]>,
    ) -> Result<Self, DirectChunkedError> {
        let recursive_snark = BlockJoltNovaSnark::new(&public_params, &first, &initial_z)
            .map_err(|error| folding_error("D18 Nova initialization failed", error))?;
        Ok(Self {
            config,
            public_params,
            setup_id,
            recursive_snark,
            initial_z,
            block_count: 0,
            total_active_cycles: 0,
            final_transition: None,
            terminated: false,
        })
    }

    pub fn fold_transition(
        &mut self,
        transition: &VerifiedBlockJoltTransition,
    ) -> Result<(), DirectChunkedError> {
        if self.terminated {
            return Err(DirectChunkedError::InvalidProofShape(
                "D18 cannot fold another compact block after termination".to_string(),
            ));
        }
        if transition.statement.block_index != self.block_count as u64 {
            return Err(DirectChunkedError::InvalidProofShape(format!(
                "D18 expected block {}, received {}",
                self.block_count, transition.statement.block_index
            )));
        }
        let circuit = BlockJoltVerifierStepCircuit::new(
            self.config,
            transition.statement.clone(),
            transition.proof.clone(),
        )
        .map_err(DirectChunkedError::InvalidProofShape)?;
        self.recursive_snark
            .prove_step(&self.public_params, &circuit)
            .map_err(|error| {
                folding_error(
                    &format!("D18 Nova block {} failed", self.block_count),
                    error,
                )
            })?;
        self.total_active_cycles = self
            .total_active_cycles
            .checked_add(transition.statement.active_cycles)
            .ok_or_else(|| {
                DirectChunkedError::InvalidProofShape(
                    "D18 total active-cycle counter overflow".to_string(),
                )
            })?;
        self.block_count += 1;
        self.terminated = transition.statement.terminal;
        self.final_transition = Some(transition.clone());
        Ok(())
    }

    pub fn finish(self) -> Result<BlockJoltNovaFoldingProof, DirectChunkedError> {
        let final_transition = self
            .final_transition
            .as_ref()
            .ok_or(DirectChunkedError::EmptyTrace)?;
        let final_z = self
            .recursive_snark
            .verify(&self.public_params, self.block_count, &self.initial_z)
            .map_err(|error| folding_error("D18 Nova self-verification failed", error))?;
        validate_final_output(
            self.block_count,
            self.total_active_cycles,
            final_transition,
            &final_z,
        )?;
        let artifact = BlockJoltNovaFoldingProof {
            public_params: self.public_params,
            setup_id: self.setup_id,
            recursive_snark: self.recursive_snark,
            initial_z: self.initial_z,
            final_z,
            block_count: self.block_count,
        };
        artifact.verify()?;
        Ok(artifact)
    }
}

/// A self-verifying D16/D18 internal recursive artifact. Public parameters are
/// retained here only for debug-path verification. D19 consumes this envelope
/// and returns the pinned, Spartan-compressed production artifact.
pub struct BlockJoltNovaFoldingProof {
    public_params: Arc<BlockJoltNovaPublicParams>,
    setup_id: Option<[u8; 32]>,
    pub(super) recursive_snark: BlockJoltNovaSnark,
    pub(super) initial_z: Vec<NovaScalar>,
    pub(super) final_z: Vec<NovaScalar>,
    block_count: usize,
}

impl BlockJoltNovaFoldingProof {
    pub fn block_count(&self) -> usize {
        self.block_count
    }

    pub fn initial_z_bytes(&self) -> Vec<[u8; 32]> {
        self.initial_z
            .iter()
            .map(|value| value.to_bytes())
            .collect()
    }

    pub fn setup_id(&self) -> Option<[u8; 32]> {
        self.setup_id
    }

    pub fn final_z_bytes(&self) -> Vec<[u8; 32]> {
        self.final_z.iter().map(|value| value.to_bytes()).collect()
    }

    pub fn deferred_checkpoint(&self) -> ([u8; 32], [u8; 32]) {
        (
            self.final_z[DEFERRED_STATE_SLOT].to_bytes(),
            self.final_z[DEFERRED_ROUND_SLOT].to_bytes(),
        )
    }

    pub fn recursive_snark_bytes(&self) -> Result<Vec<u8>, DirectChunkedError> {
        postcard::to_stdvec(&self.recursive_snark)
            .map_err(|error| folding_error("D16 Nova serialization failed", error))
    }

    pub fn verify(&self) -> Result<(), DirectChunkedError> {
        let output = self
            .recursive_snark
            .verify(&self.public_params, self.block_count, &self.initial_z)
            .map_err(|error| folding_error("D16 Nova verification failed", error))?;
        if output != self.final_z {
            return Err(DirectChunkedError::InvalidProofShape(
                "D16 cached public output differs from Nova verification".to_string(),
            ));
        }
        Ok(())
    }
}

/// Folds a contiguous stream of host-accepted compact block transitions. The
/// D15 circuit, rather than the host acceptance flag, enforces every relation
/// and boundary inside Nova.
pub fn fold_verified_block_jolt_transitions(
    config: BlockJoltHostConfig,
    transitions: &[VerifiedBlockJoltTransition],
) -> Result<BlockJoltNovaFoldingProof, DirectChunkedError> {
    if transitions.is_empty() {
        return Err(DirectChunkedError::EmptyTrace);
    }
    if transitions
        .iter()
        .enumerate()
        .any(|(index, transition)| transition.statement.terminal && index + 1 != transitions.len())
    {
        return Err(DirectChunkedError::InvalidProofShape(
            "D16 cannot fold another compact block after termination".to_string(),
        ));
    }

    let mut folder = BlockJoltNovaFolder::new(config, &transitions[0])?;
    for transition in transitions {
        folder.fold_transition(transition)?;
    }
    folder.finish()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use common::constants::REGISTER_COUNT;
    use tracer::{
        instruction::{
            and::AND,
            format::format_r::{FormatR, RegisterStateFormatR},
            Cycle, RISCVCycle,
        },
        MachineBoundaryState, TraceBlock,
    };

    use super::*;
    use crate::zkvm::block::{BlockJoltProver, DirectChunkedPreprocessing};

    fn and_cycle(address: u64, rd: u8) -> Cycle {
        RISCVCycle::<AND> {
            instruction: AND {
                address,
                operands: FormatR { rd, rs1: 1, rs2: 2 },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            register_state: RegisterStateFormatR {
                rd: (0, 0x0a),
                rs1: 0xaa,
                rs2: 0x0f,
            },
            ram_access: (),
        }
        .into()
    }

    fn config() -> BlockJoltHostConfig {
        BlockJoltHostConfig {
            cycle_capacity: 2,
            ram_k: 2,
        }
    }

    #[test]
    fn d20_streamed_setup_digest_matches_legacy_vec_encoding() {
        let value = vec![(0u64, vec![1u8, 2, 3]), (u64::MAX, vec![4u8; 257])];
        let encoded = postcard::to_stdvec(&value).unwrap();

        let mut legacy = Sha3_256::new();
        legacy.update((encoded.len() as u64).to_le_bytes());
        legacy.update(&encoded);

        let mut streamed = Sha3_256::new();
        update_setup_digest_with_postcard(&mut streamed, &value, "test serialization").unwrap();

        assert_eq!(legacy.finalize().as_slice(), streamed.finalize().as_slice());
    }

    fn two_transitions() -> Vec<VerifiedBlockJoltTransition> {
        let cycle0 = and_cycle(0x8000_0000, 3);
        let cycle1 = and_cycle(0x8000_0004, 4);
        let cycle2 = and_cycle(0x8000_0008, 5);
        let preprocessing = DirectChunkedPreprocessing::from_trace_cycles(
            b"d16-two-block-nova",
            8,
            &[cycle0.clone(), cycle1.clone(), cycle2.clone()],
        )
        .unwrap();
        let mut registers0 = [0i64; REGISTER_COUNT as usize];
        registers0[1] = 0xaa;
        registers0[2] = 0x0f;
        let mut registers1 = registers0;
        registers1[3] = 0x0a;
        let mut registers2 = registers1;
        registers2[4] = 0x0a;
        let block0 = TraceBlock {
            block_index: 0,
            global_cycle_start: 0,
            active_cycles: 1,
            target_size: 2,
            start_state: MachineBoundaryState {
                global_cycle: 0,
                emulator_trace_len: 0,
                pc: 0x8000_0000,
                registers: registers0,
                terminated: false,
            },
            end_state: MachineBoundaryState {
                global_cycle: 1,
                emulator_trace_len: 1,
                pc: 0x8000_0004,
                registers: registers1,
                terminated: false,
            },
            cycles: vec![cycle0],
            ended_at_tick_boundary: true,
        };
        let block1 = TraceBlock {
            block_index: 1,
            global_cycle_start: 1,
            active_cycles: 1,
            target_size: 2,
            start_state: block0.end_state.clone(),
            end_state: MachineBoundaryState {
                global_cycle: 2,
                emulator_trace_len: 2,
                pc: 0x8000_0008,
                registers: registers2,
                terminated: false,
            },
            cycles: vec![cycle1.clone()],
            ended_at_tick_boundary: true,
        };
        let mut prover = BlockJoltProver::new(preprocessing, config(), BTreeMap::new()).unwrap();
        let first = prover.prove_block(&block0, Some(&cycle1)).unwrap();
        let second = prover.prove_block(&block1, Some(&cycle2)).unwrap();
        vec![first, second]
    }

    #[test]
    fn d16_executes_one_real_nova_step_per_compact_block() {
        let transitions = two_transitions();
        let proof = fold_verified_block_jolt_transitions(config(), &transitions).unwrap();
        assert_eq!(proof.block_count(), 2);
        assert_eq!(
            proof.initial_z_bytes().len(),
            super::super::BLOCK_JOLT_VERIFIER_Z_ARITY
        );
        assert_eq!(
            proof.final_z_bytes().len(),
            super::super::BLOCK_JOLT_VERIFIER_Z_ARITY
        );
        assert!(!proof.recursive_snark_bytes().unwrap().is_empty());
        proof.verify().unwrap();
    }

    #[test]
    fn d16_rejects_a_tampered_compact_block_inside_nova() {
        let mut transitions = two_transitions();
        transitions[1].proof.ram.relation_proof.final_claim.0[0] ^= 1;
        transitions[1].proof.seal();
        assert!(fold_verified_block_jolt_transitions(config(), &transitions).is_err());
    }
}
