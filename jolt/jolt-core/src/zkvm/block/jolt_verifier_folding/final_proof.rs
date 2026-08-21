//! D19 production proof boundary for direct block-Jolt V2.
//!
//! A reusable shape-specific setup is supplied out of band. The final artifact
//! stores only the public recursive input/output, one Spartan-compressed Nova
//! proof, and the serialized deferred Dory decider. Debug RecursiveSNARKs,
//! public parameters, proving keys, trace rows, and polynomial witnesses are
//! excluded.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::{io::Write, time::Instant};

use super::super::{DirectChunkedError, NovaScalar};
use super::{
    compact_verifier_circuit::{DEFERRED_ROUND_SLOT, DEFERRED_STATE_SLOT, TERMINATED_SLOT},
    folding::{BlockJoltNovaCompressedSnark, BlockJoltNovaSetup},
    BlockJoltDeferredPcsProof, BlockJoltStreamingProof, BLOCK_JOLT_PROTOCOL_VERSION,
    BLOCK_JOLT_VERIFIER_Z_ARITY,
};

const D19_FINAL_WIRE_VERSION: u16 = 1;
const D19_FINAL_DIGEST_DOMAIN: &[u8] = b"block-jolt-final-proof-v1";

fn final_error(message: impl Into<String>) -> DirectChunkedError {
    DirectChunkedError::InvalidProofShape(message.into())
}

fn scalar_vector(bytes: &[[u8; 32]], label: &str) -> Result<Vec<NovaScalar>, DirectChunkedError> {
    bytes
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            Option::from(NovaScalar::from_bytes(bytes)).ok_or_else(|| {
                final_error(format!(
                    "D19 {label} element {index} is not a canonical Nova scalar"
                ))
            })
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockJoltFinalStatement {
    pub protocol_version: String,
    pub setup_id: [u8; 32],
    pub block_count: u64,
    pub initial_z: Vec<[u8; 32]>,
    pub final_z: Vec<[u8; 32]>,
}

impl BlockJoltFinalStatement {
    fn validate(&self) -> Result<(), DirectChunkedError> {
        if self.protocol_version != BLOCK_JOLT_PROTOCOL_VERSION {
            return Err(final_error("D19 final statement protocol mismatch"));
        }
        if self.block_count == 0
            || self.initial_z.len() != BLOCK_JOLT_VERIFIER_Z_ARITY
            || self.final_z.len() != BLOCK_JOLT_VERIFIER_Z_ARITY
        {
            return Err(final_error("D19 final statement has an invalid shape"));
        }
        let _ = scalar_vector(&self.initial_z, "initial state")?;
        let final_z = scalar_vector(&self.final_z, "final state")?;
        if final_z[TERMINATED_SLOT] != NovaScalar::from(1u64) {
            return Err(final_error("D19 final recursive state is not terminal"));
        }
        Ok(())
    }

    pub fn deferred_checkpoint(&self) -> Result<([u8; 32], [u8; 32]), DirectChunkedError> {
        self.validate()?;
        Ok((
            self.final_z[DEFERRED_STATE_SLOT],
            self.final_z[DEFERRED_ROUND_SLOT],
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockJoltFinalProof {
    wire_version: u16,
    statement: BlockJoltFinalStatement,
    compressed_nova_proof: Vec<u8>,
    deferred_pcs_proof: Vec<u8>,
    proof_digest: [u8; 32],
}

#[derive(Serialize)]
struct BlockJoltFinalDigestView<'a> {
    wire_version: u16,
    statement: &'a BlockJoltFinalStatement,
    compressed_nova_proof: &'a [u8],
    deferred_pcs_proof: &'a [u8],
    proof_digest: [u8; 32],
}

#[derive(Default)]
struct CountingWriter {
    bytes: usize,
}

impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("postcard size overflow"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct DigestWriter<'a> {
    hasher: &'a mut Sha3_256,
}

impl Write for DigestWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.hasher.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn postcard_size<T: Serialize + ?Sized>(
    value: &T,
    label: &str,
) -> Result<usize, DirectChunkedError> {
    postcard::to_io(value, CountingWriter::default())
        .map(|writer| writer.bytes)
        .map_err(|error| final_error(format!("D21 {label} size encoding failed: {error}")))
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockJoltFinalizationMetrics {
    pub streaming_debug_verify_micros: u128,
    pub spartan_prove_micros: u128,
    pub spartan_self_verify_micros: u128,
    pub spartan_serialize_micros: u128,
    pub dory_serialize_micros: u128,
    #[serde(default)]
    pub final_digest_micros: u128,
    pub final_self_verify_micros: u128,
    #[serde(default)]
    pub final_artifact_size_micros: u128,
    pub compressed_nova_bytes: usize,
    pub deferred_dory_bytes: usize,
    pub final_artifact_bytes: usize,
}

impl BlockJoltFinalProof {
    pub fn statement(&self) -> &BlockJoltFinalStatement {
        &self.statement
    }

    pub fn compressed_nova_bytes(&self) -> usize {
        self.compressed_nova_proof.len()
    }

    pub fn deferred_pcs_bytes(&self) -> usize {
        self.deferred_pcs_proof.len()
    }

    pub fn serialized_size(&self) -> Result<usize, DirectChunkedError> {
        self.validate_structure()?;
        postcard_size(self, "final proof")
    }

    fn recompute_digest(&self) -> Result<[u8; 32], DirectChunkedError> {
        let canonical = BlockJoltFinalDigestView {
            wire_version: self.wire_version,
            statement: &self.statement,
            compressed_nova_proof: &self.compressed_nova_proof,
            deferred_pcs_proof: &self.deferred_pcs_proof,
            proof_digest: [0; 32],
        };
        let encoded_len = postcard_size(&canonical, "final digest")?;
        let mut hasher = Sha3_256::new();
        hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
        hasher.update(D19_FINAL_DIGEST_DOMAIN);
        hasher.update(
            u64::try_from(encoded_len)
                .map_err(|error| final_error(format!("D21 final digest size overflow: {error}")))?
                .to_le_bytes(),
        );
        postcard::to_io(
            &canonical,
            DigestWriter {
                hasher: &mut hasher,
            },
        )
        .map_err(|error| final_error(format!("D21 final digest encoding failed: {error}")))?;
        Ok(hasher.finalize().into())
    }

    fn seal(&mut self) -> Result<(), DirectChunkedError> {
        self.proof_digest = self.recompute_digest()?;
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), DirectChunkedError> {
        if self.wire_version != D19_FINAL_WIRE_VERSION {
            return Err(final_error("D19 final proof wire version mismatch"));
        }
        self.statement.validate()?;
        if self.compressed_nova_proof.is_empty() || self.deferred_pcs_proof.is_empty() {
            return Err(final_error(
                "D19 final proof is missing a cryptographic component",
            ));
        }
        if self.proof_digest != self.recompute_digest()? {
            return Err(final_error("D19 final proof digest mismatch"));
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, DirectChunkedError> {
        self.validate_structure()?;
        postcard::to_stdvec(self)
            .map_err(|error| final_error(format!("D19 final proof serialization failed: {error}")))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DirectChunkedError> {
        let proof: Self = postcard::from_bytes(bytes)
            .map_err(|error| final_error(format!("D19 final proof decoding failed: {error}")))?;
        proof.validate_structure()?;
        Ok(proof)
    }

    pub fn verify(&self, setup: &BlockJoltNovaSetup) -> Result<(), DirectChunkedError> {
        self.validate_structure()?;
        if self.statement.setup_id != setup.setup_id() {
            return Err(final_error("D19 final proof uses a different setup"));
        }
        let block_count = usize::try_from(self.statement.block_count)
            .map_err(|error| final_error(format!("D19 block-count conversion failed: {error}")))?;
        let initial_z = scalar_vector(&self.statement.initial_z, "initial state")?;
        let expected_final_z = scalar_vector(&self.statement.final_z, "final state")?;
        let compressed: BlockJoltNovaCompressedSnark =
            postcard::from_bytes(&self.compressed_nova_proof).map_err(|error| {
                final_error(format!("D19 Spartan proof decoding failed: {error}"))
            })?;
        let observed_final_z = compressed
            .verify(setup.verifier_key(), block_count, &initial_z)
            .map_err(|error| final_error(format!("D19 Spartan verification failed: {error:?}")))?;
        if observed_final_z != expected_final_z {
            return Err(final_error(
                "D19 Spartan output differs from the final public state",
            ));
        }

        let deferred = BlockJoltDeferredPcsProof::from_bytes(&self.deferred_pcs_proof)?;
        deferred.verify_against_checkpoint(block_count, self.statement.deferred_checkpoint()?)
    }
}

/// Compresses an externally-setup-bound D18 artifact and immediately verifies
/// the standalone production proof before returning it.
pub fn compress_block_jolt_final_proof(
    setup: &BlockJoltNovaSetup,
    streaming: &BlockJoltStreamingProof,
) -> Result<BlockJoltFinalProof, DirectChunkedError> {
    compress_block_jolt_final_proof_with_metrics(setup, streaming).map(|(proof, _)| proof)
}

pub fn compress_block_jolt_final_proof_with_metrics(
    setup: &BlockJoltNovaSetup,
    streaming: &BlockJoltStreamingProof,
) -> Result<(BlockJoltFinalProof, BlockJoltFinalizationMetrics), DirectChunkedError> {
    let streaming_verify_started = Instant::now();
    streaming.verify()?;
    let streaming_debug_verify_micros = streaming_verify_started.elapsed().as_micros();
    let folding = streaming.folding();
    if folding.setup_id() != Some(setup.setup_id()) {
        return Err(final_error(
            "D19 refuses to compress a per-proof or differently configured Nova setup",
        ));
    }
    let spartan_prove_started = Instant::now();
    let compressed = BlockJoltNovaCompressedSnark::prove(
        setup.public_params(),
        setup.prover_key(),
        &folding.recursive_snark,
    )
    .map_err(|error| final_error(format!("D19 Spartan proving failed: {error:?}")))?;
    let spartan_prove_micros = spartan_prove_started.elapsed().as_micros();
    let spartan_verify_started = Instant::now();
    let observed_final_z = compressed
        .verify(
            setup.verifier_key(),
            folding.block_count(),
            &folding.initial_z,
        )
        .map_err(|error| final_error(format!("D19 Spartan self-verification failed: {error:?}")))?;
    let spartan_self_verify_micros = spartan_verify_started.elapsed().as_micros();
    if observed_final_z != folding.final_z {
        return Err(final_error(
            "D19 compressed and recursive Nova outputs differ",
        ));
    }

    let spartan_serialize_started = Instant::now();
    let compressed_nova_proof = postcard::to_stdvec(&compressed)
        .map_err(|error| final_error(format!("D19 Spartan serialization failed: {error}")))?;
    let spartan_serialize_micros = spartan_serialize_started.elapsed().as_micros();
    let dory_serialize_started = Instant::now();
    let deferred_pcs_proof = streaming.deferred_pcs().to_bytes()?;
    let dory_serialize_micros = dory_serialize_started.elapsed().as_micros();
    let mut proof = BlockJoltFinalProof {
        wire_version: D19_FINAL_WIRE_VERSION,
        statement: BlockJoltFinalStatement {
            protocol_version: BLOCK_JOLT_PROTOCOL_VERSION.to_string(),
            setup_id: setup.setup_id(),
            block_count: u64::try_from(folding.block_count()).map_err(|error| {
                final_error(format!("D19 block-count serialization failed: {error}"))
            })?,
            initial_z: folding.initial_z_bytes(),
            final_z: folding.final_z_bytes(),
        },
        compressed_nova_proof,
        deferred_pcs_proof,
        proof_digest: [0; 32],
    };
    let final_digest_started = Instant::now();
    proof.seal()?;
    let final_digest_micros = final_digest_started.elapsed().as_micros();
    let final_verify_started = Instant::now();
    proof.verify(setup)?;
    let final_self_verify_micros = final_verify_started.elapsed().as_micros();
    let final_artifact_size_started = Instant::now();
    let final_artifact_bytes = proof.serialized_size()?;
    let final_artifact_size_micros = final_artifact_size_started.elapsed().as_micros();
    let metrics = BlockJoltFinalizationMetrics {
        streaming_debug_verify_micros,
        spartan_prove_micros,
        spartan_self_verify_micros,
        spartan_serialize_micros,
        dory_serialize_micros,
        final_digest_micros,
        final_self_verify_micros,
        final_artifact_size_micros,
        compressed_nova_bytes: proof.compressed_nova_proof.len(),
        deferred_dory_bytes: proof.deferred_pcs_proof.len(),
        final_artifact_bytes,
    };
    Ok((proof, metrics))
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

    use super::super::{
        prove_block_jolt_streaming_with_setup, prove_pcs_bound_block_jolt_transition,
        BlockJoltHostConfig, BlockJoltProver,
    };
    use super::*;
    use crate::zkvm::block::DirectChunkedPreprocessing;

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
            b"d19-final-spartan",
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
    fn d19_serializes_and_verifies_the_standalone_spartan_artifact() {
        let (preprocessing, blocks) = two_terminal_blocks();
        let config = BlockJoltHostConfig {
            cycle_capacity: 2,
            ram_k: 2,
        };

        // Setup is shape-specific and independent of the later streaming run.
        let mut template_prover =
            BlockJoltProver::new(preprocessing.clone(), config, BTreeMap::new()).unwrap();
        let template = prove_pcs_bound_block_jolt_transition(
            &mut template_prover,
            &blocks[0],
            blocks[1].cycles.first(),
        )
        .unwrap();
        let setup = BlockJoltNovaSetup::new(config, template.transition()).unwrap();
        drop(template);
        drop(template_prover);

        let (streaming, metrics) =
            prove_block_jolt_streaming_with_setup(&preprocessing, &setup, blocks).unwrap();
        assert_eq!(streaming.folding().setup_id(), Some(setup.setup_id()));
        assert!(metrics.residency_is_bounded(config.cycle_capacity));

        let proof = compress_block_jolt_final_proof(&setup, &streaming).unwrap();
        let mut legacy_digest_wire = proof.clone();
        legacy_digest_wire.proof_digest = [0; 32];
        let legacy_digest_bytes = postcard::to_stdvec(&legacy_digest_wire).unwrap();
        let streamed_digest_view = BlockJoltFinalDigestView {
            wire_version: proof.wire_version,
            statement: &proof.statement,
            compressed_nova_proof: &proof.compressed_nova_proof,
            deferred_pcs_proof: &proof.deferred_pcs_proof,
            proof_digest: [0; 32],
        };
        assert_eq!(
            postcard::to_stdvec(&streamed_digest_view).unwrap(),
            legacy_digest_bytes,
            "D21 borrowed digest view must preserve the D19 wire encoding"
        );
        let encoded = proof.to_bytes().unwrap();
        assert_eq!(proof.serialized_size().unwrap(), encoded.len());
        let decoded = BlockJoltFinalProof::from_bytes(&encoded).unwrap();
        decoded.verify(&setup).unwrap();

        eprintln!("D19_FINAL_ARTIFACT_BYTES={}", encoded.len());
        eprintln!(
            "D19_COMPRESSED_NOVA_BYTES={}",
            proof.compressed_nova_bytes()
        );
        eprintln!("D19_DEFERRED_DORY_BYTES={}", proof.deferred_pcs_bytes());
        eprintln!("D19_TRACE_SPOOL_BYTES={}", metrics.spooled_trace_bytes);
        eprintln!("D19_PCS_SPOOL_BYTES={}", metrics.spooled_pcs_witness_bytes);

        let mut wire_tamper = encoded;
        *wire_tamper.last_mut().unwrap() ^= 1;
        assert!(BlockJoltFinalProof::from_bytes(&wire_tamper).is_err());

        let mut setup_tamper = proof.clone();
        setup_tamper.statement.setup_id[0] ^= 1;
        setup_tamper.seal().unwrap();
        assert!(setup_tamper.verify(&setup).is_err());

        let mut output_tamper = proof.clone();
        output_tamper.statement.final_z[0][0] ^= 1;
        output_tamper.seal().unwrap();
        assert!(output_tamper.verify(&setup).is_err());

        let mut nova_tamper = proof.clone();
        nova_tamper.compressed_nova_proof[0] ^= 1;
        nova_tamper.seal().unwrap();
        assert!(nova_tamper.verify(&setup).is_err());

        let mut dory_tamper = proof;
        dory_tamper.deferred_pcs_proof[0] ^= 1;
        dory_tamper.seal().unwrap();
        assert!(dory_tamper.verify(&setup).is_err());
    }
}
