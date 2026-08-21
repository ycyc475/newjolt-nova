//! D17 exact deferred Dory/PCS closure for compact block-Jolt proofs.
//!
//! Every endpoint polynomial is committed before any relation challenge is
//! sampled. The canonical commitment-bundle identifier is absorbed by all four
//! relation transcripts and by every deferred claim folded in D15/D16. After
//! Nova folding, equal-point openings are combined with a transcript-derived
//! random linear combination and verified against the exact final deferred
//! checkpoint.

use std::{
    collections::BTreeMap,
    io::{BufReader, Cursor, Read, Write},
    time::Instant,
};

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use tempfile::NamedTempFile;
use tracer::{instruction::Cycle, TraceBlock};

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{
                bind_opening_inputs, ArkDoryProof, ArkGT, DoryCommitmentScheme, DoryContext,
                DoryGlobals, DoryLayout, DoryOpeningProofHint,
            },
        },
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{PoseidonTranscript, Transcript},
};

use super::super::{
    direct_lookup::{compact_lookup_polynomials, field_from_digest},
    direct_pcs::direct_dory_lock,
    direct_register::compact_register_polynomials,
    DirectChunkedError,
};
use super::{
    compact_verifier_circuit::native_deferred_checkpoint, cpu::block_cpu_polynomials,
    ram::block_ram_polynomials, BlockJoltNovaFoldingProof, BlockJoltProver, DeferredPcsClaim,
    FieldElement, VerifiedBlockJoltTransition, BLOCK_JOLT_PROTOCOL_VERSION,
};

const D17_BUNDLE_DOMAIN: &[u8] = b"block-jolt-dory-bundle-v1";
const D17_OPENING_DOMAIN: &[u8] = b"block-jolt-dory-open-v1";
const D18_SPOOL_DOMAIN: &[u8] = b"block-jolt-deferred-pcs-spool-v2";
const D18_SPOOL_VERSION: u16 = 2;
const D19_DORY_WIRE_VERSION: u16 = 1;

fn pcs_error(message: impl Into<String>) -> DirectChunkedError {
    DirectChunkedError::InvalidProofShape(message.into())
}

fn commitment_bytes(commitment: &ArkGT) -> Result<Vec<u8>, DirectChunkedError> {
    let mut bytes = Vec::new();
    commitment
        .serialize_compressed(&mut bytes)
        .map_err(|error| pcs_error(format!("D17 commitment serialization failed: {error}")))?;
    Ok(bytes)
}

fn canonical_bytes<T: CanonicalSerialize>(
    value: &T,
    label: &str,
) -> Result<Vec<u8>, DirectChunkedError> {
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .map_err(|error| pcs_error(format!("D18 {label} serialization failed: {error}")))?;
    Ok(bytes)
}

fn canonical_from_bytes<T: CanonicalDeserialize>(
    bytes: &[u8],
    label: &str,
) -> Result<T, DirectChunkedError> {
    let mut cursor = Cursor::new(bytes);
    let value = T::deserialize_compressed(&mut cursor)
        .map_err(|error| pcs_error(format!("D18 {label} deserialization failed: {error}")))?;
    if cursor.position() != bytes.len() as u64 {
        return Err(pcs_error(format!(
            "D18 {label} contains trailing serialized bytes"
        )));
    }
    Ok(value)
}

fn commitment_bundle_id(commitments: &[ArkGT]) -> Result<[u8; 32], DirectChunkedError> {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(D17_BUNDLE_DOMAIN);
    hasher.update((commitments.len() as u64).to_le_bytes());
    for (index, commitment) in commitments.iter().enumerate() {
        let encoded = commitment_bytes(commitment)?;
        hasher.update((index as u64).to_le_bytes());
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(encoded);
    }
    let digest: [u8; 32] = hasher.finalize().into();
    Ok(FieldElement::from_fr(&field_from_digest(&digest)).0)
}

fn commit_polynomials(
    polynomials: &[MultilinearPolynomial<Fr>],
) -> Result<(Vec<ArkGT>, Vec<DoryOpeningProofHint>), DirectChunkedError> {
    if polynomials.is_empty() {
        return Err(pcs_error("D17 cannot commit an empty polynomial bundle"));
    }
    let mut by_dimension = BTreeMap::<usize, Vec<usize>>::new();
    for (index, polynomial) in polynomials.iter().enumerate() {
        by_dimension
            .entry(polynomial.get_num_vars())
            .or_default()
            .push(index);
    }
    let maximum_num_vars = *by_dimension
        .keys()
        .next_back()
        .ok_or_else(|| pcs_error("D17 cannot commit an empty polynomial bundle"))?;
    if maximum_num_vars == 0 {
        return Err(pcs_error(
            "D17 does not support a zero-variable Dory polynomial",
        ));
    }
    let mut committed = vec![None; polynomials.len()];
    // Dory's prepared-generator cache is process-global. Growing it after a
    // smaller commitment has been created can make a later opening use a
    // different prepared setup prefix. Initialize the largest bucket first
    // and use that single setup for every exact-dimension context in the block.
    let _lock = direct_dory_lock();
    let setup = DoryCommitmentScheme::setup_prover(maximum_num_vars);
    for (num_vars, indices) in by_dimension {
        if num_vars == 0 {
            return Err(pcs_error(
                "D17 does not support a zero-variable Dory polynomial",
            ));
        }
        let _context = DoryGlobals::initialize_context(
            1,
            1usize << num_vars,
            DoryContext::Main,
            Some(DoryLayout::CycleMajor),
        );
        let batch = indices
            .iter()
            .map(|index| &polynomials[*index])
            .collect::<Vec<_>>();
        for (index, value) in indices
            .into_iter()
            .zip(DoryCommitmentScheme::batch_commit(&batch, &setup))
        {
            committed[index] = Some(value);
        }
    }
    let (commitments, hints): (Vec<_>, Vec<_>) = committed
        .into_iter()
        .map(|value| value.ok_or_else(|| pcs_error("D17 omitted a polynomial commitment")))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    Ok((commitments, hints))
}

/// Prover-only PCS material for one compact block. It is intentionally absent
/// from the recursive and final proof artifacts.
pub struct BlockJoltDeferredPcsWitness {
    block_index: u64,
    commitment_id: [u8; 32],
    polynomials: Vec<MultilinearPolynomial<Fr>>,
    commitments: Vec<ArkGT>,
    hints: Vec<DoryOpeningProofHint>,
}

/// A compact transition whose endpoint polynomials were committed before its
/// Fiat--Shamir relation challenges.
pub struct PcsBoundBlockJoltTransition {
    pub transition: VerifiedBlockJoltTransition,
    witness: BlockJoltDeferredPcsWitness,
    metrics: BlockJoltPcsProvingMetrics,
}

impl PcsBoundBlockJoltTransition {
    pub fn transition(&self) -> &VerifiedBlockJoltTransition {
        &self.transition
    }

    pub fn metrics(&self) -> &BlockJoltPcsProvingMetrics {
        &self.metrics
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BlockJoltPcsProvingMetrics {
    pub lookup_polynomial_micros: u128,
    pub register_polynomial_micros: u128,
    pub ram_polynomial_micros: u128,
    pub cpu_polynomial_micros: u128,
    pub commitment_micros: u128,
    pub compact_transition_micros: u128,
    pub witness_validation_micros: u128,
    pub polynomial_count: usize,
    pub coefficient_count: usize,
}

impl BlockJoltPcsProvingMetrics {
    pub fn accumulate(&mut self, other: &Self) {
        self.lookup_polynomial_micros += other.lookup_polynomial_micros;
        self.register_polynomial_micros += other.register_polynomial_micros;
        self.ram_polynomial_micros += other.ram_polynomial_micros;
        self.cpu_polynomial_micros += other.cpu_polynomial_micros;
        self.commitment_micros += other.commitment_micros;
        self.compact_transition_micros += other.compact_transition_micros;
        self.witness_validation_micros += other.witness_validation_micros;
        self.polynomial_count += other.polynomial_count;
        self.coefficient_count += other.coefficient_count;
    }
}

#[derive(Serialize, Deserialize)]
struct DeferredPcsSpoolSparseCoefficient {
    index: u64,
    value: FieldElement,
}

#[derive(Serialize, Deserialize)]
enum DeferredPcsSpoolCoefficients {
    Dense(Vec<FieldElement>),
    Sparse(Vec<DeferredPcsSpoolSparseCoefficient>),
}

#[derive(Serialize, Deserialize)]
struct DeferredPcsSpoolPolynomial {
    num_vars: u32,
    coefficients: DeferredPcsSpoolCoefficients,
}

impl DeferredPcsSpoolPolynomial {
    fn from_polynomial(polynomial: &MultilinearPolynomial<Fr>) -> Result<Self, DirectChunkedError> {
        let num_vars = u32::try_from(polynomial.get_num_vars())
            .map_err(|error| pcs_error(format!("D21 polynomial dimension overflow: {error}")))?;
        let mut sparse = Vec::new();
        for index in 0..polynomial.len() {
            let coefficient = polynomial.get_coeff(index);
            if !coefficient.is_zero() {
                sparse.push(DeferredPcsSpoolSparseCoefficient {
                    index: u64::try_from(index).map_err(|error| {
                        pcs_error(format!("D21 sparse coefficient index overflow: {error}"))
                    })?,
                    value: FieldElement::from_fr(&coefficient),
                });
            }
        }
        let dense_estimate = polynomial
            .len()
            .saturating_mul(std::mem::size_of::<FieldElement>());
        let sparse_estimate = sparse
            .len()
            .saturating_mul(std::mem::size_of::<u64>() + std::mem::size_of::<FieldElement>());
        let coefficients = if sparse_estimate < dense_estimate {
            DeferredPcsSpoolCoefficients::Sparse(sparse)
        } else {
            DeferredPcsSpoolCoefficients::Dense(
                (0..polynomial.len())
                    .map(|index| FieldElement::from_fr(&polynomial.get_coeff(index)))
                    .collect(),
            )
        };
        Ok(Self {
            num_vars,
            coefficients,
        })
    }

    fn coefficient_count(&self) -> Result<usize, DirectChunkedError> {
        if self.num_vars == 0 {
            return Err(pcs_error(
                "D21 spooled polynomial cannot have zero variables",
            ));
        }
        1usize
            .checked_shl(self.num_vars)
            .ok_or_else(|| pcs_error("D21 spooled polynomial dimension exceeds the platform limit"))
    }

    fn nonzero_coefficient_count(&self) -> usize {
        match &self.coefficients {
            DeferredPcsSpoolCoefficients::Dense(coefficients) => coefficients
                .iter()
                .filter(|coefficient| !coefficient.to_fr().is_zero())
                .count(),
            DeferredPcsSpoolCoefficients::Sparse(coefficients) => coefficients.len(),
        }
    }

    fn is_sparse(&self) -> bool {
        matches!(&self.coefficients, DeferredPcsSpoolCoefficients::Sparse(_))
    }

    fn into_polynomial(
        self,
        maximum_coefficient_count: usize,
    ) -> Result<MultilinearPolynomial<Fr>, DirectChunkedError> {
        let expected_len = self.coefficient_count()?;
        if expected_len > maximum_coefficient_count {
            return Err(pcs_error(
                "D21 spooled polynomial exceeds the captured coefficient bound",
            ));
        }
        let coefficients = match self.coefficients {
            DeferredPcsSpoolCoefficients::Dense(coefficients) => {
                if coefficients.len() != expected_len {
                    return Err(pcs_error(
                        "D21 dense spooled polynomial has an invalid dimension",
                    ));
                }
                coefficients
                    .into_iter()
                    .map(|coefficient| coefficient.to_fr())
                    .collect::<Vec<_>>()
            }
            DeferredPcsSpoolCoefficients::Sparse(entries) => {
                let mut coefficients = vec![Fr::zero(); expected_len];
                let mut previous = None;
                for entry in entries {
                    let index = usize::try_from(entry.index).map_err(|error| {
                        pcs_error(format!(
                            "D21 sparse coefficient index conversion failed: {error}"
                        ))
                    })?;
                    let value = entry.value.to_fr();
                    if index >= expected_len
                        || previous.is_some_and(|previous| index <= previous)
                        || value.is_zero()
                    {
                        return Err(pcs_error("D21 sparse spooled polynomial is not canonical"));
                    }
                    coefficients[index] = value;
                    previous = Some(index);
                }
                coefficients
            }
        };
        Ok(MultilinearPolynomial::from(coefficients))
    }
}

#[derive(Serialize, Deserialize)]
struct DeferredPcsSpoolRecord {
    version: u16,
    block_index: u64,
    commitment_id: [u8; 32],
    transition: VerifiedBlockJoltTransition,
    polynomials: Vec<DeferredPcsSpoolPolynomial>,
    commitments: Vec<Vec<u8>>,
    hints: Vec<Vec<u8>>,
}

impl DeferredPcsSpoolRecord {
    fn from_bound(bound: PcsBoundBlockJoltTransition) -> Result<Self, DirectChunkedError> {
        let BlockJoltDeferredPcsWitness {
            block_index,
            commitment_id,
            polynomials,
            commitments,
            hints,
        } = bound.witness;
        let polynomials = polynomials
            .iter()
            .map(DeferredPcsSpoolPolynomial::from_polynomial)
            .collect::<Result<Vec<_>, DirectChunkedError>>()?;
        let commitments = commitments
            .iter()
            .map(|value| canonical_bytes(value, "Dory commitment"))
            .collect::<Result<Vec<_>, _>>()?;
        let hints = hints
            .iter()
            .map(|value| canonical_bytes(value, "Dory opening hint"))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            version: D18_SPOOL_VERSION,
            block_index,
            commitment_id,
            transition: bound.transition,
            polynomials,
            commitments,
            hints,
        })
    }

    fn into_bound(
        self,
        maximum_coefficient_count: usize,
    ) -> Result<PcsBoundBlockJoltTransition, DirectChunkedError> {
        if self.version != D18_SPOOL_VERSION {
            return Err(pcs_error("D18 PCS spool record has an unsupported version"));
        }
        let polynomials = self
            .polynomials
            .into_iter()
            .map(|polynomial| polynomial.into_polynomial(maximum_coefficient_count))
            .collect::<Result<Vec<_>, DirectChunkedError>>()?;
        let commitments = self
            .commitments
            .iter()
            .map(|bytes| canonical_from_bytes(bytes, "Dory commitment"))
            .collect::<Result<Vec<ArkGT>, _>>()?;
        let hints = self
            .hints
            .iter()
            .map(|bytes| canonical_from_bytes(bytes, "Dory opening hint"))
            .collect::<Result<Vec<DoryOpeningProofHint>, _>>()?;
        if commitments.len() != polynomials.len()
            || hints.len() != polynomials.len()
            || commitment_bundle_id(&commitments)? != self.commitment_id
        {
            return Err(pcs_error("D18 spooled PCS witness bundle is inconsistent"));
        }
        let bound = PcsBoundBlockJoltTransition {
            transition: self.transition,
            witness: BlockJoltDeferredPcsWitness {
                block_index: self.block_index,
                commitment_id: self.commitment_id,
                polynomials,
                commitments,
                hints,
            },
            metrics: BlockJoltPcsProvingMetrics::default(),
        };
        validate_bound_transition(&bound)?;
        Ok(bound)
    }
}

fn spool_record_digest(encoded: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(BLOCK_JOLT_PROTOCOL_VERSION.as_bytes());
    hasher.update(D18_SPOOL_DOMAIN);
    hasher.update((encoded.len() as u64).to_le_bytes());
    hasher.update(encoded);
    hasher.finalize().into()
}

fn spool_io_error(error: impl core::fmt::Display) -> DirectChunkedError {
    pcs_error(format!("D18 PCS spool I/O failed: {error}"))
}

/// Disk-backed prover-only witness store. Records are length-delimited and
/// checksum-bound to the protocol so truncation, reordering, or mutation is
/// detected before a Dory opening is produced.
pub struct BlockJoltDeferredPcsSpool {
    file: NamedTempFile,
    block_count: usize,
    bytes_written: usize,
    max_record_bytes: usize,
    logical_coefficient_bytes: usize,
    nonzero_coefficient_count: usize,
    dense_polynomial_count: usize,
    sparse_polynomial_count: usize,
    max_polynomial_coefficients: usize,
}

impl BlockJoltDeferredPcsSpool {
    pub fn new() -> Result<Self, DirectChunkedError> {
        Ok(Self {
            file: NamedTempFile::new().map_err(spool_io_error)?,
            block_count: 0,
            bytes_written: 0,
            max_record_bytes: 0,
            logical_coefficient_bytes: 0,
            nonzero_coefficient_count: 0,
            dense_polynomial_count: 0,
            sparse_polynomial_count: 0,
            max_polynomial_coefficients: 0,
        })
    }

    pub fn push(&mut self, bound: PcsBoundBlockJoltTransition) -> Result<(), DirectChunkedError> {
        if bound.transition.statement.block_index != self.block_count as u64
            || bound.witness.block_index != self.block_count as u64
        {
            return Err(pcs_error("D18 PCS spool block sequence is not canonical"));
        }
        let record = DeferredPcsSpoolRecord::from_bound(bound)?;
        for polynomial in &record.polynomials {
            let coefficient_count = polynomial.coefficient_count()?;
            self.max_polynomial_coefficients =
                self.max_polynomial_coefficients.max(coefficient_count);
            self.logical_coefficient_bytes = self.logical_coefficient_bytes.saturating_add(
                coefficient_count.saturating_mul(std::mem::size_of::<FieldElement>()),
            );
            self.nonzero_coefficient_count = self
                .nonzero_coefficient_count
                .saturating_add(polynomial.nonzero_coefficient_count());
            if polynomial.is_sparse() {
                self.sparse_polynomial_count += 1;
            } else {
                self.dense_polynomial_count += 1;
            }
        }
        let encoded = postcard::to_stdvec(&record)
            .map_err(|error| pcs_error(format!("D18 PCS spool encoding failed: {error}")))?;
        let encoded_len = u64::try_from(encoded.len())
            .map_err(|error| pcs_error(format!("D18 PCS spool record is too large: {error}")))?;
        let digest = spool_record_digest(&encoded);
        self.file
            .as_file_mut()
            .write_all(&encoded_len.to_le_bytes())
            .and_then(|_| self.file.as_file_mut().write_all(&digest))
            .and_then(|_| self.file.as_file_mut().write_all(&encoded))
            .map_err(spool_io_error)?;
        self.block_count += 1;
        self.bytes_written = self
            .bytes_written
            .saturating_add(8 + digest.len() + encoded.len());
        self.max_record_bytes = self.max_record_bytes.max(encoded.len());
        Ok(())
    }

    pub fn block_count(&self) -> usize {
        self.block_count
    }

    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }

    pub fn max_record_bytes(&self) -> usize {
        self.max_record_bytes
    }

    pub fn logical_coefficient_bytes(&self) -> usize {
        self.logical_coefficient_bytes
    }

    pub fn nonzero_coefficient_count(&self) -> usize {
        self.nonzero_coefficient_count
    }

    pub fn dense_polynomial_count(&self) -> usize {
        self.dense_polynomial_count
    }

    pub fn sparse_polynomial_count(&self) -> usize {
        self.sparse_polynomial_count
    }

    fn replay(&self) -> Result<DeferredPcsSpoolReplay, DirectChunkedError> {
        self.file.as_file().sync_data().map_err(spool_io_error)?;
        Ok(DeferredPcsSpoolReplay {
            reader: BufReader::new(self.file.reopen().map_err(spool_io_error)?),
            max_record_bytes: self.max_record_bytes,
            max_polynomial_coefficients: self.max_polynomial_coefficients,
            next_block_index: 0,
        })
    }
}

struct DeferredPcsSpoolReplay {
    reader: BufReader<std::fs::File>,
    max_record_bytes: usize,
    max_polynomial_coefficients: usize,
    next_block_index: usize,
}

impl DeferredPcsSpoolReplay {
    fn read_next(&mut self) -> Result<Option<PcsBoundBlockJoltTransition>, DirectChunkedError> {
        let mut length = [0u8; 8];
        match self.reader.read(&mut length[..1]) {
            Ok(0) => return Ok(None),
            Ok(1) => {}
            Ok(_) => unreachable!("one-byte read returned more than one byte"),
            Err(error) => return Err(spool_io_error(error)),
        }
        self.reader
            .read_exact(&mut length[1..])
            .map_err(spool_io_error)?;
        let length = usize::try_from(u64::from_le_bytes(length))
            .map_err(|error| pcs_error(format!("D18 PCS spool length overflow: {error}")))?;
        if length == 0 || length > self.max_record_bytes {
            return Err(pcs_error("D18 PCS spool record length is invalid"));
        }
        let mut expected_digest = [0u8; 32];
        self.reader
            .read_exact(&mut expected_digest)
            .map_err(spool_io_error)?;
        let mut encoded = vec![0u8; length];
        self.reader
            .read_exact(&mut encoded)
            .map_err(spool_io_error)?;
        if spool_record_digest(&encoded) != expected_digest {
            return Err(pcs_error("D18 PCS spool record digest mismatch"));
        }
        let record: DeferredPcsSpoolRecord = postcard::from_bytes(&encoded)
            .map_err(|error| pcs_error(format!("D18 PCS spool decoding failed: {error}")))?;
        if record.block_index != self.next_block_index as u64
            || record.transition.statement.block_index != self.next_block_index as u64
        {
            return Err(pcs_error("D18 PCS spool replay order is not canonical"));
        }
        self.next_block_index += 1;
        record
            .into_bound(self.max_polynomial_coefficients)
            .map(Some)
    }
}

/// Commits the exact block-local lookup/register/RAM/CPU endpoint
/// polynomials, then generates the compact proof under that commitment root.
pub fn prove_pcs_bound_block_jolt_transition(
    prover: &mut BlockJoltProver,
    block: &TraceBlock,
    lookahead: Option<&Cycle>,
) -> Result<PcsBoundBlockJoltTransition, DirectChunkedError> {
    let config = prover.config();
    let lookup_started = Instant::now();
    let mut polynomials = compact_lookup_polynomials(block, config.cycle_capacity)?;
    let lookup_polynomial_micros = lookup_started.elapsed().as_micros();
    let register_started = Instant::now();
    polynomials.extend(compact_register_polynomials(block, config.cycle_capacity)?);
    let register_polynomial_micros = register_started.elapsed().as_micros();
    let ram_started = Instant::now();
    polynomials.extend(block_ram_polynomials(
        prover.preprocessing(),
        block,
        config.cycle_capacity,
        config.ram_k,
        prover.ram_state(),
    )?);
    let ram_polynomial_micros = ram_started.elapsed().as_micros();
    let cpu_started = Instant::now();
    polynomials.extend(block_cpu_polynomials(
        prover.preprocessing(),
        block,
        config.cycle_capacity,
        lookahead,
    )?);
    let cpu_polynomial_micros = cpu_started.elapsed().as_micros();
    let polynomial_count = polynomials.len();
    let coefficient_count = polynomials.iter().map(MultilinearPolynomial::len).sum();

    let commitment_started = Instant::now();
    let (commitments, hints) = commit_polynomials(&polynomials)?;
    let commitment_id = commitment_bundle_id(&commitments)?;
    let commitment_micros = commitment_started.elapsed().as_micros();
    let transition_started = Instant::now();
    let transition = prover.prove_block_with_commitment(block, lookahead, commitment_id)?;
    let compact_transition_micros = transition_started.elapsed().as_micros();
    let mut metrics = BlockJoltPcsProvingMetrics {
        lookup_polynomial_micros,
        register_polynomial_micros,
        ram_polynomial_micros,
        cpu_polynomial_micros,
        commitment_micros,
        compact_transition_micros,
        witness_validation_micros: 0,
        polynomial_count,
        coefficient_count,
    };
    let bound = PcsBoundBlockJoltTransition {
        transition,
        witness: BlockJoltDeferredPcsWitness {
            block_index: block.block_index as u64,
            commitment_id,
            polynomials,
            commitments,
            hints,
        },
        metrics: metrics.clone(),
    };
    let validation_started = Instant::now();
    validate_bound_transition(&bound)?;
    metrics.witness_validation_micros = validation_started.elapsed().as_micros();
    Ok(PcsBoundBlockJoltTransition { metrics, ..bound })
}

fn validate_bound_transition(
    bound: &PcsBoundBlockJoltTransition,
) -> Result<(), DirectChunkedError> {
    bound
        .transition
        .proof
        .validate_structure(&bound.transition.statement)
        .map_err(DirectChunkedError::InvalidProofShape)?;
    let claims = &bound.transition.proof.deferred_pcs_claims;
    let witness = &bound.witness;
    if witness.block_index != bound.transition.statement.block_index
        || witness.polynomials.is_empty()
        || claims.len() != witness.polynomials.len()
        || witness.commitments.len() != witness.polynomials.len()
        || witness.hints.len() != witness.polynomials.len()
        || commitment_bundle_id(&witness.commitments)? != witness.commitment_id
        || claims
            .iter()
            .any(|claim| claim.commitment_id != witness.commitment_id)
    {
        return Err(pcs_error(
            "D17 committed polynomial order differs from the deferred claim ledger",
        ));
    }
    for (index, (polynomial, claim)) in witness.polynomials.iter().zip(claims).enumerate() {
        let point = claim
            .opening_point
            .iter()
            .map(|coordinate| coordinate.to_fr())
            .collect::<Vec<_>>();
        let evaluated = (point.len() == polynomial.get_num_vars())
            .then(|| PolynomialEvaluation::evaluate(polynomial, &point));
        if evaluated != Some(claim.claimed_value.to_fr()) {
            return Err(pcs_error(format!(
                "D17 block {} polynomial {index} ({:?}) does not evaluate to its accepted endpoint claim (point variables {}, polynomial variables {}, evaluated {:?}, claimed {:?})",
                bound.transition.statement.block_index,
                claim.relation,
                point.len(),
                polynomial.get_num_vars(),
                evaluated.map(|value| FieldElement::from_fr(&value).0),
                claim.claimed_value.0,
            )));
        }
    }
    Ok(())
}

#[derive(Clone)]
pub struct DeferredDoryOpeningGroup {
    num_vars: usize,
    claim_indices: Vec<usize>,
    proof: ArkDoryProof,
}

#[derive(Clone)]
pub struct DeferredDoryBlockProof {
    block_index: u64,
    commitment_id: [u8; 32],
    claims: Vec<DeferredPcsClaim>,
    commitments: Vec<ArkGT>,
    groups: Vec<DeferredDoryOpeningGroup>,
}

/// D17 decider artifact. It contains commitments, accepted opening claims and
/// batched Dory proofs, but no trace rows or multilinear polynomial witness.
#[derive(Clone)]
pub struct BlockJoltDeferredPcsProof {
    deferred_state: [u8; 32],
    deferred_round: [u8; 32],
    blocks: Vec<DeferredDoryBlockProof>,
}

#[derive(Serialize, Deserialize)]
struct DeferredDoryOpeningGroupWire {
    num_vars: u32,
    claim_indices: Vec<u64>,
    proof: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct DeferredDoryBlockProofWire {
    block_index: u64,
    commitment_id: [u8; 32],
    claims: Vec<DeferredPcsClaim>,
    commitments: Vec<Vec<u8>>,
    groups: Vec<DeferredDoryOpeningGroupWire>,
}

#[derive(Serialize, Deserialize)]
struct BlockJoltDeferredPcsProofWire {
    wire_version: u16,
    protocol_version: String,
    deferred_state: [u8; 32],
    deferred_round: [u8; 32],
    blocks: Vec<DeferredDoryBlockProofWire>,
}

fn point_key(claim: &DeferredPcsClaim) -> (usize, Vec<[u8; 32]>) {
    (
        claim.opening_point.len(),
        claim
            .opening_point
            .iter()
            .map(|coordinate| coordinate.0)
            .collect(),
    )
}

fn opening_group_transcript(
    deferred_state: &[u8; 32],
    deferred_round: &[u8; 32],
    block: &DeferredDoryBlockProof,
    group_index: usize,
    num_vars: usize,
    claim_indices: &[usize],
) -> PoseidonTranscript {
    let mut transcript = PoseidonTranscript::new(D17_OPENING_DOMAIN);
    transcript.append_bytes(b"deferred_state", deferred_state);
    transcript.append_bytes(b"deferred_round", deferred_round);
    transcript.append_u64(b"block_index", block.block_index);
    transcript.append_u64(b"group_index", group_index as u64);
    transcript.append_u64(b"num_vars", num_vars as u64);
    transcript.append_bytes(b"bundle", &block.commitment_id);
    transcript.append_u64(b"claim_count", claim_indices.len() as u64);
    for index in claim_indices {
        let claim = &block.claims[*index];
        transcript.append_u64(b"claim_index", *index as u64);
        transcript.append_bytes(b"polynomial_id", &claim.polynomial_id);
        transcript.append_serializable(b"commitment", &block.commitments[*index]);
        transcript.append_u64(b"point_len", claim.opening_point.len() as u64);
        for coordinate in &claim.opening_point {
            transcript.append_scalar(b"point", &coordinate.to_fr());
        }
        transcript.append_scalar(b"opening", &claim.claimed_value.to_fr());
    }
    transcript
}

fn batching_coefficients(transcript: &mut PoseidonTranscript, count: usize) -> Vec<Fr> {
    let challenge = transcript.challenge_scalar::<Fr>();
    let mut power = Fr::one();
    (0..count)
        .map(|_| {
            let coefficient = power;
            power *= challenge;
            coefficient
        })
        .collect()
}

fn combined_polynomial(
    polynomials: &[&MultilinearPolynomial<Fr>],
    coefficients: &[Fr],
) -> MultilinearPolynomial<Fr> {
    let len = polynomials[0].len();
    let mut combined = vec![Fr::zero(); len];
    for (polynomial, coefficient) in polynomials.iter().zip(coefficients) {
        for (index, value) in combined.iter_mut().enumerate() {
            *value += polynomial.get_coeff(index) * coefficient;
        }
    }
    combined.into()
}

fn round_bytes(round: u64) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&round.to_le_bytes());
    bytes
}

fn verify_deferred_checkpoint(
    observed: ([u8; 32], [u8; 32]),
    blocks: &[DeferredDoryBlockProof],
) -> Result<([u8; 32], [u8; 32]), DirectChunkedError> {
    let (state, round) =
        native_deferred_checkpoint(blocks.iter().flat_map(|block| block.claims.iter()));
    let expected = (state, round_bytes(round));
    if observed != expected {
        return Err(pcs_error(
            "D17 opening ledger does not equal Nova's final deferred checkpoint",
        ));
    }
    Ok(expected)
}

fn deferred_block_from_bound(
    position: usize,
    item: &PcsBoundBlockJoltTransition,
) -> Result<DeferredDoryBlockProof, DirectChunkedError> {
    validate_bound_transition(item)?;
    if item.witness.block_index != position as u64
        || item.transition.statement.block_index != position as u64
    {
        return Err(pcs_error("D17 block witness sequence is not canonical"));
    }
    Ok(DeferredDoryBlockProof {
        block_index: position as u64,
        commitment_id: item.witness.commitment_id,
        claims: item.transition.proof.deferred_pcs_claims.clone(),
        commitments: item.witness.commitments.clone(),
        groups: Vec::new(),
    })
}

fn append_opening_groups(
    block: &mut DeferredDoryBlockProof,
    witness: &BlockJoltDeferredPcsWitness,
    deferred_state: &[u8; 32],
    deferred_round: &[u8; 32],
) -> Result<(), DirectChunkedError> {
    let mut grouped = BTreeMap::<(usize, Vec<[u8; 32]>), Vec<usize>>::new();
    for (index, claim) in block.claims.iter().enumerate() {
        grouped.entry(point_key(claim)).or_default().push(index);
    }
    let maximum_num_vars = grouped
        .keys()
        .map(|(num_vars, _)| *num_vars)
        .max()
        .ok_or_else(|| pcs_error("D17 cannot open an empty polynomial bundle"))?;
    let _lock = direct_dory_lock();
    let setup = DoryCommitmentScheme::setup_prover(maximum_num_vars);
    for (group_index, claim_indices) in grouped.into_values().enumerate() {
        let num_vars = block.claims[claim_indices[0]].opening_point.len();
        let mut transcript = opening_group_transcript(
            deferred_state,
            deferred_round,
            block,
            group_index,
            num_vars,
            &claim_indices,
        );
        let coefficients = batching_coefficients(&mut transcript, claim_indices.len());
        let polynomials = claim_indices
            .iter()
            .map(|index| &witness.polynomials[*index])
            .collect::<Vec<_>>();
        let combined = combined_polynomial(&polynomials, &coefficients);
        let point_fields = block.claims[claim_indices[0]]
            .opening_point
            .iter()
            .map(|coordinate| coordinate.to_fr())
            .collect::<Vec<_>>();
        let point = point_fields
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<<Fr as JoltField>::Challenge>>();
        let opening = claim_indices
            .iter()
            .zip(&coefficients)
            .map(|(index, coefficient)| block.claims[*index].claimed_value.to_fr() * coefficient)
            .sum::<Fr>();

        let proof = {
            let _context = DoryGlobals::initialize_context(
                1,
                1usize << num_vars,
                DoryContext::Main,
                Some(DoryLayout::CycleMajor),
            );
            let hints = claim_indices
                .iter()
                .map(|index| witness.hints[*index].clone())
                .collect::<Vec<_>>();
            let combined_hint = DoryCommitmentScheme::combine_hints(hints, &coefficients);
            bind_opening_inputs::<Fr, _>(&mut transcript, &point, &opening);
            DoryCommitmentScheme::prove(
                &setup,
                &combined,
                &point,
                Some(combined_hint),
                &mut transcript,
            )
            .0
        };
        block.groups.push(DeferredDoryOpeningGroup {
            num_vars,
            claim_indices,
            proof,
        });
    }
    Ok(())
}

/// Produces the exact Dory closure after the compact block transitions have
/// been folded by Nova.
pub fn close_block_jolt_deferred_pcs(
    folding: &BlockJoltNovaFoldingProof,
    bound: &[PcsBoundBlockJoltTransition],
) -> Result<BlockJoltDeferredPcsProof, DirectChunkedError> {
    folding.verify()?;
    if bound.is_empty() || folding.block_count() != bound.len() {
        return Err(pcs_error("D17 block count does not match Nova folding"));
    }

    let mut blocks = bound
        .iter()
        .enumerate()
        .map(|(position, item)| deferred_block_from_bound(position, item))
        .collect::<Result<Vec<_>, _>>()?;
    let (deferred_state, deferred_round) =
        verify_deferred_checkpoint(folding.deferred_checkpoint(), &blocks)?;

    for (block_position, item) in bound.iter().enumerate() {
        append_opening_groups(
            &mut blocks[block_position],
            &item.witness,
            &deferred_state,
            &deferred_round,
        )?;
    }

    let proof = BlockJoltDeferredPcsProof {
        deferred_state,
        deferred_round,
        blocks,
    };
    proof.verify(folding)?;
    Ok(proof)
}

/// D18 bounded-memory closure. It scans compact metadata once to reconstruct
/// Nova's exact deferred ledger, then replays one prover-only PCS record at a
/// time to create the Dory opening groups.
pub fn close_block_jolt_deferred_pcs_from_spool(
    folding: &BlockJoltNovaFoldingProof,
    spool: &BlockJoltDeferredPcsSpool,
) -> Result<BlockJoltDeferredPcsProof, DirectChunkedError> {
    folding.verify()?;
    if spool.block_count() == 0 || folding.block_count() != spool.block_count() {
        return Err(pcs_error(
            "D18 PCS spool block count does not match Nova folding",
        ));
    }

    let mut metadata_replay = spool.replay()?;
    let mut blocks = Vec::with_capacity(spool.block_count());
    while let Some(bound) = metadata_replay.read_next()? {
        let position = blocks.len();
        blocks.push(deferred_block_from_bound(position, &bound)?);
    }
    if blocks.len() != spool.block_count() {
        return Err(pcs_error("D18 PCS spool ended before every block was read"));
    }
    let (deferred_state, deferred_round) =
        verify_deferred_checkpoint(folding.deferred_checkpoint(), &blocks)?;

    let mut opening_replay = spool.replay()?;
    for (position, block) in blocks.iter_mut().enumerate() {
        let bound = opening_replay
            .read_next()?
            .ok_or_else(|| pcs_error("D18 PCS opening replay ended early"))?;
        if bound.transition.statement.block_index != position as u64
            || bound.witness.commitment_id != block.commitment_id
            || bound.transition.proof.deferred_pcs_claims != block.claims
        {
            return Err(pcs_error("D18 PCS metadata changed between replay passes"));
        }
        append_opening_groups(block, &bound.witness, &deferred_state, &deferred_round)?;
    }
    if opening_replay.read_next()?.is_some() {
        return Err(pcs_error("D18 PCS opening replay contains extra blocks"));
    }

    let proof = BlockJoltDeferredPcsProof {
        deferred_state,
        deferred_round,
        blocks,
    };
    proof.verify(folding)?;
    Ok(proof)
}

impl BlockJoltDeferredPcsProof {
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn opening_group_count(&self) -> usize {
        self.blocks.iter().map(|block| block.groups.len()).sum()
    }

    pub fn deferred_checkpoint(&self) -> ([u8; 32], [u8; 32]) {
        (self.deferred_state, self.deferred_round)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, DirectChunkedError> {
        let blocks = self
            .blocks
            .iter()
            .map(|block| {
                let commitments = block
                    .commitments
                    .iter()
                    .map(|commitment| canonical_bytes(commitment, "final Dory commitment"))
                    .collect::<Result<Vec<_>, _>>()?;
                let groups = block
                    .groups
                    .iter()
                    .map(|group| {
                        Ok(DeferredDoryOpeningGroupWire {
                            num_vars: u32::try_from(group.num_vars).map_err(|error| {
                                pcs_error(format!("D19 Dory group dimension overflow: {error}"))
                            })?,
                            claim_indices: group
                                .claim_indices
                                .iter()
                                .map(|index| {
                                    u64::try_from(*index).map_err(|error| {
                                        pcs_error(format!("D19 Dory claim index overflow: {error}"))
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                            proof: canonical_bytes(&group.proof, "final Dory opening proof")?,
                        })
                    })
                    .collect::<Result<Vec<_>, DirectChunkedError>>()?;
                Ok(DeferredDoryBlockProofWire {
                    block_index: block.block_index,
                    commitment_id: block.commitment_id,
                    claims: block.claims.clone(),
                    commitments,
                    groups,
                })
            })
            .collect::<Result<Vec<_>, DirectChunkedError>>()?;
        postcard::to_stdvec(&BlockJoltDeferredPcsProofWire {
            wire_version: D19_DORY_WIRE_VERSION,
            protocol_version: BLOCK_JOLT_PROTOCOL_VERSION.to_string(),
            deferred_state: self.deferred_state,
            deferred_round: self.deferred_round,
            blocks,
        })
        .map_err(|error| pcs_error(format!("D19 Dory artifact serialization failed: {error}")))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DirectChunkedError> {
        let wire: BlockJoltDeferredPcsProofWire = postcard::from_bytes(bytes)
            .map_err(|error| pcs_error(format!("D19 Dory artifact decoding failed: {error}")))?;
        if wire.wire_version != D19_DORY_WIRE_VERSION
            || wire.protocol_version != BLOCK_JOLT_PROTOCOL_VERSION
        {
            return Err(pcs_error("D19 Dory artifact version mismatch"));
        }
        let blocks = wire
            .blocks
            .into_iter()
            .map(|block| {
                let commitments = block
                    .commitments
                    .iter()
                    .map(|bytes| canonical_from_bytes(bytes, "final Dory commitment"))
                    .collect::<Result<Vec<ArkGT>, _>>()?;
                let groups = block
                    .groups
                    .into_iter()
                    .map(|group| {
                        Ok(DeferredDoryOpeningGroup {
                            num_vars: usize::try_from(group.num_vars).map_err(|error| {
                                pcs_error(format!(
                                    "D19 Dory group dimension conversion failed: {error}"
                                ))
                            })?,
                            claim_indices: group
                                .claim_indices
                                .into_iter()
                                .map(|index| {
                                    usize::try_from(index).map_err(|error| {
                                        pcs_error(format!(
                                            "D19 Dory claim index conversion failed: {error}"
                                        ))
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                            proof: canonical_from_bytes(&group.proof, "final Dory opening proof")?,
                        })
                    })
                    .collect::<Result<Vec<_>, DirectChunkedError>>()?;
                Ok(DeferredDoryBlockProof {
                    block_index: block.block_index,
                    commitment_id: block.commitment_id,
                    claims: block.claims,
                    commitments,
                    groups,
                })
            })
            .collect::<Result<Vec<_>, DirectChunkedError>>()?;
        Ok(Self {
            deferred_state: wire.deferred_state,
            deferred_round: wire.deferred_round,
            blocks,
        })
    }

    /// Verifies Dory against a Nova checkpoint already authenticated by either
    /// a recursive SNARK or its Spartan-compressed form.
    pub fn verify_against_checkpoint(
        &self,
        block_count: usize,
        checkpoint: ([u8; 32], [u8; 32]),
    ) -> Result<(), DirectChunkedError> {
        if self.blocks.is_empty() || self.blocks.len() != block_count {
            return Err(pcs_error("D19 Dory proof block count does not match Nova"));
        }
        let expected = verify_deferred_checkpoint(checkpoint, &self.blocks)?;
        if expected != (self.deferred_state, self.deferred_round) {
            return Err(pcs_error(
                "D17 artifact stores a different deferred checkpoint",
            ));
        }

        self.verify_openings()
    }

    pub fn verify(&self, folding: &BlockJoltNovaFoldingProof) -> Result<(), DirectChunkedError> {
        folding.verify()?;
        self.verify_against_checkpoint(folding.block_count(), folding.deferred_checkpoint())
    }

    fn verify_openings(&self) -> Result<(), DirectChunkedError> {
        for (block_position, block) in self.blocks.iter().enumerate() {
            if block.block_index != block_position as u64
                || block.claims.len() != block.commitments.len()
                || block.commitment_id != commitment_bundle_id(&block.commitments)?
                || block
                    .claims
                    .iter()
                    .any(|claim| claim.commitment_id != block.commitment_id)
            {
                return Err(pcs_error("D17 commitment bundle or block order mismatch"));
            }
            let maximum_num_vars = block
                .groups
                .iter()
                .map(|group| group.num_vars)
                .max()
                .ok_or_else(|| pcs_error("D17 block contains no opening groups"))?;
            let _lock = direct_dory_lock();
            let prover_setup = DoryCommitmentScheme::setup_prover(maximum_num_vars);
            let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);
            let mut covered = vec![false; block.claims.len()];
            for (group_index, group) in block.groups.iter().enumerate() {
                if group.claim_indices.is_empty() {
                    return Err(pcs_error("D17 contains an empty opening group"));
                }
                let first = group.claim_indices[0];
                if first >= block.claims.len()
                    || block.claims[first].opening_point.len() != group.num_vars
                {
                    return Err(pcs_error("D17 opening group has an invalid dimension"));
                }
                let expected_point_key = point_key(&block.claims[first]);
                for index in &group.claim_indices {
                    if *index >= block.claims.len()
                        || covered[*index]
                        || point_key(&block.claims[*index]) != expected_point_key
                    {
                        return Err(pcs_error("D17 opening groups overlap or mix points"));
                    }
                    covered[*index] = true;
                }

                let mut transcript = opening_group_transcript(
                    &self.deferred_state,
                    &self.deferred_round,
                    block,
                    group_index,
                    group.num_vars,
                    &group.claim_indices,
                );
                let coefficients =
                    batching_coefficients(&mut transcript, group.claim_indices.len());
                let combined_commitment = DoryCommitmentScheme::combine_commitments(
                    &group
                        .claim_indices
                        .iter()
                        .map(|index| &block.commitments[*index])
                        .collect::<Vec<_>>(),
                    &coefficients,
                );
                let point_fields = block.claims[first]
                    .opening_point
                    .iter()
                    .map(|coordinate| coordinate.to_fr())
                    .collect::<Vec<_>>();
                let point = point_fields
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect::<Vec<<Fr as JoltField>::Challenge>>();
                let opening = group
                    .claim_indices
                    .iter()
                    .zip(&coefficients)
                    .map(|(index, coefficient)| {
                        block.claims[*index].claimed_value.to_fr() * coefficient
                    })
                    .sum::<Fr>();
                let _context = DoryGlobals::initialize_context(
                    1,
                    1usize << group.num_vars,
                    DoryContext::Main,
                    Some(DoryLayout::CycleMajor),
                );
                bind_opening_inputs::<Fr, _>(&mut transcript, &point, &opening);
                DoryCommitmentScheme::verify(
                    &group.proof,
                    &verifier_setup,
                    &mut transcript,
                    &point,
                    &opening,
                    &combined_commitment,
                )
                .map_err(|error| {
                    pcs_error(format!(
                        "D17 Dory opening verification failed at block {block_position}, group \
                         {group_index}, num_vars {}, claim_indices {:?}: {error}",
                        group.num_vars, group.claim_indices,
                    ))
                })?;
            }
            if covered.iter().any(|covered| !covered) {
                return Err(pcs_error("D17 does not cover every deferred opening claim"));
            }
        }
        Ok(())
    }
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
        MachineBoundaryState,
    };

    use super::*;
    use crate::zkvm::block::{
        fold_verified_block_jolt_transitions, BlockJoltHostConfig, DirectChunkedPreprocessing,
    };

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

    #[test]
    fn d21_spool_uses_canonical_sparse_or_dense_polynomial_encoding() {
        let mut sparse_coefficients = vec![Fr::zero(); 16];
        sparse_coefficients[3] = Fr::from(7u64);
        sparse_coefficients[12] = Fr::from(9u64);
        let sparse_polynomial = MultilinearPolynomial::from(sparse_coefficients.clone());
        let sparse = DeferredPcsSpoolPolynomial::from_polynomial(&sparse_polynomial).unwrap();
        assert!(sparse.is_sparse());
        assert_eq!(sparse.nonzero_coefficient_count(), 2);
        let encoded = postcard::to_stdvec(&sparse).unwrap();
        let decoded: DeferredPcsSpoolPolynomial = postcard::from_bytes(&encoded).unwrap();
        let restored = decoded.into_polynomial(16).unwrap();
        assert_eq!(
            (0..restored.len())
                .map(|index| restored.get_coeff(index))
                .collect::<Vec<_>>(),
            sparse_coefficients
        );

        let dense_coefficients = (1u64..=8).map(Fr::from).collect::<Vec<_>>();
        let dense_polynomial = MultilinearPolynomial::from(dense_coefficients.clone());
        let dense = DeferredPcsSpoolPolynomial::from_polynomial(&dense_polynomial).unwrap();
        assert!(!dense.is_sparse());
        let restored = dense.into_polynomial(8).unwrap();
        assert_eq!(
            (0..restored.len())
                .map(|index| restored.get_coeff(index))
                .collect::<Vec<_>>(),
            dense_coefficients
        );
    }

    #[test]
    fn d21_sparse_spool_rejects_duplicate_out_of_range_and_zero_entries() {
        let value = FieldElement::from_fr(&Fr::from(1u64));
        for entries in [
            vec![
                DeferredPcsSpoolSparseCoefficient { index: 2, value },
                DeferredPcsSpoolSparseCoefficient { index: 2, value },
            ],
            vec![DeferredPcsSpoolSparseCoefficient { index: 8, value }],
            vec![DeferredPcsSpoolSparseCoefficient {
                index: 2,
                value: FieldElement::from_fr(&Fr::zero()),
            }],
        ] {
            let malformed = DeferredPcsSpoolPolynomial {
                num_vars: 3,
                coefficients: DeferredPcsSpoolCoefficients::Sparse(entries),
            };
            assert!(malformed.into_polynomial(8).is_err());
        }

        let oversized_sparse = DeferredPcsSpoolPolynomial {
            num_vars: 4,
            coefficients: DeferredPcsSpoolCoefficients::Sparse(Vec::new()),
        };
        assert!(oversized_sparse.into_polynomial(8).is_err());
    }

    #[test]
    fn d17_closes_every_nova_accepted_endpoint_with_dory() {
        let cycle = and_cycle(0x8000_0000, 3);
        let lookahead = and_cycle(0x8000_0004, 4);
        let preprocessing = DirectChunkedPreprocessing::from_trace_cycles(
            b"d17-exact-deferred-dory",
            8,
            &[cycle.clone(), lookahead.clone()],
        )
        .unwrap();
        let mut start_registers = [0i64; REGISTER_COUNT as usize];
        start_registers[1] = 0xaa;
        start_registers[2] = 0x0f;
        let mut end_registers = start_registers;
        end_registers[3] = 0x0a;
        let block = TraceBlock {
            block_index: 0,
            global_cycle_start: 0,
            active_cycles: 1,
            target_size: 2,
            start_state: MachineBoundaryState {
                global_cycle: 0,
                emulator_trace_len: 0,
                pc: 0x8000_0000,
                registers: start_registers,
                terminated: false,
            },
            end_state: MachineBoundaryState {
                global_cycle: 1,
                emulator_trace_len: 1,
                pc: 0x8000_0004,
                registers: end_registers,
                terminated: false,
            },
            cycles: vec![cycle],
            ended_at_tick_boundary: true,
        };
        let config = BlockJoltHostConfig {
            cycle_capacity: 2,
            ram_k: 2,
        };
        let mut prover = BlockJoltProver::new(preprocessing, config, BTreeMap::new()).unwrap();
        let bound =
            prove_pcs_bound_block_jolt_transition(&mut prover, &block, Some(&lookahead)).unwrap();
        let transitions = vec![bound.transition.clone()];
        let folding = fold_verified_block_jolt_transitions(config, &transitions).unwrap();
        let proof = close_block_jolt_deferred_pcs(&folding, &[bound]).unwrap();
        assert_eq!(proof.block_count(), 1);
        assert!(proof.opening_group_count() > 0);
        proof.verify(&folding).unwrap();
        let encoded = proof.to_bytes().unwrap();
        let decoded = BlockJoltDeferredPcsProof::from_bytes(&encoded).unwrap();
        decoded.verify(&folding).unwrap();
        let mut truncated = encoded;
        truncated.pop();
        assert!(BlockJoltDeferredPcsProof::from_bytes(&truncated).is_err());

        let mut bad_claim = proof.clone();
        bad_claim.blocks[0].claims[0].claimed_value.0[0] ^= 1;
        assert!(bad_claim.verify(&folding).is_err());

        let mut bad_bundle = proof;
        bad_bundle.blocks[0].commitments.swap(0, 1);
        assert!(bad_bundle.verify(&folding).is_err());
    }
}
