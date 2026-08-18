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
                DoryGlobals, DoryOpeningProofHint,
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
const D18_SPOOL_DOMAIN: &[u8] = b"block-jolt-deferred-pcs-spool-v1";
const D18_SPOOL_VERSION: u16 = 1;
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
    let mut committed = vec![None; polynomials.len()];
    for (num_vars, indices) in by_dimension {
        if num_vars == 0 {
            return Err(pcs_error(
                "D17 does not support a zero-variable Dory polynomial",
            ));
        }
        let _lock = direct_dory_lock();
        let _context =
            DoryGlobals::initialize_context(1, 1usize << num_vars, DoryContext::Main, None);
        let setup = DoryCommitmentScheme::setup_prover(num_vars);
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
}

impl PcsBoundBlockJoltTransition {
    pub fn transition(&self) -> &VerifiedBlockJoltTransition {
        &self.transition
    }
}

#[derive(Serialize, Deserialize)]
struct DeferredPcsSpoolPolynomial {
    num_vars: u32,
    coefficients: Vec<FieldElement>,
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
            .map(|polynomial| {
                let num_vars = u32::try_from(polynomial.get_num_vars()).map_err(|error| {
                    pcs_error(format!("D18 polynomial dimension overflow: {error}"))
                })?;
                let coefficients = (0..polynomial.len())
                    .map(|index| FieldElement::from_fr(&polynomial.get_coeff(index)))
                    .collect();
                Ok(DeferredPcsSpoolPolynomial {
                    num_vars,
                    coefficients,
                })
            })
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

    fn into_bound(self) -> Result<PcsBoundBlockJoltTransition, DirectChunkedError> {
        if self.version != D18_SPOOL_VERSION {
            return Err(pcs_error("D18 PCS spool record has an unsupported version"));
        }
        let polynomials = self
            .polynomials
            .into_iter()
            .map(|polynomial| {
                let num_vars = usize::try_from(polynomial.num_vars).map_err(|error| {
                    pcs_error(format!(
                        "D18 polynomial dimension conversion failed: {error}"
                    ))
                })?;
                let expected_len = 1usize.checked_shl(polynomial.num_vars).ok_or_else(|| {
                    pcs_error("D18 polynomial dimension exceeds the platform limit")
                })?;
                if num_vars == 0 || polynomial.coefficients.len() != expected_len {
                    return Err(pcs_error("D18 spooled polynomial has an invalid dimension"));
                }
                Ok(MultilinearPolynomial::from(
                    polynomial
                        .coefficients
                        .into_iter()
                        .map(|coefficient| coefficient.to_fr())
                        .collect::<Vec<_>>(),
                ))
            })
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
}

impl BlockJoltDeferredPcsSpool {
    pub fn new() -> Result<Self, DirectChunkedError> {
        Ok(Self {
            file: NamedTempFile::new().map_err(spool_io_error)?,
            block_count: 0,
            bytes_written: 0,
            max_record_bytes: 0,
        })
    }

    pub fn push(&mut self, bound: PcsBoundBlockJoltTransition) -> Result<(), DirectChunkedError> {
        if bound.transition.statement.block_index != self.block_count as u64
            || bound.witness.block_index != self.block_count as u64
        {
            return Err(pcs_error("D18 PCS spool block sequence is not canonical"));
        }
        let record = DeferredPcsSpoolRecord::from_bound(bound)?;
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

    fn replay(&self) -> Result<DeferredPcsSpoolReplay, DirectChunkedError> {
        self.file.as_file().sync_data().map_err(spool_io_error)?;
        Ok(DeferredPcsSpoolReplay {
            reader: BufReader::new(self.file.reopen().map_err(spool_io_error)?),
            max_record_bytes: self.max_record_bytes,
            next_block_index: 0,
        })
    }
}

struct DeferredPcsSpoolReplay {
    reader: BufReader<std::fs::File>,
    max_record_bytes: usize,
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
        record.into_bound().map(Some)
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
    let mut polynomials = compact_lookup_polynomials(block, config.cycle_capacity)?;
    polynomials.extend(compact_register_polynomials(block, config.cycle_capacity)?);
    polynomials.extend(block_ram_polynomials(
        prover.preprocessing(),
        block,
        config.cycle_capacity,
        config.ram_k,
        prover.ram_state(),
    )?);
    polynomials.extend(block_cpu_polynomials(
        prover.preprocessing(),
        block,
        config.cycle_capacity,
        lookahead,
    )?);

    let (commitments, hints) = commit_polynomials(&polynomials)?;
    let commitment_id = commitment_bundle_id(&commitments)?;
    let transition = prover.prove_block_with_commitment(block, lookahead, commitment_id)?;
    let bound = PcsBoundBlockJoltTransition {
        transition,
        witness: BlockJoltDeferredPcsWitness {
            block_index: block.block_index as u64,
            commitment_id,
            polynomials,
            commitments,
            hints,
        },
    };
    validate_bound_transition(&bound)?;
    Ok(bound)
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
        if point.len() != polynomial.get_num_vars()
            || PolynomialEvaluation::evaluate(polynomial, &point) != claim.claimed_value.to_fr()
        {
            return Err(pcs_error(format!(
                "D17 polynomial {index} does not evaluate to its accepted endpoint claim"
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
            let _lock = direct_dory_lock();
            let _context =
                DoryGlobals::initialize_context(1, 1usize << num_vars, DoryContext::Main, None);
            let setup = DoryCommitmentScheme::setup_prover(num_vars);
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
                let _lock = direct_dory_lock();
                let _context = DoryGlobals::initialize_context(
                    1,
                    1usize << group.num_vars,
                    DoryContext::Main,
                    None,
                );
                let prover_setup = DoryCommitmentScheme::setup_prover(group.num_vars);
                let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);
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
                    pcs_error(format!("D17 Dory opening verification failed: {error}"))
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
