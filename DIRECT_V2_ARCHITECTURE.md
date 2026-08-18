# Direct Block-Jolt Verifier Folding Architecture

Status: D18 bounded-memory spool/replay implemented (M3 in progress)

Protocol identifier: `jolt-nova/direct-block-jolt/v4`

Baseline: `direct-stage-d8` (`04e84e9f`)

## Goal and trust boundary

V2 keeps D8's direct trace-first execution path but replaces the monolithic
row-level `DirectPcsStepCircuit` with a compact per-block Jolt proof. Each Nova
step verifies the field-arithmetic part of the block Jolt verifier and folds its
accepted boundary state. No native-Jolt proof, host-verified receipt, raw trace,
or host boolean is trusted by the final verifier.

The final proof is accepted only when both of these checks succeed:

1. the Spartan-compressed Nova proof accepts every block verifier transition;
2. the deferred PCS decider verifies the polynomial openings exported by those
   accepted transitions.

A digest identifies an object but never proves that the object is valid.

## Fixed preprocessing

Preprocessing binds:

- program and bytecode commitments;
- the fixed Lasso lookup-table commitment;
- block cycle capacity and proof shape;
- transcript protocol version and domain separators;
- the Nova circuit shape and public parameters;
- Spartan proving and verification keys; and
- the deferred PCS/Dory verification parameters.

Setup material is generated once per shape and reused. Per-proof setup is not a
valid production configuration.

## Public execution statement

The public statement contains:

- protocol version and preprocessing identifier;
- program and lookup-table commitments;
- public input, public output, advice commitment, and panic status;
- block capacity, block count, and total active cycles;
- initial and final machine/register/RAM commitments; and
- the final transcript and deferred-PCS obligation identifiers.

## Block statement and proof

`BlockJoltStatement` binds one fixed-capacity block to:

- its block index and global cycle interval;
- active-cycle count and terminal flag;
- program, bytecode, and lookup-table commitments;
- start/end machine state commitments;
- start/end register commitments;
- start/end RAM roots;
- input/output lookup accumulators;
- input/output verifier transcript states; and
- the block's deferred PCS claim root.

`BlockJoltProof` contains compact verifier messages only:

- Lasso lookup sumcheck messages and endpoint opening claims;
- register consistency sumcheck messages and endpoint opening claims;
- RAM consistency proof messages and endpoint opening claims;
- CPU/R1CS sumcheck messages and endpoint opening claims;
- the unified Fiat-Shamir checkpoint sequence; and
- deferred PCS claims.

The production proof type must not contain `TraceBlock`, cycle rows, Merkle
authentication paths, or D8 `Direct*BlockWitness` values. Those are prover-only
witnesses and are dropped after the block has been folded.

## Recursive state

The fixed-size D15 Nova state is:

```text
StreamingRecursiveState {
    protocol_version,
    preprocessing_id,
    program_digest,
    lookup_table_commitment,
    next_block_index,
    next_global_cycle,
    machine_state_commitment,
    register_state_commitment,
    ram_root,
    lookup_accumulator,
    lookup_transcript_state,
    lookup_transcript_round,
    deferred_pcs_accumulator,
    total_active_cycles,
    terminated,
}
```

For block `i`, Nova proves

```text
VerifyBlockJolt(Z_i, BlockJoltStatement_i, BlockJoltProof_i) = Z_{i+1}.
```

The transition checks verifier arithmetic and all state continuity constraints.
It does not merely hash the block proof.

## Two-pass proving schedule

V2 uses a bounded-memory two-pass trace schedule because CPU lookahead and the
RAM initial-value registry cannot be finalized safely from a single-use trace
iterator. The deferred Dory opening transcript also depends on Nova's final
checkpoint, so prover-only PCS material is moved to a separate authenticated
disk spool instead of being retained in memory.

D17 establishes the cryptographic closure with an in-memory prover-only PCS
witness. The disk-backed replay/spool implementation described below remains a
D18 optimization; it does not change the D17 verifier statement or trust
boundary.

### Pass A: trace capture and audit

1. lazily execute and hard-rechunk trace blocks;
2. retain at most the current block and one lookahead block;
3. validate block order, boundary continuity, cycle bounds, and RAM addresses;
4. write replayable block data to a length-delimited temporary spool;
5. derive the initial RAM registry and final trace audit; and
6. reject an empty, discontinuous, oversized, or non-terminal stream.

### Pass B: proof, fold, and discard

For each replayed block:

1. rebuild only the current block witness;
2. materialize and Dory-commit its endpoint polynomials before relation
   challenges;
3. generate its `BlockJoltProof` and run the host verifier consistency check;
4. execute one Nova `prove_step` over `BlockJoltVerifierStepCircuit`;
5. write the polynomial coefficients, commitments, and opening hints to an
   integrity-protected PCS spool; and
6. drop the trace block, compact transition, and PCS witness before reading the
   next block.

After the last block, the prover checks termination and finalizes Nova. It then
replays at most one PCS spool record at a time, creates the deferred Dory
opening-proof bundle grouped by common evaluation point and dimension, and
drops the spool. D19 compresses the Nova accumulator once with Spartan.

## Transcript ordering

All transcript labels are versioned and domain-separated. The order is:

1. protocol and preprocessing identity;
2. public execution statement;
3. ordered block commitment root;
4. per-block header and relation commitments;
5. lookup, register, RAM, and CPU proof messages in that fixed order;
6. per-block deferred PCS claim root;
7. folded-state checkpoint; and
8. final PCS batching challenge and decider statement.

At M1 the four native Jolt relations retain their own domain-separated
Fiat-Shamir sub-transcripts. The lookup sub-transcript is carried across block
boundaries; register, RAM, and CPU sub-transcripts restart from their versioned
relation domains. D14 also maintains a persistent serialized master transcript
as a host-side audit chain. D15 recursive acceptance instead replays the four
relation transcripts directly in fixed order and carries the lookup transcript
plus the deferred-opening accumulator; it does not trust the master chain as an
opaque verifier receipt.

The block index, previous transcript state, and previous deferred-PCS state are
absorbed before every block. A proof message cannot be replayed at another block
index, execution, relation, or protocol version.

## Security invariants

Acceptance requires all of the following:

1. block indices start at zero, increase by one, and equal the final count;
2. global cycle intervals are contiguous and active counts are in range;
3. all blocks use the same program, bytecode, table, shape, and preprocessing;
4. machine, register, and RAM end states equal the next block's start states;
5. lookup and transcript accumulators transition in the declared order;
6. lookup, register, RAM, and CPU proofs bind to one common block statement;
7. inactive padded rows are canonical and cannot affect any relation;
8. no block after a terminal block is accepted;
9. the final block is terminal and matches the public output/panic statement;
10. every deferred opening is derived from an accepted sumcheck endpoint;
11. the final Dory proof closes exactly the accumulator output by Nova; and
12. clear and future ZK proofs use distinct domains but share the same execution
    statement identity.

## Artifact boundary

The final production artifact contains only:

- the public execution statement;
- the Spartan-compressed Nova proof;
- the final recursive public output;
- the deferred PCS statement and Dory opening proof; and
- versioned transcript/shape identifiers.

Audit witnesses, block proofs, recursive debug SNARKs, trace rows, and D8
subclaim vectors are excluded from the production artifact.

## D9-D14 milestone boundary

Milestone M1 ends at D14. It delivers a host-side, fixed-capacity
`BlockJoltProver`/`BlockJoltVerifier` pipeline with real Jolt lookup/register
sumchecks, authenticated RAM and CPU relation messages, unified transcript
binding, deferred PCS claims, serialization, and adversarial tests. D15 and
later circuitize and recursively fold that verifier; M1 does not claim that the
new verifier is already inside Nova.

## Compatibility and migration

- D8 remains unchanged and callable as a benchmark oracle through M1.
- V2 lives in `zkvm::block::jolt_verifier_folding`.
- D8-to-V2 adapters are test-only or explicitly marked audit-only.
- V2 becomes the production default only after D19 final-proof verification.
- Removing D8 row-level circuits is deferred until the V2 security and
  performance comparison is complete.

## D9 acceptance evidence

D9 is accepted when this document, the protocol identifier, transcript test
vectors, and a machine-readable architecture manifest agree on the statement,
state, ordering, and security invariants while the D8 regression tests remain
unchanged.

## D17 implementation boundary

D17 replaces the provisional row/query hashes on the secure path with a
canonical identifier of real Dory commitments. For each block, the prover:

1. materializes the exact lookup, register, RAM, and CPU endpoint polynomials;
2. Dory-commits them before sampling any relation challenge;
3. absorbs the reduced 254-bit commitment-bundle identifier into all four
   relation transcripts and every deferred claim;
4. proves that each accepted endpoint is an evaluation of its committed
   polynomial; and
5. batches only openings that share the same dimension and point, using a
   transcript-derived random linear combination.

The Dory opening transcript includes Nova's final
`(deferred_state, deferred_round)` output. The standalone D17 verifier
recomputes the complete canonical opening ledger, checks that it equals this
Nova checkpoint, verifies the commitment-bundle identifier, requires every
claim to be covered exactly once, and verifies every batched Dory proof.

The D17 decider artifact contains commitments, claims, and Dory proofs, but no
trace rows or polynomial coefficients. D18 removes the D17 prover-side
retention of polynomial witnesses through the disk-backed schedule above.

## D18 implementation boundary

D18 adds an incremental `BlockJoltNovaFolder` and the production
`prove_block_jolt_streaming` path. The source iterator is consumed once into a
temporary trace spool and replayed once with at most the current and lookahead
blocks resident. Every accepted transition is folded immediately; no vector of
prior transitions is retained.

The corresponding `BlockJoltDeferredPcsSpool` stores one block's dense endpoint
coefficients, canonical Dory commitments, and opening hints in each record.
Every record is versioned, length-delimited, and SHA3-bound to the protocol and
its payload. Replay revalidates the block order, commitment bundle, proof
structure, polynomial dimensions, and exact endpoint evaluations. The final
Dory closure scans metadata and opening witnesses one record at a time. The
returned verifier artifact contains neither temporary spool nor polynomial
coefficients.

D18 does not claim that operating-system RSS is independent of Nova's circuit
shape or Dory setup caches. Its bounded-memory invariant is that trace and PCS
witness residency is independent of the number of blocks: at most two trace
blocks and one PCS witness record are live in the streaming layer.
