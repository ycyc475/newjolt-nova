# Direct Block-Jolt Verifier Folding Architecture

Status: D16 real per-block Nova folding implemented

Protocol identifier: `jolt-nova/direct-block-jolt/v3`

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

V2 uses a bounded-memory two-pass schedule because the global Fiat-Shamir PCS
challenge is available only after the commitment set is fixed.

### Pass A: capture and commitment planning

1. lazily execute and hard-rechunk trace blocks;
2. retain at most the current block and one lookahead block;
3. derive compact block commitment metadata;
4. write replayable block data and compact metadata to a disk spool;
5. finalize block count, public output, final roots, and commitment root; and
6. derive the global PCS batching/opening challenge.

### Pass B: proof, fold, and discard

For each replayed block:

1. rebuild only the current block witness;
2. generate its `BlockJoltProof`;
3. run the host verifier as a prover-side consistency check;
4. execute one Nova `prove_step` over `BlockJoltVerifierStepCircuit`;
5. update the streaming deferred-PCS accumulator; and
6. drop the trace block, witness, and block proof before reading the next block.

After the last block, the prover checks termination, creates one deferred Dory
opening proof, and compresses the Nova accumulator once with Spartan.

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
