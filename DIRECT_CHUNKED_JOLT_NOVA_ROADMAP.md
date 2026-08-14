# Direct Chunked Jolt-Nova

This worktree starts the single-data-flow Jolt-Nova architecture. It is forked
from the verified Stage 19 implementation, but it has a different end goal:
the prover must begin with bounded trace blocks and must not first create or
verify a full native `JoltProof` for the same execution.

## Baseline

- Base commit: `c26b20173bd4dbf756c2d516f9426f3d9f53e56c`
- Base architecture: Stage 19 dual-data-flow Jolt-Nova
- New branch: `codex/direct-chunked-jolt-nova`
- New worktree: `direct-chunked-jolt-nova`
- Old Stage 19 worktree and its uncommitted experiment artifacts remain
  untouched.

## Target proof flow

```text
ELF + inputs + advice
        |
        v
streaming trace_blocks
        |
        v
block-native Jolt relations
  - CPU/R1CS
  - register consistency
  - RAM consistency
  - lookup/Lasso
  - public I/O and advice
        |
        v
Nova step verification and folding
        |
        v
global claim and PCS closure
        |
        v
Spartan-compressed final proof
```

The production flow must not contain the old detour:

```text
full trace -> RV64IMACProver::prove -> full Jolt verifier -> receipt
```

## Security invariant

Removing the native Jolt proof is permitted only after every property that it
currently authenticates has an equivalent block-native cryptographic relation.
A host-side check, digest, capsule, or boolean `verified` flag is not a
replacement for a constrained relation.

Each accepted block must cryptographically bind:

1. its CPU cycles to the committed program and the previous machine state;
2. lookup queries to those same CPU cycles and to one shared lookup-table
   commitment;
3. register and RAM accesses to the previous authenticated memory state;
4. its public I/O and advice consumption to the execution statement;
5. its sumcheck and PCS opening obligations to the running transcript; and
6. its end state to the next Nova public state.

The final closure step must establish legal termination and close every CPU,
lookup, register, RAM, I/O, sumcheck, and PCS accumulator.

## Migration stages

### Stage D0: isolated baseline

- Create the dedicated branch and worktree.
- Record the single-data-flow security boundary.
- Keep the Stage 19 implementation buildable as a comparison baseline.

### Stage D1: architecture seam

- Add a `DirectChunkedProver` entry point that consumes `TraceBlockIterator`.
- Introduce a direct-proof configuration that has no native-Jolt receipt field.
- Initially return an explicit `UnsupportedRelation` error for relations that
  are not yet cryptographically internalized; do not silently trust host checks.

Status: completed and tagged `direct-stage-d1`.

### Stage D2: block-native lookup/Lasso

- Commit once to the shared lookup tables.
- Derive each lookup query inside the block relation from the CPU-cycle witness.
- Produce and verify a real per-block lookup subclaim.
- Fold block subclaims and defer their aggregate PCS obligation.
- Add a final lookup closure check.

Status: completed and tagged `direct-stage-d2`; the retained native audit
subclaims were additionally bound to the recursive transcript output in the
`direct-stage-d2.1` security patch.

### Stage D3: block-native register relation

- Authenticate the complete register boundary state.
- Constrain every block read and write against that state.
- Carry the authenticated end state into the next Nova step.

Status: completed on `codex/direct-chunked-jolt-nova`.

- The direct runner derives `rs1`, `rs2`, `rd`, and `RdInc` witnesses from each
  raw `TraceBlock`; no native proof/receipt is accepted.
- Jolt's real `RegistersReadWriteChecking` sumcheck now supports an
  authenticated non-zero block-initial register vector.
- One combined Nova step verifies the D2 lookup relation, all 128 register
  boundary values, every in-block read/write transition, the register
  sumcheck, and all five endpoint openings.
- The recursive state carries the complete end-register vector into the next
  block, and one Spartan proof closes the combined lookup+register stage.
- D3 deliberately leaves RAM/CPU unsupported and the Jolt PCS obligation
  deferred; those states remain explicit and fail closed until D4-D6.

### Stage D4: block-native RAM relation

- Replace the host `latest_ram_values` trust boundary with an algebraic or
  authenticated-memory relation.
- Fold read/write subclaims across blocks.
- Add a final RAM closure check.

Implemented in `direct-stage-d4`:

- The prover builds a canonical sorted registry for every RAM address touched
  by the direct trace. The immutable registry root is public and is carried by
  every Nova step.
- Initial values and subsequent writes are committed by a domain-separated
  Poseidon Merkle tree. Each read verifies its current value path; each write
  verifies the old path and derives the next root along that same path.
- The mutable RAM root is part of the Nova public state, so a block must start
  at the exact root emitted by its predecessor. One final Spartan proof closes
  lookup, register, and RAM relations together.
- The address registry, authentication paths, roots, and block metadata are
  transcript-bound. Padding accesses and paths have a canonical zero encoding.
- D4 intentionally leaves the binding of the initial RAM root to program image,
  inputs, and advice as a deferred PCS obligation for D6. D5 binds each RAM
  access tuple to the corresponding CPU/R1CS row.

### Stage D5: block-native CPU/R1CS relation

- Bind block rows, bytecode, lookup operands, RAM/register accesses, and
  lookahead directly inside the recursive relation.
- Fold the CPU/R1CS subclaim instead of carrying only row counts and digests.

Implemented in `direct-stage-d5`:

- Every active trace row is materialized as Jolt's complete 35-column
  `R1CSCycleInputs` row. The Nova step checks all native `R1CS_CONSTRAINTS`
  plus the three product-virtualization relations; inactive fixed-shape rows
  are canonical padding and are excluded from execution semantics.
- CPU lookup operands/output/table/index are constrained equal to the D2
  lookup witnesses; register values, operand metadata, and write values are
  constrained equal to D3; RAM access kind/address/read/write values are
  constrained equal to D4. These are the same allocated variables returned by
  the lower-stage circuits, not duplicate host-verified receipts.
- Verifier-known bytecode is committed by a domain-separated Poseidon Merkle
  root. Every active CPU row proves its static instruction metadata and flags
  against that root, and the initial PC/address is fixed by preprocessing.
- The last active row selects the next PC, unexpanded PC, virtual/sequence
  flags, and NoOp status into the Nova public state. The next block must consume
  exactly that state; the final block closes against the terminal NoOp.
- D5 reuses a shared D2-D4 subclaim preparation path and creates one combined
  CPU+lookup+register+RAM Nova/Spartan proof. It does not first create a
  separate D4 recursive proof or a monolithic native Jolt proof.
- PCS, initial program/input/advice memory binding, public I/O, and the unified
  global Fiat-Shamir/PCS closure remain explicitly deferred to D6.

### Stage D6: transcript and PCS aggregation

- Define one cross-block Fiat-Shamir transcript.
- Aggregate scalar opening claims and group obligations safely.
- Verify one final PCS closure without a full native Jolt verifier invocation.

Status: completed and tagged `direct-stage-d6`.

### Stage D7: remove the native-Jolt production path

- Delete `RV64IMACProver::prove` and
  `verify_with_recursive_zk_complete_artifacts` from the direct production
  runner.
- Remove receipt/capsule fields that no longer belong to the security model.
- Keep the old Stage 19 runner only as an explicitly named comparison oracle.

Status: completed and tagged `direct-stage-d7`.

### Stage D8: end-to-end validation and evaluation

- Add adversarial tests for dropped, reordered, duplicated, and forged blocks.
- Add lookup, RAM, CPU, transcript, and PCS tamper tests.
- Compare proof size, proving time, verification time, and peak RSS against
  native Jolt and the Stage 19 dual-data-flow baseline.
- Demonstrate that trace-resident memory is `O(block_size)` and measure all
  remaining non-trace memory.

Status: completed. The validated Fibonacci-2 artifact and exact measurements
are recorded in `DIRECT_STAGE_D8_REPORT.md`; this stage establishes correctness
and a baseline, while also showing that the current Nova/Spartan circuit is the
dominant time and memory bottleneck.

## Definition of completion

The direct architecture is complete only when its production benchmark can:

1. start from an ELF and inputs;
2. consume execution as bounded trace blocks;
3. prove all Jolt relations without constructing a full native `JoltProof`;
4. fold every block with Nova;
5. close all global and PCS obligations;
6. emit one Spartan-compressed proof; and
7. verify that proof using only public inputs, preprocessing commitments, and
   the direct Jolt-Nova verification key.
