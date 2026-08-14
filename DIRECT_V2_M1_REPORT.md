# Direct Block-Jolt V2 — Milestone M1 Report

Status: D9–D14 implemented and server-validated

Protocol: `jolt-nova/direct-block-jolt/v2`
Baseline retained: `direct-stage-d8`

## Delivered at M1

- D9 freezes the direct block-Jolt architecture, transcript order, recursive
  state, two-pass schedule, and security invariants.
- D10 defines versioned, serializable `BlockJoltStatement`, `BlockJoltProof`,
  relation proof, deferred PCS, and streaming-state types. The D8 adapter can
  create only an audit statement and cannot create a V2 proof.
- D11 invokes Jolt's real `InstructionReadRaf` Lasso relation per block and
  returns only compact Sumcheck messages and opening endpoints.
- D12 invokes Jolt's real register and RAM read/write-checking Sumchecks and
  carries authenticated register/RAM boundary commitments between blocks.
- D13 invokes Jolt's original Spartan outer UniSkip and streaming Sumcheck for
  the CPU/R1CS relation. Block lookahead is used consistently in both prover
  messages and final R1CS opening computation.
- D14 provides transactional `BlockJoltProver` and `BlockJoltVerifier` host
  pipelines. A master transcript binds relation commitments and compact proofs
  in Lookup → Register → RAM → CPU order; lookup, machine, register, RAM,
  transcript, cycle, program, bytecode, and terminal boundaries are checked.

The production V2 proof types contain no `TraceBlock`, cycle rows, Merkle paths,
or D8 `Direct*BlockWitness` objects. Trace and RAM witness data exist only while
the current block is being proved.

## Security and regression evidence

The M1 suite includes:

- serialization and statement-binding tests;
- honest proof/verification tests for every relation;
- two-block continuity across every recursive-state field;
- malformed UniSkip and Sumcheck rejection;
- lookup/register/RAM/CPU opening tampering rejection;
- statement, transcript, terminal, deferred-PCS, replay, and cross-program
  attack rejection;
- transactional verifier tests proving rejected input does not advance state;
  and
- a source-level assertion that production proof types exclude row witnesses.

Server validation command:

```text
cargo test -p jolt-core --features nova jolt_verifier_folding -- --nocapture
```

Observed server results:

- M1 `jolt_verifier_folding` suite: 14 passed, 0 failed;
- D8 regression suite: 10 passed, 0 failed;
- full `jolt-core --features nova` run: 734 passed and one environment-only
  failure while installing the external ZeroOS guest toolchain;
- after installing the cached guest toolchain and Rust `rust-src`, the isolated
  `stdlib_e2e_dory` test passed (1 passed, 0 failed, 734 filtered out); and
- the full run took about 25 minutes 12 seconds and peaked at 14,947,480 KiB
  RSS (about 14.25 GiB), with no swap use.

Together the full run and the isolated environment retry cover all 735
`jolt-core` tests. The ZeroOS cache and temporary directory live under the
large `/public` volume and are excluded from version control.

## Exact M1 trust boundary

M1 is a real host-side verifier milestone, not the final recursive proof. The
Sumcheck arithmetic and Fiat-Shamir transcripts are verified now, but endpoint
polynomial openings are deliberately exported as `DeferredPcsClaim` values.
Consequently M1 is conditionally sound on the later PCS closure and must not be
advertised as a standalone final SNARK.

## Post-M1 route

- D15: circuitize the compact block verifier without row-level witnesses.
- D16: execute one Nova folding step per accepted block transition.
- D17: close the exact deferred endpoint set with Dory/PCS and bind its decider
  to the Nova output.
- D18: implement the bounded-memory two-pass spool/replay pipeline and measure
  per-relation time, memory, and proof size.
- D19: produce and verify the final Spartan-compressed Nova artifact, remove V2
  audit artifacts from the production format, and run the final security audit.

D8 remains callable only as a regression/performance oracle until D19.
