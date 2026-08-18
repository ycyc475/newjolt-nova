# Direct Block-Jolt V2 — D18 Report

Status: complete

Protocol identifier: `jolt-nova/direct-block-jolt/v4`

## Delivered

D18 removes the linear in-memory retention that remained after D17:

- `BlockJoltNovaFolder` executes one real Nova step per compact transition and
  retains only the recursive accumulator and final compact transition;
- `prove_block_jolt_streaming` consumes a source iterator through a two-pass
  trace spool with current-plus-lookahead residency;
- `BlockJoltDeferredPcsSpool` moves each block's endpoint polynomials,
  commitments, and opening hints to an integrity-protected temporary file;
- the Dory closer reconstructs the final deferred ledger, then replays one PCS
  witness record at a time; and
- the returned proof contains only Nova and Dory verifier artifacts, never
  trace rows, polynomial coefficients, or temporary file handles.

Each PCS record is versioned, length-delimited, SHA3-authenticated, and checked
for canonical block order. Replay also verifies its commitment bundle,
polynomial dimensions, compact proof structure, and exact endpoint values.

## Bounded-residency invariant

The streaming layer retains at most:

- two trace blocks (current plus CPU lookahead);
- one compact block transition while its Nova step is proved; and
- one decoded PCS witness record while its Dory groups are opened.

These counts do not grow with the number of blocks. Nova's recursive
accumulator, public parameters, circuit allocations, Dory setup/cache material,
and the final verifier artifact remain resident and are shape-dependent.

## Validation

`cargo check -p jolt-core --features nova` passed with only the nine pre-existing
legacy dead-code warning groups.

The non-terminal-stream negative test passed and rejects the source before any
cryptographic proving work:

```text
cargo test -p jolt-core --features nova \
  d18_rejects_a_nonterminal_stream_before_proving -- --nocapture
```

The two-block end-to-end D18 test passed:

```text
/usr/bin/time -v cargo test -p jolt-core --features nova \
  d18_streams_trace_folding_and_pcs_witnesses_with_bounded_residency \
  -- --nocapture --test-threads=1
```

The test body completed in 218.04 seconds. The complete command, including an
incremental rebuild, took 7 minutes 27.50 seconds. Peak RSS was 3,952,468 KiB
(about 3.77 GiB) with no swap.

The refactored legacy paths were also checked:

- D16 real two-block Nova folding: passed in 76.71 seconds;
- D17 exact deferred Dory closure and tamper checks: passed in 142.88 seconds.

## Remaining D19 boundary

D18 still returns the debug `RecursiveSNARK` envelope and keeps Nova public
parameters inside it. D19 must use an externally reusable shape-specific setup,
Spartan-compress the recursive accumulator, serialize the Nova and Dory proof
bundle into one versioned final artifact, and verify it using only the public
setup and public execution state.
