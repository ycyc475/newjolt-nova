# Direct Block-Jolt V2 — D19 Report

Status: complete

Protocol identifier: `jolt-nova/direct-block-jolt/v4`

Final wire version: 1

## Delivered

D19 replaces the D18 debug recursive envelope with a standalone production
proof boundary:

- `BlockJoltNovaSetup` creates shape-specific Nova public parameters and
  Spartan keys once and exposes a protocol/config/key-bound setup identifier;
- `prove_block_jolt_streaming_with_setup` reuses that setup for one Nova step
  per block while preserving D18's bounded trace/PCS witness residency;
- `compress_block_jolt_final_proof` Spartan-compresses the final recursive
  accumulator and serializes the deferred Dory decider;
- `BlockJoltFinalProof::{to_bytes,from_bytes,verify}` defines one versioned
  transport and standalone verifier API; and
- the final verifier authenticates the Dory checkpoint from the verified Nova
  output instead of receiving a debug `RecursiveSNARK` or trusting a host
  receipt.

The final artifact stores the protocol/setup identity, block count, recursive
public input/output, compressed Nova proof, deferred Dory proof, and a complete
envelope digest. It excludes Nova public parameters and proving keys, the
uncompressed recursive proof, block proofs/transitions, trace rows, endpoint
polynomials, opening hints, and temporary spool contents.

## Security checks

The D19 test covers:

- setup generation from the fixed compact-verifier shape and reuse by a
  separate two-block streaming proof;
- exact setup-identifier binding between folding and compression;
- Spartan compression, self-verification, serialization round trip, and
  independent final verification;
- final termination and exact recursive-output equality;
- Dory verification against the deferred checkpoint extracted from the
  Spartan-authenticated final state; and
- rejection of a corrupted wire digest, forged setup identifier, forged final
  recursive output, corrupted compressed Nova component, and corrupted Dory
  component.

## Measured artifact

The canonical two-block fixture produced:

| Component | Bytes |
|---|---:|
| Spartan-compressed Nova proof | 13,184 |
| Deferred Dory proof | 580,331 |
| Complete final artifact | 595,667 |
| Trace spool (prover-only) | 566 |
| PCS witness spool (prover-only) | 67,525,258 |

The two spool sizes are diagnostics and are not included in the final proof.
The Dory component dominates this minimal fixture because it carries the
canonical commitments and opening proofs for every deferred lookup, register,
RAM, and CPU endpoint.

## Validation

The focused D19 command was:

```text
/usr/bin/time -v cargo test -p jolt-core --features nova \
  d19_serializes_and_verifies_the_standalone_spartan_artifact \
  -- --nocapture --test-threads=1
```

It passed (1 passed, 0 failed). On the final code, the test body took 385.63
seconds; the complete command, including a test-binary relink, took 10 minutes
2.32 seconds. Peak RSS was 7,181,856 KiB (about 6.85 GiB), with no swap.

The complete direct-V2 module regression also passed:

```text
/usr/bin/time -v cargo test -p jolt-core --features nova \
  jolt_verifier_folding:: -- --nocapture --test-threads=1
```

All 22 tests passed in 952.35 seconds (16 minutes 35.36 seconds for the command).
Peak RSS was 7,424,408 KiB (about 7.08 GiB), with no swap. The nine warning
groups are unchanged dead-code warnings in the legacy `zkvm::block` path.

The D8 compatibility regression passed separately: 10 passed, 0 failed in
88.84 seconds (2 minutes 11.42 seconds for the command), with peak RSS
3,637,876 KiB (about 3.47 GiB) and no swap.
