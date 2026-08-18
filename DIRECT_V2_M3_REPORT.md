# Direct Block-Jolt V2 — Milestone M3

Status: complete through D19

Protocol identifier: `jolt-nova/direct-block-jolt/v4`

M3 completes the production proof boundary planned after M2:

- D18 replaces linear trace, transition, and PCS-witness retention with a
  two-pass authenticated spool/replay pipeline. The streaming layer retains at
  most current-plus-lookahead trace blocks and one decoded PCS witness record,
  independently of execution block count.
- D19 moves Nova public parameters and Spartan keys into a reusable
  shape-specific setup, compresses the final Nova accumulator with Spartan,
  serializes the Dory decider, and verifies one standalone final artifact.

The realized production data path is:

```text
lazy trace iterator
  -> authenticated trace spool
  -> current block + lookahead
  -> real compact Jolt lookup/register/RAM/CPU proof
  -> one Nova step and immediate block-witness discard
  -> authenticated PCS-witness spool
  -> one-record-at-a-time Dory closure
  -> one Spartan compression
  -> BlockJoltFinalProof
```

Final acceptance requires both the Spartan-compressed recursive proof and the
Dory deferred-opening proof. The Dory checkpoint is read from the
Spartan-authenticated final recursive state. No host acceptance bit, receipt
digest, native-Jolt proof, raw row witness, or polynomial witness is trusted by
the final verifier.

## M3 acceptance evidence

- `cargo check -p jolt-core --features nova`: passed;
- direct-V2 D10–D19 serial regression: 22 passed, 0 failed;
- D8 compatibility regression: 10 passed, 0 failed;
- focused D19 serialization/verification and component-tamper test: passed;
- D18 measured trace residency: at most two blocks;
- D18 measured PCS-witness residency: at most one decoded record;
- final artifact round trip and standalone verification: passed; and
- all measured commands completed with no swap.

The measured two-block D19 final artifact is 595,667 bytes: 13,184 bytes of
Spartan-compressed Nova proof and 580,331 bytes of deferred Dory proof. The
serial D10–D19 suite peaked at about 7.08 GiB RSS. These are correctness
fixtures, not optimized performance claims.

## Exact M3 boundary

M3 completes the requested D18–D19 implementation and makes the V2
cryptographic artifact self-contained relative to its reusable public setup.
Future work is evaluation and optimization rather than filling a missing trust
edge in this clear-proof path. The main measured optimization targets are Dory
artifact size, the dense on-disk PCS witness representation, setup/circuit
memory, Spartan proving time, and realistic-workload scaling. BlindFold/ZK
equivalence and removal of the retained D8 comparison implementation remain
separate post-M3 work, not hidden M3 claims.
