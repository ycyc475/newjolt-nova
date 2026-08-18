# Direct Block-Jolt V2 — Milestone M2

Status: complete through D17

Protocol identifier: `jolt-nova/direct-block-jolt/v4`

M2 delivers the first end-to-end cryptographic composition of the new direct
block-Jolt architecture:

- D15 circuitizes the compact lookup/register/RAM/CPU verifier and folds the
  exact deferred-opening ledger into fixed-size public state;
- D16 executes one real Microsoft Nova folding step per compact block and
  exposes the final ledger checkpoint; and
- D17 commits the real endpoint polynomials before relation challenges and
  verifies their exact Dory openings against that Nova checkpoint.

Consequently, the M2 verifier no longer treats a host acceptance bit, opaque
receipt digest, or row hash as proof of polynomial validity. Acceptance needs
both the recursive Nova proof and the Dory deferred-opening decider.

M2 remains a research/debug artifact, not the production release. It retains
Nova public parameters inside the folding object, keeps PCS witnesses in memory
during proving, and has not yet Spartan-compressed and serialized the unified
final proof. Those are D18 and D19 responsibilities.

The complete `jolt_verifier_folding::` regression contains 19 tests and passes
with `--test-threads=1`. Heavy Nova/Dory tests must remain serialized on the
current server; default parallel scheduling can exceed its available memory.
