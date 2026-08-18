# Direct Block-Jolt V2 — Stage D17 Report

Status: implemented and server-validated

Protocol identifier: `jolt-nova/direct-block-jolt/v4`

## Delivered

D17 closes every deferred endpoint accepted by the D15 compact verifier and
D16 Nova folding with a real Dory polynomial opening. The secure path now:

1. materializes the exact endpoint polynomials in canonical claim order;
2. commits them before lookup/register/RAM/CPU Fiat--Shamir challenges;
3. hashes the ordered Dory commitment bundle to a canonical BN254 field
   identifier used by all four relation transcripts and deferred claims;
4. checks every claimed endpoint against its polynomial before folding;
5. groups only claims with identical dimensions and opening points;
6. derives random-linear-combination coefficients from a transcript bound to
   the final Nova deferred checkpoint, block, claims, and commitments; and
7. verifies that every accepted claim is covered exactly once by a Dory proof.

The committed polynomial set is complete:

- lookup: output/operands, all table flags, eight instruction-address chunks,
  and the RAF flag;
- register: three input-value polynomials, RegistersVal, Rs1Ra, Rs2Ra, RdWa,
  and RdInc;
- RAM: read/write values, RamRa, RamVal, and RamInc; and
- CPU/R1CS: all 35 polynomials in `ALL_R1CS_INPUTS` order.

The final D17 artifact contains only the accepted opening ledger, Dory
commitments, and batched opening proofs. It contains no `TraceBlock`, cycle
rows, polynomial coefficients, or D8 row-level witness.

## Security checks

Verification rejects if:

- the opening ledger differs from Nova's final
  `(deferred_state, deferred_round)`;
- a commitment is changed, reordered, omitted, or added;
- a claim points to a different commitment bundle;
- an opening group overlaps another group, mixes points, or omits a claim; or
- a claimed value, opening point, polynomial commitment, or Dory proof is
  invalid.

The protocol was bumped from v3/wire 3 to v4/wire 4 because commitment material
now enters the Fiat--Shamir relation headers before challenge derivation.

## Validation

Server environment:

```text
ssh myserver
conda activate jolt-nova-build
cd /public/share/td20062985/dyc/newproject/jolt
```

Primary command:

```text
/usr/bin/time -v cargo test -p jolt-core --features nova \
  d17_closes_every_nova_accepted_endpoint_with_dory -- --nocapture
```

Observed result:

- one real compact block proof generated under precommitted endpoints;
- one real Nova `prove_step` completed and verified;
- every endpoint polynomial evaluation matched its accepted claim;
- all same-point Dory opening groups proved and verified;
- modified claim and reordered-commitment attacks were rejected;
- test body: 175.12 seconds;
- full command wall time: 3 minutes 37.58 seconds, including startup/linking;
- peak RSS: 4,173,020 KiB (about 3.98 GiB); and
- swap: zero.

D11--D15 targeted regressions and the D16 real two-block Nova folding test also
passed under protocol v4. The existing dead-code warnings in legacy
`zkvm::block` code are unchanged and unrelated to D17.

The complete module regression was then run serially:

```text
/usr/bin/time -v cargo test -p jolt-core --features nova \
  jolt_verifier_folding:: -- --nocapture --test-threads=1
```

All 19 tests passed in 427.55 seconds (7 minutes 12.20 seconds for the complete
command), with peak RSS 6,598,340 KiB (about 6.29 GiB) and no swap. Running the
same heavy suite with Rust's default parallel test scheduler is not supported
on the current server: concurrent D15/D16/D17 tests reached 12,128,344 KiB and
failed a subsequent 32 MiB allocation. That is test-process concurrency, not a
failed proof relation; CI and server validation should retain
`--test-threads=1` for this suite.

## Remaining work

D17 deliberately keeps prover-only polynomial coefficients and Dory hints in
memory until Nova has produced its final deferred checkpoint. D18 must replace
that retention with the architecture's two-pass disk spool/replay schedule.
D19 will pin setup identifiers, Spartan-compress the recursive proof, serialize
the production artifact, and expose one final verification API.
