# Direct Block-Jolt V2 — Stage D16 Report

Status: implemented and server-validated

Protocol identifier: `jolt-nova/direct-block-jolt/v3`

## Delivered

D16 connects the fixed-shape D15 compact verifier to Microsoft Nova's real
`RecursiveSNARK` implementation. The folding function:

1. constructs the first `BlockJoltVerifierStepCircuit` and its public input;
2. generates Nova public parameters once for that fixed shape;
3. invokes exactly one `RecursiveSNARK::prove_step` per compact block;
4. verifies the recursive proof for the exact block count;
5. compares Nova's final public output with the final block/cycle,
   machine/register/RAM boundary, lookup transcript, total-cycle, and terminal
   values; and
6. exposes the final deferred-opening checkpoint for D17.

The folding artifact contains Nova public parameters, the recursive SNARK, and
the initial/final public states. It does not contain trace rows or any D8
`Direct*BlockWitness`. This is a D16 debug/self-verifying envelope; D19 will use
pinned setup identifiers and Spartan compression for the production artifact.

Folding accepts a valid non-terminal execution prefix, which is required by an
online prover. A terminal block may only appear last; no later block can be
folded because the D15 state transition constrains the carried terminal flag.

## Validation

Server command:

```text
/usr/bin/time -v cargo test -p jolt-core --features nova \
  d16_executes_one_real_nova_step_per_compact_block -- --nocapture
```

Observed result:

- two real compact M1 block proofs generated;
- two real Nova `prove_step` calls completed;
- recursive verification and final-state comparison passed;
- test body: about 64.82 seconds after compilation;
- peak RSS for the measured command: 3,952,028 KiB (about 3.77 GiB);
- no swap.

The negative test modifies the second block's RAM Sumcheck final claim and
confirms that folding fails inside the D15/Nova relation. It passed in about
58.86 seconds with peak RSS about 3.77 GiB.

During validation, the initial terminal fixture was rejected by Jolt's native
`NextUnexpandedPC` constraint because an ordinary `AND` row had been incorrectly
marked terminal. The final test uses a valid non-terminal prefix with the
required lookahead row; this was a fixture correction, not a bypass.

## Next stage

D17 must replace hash-like commitment identifiers with real committed
polynomials, produce Dory opening proofs for every canonical deferred endpoint,
verify them, and bind the exact accepted opening ledger to Nova's final
`(deferred_state, deferred_round)` checkpoint.
