# Direct Block-Jolt V2 — Stage D15 Report

Status: implemented and server-validated

Protocol identifier: `jolt-nova/direct-block-jolt/v3`

## Delivered

D15 replaces the D8 row-witness folding relation with a fixed-shape Nova step
that consumes only the compact D11–D14 verifier messages. The circuit replays
the native BN254 Poseidon Fiat–Shamir transcripts and constrains:

- Lasso lookup, register, RAM, and CPU/Spartan clear-Sumcheck arithmetic;
- the exact relation-specific endpoint formulas;
- the CPU UniSkip first round and its pinned interpolation domain;
- protocol, preprocessing, program, bytecode, lookup-table, block, cycle,
  machine, register, RAM, terminal, and lookup-transcript continuity;
- full-width 32-byte identities as four range-constrained `u64` limbs;
- the lookup transcript state together with its Poseidon round counter; and
- every accepted deferred opening in one circuit-native Poseidon accumulator.

The production step witness contains no `TraceBlock`, cycle row, D8
`Direct*BlockWitness`, Merkle path, or host `accepted` bit. The four relation
proofs have fixed dimensions determined by `cycle_capacity`, `ram_k`, Jolt's
lookup shape, and the 35 Spartan outer input polynomials.

## Protocol correction

D15 exposed that a Poseidon checkpoint is the pair `(state, n_rounds)`, not the
state alone. The lookup proof, block statement, streaming state, and circuit
state now carry both values. This changes the wire/protocol identifier from v2
to v3 and prevents the same field state from being interpreted at another
transcript round.

The recursive relation directly replays each relation transcript in its fixed
Lookup → Register → RAM → CPU order. The D14 serialized master transcript
remains an audit/host consistency chain; recursive acceptance does not trust an
opaque master-transcript receipt.

## Deferred PCS boundary

D15 verifies the arithmetic that creates each endpoint value but deliberately
does not claim that those values are polynomial openings. The canonical ordered
opening ledger is folded into the recursive state. D17 must verify the exact
Dory/PCS openings and bind its decider to that accumulator before the system is
cryptographically complete.

## Validation

Server environment:

```text
ssh myserver
conda activate jolt-nova-build
cd /public/share/td20062985/dyc/newproject/jolt
```

Commands:

```text
cargo check -p jolt-core --features nova
cargo test -p jolt-core --features nova d15_compact_verifier -- --nocapture
```

Observed result: 2 D15 tests passed. The positive test builds real compact M1
proofs and satisfies the complete circuit. Negative tests reject a modified
lookup Sumcheck final claim and a modified carried lookup transcript round.

## Next stage

D16 will set up Nova for this fixed circuit shape and execute exactly one real
`RecursiveSNARK::prove_step` for every accepted block transition. It will check
the recursive public output against the host streaming state and add a
two-block continuity/tamper regression.
