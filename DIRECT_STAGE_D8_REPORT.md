# Direct Chunked Jolt-Nova: Stage D8 report

Date: 2026-08-14
Branch: `codex/direct-chunked-jolt-nova`
Server workspace: `/public/share/td20062985/dyc/newproject`
Threads: `RAYON_NUM_THREADS=8`

## Outcome

D8 completes the first measured end-to-end direct path:

```text
ELF + public input
  -> lazy soft trace blocks
  -> bounded hard-rechunk spool
  -> lookup + register + RAM + CPU/R1CS subclaims
  -> one Nova step per fixed-capacity block
  -> global transcript and Dory PCS closure
  -> one Spartan-compressed recursive proof
  -> direct verification
```

The production runner does not call `RV64IMACProver::prove`, construct a full
native `JoltProof`, or accept a native-Jolt receipt. Stage 19 remains only as a
separately executed comparison oracle.

## D8 implementation

- A temporary length-prefixed postcard spool consumes the lazy trace without
  collecting all `TraceBlock`s in memory. A dedicated serializable DTO avoids
  changing the upstream trace representation.
- A soft source block may be hard-rechunked into fixed-capacity proof blocks.
  The adapter reconstructs the exact intermediate register vector and relies
  on D5's exact next-row witness for a CPU boundary inside an expanded emulator
  tick. A source expansion larger than twice the configured capacity fails
  closed.
- Capture holds at most one bounded soft source block plus one emitted hard
  block. Relation preparation replays at most the current hard block plus one
  lookahead block. A block-count scaling test checks that trace residency does
  not grow with the number of blocks.
- RAM addresses are required to be verifier-layout-resident and word-aligned,
  closing unaligned cross-block aliases and out-of-layout accesses.
- Production records trace, relation, PCS, Nova, Spartan, proof-size, timing,
  and RSS metrics. Proof size uses a deterministic wire-accounting model for
  every verifier-consumed field.
- The D8 production artifact omits the uncompressed recursive Nova SNARK. The
  final recursive result is carried by the Spartan-compressed proof; the older
  D6 API can still retain the recursive object for debugging.
- Attack tests cover dropped, reordered, duplicated, and forged blocks, plus
  lookup, RAM, CPU, transcript, Dory opening, PCS point, entry-PC, and execution
  statement tampering.

## Validated workload

Workload: `fibonacci-2`
Workload SHA3-256:
`bc9c113b76c0bb986dbe4ba76ddb099aa729319282f63d2eaa28c51d1585b3ab`
Fixed circuit capacity: 128 rows

The lazy tracer produced four soft source blocks with 451 active cycles. One
source block contained 138 cycles, so hard rechunking produced five proof
blocks. The production proof and its direct verification both succeeded.

## Comparison

The native-Jolt and Stage-19 rows come from a validated Stage-19 artifact with
the same machine, build profile, workload digest, input, and block target. The
direct row comes from the validated D8 artifact.

| Flow | Proof bytes | Prove time | Verify time | Peak RSS delta |
|---|---:|---:|---:|---:|
| Native Jolt | 67,787 | 2.417 s | 1.418 s | 60.47 MiB |
| Stage 19 dual flow | 22,789 | 83.647 s | 2.641 s | 3.12 GiB |
| Direct chunked D8 | 1,228,289 | 3,266.996 s | 381.388 s | 100.91 GiB |

The direct command's independent `/usr/bin/time -v` measurement reported
105,819,208 KiB maximum RSS (100.92 GiB), zero swaps, and 1:01:30 wall time.
This agrees with the runner's 10 ms sampler (100.91 GiB delta). Release
compilation was measured separately and is not included.

## Direct proof-size decomposition

| Component | Bytes |
|---|---:|
| Statement and execution | 3,079 |
| Lookup subclaims | 122,633 |
| Register subclaims | 47,353 |
| RAM subclaims | 360,973 |
| CPU/R1CS subclaims | 641,733 |
| Dory PCS | 22,553 |
| Empty Nova debug-vector framing | 8 |
| Spartan compressed proof | 18,869 |
| Recursive public state | 11,088 |
| Total | 1,228,289 |

The Rust proof currently retains relation subclaims for direct standalone
verification and adversarial auditing. Consequently this is an exact size for
the current verifier-consumed artifact, not the intended final compact
transport format.

## Memory decomposition

| Measurement | Result |
|---|---:|
| Source blocks / hard proof blocks | 4 / 5 |
| Maximum source-block cycles | 138 |
| Maximum simultaneously resident trace blocks | 2 |
| Maximum resident trace cycles | 266 |
| Estimated resident trace bytes | 29,888 |
| Spool bytes | 13,515 |
| RAM registry entries / bytes | 73 / 1,168 |
| Retained relation subclaims | 20 |
| PCS polynomial coefficients / bytes | 65,536 / 2,097,152 |
| RSS immediately after capture | 4,075,520 bytes |
| RSS immediately after relations | 16,850,944 bytes |
| RSS after PCS/folding construction | 59,434,569,728 bytes |
| Sampled peak RSS delta | 108,355,506,176 bytes |

Trace residency is therefore bounded by the configured block capacity and did
not cause the process peak. The current peak is non-trace memory from Nova
public-parameter/circuit construction, recursive proving, and Spartan
compression.

Major timed phases were:

| Phase | Time |
|---|---:|
| Relation preparation | 235.577 s |
| Nova folding | 1,299.744 s |
| Spartan compression | 1,256.493 s |
| Internal self-verification | 395.569 s |
| Dory commit + opening | 1.246 s |

## Verification evidence

The following commands passed on the server:

```bash
RAYON_NUM_THREADS=8 cargo test -p jolt-core --features nova \
  'direct_streaming::tests::d8_' -- --nocapture

RAYON_NUM_THREADS=8 cargo test -p jolt-core --features nova \
  d7_public_production_api_proves_and_verifies_direct_artifact -- --nocapture

RAYON_NUM_THREADS=8 cargo test -p jolt-core --features nova d8_ -- --nocapture

RAYON_NUM_THREADS=8 cargo test -p jolt-core --features nova \
  --example direct_chunked_stage8_benchmark -- --nocapture

RAYON_NUM_THREADS=8 cargo test -p jolt-core --features nova -- \
  --nocapture --skip zkvm::prover::tests::stdlib_e2e_dory

RAYON_NUM_THREADS=8 target/release/examples/direct_chunked_stage8_benchmark \
  --scale 2 \
  --block-capacity 128 \
  --stage19-baseline benchmark-runs/direct-d8/stage19-fibonacci-2.json \
  --output benchmark-runs/direct-d8/direct-fibonacci-2.json \
  --guest-target target-stage19-d8-guest \
  --memory-sample-interval-ms 10
```

The final artifact digest is:
`2b07bce05e43f1dbb4bbae974e500e5e64b07b2a60248bc2d49f50aba6596f8c`.

The final regression command passed all 720 selected unit tests and the
applicable doc tests. `zkvm::prover::tests::stdlib_e2e_dory` was explicitly
excluded because its external ZeroOS musl installer invokes
`curl --retry-all-errors`, which is unsupported by the server's installed curl
version; an unfiltered run reached that installer and failed there. This is an
environmental guest-toolchain failure outside the D8 code path, not a suppressed
Rust test failure.

## Interpretation and next optimization target

D8 proves that removing the native-Jolt detour and bounding trace residency are
functionally viable. It does not show a performance win: this first combined
recursive circuit is much slower and more memory-intensive than both
baselines. The next optimization work should target the actual measured
bottlenecks rather than trace storage:

1. cache/reuse the direct Nova public parameters and Spartan keys;
2. reduce the combined step-circuit constraint count and eliminate duplicated
   in-circuit hashing;
3. stream or commit relation subclaims instead of retaining all four vectors;
4. define a compact verifier proof type that excludes audit witnesses;
5. profile Nova folding and Spartan compression allocations separately; and
6. add larger workloads and several block capacities after the single-run
   correctness baseline is stable.

The validated workload uses empty advice. A later ZK transport pass must avoid
placing private advice in the verifier-facing artifact while preserving the D6
initial-memory binding.
