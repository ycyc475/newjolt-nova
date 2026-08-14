# Jolt-Nova Stage 19 benchmark report

- Workload: `fibonacci-2`
- Workload SHA3-256: `bc9c113b76c0bb986dbe4ba76ddb099aa729319282f63d2eaa28c51d1585b3ab`
- Platform: `linux/x86_64/release/1.95-x86_64-unknown-linux-gnu`
- One-time preprocessing: 53600.43 ms
- Runs per block size: 1
- Matrix digest: `e18d2a563e61466f1824d1141521e51eb32df9e7f312498386b9ed91fb0e3e68`

All rows use a verifier-exported production ZK receipt and passed native Jolt, streaming Spartan, recursive BlindFold, group, and Dory PCS verification.

| block size | blocks | cycles | native Jolt prove (ms) | recursive extension (ms) | total (ms) | cycles/s | native peak delta (MiB) | recursive peak delta (MiB) | Jolt proof bytes | recursive payload bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 451 | 2417.38 | 83647.27 | 87551.08 | 5.15 | 60.47 | 3193.24 | 67787 | 22789 |

## Measured relation attribution

| relation | calls | total (ms) | max call (ms) |
|---|---:|---:|---:|
| `cpu-r1cs` | 4 | 0.825 | 0.297 |
| `jolt-lasso-lookup-claim` | 4 | 2.288 | 0.829 |
| `nova-block-fold` | 4 | 3286.707 | 2406.314 |
| `ram-read-write` | 4 | 0.356 | 0.159 |
| `register-read-write` | 4 | 0.599 | 0.213 |
| `spartan-final-compression` | 1 | 1450.844 | 1450.844 |
| `trace-io-claims` | 4 | 0.333 | 0.110 |
| `verified-jolt-receipt-binding` | 4 | 2.151 | 0.647 |

Dominant measured phase: **recursive-blindfold-prove** (76075.12 ms, 86.89% of measured total).

Timing note: `trace` records lazy iterator setup; streamed trace generation is included in `nova_spartan_stream`. Memory values are sampled process physical-memory deltas. Recursive payload bytes are the folded Spartan and BlindFold Spartan byte strings; typed group and Dory PCS objects are verified but not serialized into that subtotal.
