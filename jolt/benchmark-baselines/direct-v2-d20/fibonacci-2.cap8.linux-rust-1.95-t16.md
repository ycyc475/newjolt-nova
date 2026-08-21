# Direct Block-Jolt V2 workload benchmark

- Workload: `fibonacci-2`
- Workload digest: `ed552887acb91562e3d3f0a14ba4f16b4e759a91b20cb3d890279597c9775156`
- Block capacity / blocks / cycles: 8 / 63 / 451
- Derived RAM K: 8192
- Setup ID: `35ada68405dd324d4aba8533a0d9fba274103bd8653b0ae7f1037f59fd9ee28e`
- Verified: true

| Phase | Time (s) |
|---|---:|
| Reusable setup | 150.450 |
| Trace capture | 0.003 |
| Lookup relation | 18.534 |
| Register relation | 1.288 |
| RAM relation | 164.841 |
| CPU/R1CS relation | 2687.221 |
| PCS polynomial materialization | 0.129 |
| PCS commitments | 32.937 |
| Nova folding | 266.427 |
| PCS spool writes | 68.290 |
| Dory closure | 1185.278 |
| Spartan compression | 41.040 |
| Final decode + verify | 237.041 |

| Size | Bytes |
|---|---:|
| Trace spool | 35869 |
| PCS witness spool (prover-only) | 8746840360 |
| Spartan proof | 13184 |
| Dory proof | 22484923 |
| Final artifact | 22500260 |
