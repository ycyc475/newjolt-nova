# Direct Block-Jolt V2 workload benchmark

- Workload: `fibonacci-2`
- Workload digest: `ed552887acb91562e3d3f0a14ba4f16b4e759a91b20cb3d890279597c9775156`
- Block capacity / blocks / cycles: 8 / 63 / 451
- Derived RAM K: 8192
- Setup ID: `35ada68405dd324d4aba8533a0d9fba274103bd8653b0ae7f1037f59fd9ee28e`
- Source commit: `63f54ce63ffab53fbf1adddb7ae830423f99b835`
- Rayon threads: `16`
- Verified: true
- Prove-stream throughput: 0.098034 cycles/s

| Phase | Time (s) |
|---|---:|
| Reusable setup | 167.803 |
| Trace capture | 0.003 |
| Lookup relation | 17.223 |
| Register relation | 1.388 |
| RAM relation | 172.796 |
| CPU/R1CS relation | 2800.308 |
| PCS polynomial materialization | 0.228 |
| PCS commitments | 33.259 |
| Nova folding | 336.830 |
| PCS spool writes | 1.185 |
| Dory closure | 916.223 |
| Spartan compression | 50.451 |
| Final digest sealing | 0.137 |
| Final artifact size pass | 0.139 |
| Final decode + verify | 241.177 |

| Size | Bytes |
|---|---:|
| Trace spool | 35869 |
| Logical dense PCS coefficients | 8729602560 |
| PCS witness spool (prover-only) | 18550169 |
| Spartan proof | 13184 |
| Dory proof | 22484923 |
| Final artifact | 22500260 |

| PCS encoding | Count |
|---|---:|
| Non-zero coefficients | 38392 |
| Dense polynomials | 440 |
| Sparse polynomials | 5860 |
