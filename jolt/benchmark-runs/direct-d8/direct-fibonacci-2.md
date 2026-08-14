# Direct Chunked Jolt-Nova D8

Workload: `fibonacci-2`
Block capacity: `128`
Artifact digest: `2b07bce05e43f1dbb4bbae974e500e5e64b07b2a60248bc2d49f50aba6596f8c`

| Flow | Proof bytes | Prove us | Verify us | Peak RSS delta bytes |
|---|---:|---:|---:|---:|
| Native Jolt | 67787 | 2417376 | 1418332 | 63406080 |
| Stage 19 dual flow | 22789 | 83647272 | 2641132 | 3348357120 |
| Direct chunked | 1228289 | 3266995966 | 381388490 | 108355506176 |

## Direct memory decomposition

- source trace blocks: 4
- fixed-capacity proof blocks: 5
- trace passes: 2
- max source trace cycles: 138
- max resident trace blocks: 2
- max resident trace cycles: 266
- estimated peak resident trace bytes: 29888
- spooled trace bytes: 13515
- tracked RAM addresses: 73
- initial RAM registry bytes: 1168
- retained relation subclaims: 20
- PCS polynomial bytes: 2097152
- RSS after capture: 4075520
- RSS after relations: 16850944
- RSS after PCS: 59434569728
- peak observed RSS: 59500417024
