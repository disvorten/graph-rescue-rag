# Extended validation results

All values below are generated from completed machine-readable summaries.

## Frozen global-development transfer (not full-wiki)

| Dataset | Queries | Passages | Hybrid FE | Gated FE | Delta FE | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 5,199 | 74,593 | 0.568 | 0.650 | +0.082 | [0.073, 0.091] |
| 2Wiki | 11,160 | 58,432 | 0.324 | 0.397 | +0.073 | [0.068, 0.078] |
| MuSiQue | 1,016 | 33,439 | 0.206 | 0.322 | +0.116 | [0.094, 0.140] |

## Target gate calibration

| Dataset | Cal/Test | Recall frozen->recal. | ECE frozen->recal. | Open frozen->recal. | FE frozen->recal. |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 200/4999 | 0.750->0.969 | 0.122->0.076 | 0.558->0.897 | 0.647->0.665 |
| 2Wiki | 200/10960 | 0.583->0.982 | 0.168->0.076 | 0.435->0.935 | 0.411->0.457 |
| MuSiQue | 200/816 | 0.977->0.960 | 0.081->0.036 | 0.960->0.928 | 0.321->0.322 |

## Official-code released-MuSiQue comparison

| System | FE@7 | SR@7 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|
| StandardRAG official code | 0.227 | 0.580 | 149.2 | 212.0 |
| HippoRAG official code | 0.269 | 0.605 | 3727.5 | 5171.2 |
| GraphRescue hybrid | 0.229 | 0.571 | 37.5 | 57.4 |
| GraphRescue gated MRV | 0.366 | 0.643 | 103.3 | 145.9 |

| Paired FE@7 comparison | Delta | 95% CI |
|---|---:|---:|
| Gated MRV minus HippoRAG | +0.097 | [0.071, 0.125] |
| Gated MRV minus StandardRAG | +0.139 | [0.111, 0.167] |
| Gated MRV minus shared hybrid | +0.137 | [0.112, 0.163] |

## Clean online retrieval latency

| Dataset | Hybrid p50 | Always p50 | Gated p50 | Gated p95 | Mean delta gated-always | 95% CI | Actions always-to-gated |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 179.1 | 212.2 | 208.2 | 284.7 | -6.6 | [-8.2, -5.0] | 1.95-to-1.58 |
| 2Wiki | 178.1 | 220.8 | 201.6 | 265.1 | -12.0 | [-15.1, -9.1] | 1.97-to-1.50 |
| MuSiQue | 220.1 | 288.5 | 282.8 | 363.5 | -6.5 | [-8.9, -4.3] | 3.00-to-2.77 |

Latency is online retrieval only and does not establish that this method is faster than a full external GraphRAG lifecycle.
