# Frozen-model global-development validation

These are external transfer results on previously unseen official-dev queries. The searchable pool is the complete development/distractor corpus plus the frozen training sample; it is **not full-wiki**.

| Dataset | Queries | Passages | Hybrid full evidence | Gated MRV | Δ | 95% CI | Gate open |
|---|---:|---:|---:|---:|---:|---:|---:|
| hotpot | 5,199 | 74,593 | 0.568 | 0.650 | +0.082 | [0.073, 0.091] | 53.5% |
| 2wiki | 11,160 | 58,432 | 0.324 | 0.397 | +0.073 | [0.068, 0.078] | 42.7% |
| musique | 1,016 | 33,439 | 0.206 | 0.322 | +0.116 | [0.094, 0.140] | 96.3% |

Intervals are paired 95% bootstrap confidence intervals; the three metric tests within each dataset are Holm-adjusted. Frozen models and gate thresholds are not refitted on these queries.
