# Target-domain gate recalibration

A deterministic target calibration subset is disjoint from the reported held-out queries. This is a label-efficient adaptation diagnostic, not the primary frozen-transfer result.

| Dataset | Cal/Test | Frozen recall | Recal. recall | Frozen ECE | Recal. ECE |
|---|---:|---:|---:|---:|---:|
| hotpot | 200/4999 | 0.750 | 0.969 | 0.122 | 0.076 |
| 2wiki | 200/10960 | 0.583 | 0.982 | 0.168 | 0.076 |
| musique | 200/816 | 0.977 | 0.960 | 0.081 | 0.036 |

| Dataset | Frozen open | Recal. open | Frozen actions | Recal. actions | Frozen FE | Recal. FE |
|---|---:|---:|---:|---:|---:|---:|
| hotpot | 0.558 | 0.897 | 1.046 | 1.635 | 0.647 | 0.665 |
| 2wiki | 0.435 | 0.935 | 0.836 | 1.740 | 0.411 | 0.457 |
| musique | 0.960 | 0.928 | 2.868 | 2.772 | 0.321 | 0.322 |
