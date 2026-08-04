# Official-code baseline on released MuSiQue

| System | Full evidence@7 | Support recall@7 | Median ms | p95 ms |
|---|---:|---:|---:|---:|
| StandardRAG official code | 0.227 | 0.580 | 149.2 | 212.0 |
| HippoRAG official code | 0.269 | 0.605 | 3727.5 | 5171.2 |
| GraphRescue hybrid | 0.229 | 0.571 | 37.5 | 57.4 |
| GraphRescue gated MRV | 0.366 | 0.643 | 103.3 | 145.9 |

All systems use the released HippoRAG MuSiQue corpus and aligned query IDs. HippoRAG/StandardRAG are official code with local Qwen models and the released OpenIE artifact; these are not published-paper number reproductions. Online latency is descriptive because systems were executed sequentially in separate processes.
