# Baseline search and reproducibility log

Last checked: 2026-07-30.

This log records why each comparison was included, adapted, or deferred. Code
availability can change; recheck every entry immediately before submission.

| Work | Relevant overlap | Public artifact at check date | Decision |
|---|---|---|---|
| KG²RAG (NAACL 2025) | semantic seeds followed by graph expansion and context organization | Official GPL-3.0 repository: <https://github.com/nju-websoft/KG2RAG> | Included as an independently implemented, equal-budget **style adaptation**. It is not an exact reproduction and copies no source code. |
| A2RAG (arXiv 2026) | adaptive cost-aware graph retrieval, evidence-sufficiency controller, progressive escalation | Paper/HTML: <https://arxiv.org/abs/2601.21162>; no code repository was linked from the paper page at check date | Cite and discuss. Exact baseline deferred until an official implementation or sufficiently complete specification is available. |
| CatRAG (Findings of ACL 2026) | query-aware graph transitions, hub suppression, complete evidence-chain retrieval | Repository: <https://github.com/kwunhang/CatRAG>; README states that the core CatRAG logic will be released later | Cite and discuss. Recheck the repository before submission; run an exact baseline if code becomes available and can be adapted without changing the frozen protocol. |
| HHS-RAG (JIIS 2026) | hierarchical hypergraph retrieval and subgraph-level decisions | Article: <https://doi.org/10.1007/s10844-026-01077-0>; no public code artifact was found on the article page at check date | Cite and discuss. Do not claim superiority without a reproducible implementation and aligned protocol. |
| PruneRAG (arXiv 2026) | confidence-guided adaptive expansion and pruning | Paper: <https://arxiv.org/abs/2601.11024> | Related adaptive-retrieval work, but not a graph-expansion baseline under the current passage-graph protocol. |

## Inclusion rule

An external system is eligible for the central comparison only if:

1. its implementation and license permit execution and publication of results;
2. its inputs can be mapped to the frozen train/evaluation split without using
   evaluation labels during construction or tuning;
3. initial corpus, seed count, final passage count, token budget, and online
   action budget can be matched or their mismatch can be reported explicitly;
4. failures to run are documented rather than silently replaced by a
   hand-built approximation.

## Current interpretation

The broad ideas “adaptive graph retrieval,” “query-aware traversal,” and
“subgraph-level decision” are already present in 2026 literature. The defensible
contribution of Graph Rescue RAG is narrower:

- a strong hybrid retriever is frozen first and graph work is framed as a
  local rescue operation;
- preflight and continuation gates are calibrated separately against a target
  recall for rescuable states;
- candidates are ranked by conditional marginal rescue value relative to the
  evidence already selected;
- comparisons preserve the final evidence budget and report paired wins,
  losses, action cost, calibration, and sensitivity to false versus missing
  graph edges.

This framing must remain explicit in the title, abstract, introduction,
related work, limitations, and cover letter.
