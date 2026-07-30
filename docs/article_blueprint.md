# Article blueprint

## Working title

**Selective Graph Rescue for Hybrid Retrieval: Calibrated Marginal-Value
Expansion in Multi-Hop Question Answering**

Avoid titles implying that adaptive graph retrieval itself is new. A2RAG,
CatRAG, PruneRAG, and HHS-RAG establish overlapping adaptive or
structure-aware ideas; see `docs/baseline_search_log.md`.

## Defensible claims

1. In the frozen pooled-corpus protocol, local graph expansion recovers
   supporting passages missed by a strong BM25+dense+RRF baseline.
2. Conditional marginal rescue value outperforms relevance-only selection
   under the same seeds, frontier, final context, token budget, and graph-action
   cap.
3. Separate calibrated preflight and continuation gates reduce graph actions
   relative to always-expand MRV with little retrieval loss.
4. The effect is stable across three training seeds and a second embedding
   backbone; false graph edges are a larger risk than missing edges in the
   tested corruption regimes.
5. An independent KG²RAG-style control confirms that generic seed expansion is
   useful but does not explain the complete gated-MRV gain.

These are protocol-bounded empirical claims, not SOTA or universal claims.

## Novelty boundary

The paper does **not** claim to invent:

- semantic seed retrieval followed by graph expansion;
- adaptive allocation of retrieval effort;
- query-aware graph traversal;
- subgraph-level decision-making.

The narrower contribution is the combination of:

- graph traversal framed as repair after a frozen hybrid retriever;
- candidate scoring by marginal utility conditioned on already selected
  evidence;
- separate recall-calibrated start/continue decisions;
- an equal-budget factorial and corruption analysis with paired uncertainty,
  wins/losses, and downstream reader validation.

## Method section

### Problem definition

- Hybrid retrieval returns ranked passages and `seed_k` anchors.
- A query is *rescuable* when missing supporting evidence is absent from the
  flat final context but reachable from a seed within the graph-action budget.
- A graph action selects one candidate path and adds its target passage.
- Every policy shares initial rankings, graph, frontier, action cap, passage
  cap, and token budget.

### Marginal Rescue Value

Describe the conditional feature representation and four modeled outcomes:
support addition, evidence completion, reader gain, and harmful expansion.
Report explicit token, hop, noise, and hub penalties. Distinguish MRV from
semantic relevance and from utility measured after retrieval.

### Graph Rescuability Gate

Describe separate preflight and continuation targets, train/calibration split,
cross-validated calibration choice, bootstrap-conservative threshold selection,
and the target-recall constraint. Thresholds must never use evaluation
questions.

## Experimental questions

- **RQ1:** How much reachable oracle opportunity remains after hybrid search?
- **RQ2:** Does MRV outperform relevance on identical local frontiers?
- **RQ3:** How much work does calibrated gating avoid at fixed rescue recall?
- **RQ4:** Is the MRV × gate interaction positive, neutral, or negative?
- **RQ5:** Does retrieval improvement transfer to a fixed local 8B reader?
- **RQ6:** How stable are conclusions across seeds and embedding backbones?
- **RQ7:** How do missing and false graph edges change rescue quality?
- **RQ8:** Does generic KG²RAG-style expansion explain the gain?

## Required evidence

Already available:

1. frozen dataset/graph statistics and protocol hashes;
2. main equal-budget retrieval table with paired confidence intervals;
3. selector × gate factorial metrics and calibration;
4. three training seeds and BGE-M3 sensitivity;
5. paired win/loss/tie analysis and diagnostic slices;
6. graph-corruption dose response;
7. equal-budget KG²RAG-style control;
8. Qwen3-8B reader results and official scorers;
9. reproducible code, tests, compact outputs, and manuscript.

Still required before submission:

1. five-seed graph-corruption aggregate;
2. completed 1,000-query reader results on all three datasets;
3. human audit of 100 stratified traces;
4. offline graph/index construction time and clean equal-hardware p50/p95;
5. human verification of references, licenses, statistics, and claims;
6. exact comparison with a 2026 adaptive graph baseline if official code
   becomes available before submission.

## Negative-result path

Retain null and harmful outcomes. Diagnose:

- missing or weak anchors;
- absent/false graph edges;
- hub-driven semantic drift;
- MRV ranking errors;
- preflight/continuation false negatives;
- context eviction and reader evidence forgetting.

This framing supports a useful diagnostic paper even when a component fails.
