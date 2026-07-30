# Graph Rescue RAG v0.1.0

First public research-artifact release for selective graph rescue after hybrid
retrieval.

Suggested GitHub description:

> Selective, budget-aware graph rescue after BM25+dense+reranker retrieval for multi-hop evidence chains.

Suggested topics:

`rag`, `graph-rag`, `information-retrieval`, `multi-hop-qa`,
`knowledge-graph`, `ollama`, `reproducible-research`

## Included

- BM25 + dense + RRF + feature reranking baseline;
- passage/entity graph with provenance and deterministic corruption;
- marginal rescue value selector;
- calibrated preflight and continuation gates;
- frozen 1,000-train/1,000-evaluation protocols for HotpotQA,
  2WikiMultiHopQA, and MuSiQue;
- three training seeds and BGE-M3 representation-sensitivity results;
- equal-budget independent KG²RAG-style control;
- Qwen3-8B reader evaluation;
- compact aggregate tables, tests, and reproducibility scripts. Full
  manuscript drafts remain local until the selected venue confirms its
  preprint/public-repository policy.

## Important scope notes

- This is a pooled-corpus research protocol, not an official leaderboard
  submission or SOTA claim.
- Benchmark data, model caches, model outputs, and upstream evaluator snapshots
  are not distributed in this release.
- The KG²RAG-style control is an independent adaptation, not an execution of
  the official KG²RAG repository.
- AI tools assisted implementation, experiment orchestration, analysis, and
  drafting. Human authors are responsible for verification and all scholarly
  claims.

## Reproduction

Start with `REPRODUCING.md` and verify input files against `DATA.md`. Run:

```text
python scripts/release_audit.py
```

before using or redistributing a modified snapshot.

## Citation

Use the release `CITATION.cff` and Zenodo DOI. The human author metadata are
filled; do not create the archival release until the scientific claims,
licences, and clean-environment reproduction have been checked.
