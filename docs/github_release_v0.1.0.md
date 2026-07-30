# Graph Rescue RAG v0.1.0

First public research-artifact release for selective graph rescue after hybrid
retrieval.

Suggested GitHub description:

> Selective local graph rescue after hybrid retrieval for multi-hop RAG, with calibrated gating and equal-budget evaluation.

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

## Reproducible claim

Under the frozen pooled-corpus protocol with 1,000 disjoint evaluation
questions per dataset, gated MRV increased the full-evidence rate over the
shared hybrid baseline on all three datasets:

- HotpotQA: `0.640 → 0.810`, Δ `+0.170`, 95% paired bootstrap CI
  `[0.143, 0.197]`;
- 2WikiMultiHopQA: `0.367 → 0.572`, Δ `+0.205`,
  `[0.178, 0.231]`;
- MuSiQue: `0.156 → 0.252`, Δ `+0.096`,
  `[0.074, 0.118]`.

## Negative finding

Graph expansion is substantially more sensitive to false edges than to missing
edges. In the five-seed corruption study, at a 25% corruption dose the
full-evidence disadvantage of false-edge injection relative to edge dropout
was `0.172` on HotpotQA, `0.178` on 2WikiMultiHopQA, and `0.082` on MuSiQue.
This makes edge precision and denoising a higher priority than increasing graph
recall indiscriminately.

## Important scope notes

- This is a pooled-corpus research protocol, not an official leaderboard
  submission or SOTA claim.
- Benchmark data, model caches, model outputs, and upstream evaluator snapshots
  are not distributed in this release.
- The KG²RAG-style control is an independent adaptation, not an execution of
  the official KG²RAG repository.

## Reproduction

Start with `REPRODUCING.md` and verify input files against `DATA.md`. Run:

```text
python scripts/release_audit.py
```

before using or redistributing a modified snapshot.

## Citation

Use the release `CITATION.cff` and the DOI assigned by Zenodo.
