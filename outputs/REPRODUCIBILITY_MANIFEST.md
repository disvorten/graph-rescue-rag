# Graph Rescue RAG: reproducibility manifest

Snapshot date: 2026-07-30

## Frozen protocols

| Dataset | Train | Eval | Corpus passages | Graph edges | Protocol ID |
|---|---:|---:|---:|---:|---|
| HotpotQA | 1 000 | 1 000 | 19 189 | 144 586 | `02fea6d4337bd7105f38aba29ae0b13c8830d1a0031cb0c23410b3b90d70c12f` |
| 2WikiMultiHopQA | 1 000 | 1 000 | 11 347 | 74 676 | `914a116c9fbf473b4d806a3525e46b8805f9cf56cb5c7f7f9174dca6ea345a32` |
| MuSiQue | 1 000 | 1 000 | 23 630 | 180 645 | `06f341a90ccf54c80cb8c039cee720d380dafbcf1386a2f243f7f549b6bf68e7` |

Protocol manifests and audits are stored in `outputs/final_protocols/<dataset>/`.

## Models

- Primary embeddings: `qwen3-embedding:0.6b`
- Secondary embeddings: `bge-m3:latest`
- Counterfactual labels and reader diagnostic: `qwen3:8b`
- Runtime: local Ollama

## Runs

- Primary experiment: 3 datasets × seeds 101, 202, 303.
- Representation sensitivity: 3 datasets × BGE-M3 × seed 101.
- Full reader evaluation: 3 datasets × 2 policies × 1 000 fixed eval queries,
  with official scorers and official per-query paired inference.
- Robustness: clean plus dropout 10/25/50%, false edges 10/25/50%, and mixed
  25/25 over fixed corruption seeds 101, 202, 303, 404, and 505.
- Published-pattern control: KG²RAG-style equal-budget adaptation, 3 datasets × 1 000 eval queries, seed 101/Qwen embedding.
- Unit/integration tests: 38 passed.

## Primary result artifacts

- `outputs/final_v1/analysis/policy_metrics.csv`
- `outputs/final_v1/analysis/across_seed_metrics.csv`
- `outputs/final_v1/analysis/paired_outcomes.csv`
- `outputs/final_v1/analysis/robustness_metrics.csv`
- `outputs/final_v1/analysis/robustness_multiseed_metrics.csv`
- `outputs/final_v1/analysis/robustness_multiseed_summary.json`
- `outputs/final_v1/analysis/reader_official_metrics.csv`
- `outputs/final_v1/analysis/reader_full_metrics.csv`
- `outputs/final_v1/analysis/reader_full_summary.json`
- `outputs/final_v1/analysis/error_examples.json`
- `outputs/final_v1/analysis/full_evidence_by_dataset.png`
- `outputs/published_baselines/summary.json`
- `outputs/published_baselines/comparison.csv`
- `outputs/published_baselines/REPORT_RU.md`
- `outputs/published_baselines/<dataset>/kg2rag_style_qwen_seed101/query_results.jsonl`
- `outputs/published_baselines/<dataset>/kg2rag_style_qwen_seed101/retrieval_traces.jsonl`

## Manuscript artifacts

Full LaTeX and Word manuscript drafts are retained locally pending written
confirmation of the selected venue's preprint/public-repository policy. Central
numeric tables in the local draft are generated from experiment outputs and
must not be edited manually.

## Remaining release work

- Add exact environment lock/export.
- Add top-level run commands and expected runtime.
- Confirm dataset licences and avoid prohibited redistribution.
- Complete a stratified manual audit of at least 100 traces.
- Measure offline graph/index construction and equal-hardware end-to-end latency.
- Add independent human reproduction record.
- Create a clean public repository and archive a tagged release in Zenodo.
