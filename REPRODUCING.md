# Reproducing Graph Rescue RAG

The commands below separate a fast software check from the multi-hour research
protocol. Run them from the repository root on Windows PowerShell.

## 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-research.txt
```

Start Ollama and install the frozen local models:

```powershell
ollama pull qwen3-embedding:0.6b
ollama pull bge-m3
ollama pull qwen3:8b
```

Record exact versions before a replication:

```powershell
python --version
python -m pip freeze
ollama --version
ollama list
```

## 2. Fast verification

```powershell
python -m unittest discover -s tests -v
python -m graph_rescue doctor --config examples/demo_config.json
python -m graph_rescue run --config examples/demo_config.json
```

Expected software-test result for the current research snapshot: 58 tests pass.

## 3. Prepare frozen benchmark protocols

Download the benchmark files described in [DATA.md](DATA.md), verify their
SHA-256 hashes, then run:

```powershell
python work/prepare_final_protocol.py --datasets hotpot 2wiki musique
```

Expected protocol sizes:

| Dataset | Train | Eval | Passages | Graph edges |
|---|---:|---:|---:|---:|
| HotpotQA | 1,000 | 1,000 | 19,189 | 144,586 |
| 2WikiMultiHopQA | 1,000 | 1,000 | 11,347 | 74,676 |
| MuSiQue | 1,000 | 1,000 | 23,630 | 180,645 |

Compare the generated protocol IDs with
`outputs/REPRODUCIBILITY_MANIFEST.md`. Stop if any input hash, split ID, or
protocol ID differs.

## 4. Primary retrieval suite

```powershell
python work/run_final_suite.py --dataset hotpot --base-config examples/final_hotpot_qwen_config.json --embedding-models qwen3-embedding:0.6b --seeds 101 202 303
python work/run_final_suite.py --dataset 2wiki --base-config examples/final_2wiki_qwen_config.json --embedding-models qwen3-embedding:0.6b --seeds 101 202 303
python work/run_final_suite.py --dataset musique --base-config examples/final_musique_qwen_config.json --embedding-models qwen3-embedding:0.6b --seeds 101 202 303
python work/run_final_suite.py --dataset hotpot --base-config examples/final_hotpot_qwen_config.json --embedding-models bge-m3:latest --seeds 101
python work/run_final_suite.py --dataset 2wiki --base-config examples/final_2wiki_qwen_config.json --embedding-models bge-m3:latest --seeds 101
python work/run_final_suite.py --dataset musique --base-config examples/final_musique_qwen_config.json --embedding-models bge-m3:latest --seeds 101
python work/analyze_final_results.py
```

The exact CLI options can be inspected with `--help`. The Qwen run is the
primary analysis; BGE-M3 is a representation-sensitivity check.

## 5. Robustness and published-pattern control

```powershell
python work/run_final_robustness.py --dataset hotpot --clean-config work/final_protocol/generated_configs/hotpot/qwen3-embedding_0.6b_seed_101.json --corruption-seeds 101 202 303 404 505 --output-root outputs/final_v1_robustness_multiseed
python work/run_final_robustness.py --dataset 2wiki --clean-config work/final_protocol/generated_configs/2wiki/qwen3-embedding_0.6b_seed_101.json --corruption-seeds 101 202 303 404 505 --output-root outputs/final_v1_robustness_multiseed
python work/run_final_robustness.py --dataset musique --clean-config work/final_protocol/generated_configs/musique/qwen3-embedding_0.6b_seed_101.json --corruption-seeds 101 202 303 404 505 --output-root outputs/final_v1_robustness_multiseed
python work/analyze_multiseed_robustness.py
python work/run_kg2rag_style_baseline.py --config work/final_protocol/generated_configs/hotpot/qwen3-embedding_0.6b_seed_101.json --source-run outputs/final_v1/hotpot/qwen3-embedding_0.6b/seed_101 --output-dir outputs/published_baselines/hotpot/kg2rag_style_qwen_seed101
python work/run_kg2rag_style_baseline.py --config work/final_protocol/generated_configs/2wiki/qwen3-embedding_0.6b_seed_101.json --source-run outputs/final_v1/2wiki/qwen3-embedding_0.6b/seed_101 --output-dir outputs/published_baselines/2wiki/kg2rag_style_qwen_seed101
python work/run_kg2rag_style_baseline.py --config work/final_protocol/generated_configs/musique/qwen3-embedding_0.6b_seed_101.json --source-run outputs/final_v1/musique/qwen3-embedding_0.6b/seed_101 --output-dir outputs/published_baselines/musique/kg2rag_style_qwen_seed101
python work/analyze_published_baselines.py
python work/analyze_final_results.py
```

The KG²RAG-style run is an independent equal-budget adaptation, not an exact
reproduction of the original implementation.

## 6. External global-development validation

The following protocol expands the searchable corpus and evaluates every
previously unseen official-development question. It is deliberately named a
global development/distractor setting: none of these runs is a HotpotQA
full-wiki experiment.

```powershell
python work/prepare_global_corpus_protocol.py --datasets hotpot 2wiki musique
python -m graph_rescue evaluate --config examples/global_hotpot_qwen_config.json
python -m graph_rescue evaluate --config examples/global_2wiki_qwen_config.json
python -m graph_rescue evaluate --config examples/global_musique_qwen_config.json
python -m work.analyze_global_results
```

The learned MRV, gate, and threshold are loaded from the frozen seed-101
primary runs. No global-evaluation labels are used for training, calibration,
graph construction, or hyperparameter selection.

The primary transfer result keeps the gate fully frozen. A separate,
label-efficient diagnostic reserves 200 deterministic target queries for
recalibration and reports only on the remaining queries:

```powershell
python -m work.analyze_gate_transfer --calibration-queries 200 --target-recall 0.95
```

Its retrieval rows are explicitly a preflight-only ablation that switches
between the already computed hybrid and MRV-always traces. They do not replace
the primary two-stage-gate result.

| Dataset | Unseen eval queries | Searchable passages | Graph edges |
|---|---:|---:|---:|
| HotpotQA | 5,199 | 74,593 | 662,078 |
| 2WikiMultiHopQA | 11,160 | 58,432 | 466,837 |
| MuSiQue | 1,016 | 33,439 | 261,985 |

## 7. Online latency and resource protocol

```powershell
python work/run_latency_benchmark.py --config examples/final_hotpot_qwen_config.json --output outputs/latency_v1/hotpot.json --queries 200 --repetitions 3
python work/run_latency_benchmark.py --config examples/final_2wiki_qwen_config.json --output outputs/latency_v1/2wiki.json --queries 200 --repetitions 3
python work/run_latency_benchmark.py --config examples/final_musique_qwen_config.json --output outputs/latency_v1/musique.json --queries 200 --repetitions 3
```

This benchmark forces a fresh query embedding, excludes warm-up, randomizes
query and policy order, and reports p50/p95 latency. For paired bootstrap
intervals, repetitions are first averaged within query and the query is the
independent resampling unit. The JSON output also records one-time
corpus/index/graph initialization, cache hits/misses, resident memory, graph
reads, and graph actions. Answer generation is outside this latency boundary
and must be reported separately.

## 8. Official HippoRAG 2 code on the released MuSiQue artifact

Clone the official MIT-licensed HippoRAG repository at a recorded revision
under `work/vendor/HippoRAG`, check out commit
`c617143f01477243992a63b2e2151cc003dd3b21`, and apply the tracked Windows/
Ollama compatibility patch. The patch only makes optional backend imports lazy
and sanitizes model names used as Windows paths. It also requests
`reasoning_effort=none` for local Qwen3 recognition memory and separates that
mode in the response-cache key. It does not change retrieval, graph, PPR, or
evaluation logic.

```powershell
git -C work/vendor/HippoRAG apply ../../../patches/hipporag_windows_ollama_compat.patch
py -3.12 -m venv work/venvs/hipporag
work/venvs/hipporag/Scripts/python.exe -m pip install -r requirements-hipporag-windows.txt
```

Then prepare a dedicated environment and adapt only the released corpus/query
identifiers:

```powershell
python work/prepare_hipporag_benchmark_adapter.py
python -m graph_rescue evaluate --config examples/hipporag_released_musique_qwen_config.json
```

The adapter preserves all 11,656 released passages. One pair differs only by
Unicode whitespace after normalization; the adapter assigns a deterministic
suffix instead of silently collapsing the pair and records the collision in
its manifest.

Run the official code from its dedicated environment, first as StandardRAG and
then as HippoRAG:

```powershell
$env:PYTHONPATH=(Resolve-Path 'work/vendor/HippoRAG/src').Path
work/venvs/hipporag/Scripts/python.exe -m work.run_official_hipporag_baseline --mode standard --max-queries 1000
work/venvs/hipporag/Scripts/python.exe -m work.run_official_hipporag_baseline --mode hipporag --max-queries 1000
```

The wrapper uses the official retrieval, PPR, and evaluator paths together
with the released GPT-4o-mini OpenIE artifact. It substitutes the documented
local Ollama models, disables Qwen3 thinking for structured recognition-memory
output, and checkpoints every query. Therefore the result is an official-code
local-model reproduction, not a claim to reproduce the published HippoRAG 2
numbers. The first HippoRAG build is substantial and creates roughly 1.9 GB of
chunk, entity, and fact stores in this environment; per-invocation timings and
exact versions are retained in the local summary and logs.

After the two official-code runs and the aligned Graph Rescue run complete,
generate the compact comparison and paired full-evidence intervals:

```powershell
python -m work.analyze_official_baselines
```

Evaluation checkpoint fingerprints use schema version 2 and include the
SHA-256 content hashes of corpus, training, evaluation, model, and optional
label inputs plus relevant source hashes. Resuming with changed file contents
therefore fails closed even when paths and configuration text are unchanged.

## 9. Reader evaluation

Prepare fixed full evaluation subsets:

```powershell
python work/prepare_reader_subset.py --datasets hotpot 2wiki musique --size 1000
```

Then run Qwen3-8B separately for each dataset:

```powershell
python work/run_reader_analysis.py --dataset hotpot --base-config examples/final_hotpot_qwen_config.json --size 1000
python work/run_reader_analysis.py --dataset 2wiki --base-config examples/final_2wiki_qwen_config.json --size 1000
python work/run_reader_analysis.py --dataset musique --base-config examples/final_musique_qwen_config.json --size 1000
python work/analyze_full_reader.py
```

Reader generations are cached locally and excluded from Git. Report the number
of generation calls, cache hits, parsing failures, official answer/support
metrics, paired bootstrap confidence intervals, and the exact question-ID hash.

## 10. Regenerate manuscript assets

```powershell
python work/generate_paper_assets.py
python work/generate_jiis_extended_assets.py
python work/update_theory_doc_v5.py
python work/update_theory_doc_v6.py
Set-Location outputs/jiis_submission
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Set-Location ../..
python work/build_jiis_submission.py
```

Do not manually edit numeric LaTeX tables generated from experiment outputs.
Review the PDF visually and check the LaTeX log for overflow, undefined
references, warnings, and errors.

The extended-asset and Word-v6 builders are readiness gates: they stop when a
global-transfer, official-baseline, or clean-latency summary is missing. Render
the Word artifact page by page before treating it as a deliverable.

## 11. Release audit

```powershell
python scripts/release_audit.py
```

The audit checks the Git allowlist, file sizes, forbidden data/cache paths,
common secret patterns, required publication artifacts, tests, and author
metadata. It does not replace human inspection.
