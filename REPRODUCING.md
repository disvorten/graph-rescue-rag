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

Expected software-test result for the release snapshot: 36 tests pass.

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

## 6. Reader evaluation

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

## 7. Regenerate manuscript assets

```powershell
python work/generate_paper_assets.py
Set-Location outputs/graph_rescue_article
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
Set-Location ../..
```

Do not manually edit numeric LaTeX tables generated from experiment outputs.
Review the PDF visually and check the LaTeX log for overflow, undefined
references, warnings, and errors.

## 8. Release audit

```powershell
python scripts/release_audit.py
```

The audit checks the Git allowlist, file sizes, forbidden data/cache paths,
common secret patterns, required publication artifacts, tests, and author
metadata. It does not replace human inspection.
