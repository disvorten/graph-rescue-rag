from __future__ import annotations

"""Run HippoRAG 2's official code with released OpenIE and local Ollama.

This is an official-code reproduction under a resource-matched local-model
setting, not a reproduction of the paper's NV-Embed-v2/large-LLM numbers.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "work" / "vendor" / "HippoRAG"
VENDOR_SRC = VENDOR_ROOT / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

os.environ.setdefault("OPENAI_API_KEY", "sk-local-ollama")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "work" / "hf_cache"))
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(PROJECT_ROOT / "work" / "hf_cache" / "datasets"),
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from hipporag.HippoRAG import HippoRAG
from hipporag.StandardRAG import StandardRAG
from hipporag.utils.config_utils import BaseConfig


DATASET_ROOT = VENDOR_ROOT / "reproduce" / "dataset"
RELEASED_OPENIE = (
    VENDOR_ROOT
    / "outputs"
    / "musique"
    / "openie_results_ner_gpt-4o-mini.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_diff_sha256(path: Path) -> str | None:
    """Hash compatibility edits applied on top of the recorded upstream commit."""

    try:
        diff = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "diff",
                "--binary",
                "--no-ext-diff",
            ]
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(diff).hexdigest()


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def document(sample: dict) -> str:
    return f"{sample['title']}\n{sample['text']}"


def gold_documents(sample: dict) -> list[str]:
    return sorted(
        {
            f"{item['title']}\n{item.get('paragraph_text', item.get('text', ''))}"
            for item in sample["paragraphs"]
            if item.get("is_supporting", True)
        }
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        result[str(row["query_id"])] = row
    return result


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("standard", "hipporag"), required=True)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument(
        "--save-dir",
        default="work/official_baselines/hipporag_official_musique",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/official_baselines/hipporag_official_musique",
    )
    args = parser.parse_args()

    corpus_path = DATASET_ROOT / "musique_corpus.json"
    queries_path = DATASET_ROOT / "musique.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    samples = json.loads(queries_path.read_text(encoding="utf-8"))[
        : args.max_queries
    ]
    docs = [document(item) for item in corpus]
    doc_ids: dict[str, int] = {}
    for index, doc in enumerate(docs):
        doc_ids.setdefault(doc, index)

    save_dir = (PROJECT_ROOT / args.save_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    renamed_openie = save_dir / "openie_results_ner_qwen3_8b.json"
    if not renamed_openie.exists():
        shutil.copy2(RELEASED_OPENIE, renamed_openie)

    config = BaseConfig(
        save_dir=str(save_dir),
        llm_base_url="http://127.0.0.1:11434/v1",
        llm_name="qwen3:8b",
        embedding_base_url="http://127.0.0.1:11434/v1/embeddings",
        embedding_model_name="VLLM/qwen3-embedding:0.6b",
        dataset="musique",
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        rerank_dspy_file_path=str(
            VENDOR_ROOT
            / "src"
            / "hipporag"
            / "prompts"
            / "dspy_prompts"
            / "filter_llama3.3-70B-Instruct.json"
        ),
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=5,
        embedding_batch_size=64,
        openie_mode="online",
    )
    logging.basicConfig(level=logging.INFO)
    rag_class = HippoRAG if args.mode == "hipporag" else StandardRAG
    initialized_started = time.perf_counter()
    rag = rag_class(global_config=config)
    initialization_seconds = time.perf_counter() - initialized_started
    indexing_started = time.perf_counter()
    rag.index(docs)
    indexing_seconds = time.perf_counter() - indexing_started

    raw_path = output_dir / f"{args.mode}_query_results.jsonl"
    completed = load_completed(raw_path)
    for index, sample in enumerate(samples, start=1):
        query_id = str(sample["id"])
        if query_id in completed:
            continue
        started = time.perf_counter()
        result = rag.retrieve([str(sample["question"])], num_to_retrieve=200)[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        retrieved = list(result.docs)
        gold = set(gold_documents(sample))
        row: dict[str, Any] = {
            "query_id": query_id,
            "question": str(sample["question"]),
            "mode": args.mode,
            "latency_ms": latency_ms,
            "support_count": len(gold),
            "retrieved_doc_indices_top200": [
                doc_ids.get(doc, -1) for doc in retrieved
            ],
        }
        for k in (1, 2, 5, 7, 10, 20, 50, 100, 200):
            selected = set(retrieved[:k])
            row[f"support_recall_at_{k}"] = len(gold & selected) / max(1, len(gold))
            row[f"full_evidence_at_{k}"] = float(bool(gold) and gold.issubset(selected))
        append_jsonl(raw_path, row)
        completed[query_id] = row
        if index % 10 == 0 or index == len(samples):
            print(
                f"{args.mode} progress: {index}/{len(samples)} "
                f"({len(completed)} checkpointed)",
                flush=True,
            )

    rows = [completed[str(sample["id"])] for sample in samples]
    aggregate = {}
    for metric in (
        "support_recall_at_5",
        "support_recall_at_7",
        "support_recall_at_10",
        "support_recall_at_20",
        "full_evidence_at_5",
        "full_evidence_at_7",
        "full_evidence_at_10",
        "full_evidence_at_20",
    ):
        values = [float(row[metric]) for row in rows]
        aggregate[metric] = statistics.fmean(values)
    latencies = [float(row["latency_ms"]) for row in rows]
    aggregate["latency_ms"] = {
        "mean": statistics.fmean(latencies),
        "median": percentile(latencies, 0.5),
        "p95": percentile(latencies, 0.95),
    }
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "status": "official-code local-model reproduction",
        "not_paper_number_reproduction": True,
        "official_repository": "https://github.com/OSU-NLP-Group/HippoRAG",
        "official_repository_revision": git_revision(VENDOR_ROOT),
        "official_repository_worktree_diff_sha256": git_worktree_diff_sha256(
            VENDOR_ROOT
        ),
        "paper": "https://arxiv.org/abs/2502.14802",
        "queries": len(rows),
        "corpus_passages": len(corpus),
        "models": {
            "embedding": "qwen3-embedding:0.6b via Ollama OpenAI-compatible endpoint",
            "recognition_memory_llm": (
                "qwen3:8b via Ollama" if args.mode == "hipporag" else None
            ),
            "released_openie": "gpt-4o-mini artifact from official repository",
        },
        "compatibility_patch": (
            "Eager imports of unused GritLM/vLLM backends were made lazy and "
            "model identifiers were sanitized only when used in Windows paths. "
            "Ollama Qwen3 recognition memory used reasoning_effort=none with "
            "cache-key separation; "
            "retrieval, graph, PPR and evaluation algorithms are unchanged."
        ),
        "source_hashes": {
            "corpus": sha256(corpus_path),
            "queries": sha256(queries_path),
            "released_openie": sha256(RELEASED_OPENIE),
            "hipporag_py": sha256(VENDOR_SRC / "hipporag" / "HippoRAG.py"),
            "compatibility_patch": sha256(
                PROJECT_ROOT / "patches" / "hipporag_windows_ollama_compat.patch"
            ),
            "runner": sha256(Path(__file__).resolve()),
        },
        "initialization_seconds": initialization_seconds,
        "indexing_seconds_this_invocation": indexing_seconds,
        "retrieval_configuration": {
            "retrieval_top_k": 200,
            "evaluation_k": [5, 7, 10, 20],
            "linking_top_k": 5,
            "qa_top_k": 5,
            "embedding_batch_size": 64,
            "recognition_memory_reasoning_effort": (
                "none" if args.mode == "hipporag" else None
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "logical_cpu_count": os.cpu_count(),
            "packages": package_versions(
                (
                    "numpy",
                    "python-igraph",
                    "openai",
                    "litellm",
                    "torch",
                    "transformers",
                    "networkx",
                    "pandas",
                    "tiktoken",
                )
            ),
        },
        "aggregate": aggregate,
        "checkpoint": str(raw_path),
    }
    summary_path = output_dir / f"{args.mode}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
