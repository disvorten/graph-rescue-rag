from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

from .config import ExperimentConfig
from .io import read_jsonl
from .ollama import OllamaClient
from .text import normalize


FORBIDDEN_CORPUS_KEYS = {
    "answer",
    "answers",
    "supporting_facts",
    "supporting_passage_ids",
    "evidences",
    "evidence_triples",
    "is_supporting",
    "question_decomposition",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _source_records(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "graph_rescue").glob("*.py"))
    paths.extend(sorted((root / "tests").glob("test_*.py")))
    return [_file_record(path) for path in paths]


def _finding(
    findings: list[dict[str, Any]],
    *,
    severity: str,
    kind: str,
    values: Iterable[str],
    message: str,
) -> None:
    examples = sorted(set(str(value) for value in values))
    if not examples:
        return
    findings.append(
        {
            "severity": severity,
            "kind": kind,
            "message": message,
            "count": len(examples),
            "examples": examples[:10],
        }
    )


def freeze_protocol(
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a content-addressed protocol manifest and leakage audit."""

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corpus_rows = list(read_jsonl(config.corpus_path))
    train_rows = list(read_jsonl(config.train_queries_path))
    eval_rows = list(read_jsonl(config.eval_queries_path))
    findings: list[dict[str, Any]] = []

    train_ids = {str(row["id"]) for row in train_rows}
    eval_ids = {str(row["id"]) for row in eval_rows}
    _finding(
        findings,
        severity="high",
        kind="query_id_overlap",
        values=train_ids & eval_ids,
        message="Train and evaluation query IDs overlap.",
    )
    train_questions = {
        normalize(str(row.get("question", ""))): str(row["id"])
        for row in train_rows
    }
    eval_questions = {
        normalize(str(row.get("question", ""))): str(row["id"])
        for row in eval_rows
    }
    duplicated_questions = [
        f"{train_questions[text]}->{eval_questions[text]}"
        for text in train_questions.keys() & eval_questions.keys()
        if text
    ]
    _finding(
        findings,
        severity="high",
        kind="normalized_question_overlap",
        values=duplicated_questions,
        message="Normalized question text occurs in both train and evaluation.",
    )

    corpus_ids = {str(row["id"]) for row in corpus_rows}
    unknown_support = {
        str(support_id)
        for row in train_rows + eval_rows
        for support_id in row.get("supporting_passage_ids", [])
        if str(support_id) not in corpus_ids
    }
    _finding(
        findings,
        severity="high",
        kind="unknown_support_passage",
        values=unknown_support,
        message="A gold supporting passage is missing from the corpus.",
    )
    forbidden = {
        f"{row.get('id', '<missing>')}:{key}"
        for row in corpus_rows
        for key in FORBIDDEN_CORPUS_KEYS & row.keys()
    }
    _finding(
        findings,
        severity="high",
        kind="gold_field_in_corpus",
        values=forbidden,
        message="Corpus rows contain fields that can expose gold supervision.",
    )
    unexplained_entities = set()
    for row in corpus_rows:
        available = normalize(
            f"{row.get('title', '')} {row.get('text', '')}"
        )
        for entity in row.get("entities", []):
            normalized_entity = normalize(str(entity))
            if normalized_entity and normalized_entity not in available:
                unexplained_entities.add(f"{row.get('id')}:{entity}")
    _finding(
        findings,
        severity="medium",
        kind="entity_not_in_passage_text",
        values=unexplained_entities,
        message=(
            "An entity feature cannot be reconstructed from its passage title/text; "
            "document the external entity source or remove it."
        ),
    )

    labels_path = config.learning.counterfactual_labels_path
    if labels_path and Path(labels_path).exists():
        label_query_ids = {
            str(row["query_id"]) for row in read_jsonl(labels_path)
        }
        _finding(
            findings,
            severity="high",
            kind="eval_query_in_counterfactual_labels",
            values=label_query_ids & eval_ids,
            message="Counterfactual training labels contain evaluation queries.",
        )

    inputs = [
        _file_record(config.corpus_path),
        _file_record(config.train_queries_path),
        _file_record(config.eval_queries_path),
    ]
    if labels_path and Path(labels_path).exists():
        inputs.append(_file_record(labels_path))
    sources = _source_records(root)
    source_tree_hash = hashlib.sha256(
        "\n".join(
            f"{record['path']}:{record['sha256']}" for record in sources
        ).encode("utf-8")
    ).hexdigest()
    try:
        ollama_models = OllamaClient(
            config.ollama.base_url, config.ollama.timeout_seconds
        ).models()
    except Exception as exc:  # pragma: no cover - environment dependent
        ollama_models = [f"unavailable:{type(exc).__name__}"]

    protocol_core = {
        "schema_version": 1,
        "config": asdict(config),
        "inputs": inputs,
        "source_tree_sha256": source_tree_hash,
        "counts": {
            "corpus_passages": len(corpus_rows),
            "train_queries": len(train_rows),
            "eval_queries": len(eval_rows),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "ollama_models": ollama_models,
        },
    }
    protocol_id = hashlib.sha256(
        json.dumps(
            protocol_core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    high_count = sum(item["severity"] == "high" for item in findings)
    medium_count = sum(item["severity"] == "medium" for item in findings)
    result = {
        **protocol_core,
        "protocol_id": protocol_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": {
            "passed": high_count == 0,
            "high_findings": high_count,
            "medium_findings": medium_count,
            "findings": findings,
        },
        "source_files": sources,
    }
    (output / "protocol_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "leakage_audit.json").write_text(
        json.dumps(result["audit"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config.save(output / "frozen_config.json")
    return result
