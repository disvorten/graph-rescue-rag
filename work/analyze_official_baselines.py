"""Create an aligned compact report for the official HippoRAG experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from graph_rescue.io import read_jsonl
from graph_rescue.metrics import paired_bootstrap_difference
from graph_rescue.profiling import percentile


def load_official_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in read_jsonl(path)}


def load_graph_rows(path: Path, policy: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["query_id"]): row
        for row in read_jsonl(path)
        if row.get("policy") == policy
    }


def latency(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
    }


def aggregate_official(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict:
    latencies = [float(rows[query_id]["latency_ms"]) for query_id in ids]
    return {
        "full_evidence_at_7": statistics.fmean(
            float(rows[query_id]["full_evidence_at_7"]) for query_id in ids
        ),
        "support_recall_at_7": statistics.fmean(
            float(rows[query_id]["support_recall_at_7"]) for query_id in ids
        ),
        "retrieval_latency": latency(latencies),
    }


def aggregate_graph(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict:
    latencies = [
        float(rows[query_id]["metrics"]["total_latency_ms"])
        for query_id in ids
    ]
    return {
        "full_evidence_at_7": statistics.fmean(
            float(rows[query_id]["metrics"]["full_evidence"])
            for query_id in ids
        ),
        "support_recall_at_7": statistics.fmean(
            float(rows[query_id]["metrics"]["support_recall"])
            for query_id in ids
        ),
        "retrieval_latency": latency(latencies),
        "mean_graph_actions": statistics.fmean(
            float(rows[query_id]["metrics"].get("graph_actions", 0.0))
            for query_id in ids
        ),
        "graph_open_rate": statistics.fmean(
            float(rows[query_id]["metrics"].get("graph_actions", 0.0) > 0.0)
            for query_id in ids
        ),
    }


def comparison(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    ids: list[str],
    *,
    left_kind: str,
    right_kind: str,
    samples: int,
    seed: int,
) -> dict:
    def value(row: dict[str, Any], kind: str) -> float:
        if kind == "official":
            return float(row["full_evidence_at_7"])
        return float(row["metrics"]["full_evidence"])

    return paired_bootstrap_difference(
        [value(left[query_id], left_kind) for query_id in ids],
        [value(right[query_id], right_kind) for query_id in ids],
        samples=samples,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("outputs/official_baselines/hipporag_official_musique"),
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("outputs/official_baselines/graph_rescue_musique"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/official_baselines/aligned_analysis"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    standard = load_official_rows(
        args.official_root / "standard_query_results.jsonl"
    )
    hipporag = load_official_rows(
        args.official_root / "hipporag_query_results.jsonl"
    )
    graph_query_path = args.graph_root / "query_results.jsonl"
    hybrid = load_graph_rows(graph_query_path, "hybrid")
    gated = load_graph_rows(graph_query_path, "mrv_gated")
    ids = sorted(set(standard) & set(hipporag) & set(hybrid) & set(gated))
    if not ids:
        raise ValueError("No aligned official-baseline query IDs")

    systems = {
        "StandardRAG_official_code": aggregate_official(standard, ids),
        "HippoRAG_official_code": aggregate_official(hipporag, ids),
        "GraphRescue_hybrid": aggregate_graph(hybrid, ids),
        "GraphRescue_gated_MRV": aggregate_graph(gated, ids),
    }
    comparisons = {
        "GraphRescue_gated_minus_HippoRAG_full_evidence_at_7": comparison(
            gated,
            hipporag,
            ids,
            left_kind="graph",
            right_kind="official",
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "GraphRescue_gated_minus_StandardRAG_full_evidence_at_7": comparison(
            gated,
            standard,
            ids,
            left_kind="graph",
            right_kind="official",
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "GraphRescue_gated_minus_GraphRescue_hybrid_full_evidence_at_7": comparison(
            gated,
            hybrid,
            ids,
            left_kind="graph",
            right_kind="graph",
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    standard_summary = json.loads(
        (args.official_root / "standard_summary.json").read_text(encoding="utf-8")
    )
    hipporag_summary = json.loads(
        (args.official_root / "hipporag_summary.json").read_text(encoding="utf-8")
    )
    graph_summary = json.loads(
        (args.graph_root / "summary.json").read_text(encoding="utf-8")
    )
    result = {
        "protocol": "official-hipporag-released-musique-aligned-v1",
        "queries": len(ids),
        "unique_query_ids": len(ids),
        "corpus_passages": int(hipporag_summary["corpus_passages"]),
        "evaluation_k": 7,
        "status": "official-code local-model reproduction",
        "not_paper_number_reproduction": True,
        "official_repository_revision": hipporag_summary[
            "official_repository_revision"
        ],
        "recognition_memory_reasoning_effort": hipporag_summary[
            "retrieval_configuration"
        ]["recognition_memory_reasoning_effort"],
        "systems": systems,
        "paired_full_evidence_comparisons": comparisons,
        "offline_seconds": {
            "StandardRAG_initialization": standard_summary["initialization_seconds"],
            "StandardRAG_indexing_this_invocation": standard_summary[
                "indexing_seconds_this_invocation"
            ],
            "HippoRAG_initialization": hipporag_summary["initialization_seconds"],
            "HippoRAG_indexing_this_invocation": hipporag_summary[
                "indexing_seconds_this_invocation"
            ],
            "GraphRescue_initialization": graph_summary.get(
                "initialization_profile", {}
            ).get("total_ms", 0.0)
            / 1000.0,
        },
        "comparability_note": (
            "All systems use the released HippoRAG MuSiQue corpus and aligned "
            "query IDs. HippoRAG/StandardRAG are official code with local Qwen "
            "models and the released OpenIE artifact; these are not published-paper "
            "number reproductions. Online latency is descriptive because systems "
            "were executed sequentially in separate processes."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "system",
                "full_evidence_at_7",
                "support_recall_at_7",
                "median_latency_ms",
                "p95_latency_ms",
            ),
        )
        writer.writeheader()
        for name, value in systems.items():
            writer.writerow(
                {
                    "system": name,
                    "full_evidence_at_7": value["full_evidence_at_7"],
                    "support_recall_at_7": value["support_recall_at_7"],
                    "median_latency_ms": value["retrieval_latency"]["median_ms"],
                    "p95_latency_ms": value["retrieval_latency"]["p95_ms"],
                }
            )
    lines = [
        "# Official-code baseline on released MuSiQue",
        "",
        "| System | Full evidence@7 | Support recall@7 | Median ms | p95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in systems.items():
        lines.append(
            "| {name} | {fe:.3f} | {sr:.3f} | {p50:.1f} | {p95:.1f} |".format(
                name=name.replace("_", " "),
                fe=value["full_evidence_at_7"],
                sr=value["support_recall_at_7"],
                p50=value["retrieval_latency"]["median_ms"],
                p95=value["retrieval_latency"]["p95_ms"],
            )
        )
    lines.extend(["", result["comparability_note"], ""])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
