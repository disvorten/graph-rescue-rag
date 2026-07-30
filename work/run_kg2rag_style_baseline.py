from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_rescue.config import ExperimentConfig
from graph_rescue.experiment import Experiment
from graph_rescue.io import read_jsonl, write_jsonl
from graph_rescue.metrics import (
    aggregate_rows,
    holm_bonferroni,
    paired_bootstrap_difference,
    retrieval_metrics,
)
from graph_rescue.policy import KG2RAGStylePolicy
from graph_rescue.reader import AnswerPresenceReader


POLICY = "kg2rag_style_equal_budget"
REFERENCE_URLS = {
    "paper": "https://aclanthology.org/2025.naacl-long.449/",
    "official_repository": "https://github.com/nju-websoft/KG2RAG",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(
    config_path: Path,
    source_results_path: Path,
    source_traces_path: Path,
) -> str:
    payload = {
        "schema_version": 1,
        "config_sha256": sha256(config_path),
        "source_results_sha256": sha256(source_results_path),
        "source_traces_sha256": sha256(source_traces_path),
        "policy_source_sha256": sha256(
            PROJECT_ROOT / "graph_rescue/policy.py"
        ),
        "runner_source_sha256": sha256(Path(__file__).resolve()),
        "policy": POLICY,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_source_rows(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(path):
        result.setdefault(str(row["query_id"]), {})[str(row["policy"])] = row
    return result


def load_source_traces(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for trace in read_jsonl(path):
        result.setdefault(str(trace["query_id"]), {})[
            str(trace["policy"])
        ] = trace
    return result


def append_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if (
            value.get("query_id")
            and value.get("row", {}).get("policy") == POLICY
            and value.get("trace", {}).get("policy") == POLICY
        ):
            result[str(value["query_id"])] = value
    return result


def paired_values(
    kg_rows: list[dict[str, Any]],
    source_rows: dict[str, dict[str, dict[str, Any]]],
    source_policy: str,
    metric: str,
) -> tuple[list[float], list[float]]:
    ordered = sorted(kg_rows, key=lambda row: row["query_id"])
    left = [float(row["metrics"][metric]) for row in ordered]
    right = [
        float(source_rows[row["query_id"]][source_policy]["metrics"][metric])
        for row in ordered
    ]
    return left, right


def render_report(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate"][POLICY]
    comparisons = summary["comparisons"]
    lines = [
        "# KG²RAG-style equal-budget baseline",
        "",
        "Это независимая адаптация паттерна KG²RAG, а не точное "
        "воспроизведение опубликованной системы. В ней сохранены три идеи: "
        "семантические seed passages, графовое расширение и организация "
        "контекста по seed-центричным группам. Исходный triplet KG и "
        "FlagReranker заменены графом passage/entity и тем же семантическим "
        "скорером, который доступен основному протоколу.",
        "",
        "## Равенство бюджета",
        "",
        f"- final_k: {summary['budget']['final_k']}",
        f"- token budget: {summary['budget']['token_budget']}",
        f"- max graph additions: {summary['budget']['max_graph_additions']}",
        f"- budget violations: {summary['budget_audit']['violations']}",
        f"- seed mismatches against frozen run: "
        f"{summary['reproducibility_audit']['seed_mismatches']}",
        "",
        "## Результат",
        "",
        "| Policy | Full evidence | Support recall | Actions | Policy ms |",
        "|---|---:|---:|---:|---:|",
        (
            f"| {POLICY} | {aggregate['full_evidence']:.3f} | "
            f"{aggregate['support_recall']:.3f} | "
            f"{aggregate['graph_actions']:.3f} | "
            f"{aggregate['policy_latency_ms']:.3f} |"
        ),
        "",
        "## Парные различия",
        "",
        "| Comparison | Metric | Delta | 95% CI | p (Holm) |",
        "|---|---|---:|---:|---:|",
    ]
    for name, item in sorted(comparisons.items()):
        left, right, metric = name.split("_vs_", 1)[0], "", ""
        del left, right, metric
        label = item["label"]
        lines.append(
            f"| {label} | {item['metric']} | {item['difference']:.3f} | "
            f"[{item['ci95_low']:.3f}, {item['ci95_high']:.3f}] | "
            f"{item['p_value_holm']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Положительная delta означает преимущество KG²RAG-style baseline.",
            "",
            "## Источники",
            "",
            f"- Paper: {REFERENCE_URLS['paper']}",
            f"- Official repository: {REFERENCE_URLS['official_repository']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    source_run = Path(args.source_run).resolve()
    source_results_path = source_run / "query_results.jsonl"
    source_traces_path = source_run / "retrieval_traces.jsonl"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_fingerprint = fingerprint(
        config_path, source_results_path, source_traces_path
    )
    meta_path = output_dir / "checkpoint_meta.json"
    checkpoint_path = output_dir / "evaluation_checkpoint.jsonl"
    if args.force:
        checkpoint_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != run_fingerprint:
            raise ValueError(
                "Checkpoint fingerprint mismatch; use a new output directory "
                "or --force."
            )
    else:
        meta_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "fingerprint": run_fingerprint,
                    "policy": POLICY,
                    "config": str(config_path),
                    "source_run": str(source_run),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    config = ExperimentConfig.load(config_path).resolve_paths(config_path.parent)
    source_rows = load_source_rows(source_results_path)
    source_traces = load_source_traces(source_traces_path)
    expected_policies = {"hybrid", "mrv_gated"}
    missing_source = [
        query_id
        for query_id, policies in source_rows.items()
        if not expected_policies.issubset(policies)
    ]
    if missing_source:
        raise ValueError(
            f"Source run lacks required policies for {len(missing_source)} queries"
        )

    experiment = Experiment(config)
    policy = KG2RAGStylePolicy(
        passages=experiment.passages,
        graph=experiment.graph,
        feature_extractor=experiment.extractor,
        config=experiment._policy_config(),
    )
    reader = AnswerPresenceReader()
    completed = load_checkpoint(checkpoint_path)
    rows = [value["row"] for value in completed.values()]
    traces = [value["trace"] for value in completed.values()]
    if completed:
        print(
            f"baseline resume: {len(completed)}/{len(experiment.eval_queries)}",
            flush=True,
        )

    seed_mismatches = 0
    for index, example in enumerate(experiment.eval_queries, start=1):
        if example.id in completed:
            continue
        retrieval_start = time.perf_counter()
        ranking = experiment.retriever.retrieve(
            example.question, config.retrieval.rerank_k
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
        trace = policy.run(example, ranking)
        source_hybrid = source_rows[example.id]["hybrid"]
        if (
            trace.seed_passage_ids
            != source_traces[example.id]["hybrid"]["seed_passage_ids"]
        ):
            seed_mismatches += 1

        reader_start = time.perf_counter()
        prediction = reader.predict(
            example, trace.final_passage_ids, experiment.passages
        )
        reader_ms = (time.perf_counter() - reader_start) * 1000.0
        trace.retrieval_latency_ms = retrieval_ms
        trace.reader_latency_ms = reader_ms
        trace.total_latency_ms = retrieval_ms + trace.policy_latency_ms + reader_ms
        metrics = retrieval_metrics(example, trace, prediction.answer)
        metrics.update(
            {
                "retrieval_latency_ms": retrieval_ms,
                "policy_latency_ms": trace.policy_latency_ms,
                "reader_latency_ms": reader_ms,
                "total_latency_ms": trace.total_latency_ms,
            }
        )
        hybrid_metrics = source_hybrid["metrics"]
        missing_from_hybrid = set(example.supporting_passage_ids) - set(
            source_hybrid["retrieved_ids"]
        )
        newly_recovered = missing_from_hybrid & set(trace.final_passage_ids)
        metrics.update(
            {
                "graph_support_gain": (
                    metrics["support_recall"]
                    - float(hybrid_metrics["support_recall"])
                ),
                "graph_query_rescued": float(
                    metrics["full_evidence"]
                    > float(hybrid_metrics["full_evidence"])
                ),
                "graph_query_harmed": float(
                    metrics["support_recall"]
                    < float(hybrid_metrics["support_recall"])
                ),
                "rescue_recall": (
                    len(newly_recovered) / len(missing_from_hybrid)
                    if missing_from_hybrid
                    else 0.0
                ),
            }
        )
        row = {
            "query_id": example.id,
            "question": example.question,
            "question_type": example.question_type,
            "dataset": example.dataset,
            "support_count": len(example.supporting_passage_ids),
            "policy": POLICY,
            "supporting_ids": list(example.supporting_passage_ids),
            "retrieved_ids": trace.final_passage_ids,
            "predictions": {reader.name: prediction.answer},
            "metrics": metrics,
        }
        bundle = {
            "query_id": example.id,
            "row": row,
            "trace": trace.to_dict(),
        }
        append_checkpoint(checkpoint_path, bundle)
        rows.append(row)
        traces.append(bundle["trace"])
        if index % max(1, config.evaluation.progress_every) == 0:
            print(
                f"baseline progress: {index}/{len(experiment.eval_queries)}",
                flush=True,
            )

    rows.sort(key=lambda row: row["query_id"])
    traces.sort(key=lambda trace: trace["query_id"])
    # Include mismatches from resumed rows in the final audit.
    seed_mismatches = sum(
        trace["seed_passage_ids"]
        != source_traces[trace["query_id"]]["hybrid"]["seed_passage_ids"]
        for trace in traces
    )
    violations = sum(
        len(row["retrieved_ids"]) > config.retrieval.final_k
        or float(row["metrics"]["evidence_tokens"])
        > config.retrieval.evidence_token_budget
        or float(row["metrics"]["graph_actions"])
        > config.graph.max_actions
        for row in rows
    )

    comparisons: dict[str, dict[str, Any]] = {}
    for source_policy in ("hybrid", "mrv_gated"):
        for metric in ("full_evidence", "support_recall"):
            left, right = paired_values(
                rows, source_rows, source_policy, metric
            )
            name = f"{POLICY}_vs_{source_policy}_{metric}"
            result = paired_bootstrap_difference(
                left,
                right,
                samples=config.evaluation.bootstrap_samples,
                seed=config.learning.random_seed,
            )
            result["label"] = f"{POLICY} vs {source_policy}"
            result["metric"] = metric
            comparisons[name] = result
    adjusted = holm_bonferroni(
        {
            name: float(value["p_value_two_sided"])
            for name, value in comparisons.items()
        }
    )
    for name, value in adjusted.items():
        comparisons[name]["p_value_holm"] = value

    outcomes = {}
    for source_policy in ("hybrid", "mrv_gated"):
        wins = losses = ties = 0
        for row in rows:
            left = float(row["metrics"]["full_evidence"])
            right = float(
                source_rows[row["query_id"]][source_policy]["metrics"][
                    "full_evidence"
                ]
            )
            if left > right:
                wins += 1
            elif left < right:
                losses += 1
            else:
                ties += 1
        outcomes[source_policy] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }

    summary = {
        "schema_version": 1,
        "policy": POLICY,
        "adaptation_status": (
            "Independent KG²RAG-style adaptation; not an exact reproduction."
        ),
        "reference_urls": REFERENCE_URLS,
        "config": asdict(config),
        "source_run": str(source_run),
        "queries": len(rows),
        "budget": {
            "seed_k": config.retrieval.seed_k,
            "final_k": config.retrieval.final_k,
            "token_budget": config.retrieval.evidence_token_budget,
            "max_graph_additions": config.graph.max_actions,
            "max_hops": config.graph.max_hops,
            "frontier_cap": config.graph.frontier_cap,
        },
        "budget_audit": {"violations": violations},
        "reproducibility_audit": {
            "fingerprint": run_fingerprint,
            "seed_mismatches": seed_mismatches,
        },
        "aggregate": aggregate_rows(rows),
        "comparisons": comparisons,
        "paired_full_evidence_outcomes": outcomes,
    }
    write_jsonl(output_dir / "query_results.jsonl", rows)
    write_jsonl(output_dir / "retrieval_traces.jsonl", traces)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )
    print(
        f"baseline complete: {len(rows)} queries; "
        f"full_evidence={summary['aggregate'][POLICY]['full_evidence']:.6f}; "
        f"budget_violations={violations}; seed_mismatches={seed_mismatches}",
        flush=True,
    )


if __name__ == "__main__":
    main()
