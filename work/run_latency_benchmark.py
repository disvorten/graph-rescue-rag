from __future__ import annotations

"""Measure online retrieval and graph-policy costs under a frozen protocol."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_rescue.config import ExperimentConfig
from graph_rescue.experiment import Experiment
from graph_rescue.learning import GateModel, MRVModel
from graph_rescue.metrics import paired_bootstrap_difference
from graph_rescue.policy import KG2RAGStylePolicy
from graph_rescue.profiling import directory_bytes, latency_summary, process_rss_bytes


def stable_sample(items: list[Any], count: int, seed: int) -> list[Any]:
    return sorted(
        items,
        key=lambda item: hashlib.sha256(
            f"{seed}|{item.id}".encode("utf-8")
        ).digest(),
    )[:count]


def load_models(config: ExperimentConfig) -> tuple[MRVModel, GateModel, GateModel]:
    model_dir = Path(config.model_dir)
    mrv = MRVModel.load(model_dir / "mrv_model.json")
    preflight_path = model_dir / "preflight_gate_model.json"
    if not preflight_path.exists():
        preflight_path = model_dir / "gate_model.json"
    preflight = GateModel.load(preflight_path)
    continue_path = model_dir / "continue_gate_model.json"
    continuation = (
        GateModel.load(continue_path)
        if continue_path.exists()
        else preflight
    )
    return mrv, preflight, continuation


def effectiveness(example: Any, final_ids: list[str]) -> dict[str, float]:
    support = set(example.supporting_passage_ids)
    retrieved = set(final_ids)
    return {
        "support_recall": len(support & retrieved) / max(1, len(support)),
        "full_evidence": float(bool(support) and support.issubset(retrieved)),
    }


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def paired_query_means(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    metric: str,
) -> tuple[list[float], list[float]]:
    """Average repetitions within query before paired query-level inference."""

    def group(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(row["query_id"])].append(float(row[metric]))
        return grouped

    left = group(left_rows)
    right = group(right_rows)
    ids = sorted(set(left) & set(right))
    if not ids:
        raise ValueError(f"No aligned queries for paired metric {metric}")
    return (
        [mean(left[query_id]) for query_id in ids],
        [mean(right[query_id]) for query_id in ids],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--cache-state",
        choices=("cold", "warm"),
        default="warm",
        help="Label for the passage-index/cache state at Experiment construction.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=("hybrid", "mrv_always", "mrv_gated", "kg2rag_style_equal_budget"),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = ExperimentConfig.load(config_path).resolve_paths(config_path.parent)
    # Validate frozen artifacts before paying the corpus indexing cost.
    mrv, preflight, continuation = load_models(config)
    experiment = Experiment(config)
    policies = {
        policy.name: policy
        for policy in experiment.policies(mrv, preflight, continuation)
    }
    policies["kg2rag_style_equal_budget"] = KG2RAGStylePolicy(
        passages=experiment.passages,
        graph=experiment.graph,
        feature_extractor=experiment.extractor,
        config=experiment._policy_config(),
    )
    unknown = set(args.policies) - set(policies)
    if unknown:
        raise ValueError(f"Unknown policies: {sorted(unknown)}")

    needed = args.warmup_queries + args.queries
    sampled = stable_sample(experiment.eval_queries, needed, args.seed)
    if len(sampled) < needed:
        raise ValueError(
            f"Requested {needed} queries, but only {len(sampled)} are available"
        )
    warmup = sampled[: args.warmup_queries]
    benchmark_queries = sampled[args.warmup_queries :]

    # Warm model kernels and graph code on questions excluded from measurement.
    for example in warmup:
        experiment.retriever.prime_uncached_query_embedding(example.question)
        ranking = experiment.retriever.retrieve(
            example.question, config.retrieval.rerank_k
        )
        for name in args.policies:
            policies[name].run(example, ranking)

    records: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    for repetition in range(args.repetitions):
        order = list(benchmark_queries)
        rng.shuffle(order)
        for index, example in enumerate(order, start=1):
            retrieval_started = time.perf_counter()
            query_embedding_ms = (
                experiment.retriever.prime_uncached_query_embedding(
                    example.question
                )
            )
            ranking = experiment.retriever.retrieve(
                example.question, config.retrieval.rerank_k
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

            policy_order = list(args.policies)
            rng.shuffle(policy_order)
            for name in policy_order:
                trace = policies[name].run(example, ranking)
                metrics = effectiveness(example, trace.final_passage_ids)
                records.append(
                    {
                        "query_id": example.id,
                        "repetition": repetition,
                        "policy": name,
                        "query_embedding_ms": query_embedding_ms,
                        "retrieval_ms": retrieval_ms,
                        "policy_ms": trace.latency_ms,
                        "online_total_ms": retrieval_ms + trace.latency_ms,
                        "graph_reads": trace.graph_reads,
                        "candidate_paths_scored": trace.candidate_paths_scored,
                        "graph_actions": sum(
                            action.selected_passage_id is not None
                            for action in trace.actions
                        ),
                        "evidence_tokens": trace.evidence_tokens,
                        **metrics,
                    }
                )
            if index % 25 == 0 or index == len(order):
                print(
                    f"latency repetition {repetition + 1}/{args.repetitions}: "
                    f"{index}/{len(order)}",
                    flush=True,
                )

    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_policy[record["policy"]].append(record)
    aggregate = {}
    for name, rows in sorted(by_policy.items()):
        first_rep = [row for row in rows if row["repetition"] == 0]
        aggregate[name] = {
            "retrieval_latency": latency_summary(
                [row["retrieval_ms"] for row in rows]
            ),
            "policy_latency": latency_summary(
                [row["policy_ms"] for row in rows]
            ),
            "online_total_latency": latency_summary(
                [row["online_total_ms"] for row in rows]
            ),
            "query_embedding_latency": latency_summary(
                [row["query_embedding_ms"] for row in rows]
            ),
            "mean_graph_reads": mean([row["graph_reads"] for row in rows]),
            "mean_candidate_paths_scored": mean(
                [row["candidate_paths_scored"] for row in rows]
            ),
            "mean_graph_actions": mean([row["graph_actions"] for row in rows]),
            "full_evidence": mean(
                [row["full_evidence"] for row in first_rep]
            ),
            "support_recall": mean(
                [row["support_recall"] for row in first_rep]
            ),
        }

    comparisons = {}
    baseline_rows = by_policy.get("hybrid", [])
    for name, rows in sorted(by_policy.items()):
        if name == "hybrid" or not baseline_rows:
            continue
        left, right = paired_query_means(rows, baseline_rows, "online_total_ms")
        comparisons[f"{name}_minus_hybrid_online_total_ms"] = (
            paired_bootstrap_difference(
                left,
                right,
                samples=5000,
                seed=args.seed,
            )
        )

    always_rows = by_policy.get("mrv_always", [])
    gated_rows = by_policy.get("mrv_gated", [])
    if always_rows and gated_rows:
        for metric in ("online_total_ms", "policy_ms", "graph_actions"):
            left, right = paired_query_means(gated_rows, always_rows, metric)
            comparisons[f"mrv_gated_minus_mrv_always_{metric}"] = (
                paired_bootstrap_difference(
                    left,
                    right,
                    samples=5000,
                    seed=args.seed,
                )
            )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_name(f"{output.stem}_raw.jsonl")
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "measurement_scope": (
            "Online retrieval only: fresh query embedding, BM25+dense+fusion+"
            "reranking, and policy/graph traversal. Answer generation excluded."
        ),
        "config": config_path.relative_to(ROOT).as_posix(),
        "cache_state_at_initialization": args.cache_state,
        "queries": args.queries,
        "warmup_queries_excluded": args.warmup_queries,
        "repetitions": args.repetitions,
        "randomized_query_and_policy_order": True,
        "fresh_query_embedding_each_repetition": True,
        "paired_inference_unit": (
            "query; repetitions are averaged within query before bootstrap"
        ),
        "initialization_profile": experiment.initialization_profile,
        "resource_snapshot": {
            "process_rss_bytes": process_rss_bytes(),
            "cache_directory_bytes": directory_bytes(config.cache_dir),
            "model_directory_bytes": directory_bytes(config.model_dir),
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "logical_cpu_count": os.cpu_count(),
            "embedding_model": config.ollama.embedding_model,
        },
        "aggregate": aggregate,
        "paired_latency_comparisons": comparisons,
        "raw_records": raw_path.relative_to(ROOT).as_posix(),
    }
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
