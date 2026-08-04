"""Aggregate frozen-model global-development transfer experiments.

The script intentionally refuses partial datasets unless ``--allow-partial``
is supplied.  It writes only compact aggregate/statistical artifacts; raw
benchmark text and per-query outputs remain ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from graph_rescue.io import read_jsonl
from graph_rescue.metrics import holm_bonferroni, paired_bootstrap_difference


DATASETS = ("hotpot", "2wiki", "musique")
POLICIES = ("hybrid", "mrv_always", "mrv_gated")
METRICS = (
    "full_evidence",
    "support_recall",
    "support_mrr",
    "support_ndcg",
    "graph_actions",
    "graph_reads",
    "candidate_paths_scored",
    "evidence_tokens",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(dataset: str, run: Path, samples: int) -> tuple[dict, list[dict]]:
    summary_path = run / "summary.json"
    query_path = run / "query_results.jsonl"
    if not summary_path.exists() or not query_path.exists():
        raise FileNotFoundError(f"Incomplete global run: {run}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(read_jsonl(query_path))
    by_policy = {
        policy: {
            str(row["query_id"]): row
            for row in rows
            if row.get("policy") == policy
        }
        for policy in POLICIES
    }
    common = sorted(set.intersection(*(set(value) for value in by_policy.values())))
    if not common:
        raise ValueError(f"No aligned policies for {dataset}")

    comparisons = {}
    p_values = {}
    paired_rows = []
    for query_id in common:
        base = by_policy["hybrid"][query_id]
        gated = by_policy["mrv_gated"][query_id]
        paired_rows.append(
            {
                "dataset": dataset,
                "query_id": query_id,
                "support_count": base.get("support_count", ""),
                "question_type": base.get("question_type", ""),
                "hybrid_full_evidence": base["metrics"]["full_evidence"],
                "gated_full_evidence": gated["metrics"]["full_evidence"],
                "full_evidence_delta": (
                    gated["metrics"]["full_evidence"]
                    - base["metrics"]["full_evidence"]
                ),
                "support_recall_delta": (
                    gated["metrics"]["support_recall"]
                    - base["metrics"]["support_recall"]
                ),
                "gated_graph_actions": gated["metrics"]["graph_actions"],
            }
        )
    for metric in ("full_evidence", "support_recall", "support_ndcg"):
        left = [by_policy["mrv_gated"][qid]["metrics"][metric] for qid in common]
        right = [by_policy["hybrid"][qid]["metrics"][metric] for qid in common]
        key = f"mrv_gated_vs_hybrid_{metric}"
        value = paired_bootstrap_difference(
            left, right, samples=samples, seed=20260804
        )
        comparisons[key] = value
        p_values[key] = value["p_value_two_sided"]
    adjusted = holm_bonferroni(p_values)
    for key, value in adjusted.items():
        comparisons[key]["p_value_holm"] = value

    aggregate = summary["aggregate"]
    compact = {
        "dataset": dataset,
        "queries": len(common),
        "corpus_passages": summary["graph"]["passages"],
        "graph_entities": summary["graph"]["entities"],
        "graph_edges": summary["graph"]["edges"],
        "policies": {
            policy: {
                metric: aggregate[policy].get(metric)
                for metric in METRICS
                if metric in aggregate[policy]
            }
            for policy in POLICIES
        },
        "gate_open_rate": sum(
            by_policy["mrv_gated"][qid]["metrics"]["graph_actions"] > 0
            for qid in common
        )
        / len(common),
        "full_evidence_outcomes": {
            "wins": sum(row["full_evidence_delta"] > 0 for row in paired_rows),
            "losses": sum(row["full_evidence_delta"] < 0 for row in paired_rows),
            "ties": sum(row["full_evidence_delta"] == 0 for row in paired_rows),
        },
        "comparisons": comparisons,
        "initialization_profile": summary.get("initialization_profile", {}),
    }
    return compact, paired_rows


def markdown_report(result: dict) -> str:
    lines = [
        "# Frozen-model global-development validation",
        "",
        "These are external transfer results on previously unseen official-dev "
        "queries. The searchable pool is the complete development/distractor "
        "corpus plus the frozen training sample; it is **not full-wiki**.",
        "",
        "| Dataset | Queries | Passages | Hybrid full evidence | Gated MRV | Δ | 95% CI | Gate open |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        if dataset not in result["datasets"]:
            continue
        item = result["datasets"][dataset]
        comparison = item["comparisons"]["mrv_gated_vs_hybrid_full_evidence"]
        lines.append(
            "| {dataset} | {queries:,} | {passages:,} | {base:.3f} | "
            "{gated:.3f} | {delta:+.3f} | [{low:.3f}, {high:.3f}] | {gate:.1%} |".format(
                dataset=dataset,
                queries=item["queries"],
                passages=item["corpus_passages"],
                base=item["policies"]["hybrid"]["full_evidence"],
                gated=item["policies"]["mrv_gated"]["full_evidence"],
                delta=comparison["difference"],
                low=comparison["ci95_low"],
                high=comparison["ci95_high"],
                gate=item["gate_open_rate"],
            )
        )
    lines.extend(
        [
            "",
            "Intervals are paired 95% bootstrap confidence intervals; the three "
            "metric tests within each dataset are Holm-adjusted. Frozen models and "
            "gate thresholds are not refitted on these queries.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/global_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/global_v1/analysis"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    result = {
        "protocol": "graph-rescue-global-dev-v1",
        "setting": "global development/distractor; not full-wiki",
        "model_transfer": "frozen seed-101 models and thresholds; no refit",
        "bootstrap_samples": args.bootstrap_samples,
        "datasets": {},
    }
    paired = []
    for dataset in DATASETS:
        run = args.root / dataset / "frozen_seed_101"
        try:
            item, rows = analyze_run(dataset, run, args.bootstrap_samples)
        except FileNotFoundError:
            if args.allow_partial:
                continue
            raise
        result["datasets"][dataset] = item
        paired.extend(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output / "paired_outcomes.csv", paired)
    (args.output / "REPORT.md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
