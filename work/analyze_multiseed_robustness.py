"""Aggregate graph-corruption robustness over deterministic seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    "full_evidence",
    "support_recall",
    "graph_actions",
    "harmful_expansions",
    "policy_latency_ms",
)


def load_rows(root: Path, datasets: list[str]) -> list[dict]:
    rows: list[dict] = []
    for dataset in datasets:
        path = root / dataset / "robustness_summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            value = dict(row)
            value.setdefault("dataset", dataset)
            value.setdefault("corruption_seed", 101)
            rows.append(value)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["dataset"]),
                str(row["condition"]),
                str(row["policy"]),
            )
        ].append(row)
    result: list[dict] = []
    for (dataset, condition, policy), values in sorted(groups.items()):
        seeds = sorted({int(value["corruption_seed"]) for value in values})
        record: dict[str, object] = {
            "dataset": dataset,
            "condition": condition,
            "policy": policy,
            "seeds": len(seeds),
            "seed_ids": " ".join(str(seed) for seed in seeds),
        }
        for metric in METRICS:
            observed = [float(value[metric]) for value in values]
            record[f"{metric}_mean"] = mean(observed)
            record[f"{metric}_std"] = (
                stdev(observed) if len(observed) > 1 else 0.0
            )
            record[f"{metric}_min"] = min(observed)
            record[f"{metric}_max"] = max(observed)
        result.append(record)
    return result


def diagnostics(rows: list[dict], *, raw_count: int) -> dict:
    gated = {
        (str(row["dataset"]), str(row["condition"])): row
        for row in rows
        if row["policy"] == "mrv_gated"
    }
    checks = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for family, conditions in (
            ("dropout", ("dropout_10", "dropout_25", "dropout_50")),
            ("false_edges", ("false_edges_10", "false_edges_25", "false_edges_50")),
        ):
            values = [
                float(gated[(dataset, condition)]["full_evidence_mean"])
                for condition in conditions
                if (dataset, condition) in gated
            ]
            checks.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "conditions_present": len(values),
                    "nonincreasing_mean": (
                        len(values) == len(conditions)
                        and all(
                            left >= right
                            for left, right in zip(values, values[1:])
                        )
                    ),
                }
            )
    return {
        "schema_version": 1,
        "raw_rows": raw_count,
        "aggregate_rows": len(rows),
        "dose_response_checks": checks,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="outputs/final_v1_robustness_multiseed"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpot", "2wiki", "musique"],
    )
    parser.add_argument(
        "--output", default="outputs/final_v1/analysis"
    )
    args = parser.parse_args()
    raw = load_rows(Path(args.root), args.datasets)
    if not raw:
        raise SystemExit(f"No robustness summaries found under {args.root}")
    aggregated = aggregate(raw)
    output = Path(args.output)
    write_csv(output / "robustness_multiseed_metrics.csv", aggregated)
    report = diagnostics(aggregated, raw_count=len(raw))
    (output / "robustness_multiseed_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
