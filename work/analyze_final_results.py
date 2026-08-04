from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from graph_rescue.io import read_jsonl


PRIMARY_EMBEDDING = "qwen3-embedding_0.6b"
PRIMARY_SEED = "seed_101"
POLICY_ORDER = (
    "bm25",
    "dense",
    "rrf_fusion",
    "hybrid",
    "relevance_always",
    "mrv_always",
    "relevance_gated",
    "mrv_gated",
    "oracle_upper_bound",
)
ROBUSTNESS_CONDITIONS = (
    "clean",
    "dropout_10",
    "dropout_25",
    "dropout_50",
    "false_edges_10",
    "false_edges_25",
    "false_edges_50",
    "mixed_25_25",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover_runs(root: Path) -> list[dict]:
    records = []
    for summary_path in root.glob("*/*/seed_*/summary.json"):
        relative = summary_path.relative_to(root)
        dataset, embedding, seed = relative.parts[:3]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records.append(
            {
                "dataset": dataset,
                "embedding": embedding,
                "seed": int(seed.removeprefix("seed_")),
                "path": summary_path,
                "summary": summary,
            }
        )
    return sorted(
        records,
        key=lambda item: (
            item["dataset"],
            item["embedding"],
            item["seed"],
        ),
    )


def policy_metric_rows(records: list[dict]) -> list[dict]:
    rows = []
    selected_metrics = (
        "full_evidence",
        "support_recall",
        "support_recall_at_2",
        "support_recall_at_5",
        "support_mrr",
        "support_ndcg",
        "graph_actions",
        "harmful_expansions",
        "evidence_tokens",
        "retrieval_latency_ms",
        "policy_latency_ms",
        "total_latency_ms",
        "total_latency_ms_p95",
    )
    for record in records:
        for policy, metrics in record["summary"]["aggregate"].items():
            row = {
                "dataset": record["dataset"],
                "embedding": record["embedding"],
                "seed": record["seed"],
                "policy": policy,
            }
            row.update(
                {
                    metric: metrics.get(metric)
                    for metric in selected_metrics
                    if metric in metrics
                }
            )
            rows.append(row)
    return rows


def across_seed_rows(metric_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in metric_rows:
        grouped[
            (row["dataset"], row["embedding"], row["policy"])
        ].append(row)
    result = []
    for (dataset, embedding, policy), rows in sorted(grouped.items()):
        output = {
            "dataset": dataset,
            "embedding": embedding,
            "policy": policy,
            "runs": len(rows),
        }
        for metric in (
            "full_evidence",
            "support_recall",
            "graph_actions",
            "harmful_expansions",
            "evidence_tokens",
            "total_latency_ms",
        ):
            values = [
                float(row[metric])
                for row in rows
                if row.get(metric) is not None
            ]
            if values:
                output[f"{metric}_mean"] = mean(values)
                output[f"{metric}_std"] = (
                    stdev(values) if len(values) > 1 else 0.0
                )
        result.append(output)
    return result


def primary_query_rows(root: Path, dataset: str) -> list[dict]:
    path = (
        root
        / dataset
        / PRIMARY_EMBEDDING
        / PRIMARY_SEED
        / "query_results.jsonl"
    )
    return list(read_jsonl(path)) if path.exists() else []


def paired_and_slice_analysis(
    root: Path, datasets: Iterable[str]
) -> tuple[list[dict], list[dict], list[dict]]:
    paired = []
    slices = []
    errors = []
    for dataset in datasets:
        rows = primary_query_rows(root, dataset)
        by_policy = {
            policy: {
                str(row["query_id"]): row
                for row in rows
                if row["policy"] == policy
            }
            for policy in ("hybrid", "mrv_gated")
        }
        common_ids = sorted(
            set(by_policy["hybrid"]) & set(by_policy["mrv_gated"])
        )
        grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
        for query_id in common_ids:
            baseline = by_policy["hybrid"][query_id]
            proposed = by_policy["mrv_gated"][query_id]
            full_delta = (
                proposed["metrics"]["full_evidence"]
                - baseline["metrics"]["full_evidence"]
            )
            recall_delta = (
                proposed["metrics"]["support_recall"]
                - baseline["metrics"]["support_recall"]
            )
            outcome = (
                "win"
                if full_delta > 0 or recall_delta > 0
                else (
                    "loss"
                    if full_delta < 0 or recall_delta < 0
                    else "tie"
                )
            )
            item = {
                "dataset": dataset,
                "query_id": query_id,
                "question_type": baseline.get("question_type", ""),
                "support_count": int(baseline.get("support_count", 0)),
                "slice": baseline["slice"],
                "outcome": outcome,
                "full_evidence_delta": full_delta,
                "support_recall_delta": recall_delta,
                "action_count": proposed["metrics"]["graph_actions"],
                "harmful_expansions": proposed["metrics"][
                    "harmful_expansions"
                ],
            }
            paired.append(item)
            grouped[
                (
                    item["slice"],
                    item["question_type"],
                    item["support_count"],
                )
            ].append(item)
            if outcome != "tie":
                errors.append(
                    {
                        **item,
                        "gold_support": baseline["supporting_ids"],
                        "hybrid_retrieved": baseline["retrieved_ids"],
                        "mrv_gated_retrieved": proposed["retrieved_ids"],
                    }
                )
        for (slice_name, question_type, support_count), items in grouped.items():
            slices.append(
                {
                    "dataset": dataset,
                    "slice": slice_name,
                    "question_type": question_type,
                    "support_count": support_count,
                    "queries": len(items),
                    "win_rate": mean(
                        item["outcome"] == "win" for item in items
                    ),
                    "loss_rate": mean(
                        item["outcome"] == "loss" for item in items
                    ),
                    "mean_support_recall_delta": mean(
                        item["support_recall_delta"] for item in items
                    ),
                }
            )
    errors.sort(
        key=lambda item: (
            item["dataset"],
            item["outcome"] != "loss",
            -abs(item["support_recall_delta"]),
            item["query_id"],
        )
    )
    return paired, slices, errors


def pareto_frontier(rows: list[dict]) -> list[dict]:
    primary = [
        row
        for row in rows
        if row["embedding"] == PRIMARY_EMBEDDING
        and row["seed"] == 101
        and row.get("full_evidence") is not None
        and row.get("total_latency_ms") is not None
    ]
    result = []
    for dataset in sorted({row["dataset"] for row in primary}):
        candidates = [row for row in primary if row["dataset"] == dataset]
        for candidate in candidates:
            dominated = any(
                other["full_evidence"] >= candidate["full_evidence"]
                and other["total_latency_ms"] <= candidate["total_latency_ms"]
                and (
                    other["full_evidence"] > candidate["full_evidence"]
                    or other["total_latency_ms"] < candidate["total_latency_ms"]
                )
                for other in candidates
            )
            result.append(
                {
                    "dataset": dataset,
                    "policy": candidate["policy"],
                    "full_evidence": candidate["full_evidence"],
                    "total_latency_ms": candidate["total_latency_ms"],
                    "pareto_optimal": not dominated,
                }
            )
    return result


def robustness_rows(
    root: Path,
    robustness_root: Path,
    datasets: Iterable[str],
) -> list[dict]:
    rows = []
    for dataset in datasets:
        clean_path = (
            root
            / dataset
            / PRIMARY_EMBEDDING
            / PRIMARY_SEED
            / "summary.json"
        )
        if not clean_path.exists():
            continue
        clean = json.loads(clean_path.read_text(encoding="utf-8"))
        for condition in ROBUSTNESS_CONDITIONS:
            if condition == "clean":
                summary = clean
            else:
                path = (
                    robustness_root
                    / dataset
                    / condition
                    / "summary.json"
                )
                if not path.exists():
                    continue
                summary = json.loads(path.read_text(encoding="utf-8"))
            for policy in ("hybrid", "mrv_gated", "oracle_upper_bound"):
                metrics = summary["aggregate"][policy]
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "policy": policy,
                        "full_evidence": metrics["full_evidence"],
                        "support_recall": metrics["support_recall"],
                        "graph_actions": metrics["graph_actions"],
                        "harmful_expansions": metrics[
                            "harmful_expansions"
                        ],
                        "graph_edges": summary["graph"]["edges"],
                        "dropped_edges": summary["graph"][
                            "dropped_edges"
                        ],
                        "false_edges": summary["graph"]["false_edges"],
                    }
                )
    return rows


def reader_rows(reader_root: Path, datasets: Iterable[str]) -> list[dict]:
    rows = []
    for dataset in datasets:
        path = reader_root / dataset / "seed_101" / "reader_analysis.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        for policy in ("hybrid", "mrv_gated"):
            metrics = result["official"][policy]["metrics"]
            if dataset == "musique":
                answer_f1 = float(metrics["answer_f1"])
                support_f1 = float(metrics["support_f1"])
            else:
                answer_f1 = float(metrics["f1"])
                support_f1 = float(metrics["sp_f1"])
                if dataset == "2wiki":
                    answer_f1 /= 100.0
                    support_f1 /= 100.0
            rows.append(
                {
                    "dataset": dataset,
                    "policy": policy,
                    "answer_f1": answer_f1,
                    "support_f1": support_f1,
                    "queries": result["size"],
                    "evaluator_sha256": result["official"][policy][
                        "evaluator_sha256"
                    ],
                }
            )
    return rows


def _font(size: int, bold: bool = False):
    candidates = [
        (
            Path("C:/Windows/Fonts/arialbd.ttf")
            if bold
            else Path("C:/Windows/Fonts/arial.ttf")
        ),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def bar_figure(
    path: Path,
    rows: list[dict],
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    policies = ("hybrid", "mrv_gated", "oracle_upper_bound")
    lookup = {
        (row["dataset"], row["policy"]): float(row[metric])
        for row in rows
        if row["embedding"] == PRIMARY_EMBEDDING
        and row["seed"] == 101
        and row["policy"] in policies
        and row.get(metric) is not None
    }
    if not lookup:
        return
    width, height = 1500, 880
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(28)
    small = _font(24)
    title_font = _font(36, bold=True)
    left, top, right, bottom = 150, 110, 70, 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    draw.text((left, 35), title, fill="#111827", font=title_font)
    for tick in range(0, 11):
        value = tick / 10
        y = top + plot_h * (1 - value)
        draw.line((left, y, left + plot_w, y), fill="#e5e7eb", width=2)
        draw.text((70, y - 14), f"{value:.1f}", fill="#374151", font=small)
    colors = {
        "hybrid": "#64748b",
        "mrv_gated": "#2563eb",
        "oracle_upper_bound": "#d97706",
    }
    group_w = plot_w / max(1, len(datasets))
    bar_w = group_w * 0.20
    for d_index, dataset in enumerate(datasets):
        center = left + group_w * (d_index + 0.5)
        for p_index, policy in enumerate(policies):
            value = lookup.get((dataset, policy), 0.0)
            x0 = center + (p_index - 1) * bar_w - bar_w * 0.42
            x1 = x0 + bar_w * 0.84
            y0 = top + plot_h * (1 - value)
            draw.rectangle((x0, y0, x1, top + plot_h), fill=colors[policy])
            draw.text(
                (x0, y0 - 34),
                f"{value:.3f}",
                fill="#111827",
                font=small,
            )
        label_box = draw.textbbox((0, 0), dataset, font=font)
        draw.text(
            (center - (label_box[2] - label_box[0]) / 2, top + plot_h + 25),
            dataset,
            fill="#111827",
            font=font,
        )
    draw.text((15, top + plot_h / 2), ylabel, fill="#111827", font=font)
    legend_x = left + 20
    for policy in policies:
        draw.rectangle(
            (legend_x, height - 55, legend_x + 28, height - 27),
            fill=colors[policy],
        )
        legend_x += 38
        draw.text(
            (legend_x, height - 58),
            policy.replace("_", " "),
            fill="#111827",
            font=small,
        )
        legend_x += 230
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(180, 180))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/final_v1")
    parser.add_argument(
        "--output", default="outputs/final_v1/analysis"
    )
    parser.add_argument(
        "--robustness-root", default="outputs/final_v1_robustness"
    )
    parser.add_argument(
        "--reader-root", default="outputs/final_v1_reader"
    )
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = discover_runs(root)
    metrics = policy_metric_rows(records)
    across = across_seed_rows(metrics)
    datasets = sorted({record["dataset"] for record in records})
    paired, slices, errors = paired_and_slice_analysis(root, datasets)
    pareto = pareto_frontier(metrics)
    robustness = robustness_rows(
        root, Path(args.robustness_root), datasets
    )
    reader = reader_rows(Path(args.reader_root), datasets)
    write_csv(output / "policy_metrics.csv", metrics)
    write_csv(output / "across_seed_metrics.csv", across)
    write_csv(output / "paired_outcomes.csv", paired)
    write_csv(output / "slice_metrics.csv", slices)
    write_csv(output / "pareto_frontier.csv", pareto)
    write_csv(output / "robustness_metrics.csv", robustness)
    write_csv(output / "reader_official_metrics.csv", reader)
    (output / "error_examples.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "runs": len(records),
        "datasets": datasets,
        "paired_queries": len(paired),
        "wins": sum(item["outcome"] == "win" for item in paired),
        "losses": sum(item["outcome"] == "loss" for item in paired),
        "ties": sum(item["outcome"] == "tie" for item in paired),
        "robustness_rows": len(robustness),
        "reader_rows": len(reader),
        "primary_embedding": PRIMARY_EMBEDDING,
        "primary_seed": 101,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    bar_figure(
        output / "full_evidence_by_dataset.png",
        metrics,
        metric="full_evidence",
        title="Full supporting-evidence retrieval",
        ylabel="Rate",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
