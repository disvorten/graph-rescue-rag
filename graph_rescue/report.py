from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


MAIN_METRICS = (
    "full_evidence",
    "support_recall",
    "graph_query_rescued",
    "rescue_recall",
    "graph_query_harmed",
    "graph_actions",
    "graph_reads",
    "latency_ms",
    "retrieval_latency_ms",
    "policy_latency_ms",
    "reader_latency_ms",
    "total_latency_ms",
    "answer_em",
    "answer_f1",
    "support_recall_at_5",
    "support_ndcg",
)


def write_experiment_report(summary: dict[str, Any], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    aggregate = summary["aggregate"]
    policies = list(aggregate)

    with (target / "aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy", *MAIN_METRICS])
        writer.writeheader()
        for policy in policies:
            writer.writerow(
                {
                    "policy": policy,
                    **{
                        metric: aggregate[policy].get(metric, "")
                        for metric in MAIN_METRICS
                    },
                }
            )

    lines = [
        "# Graph Rescue experiment report",
        "",
        "## Dataset and graph",
        "",
        f"- Passages: {summary['graph']['passages']}",
        f"- Entities: {summary['graph']['entities']}",
        f"- Edges: {summary['graph']['edges']}",
        "",
        "### Diagnostic slices",
        "",
        "| Slice | Queries |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {count} |"
        for name, count in summary["slice_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Equal-budget results",
            "",
            "| Policy | Full evidence | Support recall | Graph rescue | "
            "Graph actions | Graph reads | Latency ms | Answer EM |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy in policies:
        row = aggregate[policy]
        lines.append(
            f"| {policy} | {row['full_evidence']:.3f} | "
            f"{row['support_recall']:.3f} | "
            f"{row['graph_query_rescued']:.3f} | "
            f"{row['graph_actions']:.3f} | {row['graph_reads']:.1f} | "
            f"{row['latency_ms']:.3f} | {row['answer_em']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Factorial interaction",
            "",
            "| Metric | Interaction | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for metric, value in summary["factorial_interactions"].items():
        lines.append(
            f"| {metric} | {value['interaction']:.4f} | "
            f"[{value['ci95_low']:.4f}, {value['ci95_high']:.4f}] |"
        )
    gate = summary["gate_metrics"]
    lines.extend(
        [
            "",
            "## Gate calibration",
            "",
            f"- AUROC: {gate['auroc']:.4f}",
            f"- AUPRC: {gate['auprc']:.4f}",
            f"- Brier score: {gate['brier']:.4f}",
            f"- ECE: {gate['ece']:.4f}",
            f"- Decision threshold: {gate['threshold']:.4f}",
            f"- Precision / recall: {gate['precision']:.4f} / {gate['recall']:.4f}",
            f"- Selected calibrator: "
            f"{summary.get('gate_model', {}).get('calibration_method', 'unknown')}",
            "",
            "> These are pilot results from a real local evaluation. Treat them as "
            "preliminary evidence until the experiment is replicated on official "
            "train/dev/test splits, additional datasets, and a reader-based QA metric.",
            "",
        ]
    )
    (target / "report.md").write_text("\n".join(lines), encoding="utf-8")
