"""Generate JIIS tables and a compact public extended-validation report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT / "outputs" / "global_v1" / "analysis" / "analysis_summary.json"
GATE = ROOT / "outputs" / "global_v1" / "gate_transfer" / "analysis_summary.json"
OFFICIAL = (
    ROOT
    / "outputs"
    / "official_baselines"
    / "aligned_analysis"
    / "analysis_summary.json"
)
LATENCY_ROOT = ROOT / "outputs" / "latency_v1"
TEX_OUTPUT = ROOT / "outputs" / "jiis_submission" / "extended_results.tex"
PUBLIC_OUTPUT = ROOT / "outputs" / "extended_validation_v1"

DATASETS = ("hotpot", "2wiki", "musique")
DISPLAY = {"hotpot": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue"}


def read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def portable_paths(value):
    """Replace workspace-local absolute paths in public summaries."""
    if isinstance(value, dict):
        return {key: portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_paths(item) for item in value]
    if isinstance(value, str):
        root_text = str(ROOT)
        if value.casefold().startswith(root_text.casefold()):
            return Path(value).relative_to(ROOT).as_posix()
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def global_table(result: dict) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Frozen-model transfer to previously unseen official-development queries in the global development/distractor setting. The corpus is not full-Wikipedia. CI denotes a paired 95\% bootstrap interval.}",
        r"\label{tab:global-transfer}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Queries & Passages & Hybrid FE & Gated FE & $\Delta$ FE & 95\% CI \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        item = result["datasets"][dataset]
        comp = item["comparisons"]["mrv_gated_vs_hybrid_full_evidence"]
        lines.append(
            f"{DISPLAY[dataset]} & {item['queries']:,} & {item['corpus_passages']:,} & "
            f"{item['policies']['hybrid']['full_evidence']:.3f} & "
            f"{item['policies']['mrv_gated']['full_evidence']:.3f} & "
            f"{comp['difference']:+.3f} & "
            f"[{comp['ci95_low']:.3f}, {comp['ci95_high']:.3f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def gate_table(result: dict) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Label-efficient target recalibration on a deterministic 200-query calibration subset. Metrics are reported on the disjoint remainder. Retrieval columns replay a preflight-only switch between hybrid and MRV-always traces; they are not the primary two-stage-gate result.}",
        r"\label{tab:gate-transfer}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Test $n$ & Recall F$\to$R & ECE F$\to$R & Open F$\to$R & Actions F$\to$R & FE F$\to$R \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        item = result["datasets"][dataset]
        frozen = item["frozen_gate_on_heldout"]
        recal = item["recalibrated_gate_on_heldout"]
        frozen_policy = item["preflight_only_policy_frozen"]
        recal_policy = item["preflight_only_policy_recalibrated"]
        lines.append(
            f"{DISPLAY[dataset]} & {item['heldout_test_queries']:,} & "
            f"{frozen['recall']:.3f}$\\to${recal['recall']:.3f} & "
            f"{frozen['ece']:.3f}$\\to${recal['ece']:.3f} & "
            f"{frozen_policy['open_rate']:.3f}$\\to${recal_policy['open_rate']:.3f} & "
            f"{frozen_policy['graph_actions']:.2f}$\\to${recal_policy['graph_actions']:.2f} & "
            f"{frozen_policy['full_evidence']:.3f}$\\to${recal_policy['full_evidence']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return "\n".join(lines)


def official_table(result: dict) -> str:
    display = {
        "StandardRAG_official_code": "StandardRAG official",
        "HippoRAG_official_code": "HippoRAG official",
        "GraphRescue_hybrid": "Graph Rescue hybrid",
        "GraphRescue_gated_MRV": "Graph Rescue gated MRV",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Aligned retrieval on the released HippoRAG MuSiQue corpus (same query IDs, $k=7$). The official systems use local Qwen models, non-thinking Qwen3 recognition memory, and the released OpenIE artifact; these are not published-paper-number reproductions.}",
        r"\label{tab:official-baseline}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"System & FE@7 & SR@7 & p50 ms & p95 ms \\",
        r"\midrule",
    ]
    for name, item in result["systems"].items():
        lines.append(
            f"{display.get(name, name)} & {item['full_evidence_at_7']:.3f} & "
            f"{item['support_recall_at_7']:.3f} & "
            f"{item['retrieval_latency']['median_ms']:.1f} & "
            f"{item['retrieval_latency']['p95_ms']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    comparisons = result["paired_full_evidence_comparisons"]
    comparison_names = (
        (
            "GraphRescue_gated_minus_HippoRAG_full_evidence_at_7",
            "Gated MRV $-$ HippoRAG",
        ),
        (
            "GraphRescue_gated_minus_StandardRAG_full_evidence_at_7",
            "Gated MRV $-$ StandardRAG",
        ),
        (
            "GraphRescue_gated_minus_GraphRescue_hybrid_full_evidence_at_7",
            "Gated MRV $-$ shared hybrid",
        ),
    )
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Paired full-evidence differences for the aligned released-MuSiQue comparison.}",
            r"\label{tab:official-baseline-paired}",
            r"\small",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Comparison & $\Delta$ FE@7 & 95\% CI \\",
            r"\midrule",
        ]
    )
    for key, label in comparison_names:
        item = comparisons[key]
        lines.append(
            f"{label} & {item['difference']:+.3f} & "
            f"[{item['ci95_low']:.3f}, {item['ci95_high']:.3f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def latency_table(results: dict[str, dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Sequential online retrieval benchmark: 200 queries, three repetitions, fresh query embeddings, excluded warm-up, and randomized query/policy order. Answer generation is excluded.}",
        r"\label{tab:clean-latency}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Dataset & Hybrid p50 & Always p50 & Gated p50 & Gated p95 & Mean $\Delta$ G$-$A & 95\% CI & Actions A$\to$G \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        result = results[dataset]
        base = result["aggregate"]["hybrid"]
        always = result["aggregate"]["mrv_always"]
        gated = result["aggregate"]["mrv_gated"]
        comparison = result["paired_latency_comparisons"][
            "mrv_gated_minus_mrv_always_online_total_ms"
        ]
        lines.append(
            f"{DISPLAY[dataset]} & {base['online_total_latency']['median_ms']:.1f} & "
            f"{always['online_total_latency']['median_ms']:.1f} & "
            f"{gated['online_total_latency']['median_ms']:.1f} & "
            f"{gated['online_total_latency']['p95_ms']:.1f} & "
            f"{comparison['difference']:+.1f} & "
            f"[{comparison['ci95_low']:.1f}, {comparison['ci95_high']:.1f}] & "
            f"{always['mean_graph_actions']:.2f}$\\to${gated['mean_graph_actions']:.2f} "
            f"\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return "\n".join(lines)


def markdown(
    global_result: dict,
    gate_result: dict,
    official_result: dict,
    latency_results: dict[str, dict],
) -> str:
    lines = [
        "# Extended validation results",
        "",
        "All values below are generated from completed machine-readable summaries.",
        "",
        "## Frozen global-development transfer (not full-wiki)",
        "",
        "| Dataset | Queries | Passages | Hybrid FE | Gated FE | Delta FE | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        item = global_result["datasets"][dataset]
        comp = item["comparisons"]["mrv_gated_vs_hybrid_full_evidence"]
        lines.append(
            f"| {DISPLAY[dataset]} | {item['queries']:,} | {item['corpus_passages']:,} | "
            f"{item['policies']['hybrid']['full_evidence']:.3f} | "
            f"{item['policies']['mrv_gated']['full_evidence']:.3f} | "
            f"{comp['difference']:+.3f} | "
            f"[{comp['ci95_low']:.3f}, {comp['ci95_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Target gate calibration",
            "",
            "| Dataset | Cal/Test | Recall frozen→recal. | ECE frozen→recal. | Open frozen→recal. | FE frozen→recal. |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        item = gate_result["datasets"][dataset]
        frozen = item["frozen_gate_on_heldout"]
        recal = item["recalibrated_gate_on_heldout"]
        fp = item["preflight_only_policy_frozen"]
        rp = item["preflight_only_policy_recalibrated"]
        lines.append(
            f"| {DISPLAY[dataset]} | {item['calibration_queries']}/{item['heldout_test_queries']} | "
            f"{frozen['recall']:.3f}→{recal['recall']:.3f} | "
            f"{frozen['ece']:.3f}→{recal['ece']:.3f} | "
            f"{fp['open_rate']:.3f}→{rp['open_rate']:.3f} | "
            f"{fp['full_evidence']:.3f}→{rp['full_evidence']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Official-code released-MuSiQue comparison",
            "",
            "| System | FE@7 | SR@7 | p50 ms | p95 ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in official_result["systems"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {item['full_evidence_at_7']:.3f} | "
            f"{item['support_recall_at_7']:.3f} | "
            f"{item['retrieval_latency']['median_ms']:.1f} | "
            f"{item['retrieval_latency']['p95_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "| Paired FE@7 comparison | Delta | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for key, label in (
        (
            "GraphRescue_gated_minus_HippoRAG_full_evidence_at_7",
            "Gated MRV minus HippoRAG",
        ),
        (
            "GraphRescue_gated_minus_StandardRAG_full_evidence_at_7",
            "Gated MRV minus StandardRAG",
        ),
        (
            "GraphRescue_gated_minus_GraphRescue_hybrid_full_evidence_at_7",
            "Gated MRV minus shared hybrid",
        ),
    ):
        item = official_result["paired_full_evidence_comparisons"][key]
        lines.append(
            f"| {label} | {item['difference']:+.3f} | "
            f"[{item['ci95_low']:.3f}, {item['ci95_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Clean online retrieval latency",
            "",
            "| Dataset | Hybrid p50 | Always p50 | Gated p50 | Gated p95 | Mean delta gated-always | 95% CI | Actions always-to-gated |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        result = latency_results[dataset]
        base = result["aggregate"]["hybrid"]
        always = result["aggregate"]["mrv_always"]
        gated = result["aggregate"]["mrv_gated"]
        comparison = result["paired_latency_comparisons"][
            "mrv_gated_minus_mrv_always_online_total_ms"
        ]
        lines.append(
            f"| {DISPLAY[dataset]} | {base['online_total_latency']['median_ms']:.1f} | "
            f"{always['online_total_latency']['median_ms']:.1f} | "
            f"{gated['online_total_latency']['median_ms']:.1f} | "
            f"{gated['online_total_latency']['p95_ms']:.1f} | "
            f"{comparison['difference']:+.1f} | "
            f"[{comparison['ci95_low']:.1f}, {comparison['ci95_high']:.1f}] | "
            f"{always['mean_graph_actions']:.2f}-to-{gated['mean_graph_actions']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Latency is online retrieval only and does not establish that this method "
            "is faster than a full external GraphRAG lifecycle.",
            "",
        ]
    )
    report = "\n".join(lines)
    # Normalize a legacy mojibake arrow that may survive when this source is
    # opened by a non-UTF-8 Windows console. Public artifacts stay ASCII-safe.
    return report.replace("\u2192", "->").replace("\u0432\u2020\u2019", "->")


def main() -> None:
    global_result = read(GLOBAL)
    gate_result = read(GATE)
    official_result = read(OFFICIAL)
    latency_results = {dataset: read(LATENCY_ROOT / f"{dataset}.json") for dataset in DATASETS}
    tex = (
        "% Auto-generated by work/generate_jiis_extended_assets.py.\n"
        + global_table(global_result)
        + gate_table(gate_result)
        + official_table(official_result)
        + latency_table(latency_results)
    )
    TEX_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUTPUT.write_text(tex, encoding="utf-8")

    sources = [GLOBAL, GATE, OFFICIAL, *(LATENCY_ROOT / f"{name}.json" for name in DATASETS)]
    combined = portable_paths({
        "schema_version": 1,
        "scope": "compact journal extended-validation results",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "global_transfer": global_result,
        "gate_transfer": gate_result,
        "official_baseline": official_result,
        "latency": latency_results,
    })
    PUBLIC_OUTPUT.mkdir(parents=True, exist_ok=True)
    (PUBLIC_OUTPUT / "summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PUBLIC_OUTPUT / "REPORT.md").write_text(
        markdown(global_result, gate_result, official_result, latency_results),
        encoding="utf-8",
    )
    print(TEX_OUTPUT)
    print(PUBLIC_OUTPUT / "REPORT.md")


if __name__ == "__main__":
    main()
