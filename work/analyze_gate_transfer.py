"""Evaluate label-efficient target-domain calibration on global-dev runs.

The primary global result keeps the training-domain gate fully frozen. This
secondary analysis uses a deterministic, disjoint target calibration subset
and reports performance only on the remaining queries. Because continuation
decisions cannot be retuned without replaying intermediate states, retrieval
effects are evaluated for a clearly named preflight-only policy: a query
either keeps hybrid evidence or uses the already computed MRV-always trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from graph_rescue.learning import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    bootstrap_threshold_for_recall,
    select_calibration_method,
)
from graph_rescue.graph import KnowledgeGraph
from graph_rescue.io import load_passages
from graph_rescue.metrics import binary_metrics, calibration_bins


DATASETS = ("hotpot", "2wiki", "musique")


def stable_order(query_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}|{query_id}".encode("utf-8")).digest()


def load_examples(run: Path) -> list[dict[str, Any]]:
    checkpoint = run / "evaluation_checkpoint.jsonl"
    run_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    config = run_summary["config"]
    graph_config = config["graph"]
    passages = load_passages(config["corpus_path"])
    graph = KnowledgeGraph.build(
        passages,
        min_entity_df=int(graph_config["min_entity_df"]),
        max_entity_df_ratio=float(graph_config["max_entity_df_ratio"]),
        entity_mode=str(graph_config["entity_mode"]),
    )
    if (
        float(graph_config.get("edge_dropout_rate", 0.0)) > 0.0
        or float(graph_config.get("false_edge_ratio", 0.0)) > 0.0
    ):
        graph.corrupt(
            edge_dropout_rate=float(graph_config.get("edge_dropout_rate", 0.0)),
            false_edge_ratio=float(graph_config.get("false_edge_ratio", 0.0)),
            seed=int(graph_config.get("corruption_seed", 42)),
        )
    examples = []
    with checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            bundle = json.loads(line)
            rows = {row["policy"]: row for row in bundle["rows"]}
            traces = {trace["policy"]: trace for trace in bundle["traces"]}
            gated_trace = traces["mrv_gated"]
            preflight = next(
                (
                    action
                    for action in gated_trace["actions"]
                    if action.get("gate_stage") == "preflight"
                    and action.get("gate_probability") is not None
                ),
                None,
            )
            if preflight is None:
                continue
            gated_row = rows["mrv_gated"]
            evidence_ids = list(
                preflight.get("evidence_ids_before")
                or gated_trace["seed_passage_ids"]
            )
            missing_support = set(gated_row["supporting_ids"]) - set(evidence_ids)
            candidates, _ = graph.candidate_paths(
                evidence_ids,
                excluded_passage_ids=evidence_ids,
                max_hops=int(graph_config["max_hops"]),
                cap=int(graph_config["frontier_cap"]),
            )
            reachable = {item.target_passage_id for item in candidates}
            examples.append(
                {
                    "query_id": str(bundle["query_id"]),
                    "probability": float(preflight["gate_probability"]),
                    "label": int(bool(missing_support & reachable)),
                    "hybrid": rows["hybrid"]["metrics"],
                    "mrv_always": rows["mrv_always"]["metrics"],
                }
            )
    return examples


def fit_recalibrator(
    probabilities: list[float], labels: list[int], *, seed: int
) -> tuple[str, Any]:
    method, _ = select_calibration_method(
        probabilities, labels, folds=5, seed=seed
    )
    calibrators = {
        "identity": IdentityCalibrator,
        "platt": PlattCalibrator,
        "isotonic": IsotonicCalibrator,
    }
    calibrator = calibrators[method]().fit(probabilities, labels)
    return method, calibrator


def policy_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    selected = []
    for row in rows:
        opened = row["probability"] >= threshold
        metrics = row["mrv_always"] if opened else row["hybrid"]
        selected.append((opened, metrics))
    return {
        "open_rate": statistics.fmean(float(item[0]) for item in selected),
        "full_evidence": statistics.fmean(
            float(item[1]["full_evidence"]) for item in selected
        ),
        "support_recall": statistics.fmean(
            float(item[1]["support_recall"]) for item in selected
        ),
        "graph_actions": statistics.fmean(
            float(item[1]["graph_actions"]) for item in selected
        ),
    }


def analyze_dataset(
    run: Path,
    *,
    calibration_size: int,
    seed: int,
    target_recall: float,
) -> dict[str, Any]:
    rows = load_examples(run)
    ordered = sorted(rows, key=lambda row: stable_order(row["query_id"], seed))
    if len(ordered) <= calibration_size:
        raise ValueError(
            f"{run}: {len(ordered)} gate examples, calibration_size={calibration_size}"
        )
    calibration = ordered[:calibration_size]
    test = ordered[calibration_size:]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    frozen_threshold = float(summary["gate_model"]["threshold"])
    reconstructed = binary_metrics(
        [row["probability"] for row in rows],
        [row["label"] for row in rows],
        threshold=frozen_threshold,
    )
    expected = summary["gate_metrics_by_stage"]["preflight"]
    validation_keys = (
        "auroc",
        "auprc",
        "brier",
        "ece",
        "precision",
        "recall",
        "specificity",
        "predicted_positive_rate",
    )
    disagreement = {
        key: abs(float(reconstructed[key]) - float(expected[key]))
        for key in validation_keys
    }
    if max(disagreement.values(), default=0.0) > 1e-9:
        raise ValueError(
            f"Reconstructed gate target does not match evaluator for {run}: "
            f"{disagreement}"
        )

    calibration_probabilities = [row["probability"] for row in calibration]
    calibration_labels = [row["label"] for row in calibration]
    method, calibrator = fit_recalibrator(
        calibration_probabilities, calibration_labels, seed=seed
    )
    recalibrated_calibration = calibrator.transform(calibration_probabilities)
    threshold = bootstrap_threshold_for_recall(
        recalibrated_calibration,
        calibration_labels,
        target_recall,
        samples=1000,
        quantile=0.10,
        seed=seed,
    )

    test_probabilities = [row["probability"] for row in test]
    test_labels = [row["label"] for row in test]
    recalibrated_test = list(calibrator.transform(test_probabilities))
    recalibrated_rows = [
        {**row, "probability": probability}
        for row, probability in zip(test, recalibrated_test)
    ]
    return {
        "available_preflight_examples": len(rows),
        "calibration_queries": len(calibration),
        "heldout_test_queries": len(test),
        "target_recall": target_recall,
        "selected_recalibration_method": method,
        "gate_target_reconstruction_matches_primary_evaluator": True,
        "frozen_threshold": frozen_threshold,
        "recalibrated_threshold": float(threshold),
        "frozen_gate_on_heldout": binary_metrics(
            test_probabilities, test_labels, threshold=frozen_threshold
        ),
        "recalibrated_gate_on_heldout": binary_metrics(
            recalibrated_test, test_labels, threshold=threshold
        ),
        "recalibrated_bins_on_heldout": calibration_bins(
            recalibrated_test, test_labels
        ),
        "preflight_only_policy_frozen": policy_summary(test, frozen_threshold),
        "preflight_only_policy_recalibrated": policy_summary(
            recalibrated_rows, float(threshold)
        ),
        "scope_note": (
            "Retrieval rows implement a preflight-only ablation by selecting "
            "between hybrid and the precomputed MRV-always trace. They do not "
            "replace the primary frozen two-stage-gate result."
        ),
    }


def report(result: dict[str, Any]) -> str:
    lines = [
        "# Target-domain gate recalibration",
        "",
        "A deterministic target calibration subset is disjoint from the reported "
        "held-out queries. This is a label-efficient adaptation diagnostic, not "
        "the primary frozen-transfer result.",
        "",
        "| Dataset | Cal/Test | Frozen recall | Recal. recall | Frozen ECE | Recal. ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        if dataset not in result["datasets"]:
            continue
        item = result["datasets"][dataset]
        lines.append(
            "| {dataset} | {cal}/{test} | {fr:.3f} | {rr:.3f} | {fe:.3f} | "
            "{re:.3f} |".format(
                dataset=dataset,
                cal=item["calibration_queries"],
                test=item["heldout_test_queries"],
                fr=item["frozen_gate_on_heldout"]["recall"],
                rr=item["recalibrated_gate_on_heldout"]["recall"],
                fe=item["frozen_gate_on_heldout"]["ece"],
                re=item["recalibrated_gate_on_heldout"]["ece"],
            )
        )
    lines.extend(
        [
            "",
            "| Dataset | Frozen open | Recal. open | Frozen actions | Recal. actions | Frozen FE | Recal. FE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        if dataset not in result["datasets"]:
            continue
        item = result["datasets"][dataset]
        frozen = item["preflight_only_policy_frozen"]
        recalibrated = item["preflight_only_policy_recalibrated"]
        lines.append(
            "| {dataset} | {fo:.3f} | {ro:.3f} | {fa:.3f} | {ra:.3f} | "
            "{ff:.3f} | {rf:.3f} |".format(
                dataset=dataset,
                fo=frozen["open_rate"],
                ro=recalibrated["open_rate"],
                fa=frozen["graph_actions"],
                ra=recalibrated["graph_actions"],
                ff=frozen["full_evidence"],
                rf=recalibrated["full_evidence"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/global_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/global_v1/gate_transfer")
    )
    parser.add_argument("--calibration-queries", type=int, default=200)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "protocol": "target-domain-gate-recalibration-v1",
        "calibration_queries_per_dataset": args.calibration_queries,
        "selection_seed": args.seed,
        "datasets": {},
    }
    for dataset in DATASETS:
        run = args.root / dataset / "frozen_seed_101"
        if not (run / "summary.json").exists():
            if args.allow_partial:
                continue
            raise FileNotFoundError(run / "summary.json")
        result["datasets"][dataset] = analyze_dataset(
            run,
            calibration_size=args.calibration_queries,
            seed=args.seed,
            target_recall=args.target_recall,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "REPORT.md").write_text(report(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
