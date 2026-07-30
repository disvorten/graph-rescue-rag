"""Create compact, path-free tables for the full Qwen3-8B reader runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from graph_rescue.metrics import (
    holm_bonferroni,
    paired_bootstrap_difference,
)
from graph_rescue.official_metrics import (
    best_answer_scores,
    joint_scores,
    support_fact_scores,
)


DATASETS = ("hotpot", "2wiki", "musique")
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 101


def normalized_official_metrics(
    dataset: str,
    metrics: dict,
) -> dict[str, float | None]:
    if dataset == "musique":
        return {
            "answer_em": float(metrics.get("answer_em", 0.0)),
            "answer_f1": float(metrics["answer_f1"]),
            "support_em": None,
            "support_f1": float(metrics["support_f1"]),
            "joint_f1": None,
        }
    scale = 0.01 if dataset == "2wiki" else 1.0
    return {
        "answer_em": float(metrics["em"]) * scale,
        "answer_f1": float(metrics["f1"]) * scale,
        "support_em": float(metrics["sp_em"]) * scale,
        "support_f1": float(metrics["sp_f1"]) * scale,
        "joint_f1": (
            float(metrics["joint_f1"]) * scale
            if "joint_f1" in metrics
            else None
        ),
    }


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def index_support_scores(
    predicted: list[int],
    gold: list[int],
) -> dict[str, float]:
    predicted_set = {int(value) for value in predicted}
    gold_set = {int(value) for value in gold}
    true_positive = len(predicted_set & gold_set)
    false_positive = len(predicted_set - gold_set)
    false_negative = len(gold_set - predicted_set)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    exact = float(false_positive + false_negative == 0)
    if not predicted_set and not gold_set:
        f1 = 1.0
    return {
        "em": exact,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def aliases_by_id(path: str | Path) -> dict[str, list[str]]:
    result = {}
    for row in read_jsonl(path):
        result[str(row["Q_id"])] = [
            str(value)
            for value in row.get("aliases", []) + row.get("demonyms", [])
        ]
    return result


def official_query_scores(
    dataset: str,
    record: dict,
) -> dict[str, list[float]]:
    prediction_path = Path(record["prediction_path"])
    gold_path = Path(record["gold_path"])
    scores: dict[str, list[float]] = {
        "answer_em": [],
        "answer_f1": [],
        "support_f1": [],
    }
    if dataset in {"hotpot", "2wiki"}:
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        gold_rows = json.loads(gold_path.read_text(encoding="utf-8"))
        alias_index = (
            aliases_by_id(record["alias_path"]) if dataset == "2wiki" else {}
        )
        if dataset == "hotpot":
            scores["support_em"] = []
            scores["joint_f1"] = []
        else:
            scores["support_em"] = []
        for gold in gold_rows:
            query_id = str(gold["_id"])
            answers = [str(gold["answer"])]
            if dataset == "2wiki":
                answers.extend(alias_index.get(str(gold["answer_id"]), []))
            answer = best_answer_scores(
                str(predictions["answer"][query_id]),
                answers,
            )
            support = support_fact_scores(
                predictions["sp"][query_id],
                gold["supporting_facts"],
            )
            scores["answer_em"].append(answer["em"])
            scores["answer_f1"].append(answer["f1"])
            scores["support_em"].append(support["em"])
            scores["support_f1"].append(support["f1"])
            if dataset == "hotpot":
                scores["joint_f1"].append(
                    joint_scores(answer, support)["f1"]
                )
        return scores

    predictions = {
        str(row["id"]): row for row in read_jsonl(prediction_path)
    }
    for gold in read_jsonl(gold_path):
        if not bool(gold["answerable"]):
            continue
        query_id = str(gold["id"])
        prediction = predictions[query_id]
        answer = best_answer_scores(
            str(prediction["predicted_answer"]),
            [str(gold["answer"])]
            + [str(value) for value in gold.get("answer_aliases", [])],
        )
        gold_support = [
            int(paragraph["idx"])
            for paragraph in gold["paragraphs"]
            if paragraph["is_supporting"]
        ]
        support = index_support_scores(
            prediction["predicted_support_idxs"],
            gold_support,
        )
        scores["answer_em"].append(answer["em"])
        scores["answer_f1"].append(answer["f1"])
        scores["support_f1"].append(support["f1"])
    return scores


def official_paired_comparisons(
    dataset: str,
    analysis: dict,
) -> dict[str, dict[str, float]]:
    base = official_query_scores(dataset, analysis["official"]["hybrid"])
    gated = official_query_scores(dataset, analysis["official"]["mrv_gated"])
    comparisons = {}
    for metric in sorted(set(base) & set(gated)):
        comparisons[metric] = paired_bootstrap_difference(
            gated[metric],
            base[metric],
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
    adjusted = holm_bonferroni(
        {
            metric: value["p_value_two_sided"]
            for metric, value in comparisons.items()
        }
    )
    for metric, value in comparisons.items():
        value["p_value_holm"] = adjusted[metric]
    return comparisons


def analyze_dataset(
    dataset: str,
    reader_root: Path,
    protocol_root: Path,
) -> dict | None:
    run = reader_root / dataset / "seed_101"
    analysis_path = run / "reader_analysis.json"
    summary_path = run / "summary.json"
    manifest_path = (
        protocol_root / dataset / "reader_eval_1000_manifest.json"
    )
    if not all(
        path.exists()
        for path in (analysis_path, summary_path, manifest_path)
    ):
        return None
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policies = {
        policy: normalized_official_metrics(
            dataset, analysis["official"][policy]["metrics"]
        )
        for policy in ("hybrid", "mrv_gated")
    }
    official_comparisons = official_paired_comparisons(dataset, analysis)
    stats = summary.get("reader_stats", {}).get(
        "ollama:qwen3:8b", {}
    )
    result = {
        "dataset": dataset,
        "queries": int(analysis["size"]),
        "query_id_sha256": manifest["query_id_sha256"],
        "hybrid": policies["hybrid"],
        "mrv_gated": policies["mrv_gated"],
        "reader_stats": {
            "generation_calls": int(stats.get("generation_calls", 0)),
            "cache_hits": int(stats.get("cache_hits", 0)),
        },
        "inference": {
            "source": "official_per_query_recomputation",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "holm_family": sorted(official_comparisons),
        },
        "deltas": {},
    }
    for name, source in official_comparisons.items():
        base = policies["hybrid"].get(name)
        gated = policies["mrv_gated"].get(name)
        if base is None or gated is None:
            continue
        result["deltas"][name] = {
            "difference": float(source["difference"]),
            "ci95_low": float(source["ci95_low"]),
            "ci95_high": float(source["ci95_high"]),
            "p_value_two_sided": float(source["p_value_two_sided"]),
            "p_value_holm": float(source["p_value_holm"]),
        }
    return result


def flat_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        for metric, delta in result["deltas"].items():
            rows.append(
                {
                    "dataset": result["dataset"],
                    "queries": result["queries"],
                    "metric": metric,
                    "hybrid": result["hybrid"][metric],
                    "mrv_gated": result["mrv_gated"][metric],
                    **delta,
                    "generation_calls": result["reader_stats"][
                        "generation_calls"
                    ],
                    "cache_hits": result["reader_stats"]["cache_hits"],
                    "query_id_sha256": result["query_id_sha256"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reader-root", default="outputs/final_v1_reader_full"
    )
    parser.add_argument(
        "--protocol-root", default="work/final_protocol"
    )
    parser.add_argument(
        "--output", default="outputs/final_v1/analysis"
    )
    args = parser.parse_args()
    results = [
        result
        for dataset in DATASETS
        if (
            result := analyze_dataset(
                dataset,
                Path(args.reader_root),
                Path(args.protocol_root),
            )
        )
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = flat_rows(results)
    csv_path = output / "reader_full_metrics.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    complete = {result["dataset"] for result in results}
    payload = {
        "schema_version": 1,
        "complete": len(results) == len(DATASETS),
        "datasets_complete": [result["dataset"] for result in results],
        "datasets_missing": [
            dataset for dataset in DATASETS if dataset not in complete
        ],
        "results": results,
    }
    (output / "reader_full_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
