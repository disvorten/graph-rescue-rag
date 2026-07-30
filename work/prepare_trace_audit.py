"""Create a deterministic local packet for human retrieval-trace review."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DATASETS = ("hotpot", "2wiki", "musique")
PRIMARY_RUN = Path("outputs/final_v1")
EMBEDDING = "qwen3-embedding_0.6b"
SEED = "seed_101"
QUOTAS = {
    "hotpot": {"win": 14, "loss": 10, "tie": 10},
    "2wiki": {"win": 14, "loss": 9, "tie": 10},
    "musique": {"win": 14, "loss": 9, "tie": 10},
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_policy_rows(path: Path) -> dict[str, dict[str, dict]]:
    rows: dict[str, dict[str, dict]] = {}
    for row in read_jsonl(path):
        if row["policy"] not in {"hybrid", "mrv_gated"}:
            continue
        rows.setdefault(str(row["query_id"]), {})[str(row["policy"])] = row
    return rows


def classify(policies: dict[str, dict]) -> str:
    base = float(policies["hybrid"]["metrics"]["full_evidence"])
    gated = float(policies["mrv_gated"]["metrics"]["full_evidence"])
    if gated > base:
        return "win"
    if gated < base:
        return "loss"
    return "tie"


def select_ids(
    dataset: str,
    rows: dict[str, dict[str, dict]],
    *,
    random_seed: int,
) -> list[tuple[str, str]]:
    grouped = {"win": [], "loss": [], "tie": []}
    for query_id, policies in rows.items():
        if set(policies) != {"hybrid", "mrv_gated"}:
            continue
        grouped[classify(policies)].append(query_id)
    rng = random.Random(f"{random_seed}:{dataset}")
    selected = []
    for outcome, quota in QUOTAS[dataset].items():
        candidates = sorted(grouped[outcome])
        if len(candidates) < quota:
            raise ValueError(
                f"{dataset}/{outcome}: need {quota}, found {len(candidates)}"
            )
        selected.extend((query_id, outcome) for query_id in rng.sample(candidates, quota))
    rng.shuffle(selected)
    return selected


def passage_bundle(
    ids: set[str],
    corpus: dict[str, dict],
    supporting_ids: set[str],
) -> list[dict]:
    result = []
    for passage_id in sorted(ids):
        passage = corpus.get(passage_id)
        if not passage:
            result.append(
                {
                    "id": passage_id,
                    "missing_from_corpus": True,
                    "is_gold_support": passage_id in supporting_ids,
                }
            )
            continue
        result.append(
            {
                "id": passage_id,
                "title": passage.get("title", ""),
                "text": passage.get("text", ""),
                "is_gold_support": passage_id in supporting_ids,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seed", type=int, default=20260730)
    parser.add_argument("--output-dir", default="work/trace_audit")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    worksheet_rows = []
    packet_rows = []

    for dataset in DATASETS:
        run = PRIMARY_RUN / dataset / EMBEDDING / SEED
        policies = load_policy_rows(run / "query_results.jsonl")
        traces = load_policy_rows(run / "retrieval_traces.jsonl")
        corpus = {
            str(row["id"]): row
            for row in read_jsonl(
                Path("work/final_protocol") / dataset / "corpus.jsonl"
            )
        }
        for query_id, outcome in select_ids(
            dataset, policies, random_seed=args.random_seed
        ):
            policy_rows = policies[query_id]
            trace_rows = traces[query_id]
            gated = policy_rows["mrv_gated"]
            hybrid_trace = trace_rows["hybrid"]
            gated_trace = trace_rows["mrv_gated"]
            supporting_ids = set(gated["supporting_ids"])
            involved_ids = (
                set(hybrid_trace["final_passage_ids"])
                | set(gated_trace["final_passage_ids"])
                | supporting_ids
            )
            packet_rows.append(
                {
                    "dataset": dataset,
                    "query_id": query_id,
                    "outcome": outcome,
                    "question": gated["question"],
                    "question_type": gated.get("question_type", ""),
                    "supporting_ids": sorted(supporting_ids),
                    "hybrid_final_ids": hybrid_trace["final_passage_ids"],
                    "gated_final_ids": gated_trace["final_passage_ids"],
                    "gated_actions": gated_trace["actions"],
                    "passages": passage_bundle(
                        involved_ids, corpus, supporting_ids
                    ),
                }
            )
            worksheet_rows.append(
                {
                    "dataset": dataset,
                    "query_id": query_id,
                    "outcome": outcome,
                    "question_type": gated.get("question_type", ""),
                    "hybrid_full_evidence": policy_rows["hybrid"]["metrics"][
                        "full_evidence"
                    ],
                    "gated_full_evidence": gated["metrics"][
                        "full_evidence"
                    ],
                    "graph_actions": gated["metrics"]["graph_actions"],
                    "reviewer": "",
                    "review_date": "",
                    "seed_quality_0_2": "",
                    "edge_validity_0_2": "",
                    "added_evidence_utility_0_2": "",
                    "gate_decision_appropriate_yes_no_unclear": "",
                    "primary_failure_code": "",
                    "confidence_1_3": "",
                    "notes": "",
                }
            )

    packet_path = output / "review_packet.jsonl"
    with packet_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in packet_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    worksheet_path = output / "human_annotations.csv"
    with worksheet_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(worksheet_rows[0]))
        writer.writeheader()
        writer.writerows(worksheet_rows)
    manifest = {
        "schema_version": 1,
        "random_seed": args.random_seed,
        "examples": len(packet_rows),
        "quotas": QUOTAS,
        "packet": str(packet_path),
        "worksheet": str(worksheet_path),
        "contains_benchmark_text": True,
        "redistribute": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
