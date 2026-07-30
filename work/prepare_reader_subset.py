from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_rescue.io import read_jsonl, write_jsonl


def priority(dataset: str, query_id: str) -> str:
    return hashlib.sha256(
        f"reader-v1|{dataset}|{query_id}".encode("utf-8")
    ).hexdigest()


def stratum(row: dict) -> str:
    question_type = str(row.get("question_type", "")).strip()
    if question_type:
        return question_type
    return f"support_{len(row.get('supporting_passage_ids', []))}"


def proportional_quotas(groups: dict[str, list[dict]], size: int) -> dict[str, int]:
    total = sum(len(rows) for rows in groups.values())
    exact = {
        name: size * len(rows) / total for name, rows in groups.items()
    }
    quotas = {
        name: min(len(groups[name]), int(value))
        for name, value in exact.items()
    }
    remaining = size - sum(quotas.values())
    order = sorted(
        groups,
        key=lambda name: (
            -(exact[name] - int(exact[name])),
            name,
        ),
    )
    for name in order:
        if remaining <= 0:
            break
        if quotas[name] < len(groups[name]):
            quotas[name] += 1
            remaining -= 1
    return quotas


def prepare(dataset: str, size: int) -> dict:
    root = Path("work/final_protocol") / dataset
    query_rows = list(read_jsonl(root / "eval.jsonl"))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in query_rows:
        groups[stratum(row)].append(row)
    quotas = proportional_quotas(groups, size)
    selected = []
    for name, rows in groups.items():
        rows.sort(key=lambda row: priority(dataset, str(row["id"])))
        selected.extend(rows[: quotas[name]])
    selected.sort(key=lambda row: str(row["id"]))
    selected_ids = {str(row["id"]) for row in selected}
    query_output = root / f"reader_eval_{size}.jsonl"
    write_jsonl(query_output, selected)

    if dataset == "musique":
        gold = [
            row
            for row in read_jsonl(root / "official_eval_gold.jsonl")
            if str(row["id"]) in selected_ids
        ]
        gold.sort(key=lambda row: str(row["id"]))
        gold_output = root / f"reader_eval_{size}_gold.jsonl"
        write_jsonl(gold_output, gold)
    else:
        source = json.loads(
            (root / "official_eval_gold.json").read_text(encoding="utf-8")
        )
        gold_by_id = {str(row["_id"]): row for row in source}
        gold = [gold_by_id[str(row["id"])] for row in selected]
        gold_output = root / f"reader_eval_{size}_gold.json"
        gold_output.write_text(
            json.dumps(gold, ensure_ascii=False),
            encoding="utf-8",
        )

    manifest = {
        "dataset": dataset,
        "size": len(selected),
        "source_eval_size": len(query_rows),
        "strata": {
            name: {
                "population": len(groups[name]),
                "selected": quotas[name],
            }
            for name in sorted(groups)
        },
        "query_id_sha256": hashlib.sha256(
            "\n".join(sorted(selected_ids)).encode("utf-8")
        ).hexdigest(),
        "queries": str(query_output),
        "gold": str(gold_output),
    }
    (root / f"reader_eval_{size}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("hotpot", "2wiki", "musique"),
        default=("hotpot", "2wiki", "musique"),
    )
    parser.add_argument("--size", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                dataset: prepare(dataset, args.size)
                for dataset in args.datasets
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
