from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl


def export_official_predictions(
    *,
    dataset: str,
    query_results_path: str | Path,
    queries_path: str | Path,
    output_path: str | Path,
    policy: str,
    reader: str,
) -> dict[str, Any]:
    if dataset not in {"hotpot", "2wiki", "musique"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    query_rows = {
        str(row["id"]): row for row in read_jsonl(queries_path)
    }
    selected = {
        str(row["query_id"]): row
        for row in read_jsonl(query_results_path)
        if row["policy"] == policy
    }
    missing = sorted(set(query_rows) - set(selected))
    if missing:
        raise ValueError(
            f"Missing {len(missing)} query results for policy {policy!r}: "
            f"{missing[:5]}"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if dataset in {"hotpot", "2wiki"}:
        payload: dict[str, dict[str, Any]] = {
            "answer": {},
            "sp": {},
        }
        if dataset == "2wiki":
            payload["evidence"] = {}
        for query_id in query_rows:
            row = selected[query_id]
            predictions = row.get("predictions", {})
            if reader not in predictions:
                raise ValueError(
                    f"Reader {reader!r} missing for query {query_id}"
                )
            evidence = row.get("reader_evidence", {}).get(reader, {})
            payload["answer"][query_id] = str(predictions[reader])
            payload["sp"][query_id] = [
                [str(item[0]), int(item[1])]
                for item in evidence.get("supporting_facts", [])
            ]
            if dataset == "2wiki":
                payload["evidence"][query_id] = [
                    [str(item[0]), str(item[1]), str(item[2])]
                    for item in evidence.get("evidence_triples", [])
                ]
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "dataset": dataset,
            "queries": len(query_rows),
            "output": str(output),
            "format": "json",
        }

    musique_rows = []
    for query_id, query in query_rows.items():
        row = selected[query_id]
        predictions = row.get("predictions", {})
        if reader not in predictions:
            raise ValueError(
                f"Reader {reader!r} missing for query {query_id}"
            )
        evidence = row.get("reader_evidence", {}).get(reader, {})
        passage_index = {
            str(item[0]): int(item[1])
            for item in query.get("passage_indices", [])
        }
        predicted_support = sorted(
            {
                passage_index[passage_id]
                for passage_id in evidence.get(
                    "supporting_passage_ids", []
                )
                if passage_id in passage_index
            }
        )
        musique_rows.append(
            {
                "id": query_id,
                "predicted_answer": str(predictions[reader]),
                "predicted_support_idxs": predicted_support,
                "predicted_answerable": True,
            }
        )
    write_jsonl(output, musique_rows)
    return {
        "dataset": dataset,
        "queries": len(query_rows),
        "output": str(output),
        "format": "jsonl",
    }
