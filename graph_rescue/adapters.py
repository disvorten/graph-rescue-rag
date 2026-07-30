from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .io import write_jsonl


def stable_passage_id(title: str, text: str = "") -> str:
    normalized_title = " ".join(title.casefold().split())
    normalized_text = " ".join(text.casefold().split())
    key = f"{normalized_title}\0{normalized_text}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"p_{digest}"


def _read_json(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("data", "examples", "questions"):
            if isinstance(value.get(key), list):
                return value[key]
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array or an object containing a data array")
    return value


def _context_pairs(context: Any) -> Iterable[tuple[str, str, list[str]]]:
    for item in context or []:
        if isinstance(item, dict):
            title = str(item.get("title", ""))
            text = item.get("paragraph_text", item.get("text", ""))
            sentences = item.get("sentences", [])
            if isinstance(text, list):
                sentences = [str(part) for part in text]
                text = " ".join(str(part) for part in text)
            yield title, str(text), [str(part) for part in sentences]
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sentences = item[0], item[1]
            text = " ".join(str(part) for part in sentences) if isinstance(
                sentences, list
            ) else str(sentences)
            values = (
                [str(part) for part in sentences]
                if isinstance(sentences, list)
                else []
            )
            yield str(title), text, values


def convert_hotpot_or_2wiki(
    input_path: str | Path,
    corpus_path: str | Path,
    queries_path: str | Path,
) -> dict[str, int]:
    records = _read_json(input_path)
    passages: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        title_to_id: dict[str, str] = {}
        for title, text, sentences in _context_pairs(record.get("context")):
            passage_id = stable_passage_id(title, text)
            title_to_id[title] = passage_id
            passages.setdefault(
                passage_id,
                {
                    "id": passage_id,
                    "title": title,
                    "text": text,
                    "sentences": sentences,
                },
            )
        support_titles = {
            str(item[0])
            for item in record.get("supporting_facts", [])
            if isinstance(item, (list, tuple)) and item
        }
        supporting_ids = sorted(
            title_to_id[title] for title in support_titles if title in title_to_id
        )
        queries.append(
            {
                "id": str(record.get("_id", record.get("id", index))),
                "question": str(record["question"]),
                "answers": [str(record.get("answer", ""))],
                "supporting_passage_ids": supporting_ids,
                "supporting_facts": [
                    [str(item[0]), int(item[1])]
                    for item in record.get("supporting_facts", [])
                    if isinstance(item, (list, tuple)) and len(item) >= 2
                ],
                "evidence_triples": [
                    [str(item[0]), str(item[1]), str(item[2])]
                    for item in record.get("evidences", [])
                    if isinstance(item, (list, tuple)) and len(item) >= 3
                ],
                "question_type": str(record.get("type", "")),
            }
        )
    write_jsonl(corpus_path, passages.values())
    write_jsonl(queries_path, queries)
    return {"passages": len(passages), "queries": len(queries)}


def convert_musique(
    input_path: str | Path,
    corpus_path: str | Path,
    queries_path: str | Path,
) -> dict[str, int]:
    records = _read_json(input_path)
    passages: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        supporting_ids: list[str] = []
        for paragraph in record.get("paragraphs", []):
            title = str(paragraph.get("title", ""))
            text = str(
                paragraph.get("paragraph_text", paragraph.get("text", ""))
            )
            passage_id = stable_passage_id(title, text)
            passages.setdefault(
                passage_id,
                {
                    "id": passage_id,
                    "title": title,
                    "text": text,
                    "sentences": [
                        str(item)
                        for item in paragraph.get("sentences", [])
                    ],
                },
            )
            if paragraph.get("is_supporting", False):
                supporting_ids.append(passage_id)
        queries.append(
            {
                "id": str(record.get("id", record.get("_id", index))),
                "question": str(record["question"]),
                "answers": [str(record.get("answer", ""))],
                "supporting_passage_ids": sorted(set(supporting_ids)),
                "supporting_facts": [
                    ["", int(paragraph.get("idx", paragraph_index))]
                    for paragraph_index, paragraph in enumerate(
                        record.get("paragraphs", [])
                    )
                    if paragraph.get("is_supporting", False)
                ],
                "question_type": str(record.get("type", "")),
            }
        )
    write_jsonl(corpus_path, passages.values())
    write_jsonl(queries_path, queries)
    return {"passages": len(passages), "queries": len(queries)}


def convert_dataset(
    dataset_format: str,
    input_path: str | Path,
    corpus_path: str | Path,
    queries_path: str | Path,
) -> dict[str, int]:
    if dataset_format in {"hotpot", "2wiki"}:
        return convert_hotpot_or_2wiki(input_path, corpus_path, queries_path)
    if dataset_format == "musique":
        return convert_musique(input_path, corpus_path, queries_path)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")
