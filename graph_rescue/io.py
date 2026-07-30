from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Iterator

from .models import Passage, QueryExample


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_passages(path: str | Path) -> list[Passage]:
    passages = [Passage.from_dict(item) for item in read_jsonl(path)]
    ids = [item.id for item in passages]
    if len(ids) != len(set(ids)):
        raise ValueError("Passage IDs must be unique")
    return passages


def load_queries(path: str | Path) -> list[QueryExample]:
    queries = [QueryExample.from_dict(item) for item in read_jsonl(path)]
    ids = [item.id for item in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Query IDs must be unique")
    return queries
