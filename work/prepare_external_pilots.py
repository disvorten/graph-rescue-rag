from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator

from graph_rescue.adapters import stable_passage_id
from graph_rescue.graph import KnowledgeGraph
from graph_rescue.models import Passage
from graph_rescue.text import normalize, tokenize


ROOT = Path("work/datasets")
DOWNLOAD_MANIFEST = ROOT / "multihop_download_manifest.json"
CAPITALIZED_ENTITY = re.compile(
    r"\b(?:[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,3})\b", flags=re.UNICODE
)


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    eof = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            if position:
                buffer = buffer[position:]
                position = 0
            if not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    position += 1
                    started = True
                    continue
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"Expected objects in {path}")
                yield value
                position = end
            if eof:
                if buffer[position:].strip():
                    raise ValueError(f"Unexpected trailing data in {path}")
                return


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                yield value


def deterministic_group_sample(
    records: Iterable[dict],
    *,
    group_key: Callable[[dict], str],
    id_key: Callable[[dict], str],
    per_group: dict[str, int],
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    heaps: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    observed: Counter[str] = Counter()
    for record in records:
        group = group_key(record)
        observed[group] += 1
        limit = per_group.get(group, 0)
        if limit <= 0:
            continue
        record_id = id_key(record)
        priority = int.from_bytes(
            hashlib.sha256(f"{seed}|{record_id}".encode("utf-8")).digest()[:8],
            "big",
        )
        item = (-priority, record_id, record)
        heap = heaps[group]
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)

    selected: list[dict] = []
    for group, limit in per_group.items():
        values = heaps.get(group, [])
        if len(values) != limit:
            raise RuntimeError(
                f"Group {group!r} has {len(values)} selected records, expected {limit}"
            )
        selected.extend(
            item[2] for item in sorted(values, key=lambda item: (-item[0], item[1]))
        )
    selected.sort(key=id_key)
    return selected, dict(sorted(observed.items()))


def context_2wiki(record: dict) -> list[tuple[str, str, bool]]:
    support_titles = {
        str(item[0])
        for item in record.get("supporting_facts", [])
        if isinstance(item, list) and item
    }
    result = []
    for title, sentences in record.get("context", []):
        text = (
            " ".join(str(sentence) for sentence in sentences)
            if isinstance(sentences, list)
            else str(sentences)
        )
        result.append((str(title), text, str(title) in support_titles))
    return result


def context_musique(record: dict) -> list[tuple[str, str, bool]]:
    return [
        (
            str(paragraph.get("title", "")),
            str(paragraph.get("paragraph_text", paragraph.get("text", ""))),
            bool(paragraph.get("is_supporting", False)),
        )
        for paragraph in record.get("paragraphs", [])
    ]


def build_entities(
    passages: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    title_lookup: dict[tuple[str, ...], set[str]] = defaultdict(set)
    max_title_tokens = 1
    for passage in passages.values():
        title_tokens = tuple(tokenize(passage["title"]))
        if not title_tokens or len(title_tokens) > 8:
            continue
        if len(title_tokens) == 1 and len(title_tokens[0]) < 4:
            continue
        title_lookup[title_tokens].add(normalize(passage["title"]))
        max_title_tokens = max(max_title_tokens, len(title_tokens))

    result: dict[str, list[str]] = {}
    for passage_id, passage in passages.items():
        words = tokenize(passage["text"])
        title_entities = {normalize(passage["title"])}
        for start in range(len(words)):
            for width in range(
                1, min(max_title_tokens, len(words) - start) + 1
            ):
                title_entities.update(
                    title_lookup.get(tuple(words[start : start + width]), ())
                )
        surface_entities = {
            normalize(value)
            for value in CAPITALIZED_ENTITY.findall(
                f"{passage['title']}. {passage['text']}"
            )
            if len(normalize(value)) > 2
        }
        entities = sorted((title_entities | surface_entities) - {""})
        result[passage_id] = entities[:128]
    return result


def make_query(
    record: dict,
    context: list[tuple[str, str, bool]],
    *,
    dataset: str,
) -> dict:
    supporting_ids = sorted(
        {
            stable_passage_id(title, text)
            for title, text, supporting in context
            if supporting
        }
    )
    record_id = record.get("_id", record.get("id"))
    answers = [str(record.get("answer", ""))]
    if dataset == "2wiki":
        answers.extend(str(value) for value in record.get("answer_aliases", []))
    return {
        "id": str(record_id),
        "question": str(record["question"]),
        "answers": sorted(set(answers)),
        "supporting_passage_ids": supporting_ids,
    }


def graph_diagnostics(
    graph: KnowledgeGraph,
    queries: list[dict],
    *,
    max_hops: int,
    frontier_cap: int,
) -> dict[str, float]:
    any_connected = 0
    pair_total = 0
    pair_reachable = 0
    for query in queries:
        support = list(query["supporting_passage_ids"])
        query_connected = False
        for source in support:
            candidates, _ = graph.candidate_paths(
                [source],
                excluded_passage_ids=[source],
                max_hops=max_hops,
                cap=frontier_cap,
            )
            reachable = {item.target_passage_id for item in candidates}
            for target in support:
                if target == source:
                    continue
                pair_total += 1
                if target in reachable:
                    pair_reachable += 1
                    query_connected = True
        any_connected += int(query_connected)
    return {
        "support_query_any_connection": any_connected / max(1, len(queries)),
        "support_ordered_pair_recall": pair_reachable / max(1, pair_total),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare_dataset(
    *,
    name: str,
    train_records: list[dict],
    eval_records: list[dict],
    context_fn: Callable[[dict], list[tuple[str, str, bool]]],
    source: dict,
    max_hops: int,
    frontier_cap: int,
    output_name: str | None = None,
) -> dict:
    output = Path("work") / (output_name or f"{name}_pilot")
    passages: dict[str, dict[str, str]] = {}
    contexts_by_record: dict[str, list[tuple[str, str, bool]]] = {}
    for record in train_records + eval_records:
        record_id = str(record.get("_id", record.get("id")))
        context = context_fn(record)
        contexts_by_record[record_id] = context
        for title, text, _ in context:
            passage_id = stable_passage_id(title, text)
            candidate = {"id": passage_id, "title": title, "text": text}
            previous = passages.get(passage_id)
            if previous is None or len(text) > len(previous["text"]):
                passages[passage_id] = candidate

    entities = build_entities(passages)
    corpus = []
    passage_objects = []
    for passage_id in sorted(passages):
        row = dict(passages[passage_id])
        row["entities"] = entities[passage_id]
        corpus.append(row)
        passage_objects.append(Passage.from_dict(row))

    train_queries = [
        make_query(
            record,
            contexts_by_record[str(record.get("_id", record.get("id")))],
            dataset=name,
        )
        for record in train_records
    ]
    eval_queries = [
        make_query(
            record,
            contexts_by_record[str(record.get("_id", record.get("id")))],
            dataset=name,
        )
        for record in eval_records
    ]
    graph = KnowledgeGraph.build(
        passage_objects,
        min_entity_df=2,
        max_entity_df_ratio=0.03,
    )
    diagnostics = graph_diagnostics(
        graph,
        train_queries + eval_queries,
        max_hops=max_hops,
        frontier_cap=frontier_cap,
    )

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "corpus.jsonl", corpus)
    write_jsonl(output / "train.jsonl", train_queries)
    write_jsonl(output / "eval.jsonl", eval_queries)
    manifest = {
        "dataset": name,
        "source": source,
        "train_queries": len(train_queries),
        "eval_queries": len(eval_queries),
        "corpus_passages": len(corpus),
        "support_count_distribution_train": dict(
            sorted(
                Counter(
                    len(item["supporting_passage_ids"]) for item in train_queries
                ).items()
            )
        ),
        "support_count_distribution_eval": dict(
            sorted(
                Counter(
                    len(item["supporting_passage_ids"]) for item in eval_queries
                ).items()
            )
        ),
        "graph": {
            "construction": (
                "Normalized own titles, corpus-title mentions, and capitalization-"
                "based surface entities extracted only from passage text. Gold "
                "supporting annotations are used only for diagnostics."
            ),
            "passages": graph.stats.passages,
            "entities": graph.stats.entities,
            "edges": graph.stats.edges,
            "filtered_entities": graph.stats.filtered_entities,
            **diagnostics,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    downloads = json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    source_by_name = {
        record["dataset"]: {
            "url": record["url"],
            "archive_sha256": record["sha256"],
            "archive_bytes": record["bytes"],
        }
        for record in downloads["records"]
    }

    wiki_groups = {
        "comparison": 60,
        "inference": 60,
        "compositional": 60,
        "bridge_comparison": 60,
    }
    wiki_eval_groups = {name: 30 for name in wiki_groups}
    wiki_train, wiki_train_observed = deterministic_group_sample(
        iter_json_array(ROOT / "2wiki_official" / "train.json"),
        group_key=lambda row: str(row.get("type", "unknown")),
        id_key=lambda row: str(row["_id"]),
        per_group=wiki_groups,
        seed=20260730,
    )
    wiki_eval, wiki_eval_observed = deterministic_group_sample(
        iter_json_array(ROOT / "2wiki_official" / "dev.json"),
        group_key=lambda row: str(row.get("type", "unknown")),
        id_key=lambda row: str(row["_id"]),
        per_group=wiki_eval_groups,
        seed=20260731,
    )
    wiki_manifest = prepare_dataset(
        name="2wiki",
        train_records=wiki_train,
        eval_records=wiki_eval,
        context_fn=context_2wiki,
        source={
            **source_by_name["2WikiMultiHopQA"],
            "official_train_examples_by_type": wiki_train_observed,
            "official_dev_examples_by_type": wiki_eval_observed,
            "sampling": "60 train and 30 dev examples per official question type",
        },
        max_hops=2,
        frontier_cap=160,
    )

    musique_groups = {"2": 80, "3": 80, "4": 80}
    musique_eval_groups = {"2": 40, "3": 40, "4": 40}
    hop_group = lambda row: str(
        sum(bool(item.get("is_supporting", False)) for item in row["paragraphs"])
    )
    musique_train, musique_train_observed = deterministic_group_sample(
        iter_jsonl(
            ROOT
            / "musique_official"
            / "data"
            / "musique_ans_v1.0_train.jsonl"
        ),
        group_key=hop_group,
        id_key=lambda row: str(row["id"]),
        per_group=musique_groups,
        seed=20260732,
    )
    musique_eval, musique_eval_observed = deterministic_group_sample(
        iter_jsonl(
            ROOT
            / "musique_official"
            / "data"
            / "musique_ans_v1.0_dev.jsonl"
        ),
        group_key=hop_group,
        id_key=lambda row: str(row["id"]),
        per_group=musique_eval_groups,
        seed=20260733,
    )
    musique_manifest = prepare_dataset(
        name="musique",
        train_records=musique_train,
        eval_records=musique_eval,
        context_fn=context_musique,
        source={
            **source_by_name["MuSiQue"],
            "official_train_examples_by_support_count": musique_train_observed,
            "official_dev_examples_by_support_count": musique_eval_observed,
            "sampling": "80 train and 40 dev examples per 2/3/4-hop group",
        },
        max_hops=3,
        frontier_cap=180,
    )
    print(
        json.dumps(
            {"2wiki": wiki_manifest, "musique": musique_manifest},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
