from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from graph_rescue.adapters import stable_passage_id
from graph_rescue.graph import KnowledgeGraph
from graph_rescue.io import read_jsonl, write_jsonl
from graph_rescue.models import Passage
from work.prepare_external_pilots import (
    build_entities,
    deterministic_group_sample,
    graph_diagnostics,
    iter_json_array,
    iter_jsonl,
)


ROOT = Path("work/datasets")
OUTPUT_ROOT = Path("work/final_protocol")
TRAIN_QUERIES = 1000
EVAL_QUERIES = 1000


def previous_query_ids(dataset: str) -> set[str]:
    patterns = {
        "hotpot": ("hotpot_pilot", "hotpot_large"),
        "2wiki": ("2wiki_pilot", "2wiki_large"),
        "musique": ("musique_pilot", "musique_large"),
    }
    result: set[str] = set()
    for directory in patterns[dataset]:
        for name in ("train.jsonl", "eval.jsonl"):
            path = Path("work") / directory / name
            if path.exists():
                result.update(str(row["id"]) for row in read_jsonl(path))
    return result


def excluding(
    records: Iterable[dict],
    excluded_ids: set[str],
) -> Iterable[dict]:
    for record in records:
        record_id = str(record.get("_id", record.get("id")))
        if record_id not in excluded_ids:
            yield record


def context_hotpot_or_2wiki(record: dict) -> list[dict]:
    support_titles = {
        str(item[0])
        for item in record.get("supporting_facts", [])
        if isinstance(item, (list, tuple)) and item
    }
    result = []
    for title, sentences in record.get("context", []):
        sentence_list = (
            [str(item) for item in sentences]
            if isinstance(sentences, list)
            else []
        )
        result.append(
            {
                "title": str(title),
                "text": (
                    " ".join(sentence_list)
                    if sentence_list
                    else str(sentences)
                ),
                "sentences": sentence_list,
                "is_supporting": str(title) in support_titles,
            }
        )
    return result


def context_musique(record: dict) -> list[dict]:
    return [
        {
            "title": str(paragraph.get("title", "")),
            "text": str(
                paragraph.get(
                    "paragraph_text", paragraph.get("text", "")
                )
            ),
            "sentences": [
                str(item) for item in paragraph.get("sentences", [])
            ],
            "is_supporting": bool(paragraph.get("is_supporting", False)),
            "paragraph_index": int(paragraph.get("idx", index)),
        }
        for index, paragraph in enumerate(record.get("paragraphs", []))
    ]


def query_row(
    dataset: str,
    record: dict,
    context: list[dict],
) -> dict:
    passage_ids = [
        stable_passage_id(item["title"], item["text"]) for item in context
    ]
    support_ids = sorted(
        {
            passage_id
            for passage_id, item in zip(passage_ids, context)
            if item["is_supporting"]
        }
    )
    answers = [str(record.get("answer", ""))]
    answers.extend(str(item) for item in record.get("answer_aliases", []))
    row = {
        "id": str(record.get("_id", record.get("id"))),
        "question": str(record["question"]),
        "answers": sorted(set(answers)),
        "supporting_passage_ids": support_ids,
        "question_type": str(record.get("type", "")),
        "dataset": dataset,
    }
    if dataset in {"hotpot", "2wiki"}:
        row["supporting_facts"] = [
            [str(item[0]), int(item[1])]
            for item in record.get("supporting_facts", [])
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
    if dataset == "2wiki":
        row["evidence_triples"] = [
            [str(item[0]), str(item[1]), str(item[2])]
            for item in record.get("evidences", [])
            if isinstance(item, (list, tuple)) and len(item) >= 3
        ]
    if dataset == "musique":
        row["passage_indices"] = [
            [passage_id, int(item["paragraph_index"])]
            for passage_id, item in zip(passage_ids, context)
        ]
        row["supporting_paragraph_indices"] = sorted(
            {
                int(item["paragraph_index"])
                for item in context
                if item["is_supporting"]
            }
        )
    return row


def prepare(
    *,
    dataset: str,
    train_records: list[dict],
    eval_records: list[dict],
    context_fn: Callable[[dict], list[dict]],
    source: dict,
    max_hops: int,
    frontier_cap: int,
    excluded_eval_ids: set[str],
) -> dict:
    output = OUTPUT_ROOT / dataset
    passages: dict[str, dict] = {}
    contexts: dict[str, list[dict]] = {}
    for record in train_records + eval_records:
        record_id = str(record.get("_id", record.get("id")))
        context = context_fn(record)
        contexts[record_id] = context
        for item in context:
            passage_id = stable_passage_id(item["title"], item["text"])
            candidate = {
                "id": passage_id,
                "title": item["title"],
                "text": item["text"],
                "sentences": item["sentences"],
            }
            previous = passages.get(passage_id)
            if previous is None or len(candidate["text"]) > len(previous["text"]):
                passages[passage_id] = candidate

    entity_input = {
        passage_id: {
            "id": passage_id,
            "title": item["title"],
            "text": item["text"],
        }
        for passage_id, item in passages.items()
    }
    entities = build_entities(entity_input)
    corpus = []
    passage_objects = []
    for passage_id in sorted(passages):
        row = dict(passages[passage_id])
        row["entities"] = entities[passage_id]
        corpus.append(row)
        passage_objects.append(Passage.from_dict(row))

    train_queries = [
        query_row(
            dataset,
            record,
            contexts[str(record.get("_id", record.get("id")))],
        )
        for record in train_records
    ]
    eval_queries = [
        query_row(
            dataset,
            record,
            contexts[str(record.get("_id", record.get("id")))],
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
    if dataset == "musique":
        write_jsonl(output / "official_eval_gold.jsonl", eval_records)
    else:
        (output / "official_eval_gold.json").write_text(
            json.dumps(eval_records, ensure_ascii=False),
            encoding="utf-8",
        )
    selected_eval_ids = sorted(row["id"] for row in eval_queries)
    manifest = {
        "protocol": "graph-rescue-final-v1",
        "dataset": dataset,
        "source": source,
        "status": "prepared_untouched_eval",
        "train_queries": len(train_queries),
        "eval_queries": len(eval_queries),
        "corpus_passages": len(corpus),
        "excluded_previously_seen_ids": len(excluded_eval_ids),
        "eval_id_sha256": hashlib.sha256(
            "\n".join(selected_eval_ids).encode("utf-8")
        ).hexdigest(),
        "support_count_distribution_train": dict(
            sorted(
                Counter(
                    len(row["supporting_passage_ids"])
                    for row in train_queries
                ).items()
            )
        ),
        "support_count_distribution_eval": dict(
            sorted(
                Counter(
                    len(row["supporting_passage_ids"])
                    for row in eval_queries
                ).items()
            )
        ),
        "graph": {
            "construction": (
                "Entities are reconstructed only from passage titles and text. "
                "Gold support, answers, decompositions and evidence triples are "
                "not read by graph construction."
            ),
            "passages": graph.stats.passages,
            "entities": graph.stats.entities,
            "edges": graph.stats.edges,
            "filtered_entities": graph.stats.filtered_entities,
            **diagnostics,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def sample_hotpot() -> tuple[list[dict], list[dict], set[str]]:
    train_source = ROOT / "hotpot_train_v1.1.json"
    if not train_source.exists():
        raise FileNotFoundError(
            "Official HotpotQA train file is not downloaded yet: "
            f"{train_source}"
        )
    seen = previous_query_ids("hotpot")
    train, _ = deterministic_group_sample(
        iter_json_array(train_source),
        group_key=lambda row: str(row.get("type", "unknown")),
        id_key=lambda row: str(row["_id"]),
        per_group={"bridge": 800, "comparison": 200},
        seed=20260821,
    )
    evaluation, _ = deterministic_group_sample(
        excluding(
            iter_json_array(ROOT / "hotpot_dev_distractor_v1.json"),
            seen,
        ),
        group_key=lambda row: str(row.get("type", "unknown")),
        id_key=lambda row: str(row["_id"]),
        per_group={"bridge": 800, "comparison": 200},
        seed=20260822,
    )
    return train, evaluation, seen


def sample_2wiki() -> tuple[list[dict], list[dict], set[str]]:
    seen = previous_query_ids("2wiki")
    quotas = {
        "comparison": 250,
        "inference": 250,
        "compositional": 250,
        "bridge_comparison": 250,
    }
    train, _ = deterministic_group_sample(
        iter_json_array(ROOT / "2wiki_official" / "train.json"),
        group_key=lambda row: str(row.get("type", "unknown")),
        id_key=lambda row: str(row["_id"]),
        per_group=quotas,
        seed=20260823,
    )
    evaluation, _ = deterministic_group_sample(
        excluding(
            iter_json_array(ROOT / "2wiki_official" / "dev.json"),
            seen,
        ),
        group_key=lambda row: str(row.get("type", "unknown")),
        id_key=lambda row: str(row["_id"]),
        per_group=quotas,
        seed=20260824,
    )
    return train, evaluation, seen


def musique_hops(row: dict) -> str:
    return str(
        sum(
            bool(item.get("is_supporting", False))
            for item in row.get("paragraphs", [])
        )
    )


def sample_musique() -> tuple[list[dict], list[dict], set[str]]:
    seen = previous_query_ids("musique")
    quotas = {"2": 425, "3": 375, "4": 200}
    train, _ = deterministic_group_sample(
        iter_jsonl(
            ROOT
            / "musique_official"
            / "data"
            / "musique_ans_v1.0_train.jsonl"
        ),
        group_key=musique_hops,
        id_key=lambda row: str(row["id"]),
        per_group=quotas,
        seed=20260825,
    )
    evaluation, _ = deterministic_group_sample(
        excluding(
            iter_jsonl(
                ROOT
                / "musique_official"
                / "data"
                / "musique_ans_v1.0_dev.jsonl"
            ),
            seen,
        ),
        group_key=musique_hops,
        id_key=lambda row: str(row["id"]),
        per_group=quotas,
        seed=20260826,
    )
    return train, evaluation, seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("hotpot", "2wiki", "musique"),
        default=("hotpot", "2wiki", "musique"),
    )
    args = parser.parse_args()
    results = {}
    if "hotpot" in args.datasets:
        hotpot_train, hotpot_eval, hotpot_seen = sample_hotpot()
        results["hotpot"] = prepare(
            dataset="hotpot",
            train_records=hotpot_train,
            eval_records=hotpot_eval,
            context_fn=context_hotpot_or_2wiki,
            source={
                "train": "official HotpotQA train v1.1",
                "eval": "official HotpotQA distractor dev v1",
            },
            max_hops=2,
            frontier_cap=160,
            excluded_eval_ids=hotpot_seen,
        )
    if "2wiki" in args.datasets:
        wiki_train, wiki_eval, wiki_seen = sample_2wiki()
        results["2wiki"] = prepare(
            dataset="2wiki",
            train_records=wiki_train,
            eval_records=wiki_eval,
            context_fn=context_hotpot_or_2wiki,
            source={
                "train": "official 2WikiMultiHopQA train",
                "eval": "official 2WikiMultiHopQA dev",
            },
            max_hops=2,
            frontier_cap=180,
            excluded_eval_ids=wiki_seen,
        )
    if "musique" in args.datasets:
        musique_train, musique_eval, musique_seen = sample_musique()
        results["musique"] = prepare(
            dataset="musique",
            train_records=musique_train,
            eval_records=musique_eval,
            context_fn=context_musique,
            source={
                "train": "official MuSiQue-Answerable train",
                "eval": "official MuSiQue-Answerable dev",
            },
            max_hops=3,
            frontier_cap=200,
            excluded_eval_ids=musique_seen,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
