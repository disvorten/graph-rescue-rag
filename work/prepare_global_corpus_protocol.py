from __future__ import annotations

"""Prepare leakage-auditable global-corpus evaluation protocols.

The original controlled protocol pools passages from the selected train and
evaluation questions.  This script creates a harder external validation: every
passage occurring anywhere in the official development split is indexed, while
evaluation is restricted to question IDs that were not used by any earlier
pilot or final-v1 evaluation.  Gold annotations are never read when entities or
graph edges are constructed.

This is deliberately called a *global distractor/dev corpus*.  It is not the
HotpotQA full-wiki setting and must not be reported as such.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

from graph_rescue.adapters import stable_passage_id
from graph_rescue.graph import KnowledgeGraph
from graph_rescue.io import read_jsonl, write_jsonl
from graph_rescue.models import Passage
from work.prepare_external_pilots import (
    build_entities,
    graph_diagnostics,
    iter_json_array,
    iter_jsonl,
)
from work.prepare_final_protocol import (
    context_hotpot_or_2wiki,
    context_musique,
    previous_query_ids,
    query_row,
)


DATA_ROOT = Path("work/datasets")
FINAL_ROOT = Path("work/final_protocol")
OUTPUT_ROOT = Path("work/global_corpus_protocol")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def record_id(record: dict) -> str:
    return str(record.get("_id", record.get("id")))


def seen_eval_ids(dataset: str) -> set[str]:
    result = previous_query_ids(dataset)
    final_eval = FINAL_ROOT / dataset / "eval.jsonl"
    if final_eval.exists():
        result.update(str(row["id"]) for row in read_jsonl(final_eval))
    return result


def add_passage(
    passages: dict[str, dict],
    *,
    title: str,
    text: str,
    sentences: list[str] | None = None,
) -> None:
    passage_id = stable_passage_id(title, text)
    candidate = {
        "id": passage_id,
        "title": title,
        "text": text,
        "sentences": list(sentences or []),
    }
    previous = passages.get(passage_id)
    if previous is None or len(text) > len(previous["text"]):
        passages[passage_id] = candidate


def prepare(
    *,
    dataset: str,
    source_path: Path,
    records: Iterable[dict],
    context_fn: Callable[[dict], list[dict]],
    max_hops: int,
    frontier_cap: int,
    diagnostics_limit: int,
) -> dict:
    output = OUTPUT_ROOT / dataset
    passages: dict[str, dict] = {}

    # Retain the frozen training sample and its passages.  The final-v1 corpus
    # also contains selected development passages; keeping them is harmless in
    # a global development corpus because every dev passage is indexed below.
    train_queries = list(read_jsonl(FINAL_ROOT / dataset / "train.jsonl"))
    for row in read_jsonl(FINAL_ROOT / dataset / "corpus.jsonl"):
        add_passage(
            passages,
            title=str(row.get("title", "")),
            text=str(row.get("text", "")),
            sentences=[str(item) for item in row.get("sentences", [])],
        )

    excluded = seen_eval_ids(dataset)
    eval_queries: list[dict] = []
    excluded_official_dev_queries = 0
    official_dev_queries = 0
    official_dev_context_occurrences = 0
    for record in records:
        official_dev_queries += 1
        context = context_fn(record)
        official_dev_context_occurrences += len(context)
        for item in context:
            add_passage(
                passages,
                title=str(item["title"]),
                text=str(item["text"]),
                sentences=[str(value) for value in item.get("sentences", [])],
            )
        if record_id(record) not in excluded:
            eval_queries.append(query_row(dataset, record, context))
        else:
            excluded_official_dev_queries += 1

    entity_input = {
        passage_id: {
            "id": passage_id,
            "title": item["title"],
            "text": item["text"],
        }
        for passage_id, item in passages.items()
    }
    entities = build_entities(entity_input)
    corpus: list[dict] = []
    passage_objects: list[Passage] = []
    for passage_id in sorted(passages):
        row = dict(passages[passage_id])
        row["entities"] = entities[passage_id]
        corpus.append(row)
        passage_objects.append(Passage.from_dict(row))

    graph = KnowledgeGraph.build(
        passage_objects,
        min_entity_df=2,
        max_entity_df_ratio=0.03,
    )
    diagnostic_queries = eval_queries[:diagnostics_limit]
    diagnostics = graph_diagnostics(
        graph,
        diagnostic_queries,
        max_hops=max_hops,
        frontier_cap=frontier_cap,
    )

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "corpus.jsonl", corpus)
    write_jsonl(output / "train.jsonl", train_queries)
    write_jsonl(output / "eval.jsonl", eval_queries)
    eval_ids = sorted(row["id"] for row in eval_queries)
    manifest = {
        "protocol": "graph-rescue-global-dev-v1",
        "dataset": dataset,
        "setting": "global development/distractor corpus",
        "not_fullwiki": True,
        "status": "prepared_external_validation",
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "official_dev_queries": official_dev_queries,
        "official_dev_context_occurrences": official_dev_context_occurrences,
        "frozen_train_queries": len(train_queries),
        "held_out_eval_queries": len(eval_queries),
        "excluded_previously_evaluated_official_dev_queries": (
            excluded_official_dev_queries
        ),
        "candidate_seen_ids_across_all_prior_train_eval_files": len(excluded),
        "corpus_passages": len(corpus),
        "eval_id_sha256": hashlib.sha256(
            "\n".join(eval_ids).encode("utf-8")
        ).hexdigest(),
        "support_count_distribution_eval": dict(
            sorted(
                Counter(
                    len(row["supporting_passage_ids"])
                    for row in eval_queries
                ).items()
            )
        ),
        "leakage_controls": {
            "evaluation_ids": (
                "All official development IDs not present in earlier pilot or "
                "final-v1 evaluation files."
            ),
            "corpus": (
                "All unique passages from the complete official development "
                "split plus the frozen training sample passages."
            ),
            "graph": (
                "Entities and edges use passage titles/text only; answers, "
                "support labels, decompositions and evidence triples are not "
                "read by graph construction."
            ),
        },
        "graph": {
            "passages": graph.stats.passages,
            "entities": graph.stats.entities,
            "edges": graph.stats.edges,
            "filtered_entities": graph.stats.filtered_entities,
            "diagnostics_queries": len(diagnostic_queries),
            **diagnostics,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def dataset_spec(
    dataset: str,
) -> tuple[Path, Iterator[dict], Callable[[dict], list[dict]], int, int]:
    if dataset == "hotpot":
        path = DATA_ROOT / "hotpot_dev_distractor_v1.json"
        return path, iter_json_array(path), context_hotpot_or_2wiki, 2, 160
    if dataset == "2wiki":
        path = DATA_ROOT / "2wiki_official" / "dev.json"
        return path, iter_json_array(path), context_hotpot_or_2wiki, 2, 180
    if dataset == "musique":
        path = (
            DATA_ROOT
            / "musique_official"
            / "data"
            / "musique_ans_v1.0_dev.jsonl"
        )
        return path, iter_jsonl(path), context_musique, 3, 200
    raise ValueError(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("hotpot", "2wiki", "musique"),
        default=("hotpot", "2wiki", "musique"),
    )
    parser.add_argument(
        "--diagnostics-limit",
        type=int,
        default=1000,
        help="Deterministic prefix used only for expensive graph diagnostics.",
    )
    args = parser.parse_args()
    results = {}
    for dataset in args.datasets:
        path, records, context_fn, max_hops, frontier_cap = dataset_spec(dataset)
        results[dataset] = prepare(
            dataset=dataset,
            source_path=path,
            records=records,
            context_fn=context_fn,
            max_hops=max_hops,
            frontier_cap=frontier_cap,
            diagnostics_limit=args.diagnostics_limit,
        )
        print(json.dumps({dataset: results[dataset]}, ensure_ascii=False, indent=2))
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
