from __future__ import annotations

"""Adapt HippoRAG's released MuSiQue benchmark to Graph Rescue JSONL."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from graph_rescue.adapters import stable_passage_id
from graph_rescue.io import write_jsonl
from work.prepare_external_pilots import build_entities


VENDOR_ROOT = Path("work/vendor/HippoRAG")
DATASET_ROOT = VENDOR_ROOT / "reproduce" / "dataset"
OUTPUT_ROOT = Path("work/official_baselines/hipporag_musique_adapter")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_passage_index(
    corpus_source: list[dict],
) -> tuple[dict[str, dict], dict[tuple[str, str], str], int]:
    """Preserve raw corpus rows while keeping stable IDs for normalized matches.

    ``stable_passage_id`` deliberately normalizes whitespace and case.  Released
    corpora can nevertheless contain raw variants that an official retriever
    treats as distinct documents.  The first variant keeps the stable ID and a
    later normalized collision receives a deterministic raw-content suffix.
    Exact duplicates remain deduplicated because they are not distinct retrieval
    documents in HippoRAG's hash-keyed embedding store.
    """

    passages: dict[str, dict] = {}
    raw_to_id: dict[tuple[str, str], str] = {}
    normalized_collisions = 0
    for item in corpus_source:
        title = str(item["title"])
        text = str(item["text"])
        raw_key = (title, text)
        if raw_key in raw_to_id:
            continue
        base_id = stable_passage_id(title, text)
        passage_id = base_id
        if passage_id in passages:
            raw_digest = hashlib.sha256(
                f"{title}\0{text}".encode("utf-8")
            ).hexdigest()[:8]
            passage_id = f"{base_id}__raw_{raw_digest}"
            normalized_collisions += 1
            if passage_id in passages:
                raise ValueError(f"Unresolved passage ID collision: {passage_id}")
        passages[passage_id] = {
            "id": passage_id,
            "title": title,
            "text": text,
        }
        raw_to_id[raw_key] = passage_id
    return passages, raw_to_id, normalized_collisions


def resolve_passage_id(
    title: str, text: str, raw_to_id: dict[tuple[str, str], str]
) -> str:
    return raw_to_id.get((title, text), stable_passage_id(title, text))


def main() -> None:
    corpus_path = DATASET_ROOT / "musique_corpus.json"
    queries_path = DATASET_ROOT / "musique.json"
    corpus_source = json.loads(corpus_path.read_text(encoding="utf-8"))
    query_source = json.loads(queries_path.read_text(encoding="utf-8"))

    passages, raw_to_id, normalized_collisions = build_passage_index(corpus_source)
    entities = build_entities(passages)
    corpus = []
    for passage_id in sorted(passages):
        row = dict(passages[passage_id])
        row["entities"] = entities[passage_id]
        corpus.append(row)

    queries = []
    unknown_supports = []
    for sample in query_source:
        supporting_ids = sorted(
            {
                resolve_passage_id(
                    str(paragraph["title"]),
                    str(paragraph.get("paragraph_text", paragraph.get("text", ""))),
                    raw_to_id,
                )
                for paragraph in sample["paragraphs"]
                if paragraph.get("is_supporting", True)
            }
        )
        missing = set(supporting_ids) - set(passages)
        if missing:
            unknown_supports.append({"id": sample["id"], "missing": sorted(missing)})
        queries.append(
            {
                "id": str(sample["id"]),
                "question": str(sample["question"]),
                "answers": sorted(
                    {
                        str(sample.get("answer", "")),
                        *[str(value) for value in sample.get("answer_aliases", [])],
                    }
                ),
                "supporting_passage_ids": supporting_ids,
                "supporting_paragraph_indices": sorted(
                    int(paragraph["idx"])
                    for paragraph in sample["paragraphs"]
                    if paragraph.get("is_supporting", True)
                ),
                "passage_indices": [
                    [
                        resolve_passage_id(
                            str(paragraph["title"]),
                            str(
                                paragraph.get(
                                    "paragraph_text", paragraph.get("text", "")
                                )
                            ),
                            raw_to_id,
                        ),
                        int(paragraph["idx"]),
                    ]
                    for paragraph in sample["paragraphs"]
                ],
                "dataset": "musique_hipporag_released",
            }
        )
    if unknown_supports:
        raise ValueError(
            f"Released corpus lacks support passages for {len(unknown_supports)} queries"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_ROOT / "corpus.jsonl", corpus)
    # Frozen models are loaded from final-v1; no benchmark query is used for
    # training or calibration in this external implementation comparison.
    write_jsonl(OUTPUT_ROOT / "train_empty.jsonl", [])
    write_jsonl(OUTPUT_ROOT / "eval.jsonl", queries)
    manifest = {
        "protocol": "hipporag-released-musique-adapter-v1",
        "source_repository": "https://github.com/OSU-NLP-Group/HippoRAG",
        "corpus_source": str(corpus_path),
        "corpus_sha256": sha256(corpus_path),
        "query_source": str(queries_path),
        "query_sha256": sha256(queries_path),
        "corpus_passages": len(corpus),
        "normalized_id_collisions_preserved": normalized_collisions,
        "eval_queries": len(queries),
        "support_count_distribution": dict(
            sorted(Counter(len(row["supporting_passage_ids"]) for row in queries).items())
        ),
        "training": "none; final-v1 MuSiQue seed-101 models are frozen",
        "graph_construction": (
            "Title/text entities only; released support labels are used only for evaluation."
        ),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
