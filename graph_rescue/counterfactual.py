from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .experiment import Experiment
from .ollama import OllamaClient
from .reader import OllamaReader
from .text import answer_f1
from .training import counterfactual_key


def _best_answer_f1(prediction: str, answers: tuple[str, ...]) -> float:
    return max((answer_f1(prediction, answer) for answer in answers), default=0.0)


def _query_priority(query_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{query_id}".encode("utf-8")).hexdigest()


def generate_counterfactual_labels(
    experiment: Experiment,
    output_path: str | Path,
    *,
    max_queries: int = 60,
    max_candidates: int = 3,
) -> dict:
    model = experiment.config.ollama.generation_model
    if not model:
        raise ValueError(
            "A generation_model is required to produce counterfactual labels"
        )
    client = OllamaClient(
        experiment.config.ollama.base_url,
        experiment.config.ollama.timeout_seconds,
    )
    if model not in client.models():
        raise RuntimeError(f"Generation model {model!r} is not installed")
    reader = OllamaReader(
        client,
        model,
        cache_dir=Path(experiment.config.cache_dir) / "reader",
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    existing_keys = set()
    if target.exists():
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                existing_rows.append(row)
                existing_keys.add(
                    counterfactual_key(
                        row["query_id"],
                        row["evidence_ids"],
                        row["candidate_id"],
                    )
                )

    selected_queries = sorted(
        experiment.train_queries,
        key=lambda item: (
            _query_priority(
                item.id, experiment.config.learning.random_seed
            ),
            item.id,
        ),
    )[:max_queries]
    new_rows = []
    positive_gold_candidates = 0
    for example in selected_queries:
        ranking = experiment.retriever.retrieve(
            example.question, experiment.config.retrieval.rerank_k
        )
        evidence = [
            item.passage_id
            for item in ranking[: experiment.config.retrieval.seed_k]
        ]
        candidates, _ = experiment.graph.candidate_paths(
            evidence,
            excluded_passage_ids=evidence,
            max_hops=experiment.config.graph.max_hops,
            cap=experiment.config.graph.frontier_cap,
        )
        if not candidates:
            continue
        support = set(example.supporting_passage_ids)
        positive = [
            item
            for item in candidates
            if item.target_passage_id in support - set(evidence)
        ]
        positive.sort(
            key=lambda item: (
                item.hop_count,
                -item.confidence,
                item.target_passage_id,
            )
        )
        negative = [
            item for item in candidates if item.target_passage_id not in support
        ]
        negative.sort(
            key=lambda item: (
                -experiment.extractor.relevance(
                    experiment.extractor.vector(
                        example.question, evidence, item, ranking
                    )
                ),
                item.target_passage_id,
            )
        )
        hub_negative = sorted(
            negative,
            key=lambda item: (
                -item.max_hubness,
                item.hop_count,
                item.target_passage_id,
            ),
        )

        chosen = []
        if positive:
            chosen.append(positive[0])
            positive_gold_candidates += 1
        if negative:
            chosen.append(negative[0])
        if hub_negative:
            chosen.append(hub_negative[0])
        deduplicated = []
        seen_targets = set()
        for candidate in chosen:
            if candidate.target_passage_id in seen_targets:
                continue
            seen_targets.add(candidate.target_passage_id)
            deduplicated.append(candidate)
        chosen = deduplicated[:max_candidates]
        if not chosen:
            continue

        base_prediction = reader.answer(example, evidence, experiment.passages)
        base_f1 = _best_answer_f1(base_prediction, example.answers)
        for candidate in chosen:
            key = counterfactual_key(
                example.id, evidence, candidate.target_passage_id
            )
            if key in existing_keys:
                continue
            candidate_evidence = evidence + [candidate.target_passage_id]
            candidate_prediction = reader.answer(
                example, candidate_evidence, experiment.passages
            )
            candidate_f1 = _best_answer_f1(
                candidate_prediction, example.answers
            )
            row = {
                "query_id": example.id,
                "question": example.question,
                "evidence_ids": evidence,
                "candidate_id": candidate.target_passage_id,
                "candidate_is_gold_support": (
                    candidate.target_passage_id in support
                ),
                "base_prediction": base_prediction,
                "candidate_prediction": candidate_prediction,
                "base_f1": base_f1,
                "candidate_f1": candidate_f1,
                "delta_f1": candidate_f1 - base_f1,
                "path_hops": candidate.hop_count,
                "path_confidence": candidate.confidence,
                "max_hubness": candidate.max_hubness,
            }
            new_rows.append(row)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            existing_keys.add(key)

    all_rows = existing_rows + new_rows
    return {
        "output": str(target),
        "selected_queries": len(selected_queries),
        "rows_total": len(all_rows),
        "rows_added": len(new_rows),
        "positive_gold_candidates_selected": positive_gold_candidates,
        "positive_delta_f1": sum(
            float(row["delta_f1"]) > 0.01 for row in all_rows
        ),
        "negative_delta_f1": sum(
            float(row["delta_f1"]) < -0.01 for row in all_rows
        ),
        "reader": reader.stats(),
    }
