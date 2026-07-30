from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import ExperimentConfig
from .features import (
    CANDIDATE_FEATURE_NAMES,
    CONTINUE_GATE_FEATURE_NAMES,
    GATE_FEATURE_NAMES,
    CandidateFeatureExtractor,
    continue_gate_vector,
    gate_vector,
)
from .graph import KnowledgeGraph
from .hybrid import HybridRetriever
from .learning import GateModel, MRVModel
from .io import read_jsonl
from .models import CandidatePath, CandidateScore, Passage, QueryExample, RetrievedPassage
from .text import estimate_tokens


@dataclass
class TrainingSummary:
    candidate_examples: int
    positive_add_support: int
    positive_complete: int
    harmful_examples: int
    reader_supervised_examples: int
    positive_reader_gain: int
    gate_states: int
    positive_gate_states: int
    gate_calibration_method: str
    gate_threshold: float
    preflight_gate_states: int
    positive_preflight_gate_states: int
    continue_gate_states: int
    positive_continue_gate_states: int
    continue_gate_calibration_method: str
    continue_gate_threshold: float
    mrv_model_path: str
    gate_model_path: str
    continue_gate_model_path: str


@dataclass
class GateTrainingRows:
    preflight_x: np.ndarray
    preflight_y: np.ndarray
    continue_x: np.ndarray
    continue_y: np.ndarray


def counterfactual_key(
    query_id: str, evidence_ids: Sequence[str], candidate_id: str
) -> tuple[str, tuple[str, ...], str]:
    return (str(query_id), tuple(evidence_ids), str(candidate_id))


def load_counterfactual_labels(
    path: str | Path | None,
) -> dict[tuple[str, tuple[str, ...], str], dict]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Counterfactual labels not found: {target}")
    result = {}
    for row in read_jsonl(target):
        key = counterfactual_key(
            str(row["query_id"]),
            [str(item) for item in row["evidence_ids"]],
            str(row["candidate_id"]),
        )
        result[key] = row
    return result


def split_for_calibration(
    examples: Sequence[QueryExample], fraction: float, seed: int
) -> tuple[list[QueryExample], list[QueryExample]]:
    if len(examples) < 2:
        return list(examples), list(examples)
    rng = np.random.default_rng(seed)
    calibration_size = max(1, int(round(len(examples) * fraction)))
    calibration_size = min(calibration_size, len(examples) - 1)
    multi_support = [
        index
        for index, item in enumerate(examples)
        if len(item.supporting_passage_ids) > 1
    ]
    single_support = [
        index
        for index, item in enumerate(examples)
        if len(item.supporting_passage_ids) <= 1
    ]
    rng.shuffle(multi_support)
    rng.shuffle(single_support)
    calibration_indices: set[int] = set()
    if calibration_size >= 2 and multi_support and single_support:
        calibration_indices.update((multi_support.pop(), single_support.pop()))
    remaining = multi_support + single_support
    rng.shuffle(remaining)
    calibration_indices.update(
        remaining[: calibration_size - len(calibration_indices)]
    )
    train = [item for index, item in enumerate(examples) if index not in calibration_indices]
    calibration = [
        item for index, item in enumerate(examples) if index in calibration_indices
    ]
    return train, calibration


def candidate_training_rows(
    examples: Sequence[QueryExample],
    retriever: HybridRetriever,
    graph: KnowledgeGraph,
    passages: dict[str, Passage],
    extractor: CandidateFeatureExtractor,
    config: ExperimentConfig,
    counterfactual_labels: dict[
        tuple[str, tuple[str, ...], str], dict
    ] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    feature_rows: list[np.ndarray] = []
    add_labels: list[int] = []
    complete_labels: list[int] = []
    reader_gain_labels: list[int] = []
    harmful_labels: list[int] = []
    reader_supervision: list[int] = []
    counterfactual_labels = counterfactual_labels or {}

    for example in examples:
        ranking = retriever.retrieve(example.question, config.retrieval.rerank_k)
        evidence = [item.passage_id for item in ranking[: config.retrieval.seed_k]]
        candidates, _ = graph.candidate_paths(
            evidence,
            excluded_passage_ids=evidence,
            max_hops=config.graph.max_hops,
            cap=config.graph.frontier_cap,
        )
        support = set(example.supporting_passage_ids)
        evidence_support = support & set(evidence)
        for candidate in candidates:
            vector = extractor.vector(
                example.question, evidence, candidate, ranking
            )
            target = candidate.target_passage_id
            add = int(target in support and target not in evidence)
            complete = int(
                bool(support)
                and support.issubset(set(evidence) | {target})
                and not support.issubset(set(evidence))
            )
            redundancy = vector[CANDIDATE_FEATURE_NAMES.index("evidence_redundancy")]
            harmful = int(
                target not in support
                and (
                    redundancy >= 0.45
                    or candidate.max_hubness >= 8
                    or candidate.confidence < 0.10
                )
            )
            counterfactual = counterfactual_labels.get(
                counterfactual_key(example.id, evidence, target)
            )
            reader_gain = max(add, complete)
            supervised = 0
            if counterfactual is not None:
                delta_f1 = float(counterfactual["delta_f1"])
                reader_gain = int(delta_f1 > 0.01)
                harmful = int(delta_f1 < -0.01)
                supervised = 1
            feature_rows.append(vector)
            add_labels.append(add)
            complete_labels.append(complete)
            reader_gain_labels.append(reader_gain)
            harmful_labels.append(harmful)
            reader_supervision.append(supervised)

        # Add one positive intermediate state when a graph rescue is reachable.
        reachable_support = [
            item for item in candidates if item.target_passage_id in support - evidence_support
        ]
        if reachable_support:
            evidence = evidence + [reachable_support[0].target_passage_id]
            candidates, _ = graph.candidate_paths(
                evidence,
                excluded_passage_ids=evidence,
                max_hops=config.graph.max_hops,
                cap=config.graph.frontier_cap,
            )
            for candidate in candidates[: min(50, len(candidates))]:
                vector = extractor.vector(
                    example.question, evidence, candidate, ranking
                )
                target = candidate.target_passage_id
                add = int(target in support and target not in evidence)
                complete = int(
                    bool(support)
                    and support.issubset(set(evidence) | {target})
                    and not support.issubset(set(evidence))
                )
                redundancy = vector[
                    CANDIDATE_FEATURE_NAMES.index("evidence_redundancy")
                ]
                harmful = int(
                    target not in support
                    and (
                        redundancy >= 0.45
                        or candidate.max_hubness >= 8
                        or candidate.confidence < 0.10
                    )
                )
                counterfactual = counterfactual_labels.get(
                    counterfactual_key(example.id, evidence, target)
                )
                reader_gain = max(add, complete)
                supervised = 0
                if counterfactual is not None:
                    delta_f1 = float(counterfactual["delta_f1"])
                    reader_gain = int(delta_f1 > 0.01)
                    harmful = int(delta_f1 < -0.01)
                    supervised = 1
                feature_rows.append(vector)
                add_labels.append(add)
                complete_labels.append(complete)
                reader_gain_labels.append(reader_gain)
                harmful_labels.append(harmful)
                reader_supervision.append(supervised)

    if not feature_rows:
        raise RuntimeError(
            "The graph produced no candidate paths. Check entity/link coverage and "
            "graph filtering thresholds."
        )
    return (
        np.vstack(feature_rows),
        np.asarray(add_labels, dtype=np.int64),
        np.asarray(complete_labels, dtype=np.int64),
        np.asarray(reader_gain_labels, dtype=np.int64),
        np.asarray(harmful_labels, dtype=np.int64),
        np.asarray(reader_supervision, dtype=np.int64),
    )


def score_paths(
    example: QueryExample,
    evidence: Sequence[str],
    candidates: Sequence[CandidatePath],
    ranking: Sequence[RetrievedPassage],
    extractor: CandidateFeatureExtractor,
    model: MRVModel,
    passages: dict[str, Passage],
    config: ExperimentConfig,
) -> list[CandidateScore]:
    if not candidates:
        return []
    matrix = np.vstack(
        [
            extractor.vector(example.question, evidence, candidate, ranking)
            for candidate in candidates
        ]
    )
    p_add, p_complete, p_reader_gain, p_harmful = model.predict(matrix)
    result: list[CandidateScore] = []
    for index, (candidate, vector) in enumerate(zip(candidates, matrix)):
        token_fraction = estimate_tokens(
            passages[candidate.target_passage_id].full_text
        ) / max(1, config.retrieval.evidence_token_budget)
        marginal = (
            float(p_complete[index])
            + 0.35 * float(p_add[index])
            + config.learning.reader_gain_weight
            * float(p_reader_gain[index])
            - 0.25 * float(p_harmful[index])
            - config.learning.token_cost_weight * token_fraction
            - config.learning.hop_cost_weight * candidate.hop_count
            - config.learning.noise_cost_weight
            * min(1.0, candidate.max_hubness / 100.0)
        )
        result.append(
            CandidateScore(
                path=candidate,
                relevance=extractor.relevance(vector),
                p_add_support=float(p_add[index]),
                p_complete=float(p_complete[index]),
                p_reader_gain=float(p_reader_gain[index]),
                p_harmful=float(p_harmful[index]),
                marginal_value=marginal,
                features=dict(zip(CANDIDATE_FEATURE_NAMES, vector)),
            )
        )
    return result


def gate_training_rows(
    examples: Sequence[QueryExample],
    retriever: HybridRetriever,
    graph: KnowledgeGraph,
    passages: dict[str, Passage],
    extractor: CandidateFeatureExtractor,
    mrv_model: MRVModel,
    config: ExperimentConfig,
) -> GateTrainingRows:
    preflight_rows: list[np.ndarray] = []
    preflight_labels: list[int] = []
    continue_rows: list[np.ndarray] = []
    continue_labels: list[int] = []
    for example in examples:
        ranking = retriever.retrieve(example.question, config.retrieval.rerank_k)
        seeds = list(ranking[: config.retrieval.seed_k])
        evidence = [item.passage_id for item in seeds]
        support = set(example.supporting_passage_ids)

        previous: CandidateScore | None = None
        previous_frontier_size = 0
        previous_score_margin = 0.0
        for step in range(config.graph.max_actions):
            candidates, _ = graph.candidate_paths(
                evidence,
                excluded_passage_ids=evidence,
                max_hops=config.graph.max_hops,
                cap=config.graph.frontier_cap,
            )
            evidence_tokens = sum(
                estimate_tokens(passages[item].full_text) for item in evidence
            )
            base = gate_vector(
                example.question,
                ranking,
                evidence,
                passages,
                graph,
                seed_k=config.retrieval.seed_k,
                evidence_tokens=evidence_tokens,
                token_budget=config.retrieval.evidence_token_budget,
            )
            if step == 0:
                preflight_rows.append(base)
            elif previous is not None:
                continue_rows.append(
                    continue_gate_vector(
                        example.question,
                        ranking,
                        evidence,
                        passages,
                        graph,
                        seed_k=config.retrieval.seed_k,
                        evidence_tokens=evidence_tokens,
                        token_budget=config.retrieval.evidence_token_budget,
                        previous=previous,
                        previous_frontier_size=previous_frontier_size,
                        previous_score_margin=previous_score_margin,
                    )
                )
            reachable_targets = {
                candidate.target_passage_id for candidate in candidates
            }
            missing_support = support - set(evidence)
            rescuable = int(bool(missing_support & reachable_targets))
            if step == 0:
                preflight_labels.append(rescuable)
            elif previous is not None:
                continue_labels.append(rescuable)
            if not candidates:
                break

            scores = score_paths(
                example,
                evidence,
                candidates,
                ranking,
                extractor,
                mrv_model,
                passages,
                config,
            )
            if not scores:
                break
            selected = max(
                scores,
                key=lambda item: (
                    item.marginal_value,
                    item.p_add_support,
                    item.path.confidence,
                    item.path.id,
                ),
            )
            if selected.marginal_value <= 0.0:
                break
            ordered = sorted(
                (item.marginal_value for item in scores), reverse=True
            )
            previous_score_margin = float(
                ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
            )
            previous = selected
            previous_frontier_size = len(candidates)
            evidence.append(selected.path.target_passage_id)
    if not preflight_rows:
        raise RuntimeError("No gate training states were produced")
    return GateTrainingRows(
        preflight_x=np.vstack(preflight_rows),
        preflight_y=np.asarray(preflight_labels, dtype=np.int64),
        continue_x=(
            np.vstack(continue_rows)
            if continue_rows
            else np.empty((0, len(CONTINUE_GATE_FEATURE_NAMES)))
        ),
        continue_y=np.asarray(continue_labels, dtype=np.int64),
    )


def train_models(
    examples: Sequence[QueryExample],
    retriever: HybridRetriever,
    graph: KnowledgeGraph,
    passages: dict[str, Passage],
    extractor: CandidateFeatureExtractor,
    config: ExperimentConfig,
) -> tuple[MRVModel, GateModel, GateModel, TrainingSummary]:
    train_examples, calibration_examples = split_for_calibration(
        examples,
        config.learning.calibration_fraction,
        config.learning.random_seed,
    )
    counterfactual_labels = load_counterfactual_labels(
        config.learning.counterfactual_labels_path
    )
    (
        mrv_x,
        add_y,
        complete_y,
        reader_gain_y,
        harmful_y,
        reader_supervision,
    ) = candidate_training_rows(
        train_examples,
        retriever,
        graph,
        passages,
        extractor,
        config,
        counterfactual_labels,
    )
    mrv_model = MRVModel(
        epochs=config.learning.epochs,
        learning_rate=config.learning.learning_rate,
        l2=config.learning.l2,
    ).fit(
        mrv_x,
        add_y,
        complete_y,
        harmful_y,
        CANDIDATE_FEATURE_NAMES,
        reader_gain_labels=reader_gain_y,
    )

    gate_train = gate_training_rows(
        train_examples,
        retriever,
        graph,
        passages,
        extractor,
        mrv_model,
        config,
    )
    gate_calibration = gate_training_rows(
        calibration_examples,
        retriever,
        graph,
        passages,
        extractor,
        mrv_model,
        config,
    )
    preflight_gate_model = GateModel(
        epochs=config.learning.epochs,
        learning_rate=config.learning.learning_rate,
        l2=config.learning.l2,
        calibration_method=config.learning.gate_calibration_method,
        calibration_folds=config.learning.gate_calibration_folds,
        calibration_seed=config.learning.random_seed,
        threshold_bootstrap_samples=(
            config.learning.gate_threshold_bootstrap_samples
        ),
        threshold_quantile=config.learning.gate_threshold_quantile,
    ).fit(
        gate_train.preflight_x,
        gate_train.preflight_y,
        gate_calibration.preflight_x,
        gate_calibration.preflight_y,
        GATE_FEATURE_NAMES,
        target_recall=config.learning.gate_target_recall,
    )

    action_feature_count = len(CONTINUE_GATE_FEATURE_NAMES) - len(GATE_FEATURE_NAMES)

    def continuation_or_fallback(
        rows: GateTrainingRows,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(rows.continue_y):
            return rows.continue_x, rows.continue_y
        zero_actions = np.zeros((len(rows.preflight_x), action_feature_count))
        return (
            np.hstack((rows.preflight_x, zero_actions)),
            rows.preflight_y.copy(),
        )

    continue_train_x, continue_train_y = continuation_or_fallback(gate_train)
    continue_calibration_x, continue_calibration_y = continuation_or_fallback(
        gate_calibration
    )
    continue_gate_model = GateModel(
        epochs=config.learning.epochs,
        learning_rate=config.learning.learning_rate,
        l2=config.learning.l2,
        calibration_method=config.learning.gate_calibration_method,
        calibration_folds=config.learning.gate_calibration_folds,
        calibration_seed=config.learning.random_seed + 1,
        threshold_bootstrap_samples=(
            config.learning.gate_threshold_bootstrap_samples
        ),
        threshold_quantile=config.learning.gate_threshold_quantile,
    ).fit(
        continue_train_x,
        continue_train_y,
        continue_calibration_x,
        continue_calibration_y,
        CONTINUE_GATE_FEATURE_NAMES,
        target_recall=config.learning.gate_target_recall,
    )

    model_dir = Path(config.model_dir)
    mrv_path = model_dir / "mrv_model.json"
    gate_path = model_dir / "preflight_gate_model.json"
    continue_gate_path = model_dir / "continue_gate_model.json"
    mrv_model.save(mrv_path)
    preflight_gate_model.save(gate_path)
    continue_gate_model.save(continue_gate_path)
    summary = TrainingSummary(
        candidate_examples=len(mrv_x),
        positive_add_support=int(np.sum(add_y)),
        positive_complete=int(np.sum(complete_y)),
        harmful_examples=int(np.sum(harmful_y)),
        reader_supervised_examples=int(np.sum(reader_supervision)),
        positive_reader_gain=int(
            np.sum(reader_gain_y[reader_supervision.astype(bool)])
        ),
        gate_states=(
            len(gate_train.preflight_y)
            + len(gate_calibration.preflight_y)
            + len(gate_train.continue_y)
            + len(gate_calibration.continue_y)
        ),
        positive_gate_states=int(
            np.sum(gate_train.preflight_y)
            + np.sum(gate_calibration.preflight_y)
            + np.sum(gate_train.continue_y)
            + np.sum(gate_calibration.continue_y)
        ),
        gate_calibration_method=(
            preflight_gate_model.selected_calibration_method
        ),
        gate_threshold=preflight_gate_model.threshold,
        preflight_gate_states=(
            len(gate_train.preflight_y) + len(gate_calibration.preflight_y)
        ),
        positive_preflight_gate_states=int(
            np.sum(gate_train.preflight_y)
            + np.sum(gate_calibration.preflight_y)
        ),
        continue_gate_states=(
            len(gate_train.continue_y) + len(gate_calibration.continue_y)
        ),
        positive_continue_gate_states=int(
            np.sum(gate_train.continue_y)
            + np.sum(gate_calibration.continue_y)
        ),
        continue_gate_calibration_method=(
            continue_gate_model.selected_calibration_method
        ),
        continue_gate_threshold=continue_gate_model.threshold,
        mrv_model_path=str(mrv_path),
        gate_model_path=str(gate_path),
        continue_gate_model_path=str(continue_gate_path),
    )
    return mrv_model, preflight_gate_model, continue_gate_model, summary
