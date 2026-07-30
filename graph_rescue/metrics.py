from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from .models import QueryExample, RetrievalTrace
from .official_metrics import best_answer_scores


def retrieval_metrics(
    example: QueryExample,
    trace: RetrievalTrace,
    predicted_answer: str,
) -> dict[str, float]:
    support = set(example.supporting_passage_ids)
    seeds = set(trace.seed_passage_ids)
    final_ids = list(trace.final_passage_ids)
    final = set(final_ids)
    seed_hits = len(support & seeds)
    final_hits = len(support & final)
    full_seed = float(bool(support) and support.issubset(seeds))
    full_final = float(bool(support) and support.issubset(final))
    selected_graph = {
        action.selected_passage_id
        for action in trace.actions
        if action.selected_passage_id is not None
    }
    harmful = len(selected_graph - support)
    answer_exact = max(
        (best_answer_scores(predicted_answer, example.answers)["em"],),
        default=0.0,
    )
    official_answer = best_answer_scores(predicted_answer, example.answers)
    metrics = {
        "support_recall": final_hits / max(1, len(support)),
        "full_evidence": full_final,
        "seed_support_recall": seed_hits / max(1, len(support)),
        "seed_full_evidence": full_seed,
        "rescued_support_count": float(max(0, final_hits - seed_hits)),
        "seed_to_context_rescue": float(full_final > full_seed),
        "harmful_expansions": float(harmful),
        "graph_actions": float(
            sum(action.selected_passage_id is not None for action in trace.actions)
        ),
        "latency_ms": trace.latency_ms,
        "graph_reads": float(trace.graph_reads),
        "candidate_paths_scored": float(trace.candidate_paths_scored),
        "evidence_tokens": float(trace.evidence_tokens),
        "answer_em": answer_exact,
        "answer_f1": official_answer["f1"],
        "answer_precision": official_answer["precision"],
        "answer_recall": official_answer["recall"],
    }
    for cutoff in (1, 2, 5, 10):
        selected = final_ids[:cutoff]
        hits = len(support & set(selected))
        metrics[f"support_recall_at_{cutoff}"] = hits / max(1, len(support))
        metrics[f"support_precision_at_{cutoff}"] = hits / max(1, len(selected))
        metrics[f"full_evidence_at_{cutoff}"] = float(
            bool(support) and support.issubset(selected)
        )
    first_relevant_rank = next(
        (
            rank
            for rank, passage_id in enumerate(final_ids, start=1)
            if passage_id in support
        ),
        None,
    )
    metrics["support_mrr"] = (
        1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
    )
    ideal_relevant = min(len(support), len(final_ids))
    dcg = sum(
        (1.0 if passage_id in support else 0.0) / np.log2(rank + 1.0)
        for rank, passage_id in enumerate(final_ids, start=1)
    )
    ideal_dcg = sum(
        1.0 / np.log2(rank + 1.0)
        for rank in range(1, ideal_relevant + 1)
    )
    metrics["support_ndcg"] = float(dcg / ideal_dcg) if ideal_dcg else 0.0
    return metrics


def aggregate_rows(rows: Iterable[dict]) -> dict[str, dict[str, float]]:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    result: dict[str, dict[str, float]] = {}
    for policy, policy_rows in grouped.items():
        metric_names = sorted(policy_rows[0]["metrics"])
        result[policy] = {
            name: float(np.mean([row["metrics"][name] for row in policy_rows]))
            for name in metric_names
        }
        for name in (
            "retrieval_latency_ms",
            "policy_latency_ms",
            "reader_latency_ms",
            "total_latency_ms",
        ):
            if name not in metric_names:
                continue
            values = np.asarray(
                [row["metrics"][name] for row in policy_rows],
                dtype=np.float64,
            )
            result[policy][f"{name}_p50"] = float(np.quantile(values, 0.50))
            result[policy][f"{name}_p95"] = float(np.quantile(values, 0.95))
        result[policy]["queries"] = float(len(policy_rows))
    return result


def binary_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int],
    bins: int = 10,
    threshold: float = 0.5,
) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if not len(p):
        return {
            "auroc": 0.0,
            "auprc": 0.0,
            "brier": 0.0,
            "ece": 0.0,
            "threshold": threshold,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "predicted_positive_rate": 0.0,
        }
    predicted = p >= threshold
    positives = y == 1
    negatives = y == 0
    true_positive = int(np.sum(predicted & positives))
    false_positive = int(np.sum(predicted & negatives))
    true_negative = int(np.sum((~predicted) & negatives))
    false_negative = int(np.sum((~predicted) & positives))
    return {
        "auroc": roc_auc(p, y),
        "auprc": average_precision(p, y),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(p, y, bins=bins),
        "threshold": float(threshold),
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
        "specificity": true_negative / max(1, true_negative + false_positive),
        "predicted_positive_rate": float(np.mean(predicted)),
    }


def roc_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_values = probabilities[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def average_precision(probabilities: np.ndarray, labels: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-probabilities)
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= boundaries[index]) & (
                probabilities <= boundaries[index + 1]
            )
        else:
            mask = (probabilities >= boundaries[index]) & (
                probabilities < boundaries[index + 1]
            )
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
            )
    return error


def calibration_bins(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> list[dict[str, float]]:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= boundaries[index]) & (p <= boundaries[index + 1])
        else:
            mask = (p >= boundaries[index]) & (p < boundaries[index + 1])
        count = int(np.sum(mask))
        result.append(
            {
                "lower": float(boundaries[index]),
                "upper": float(boundaries[index + 1]),
                "count": float(count),
                "mean_probability": (
                    float(np.mean(p[mask])) if count else 0.0
                ),
                "empirical_rate": (
                    float(np.mean(y[mask])) if count else 0.0
                ),
            }
        )
    return result


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if len(a) != len(b) or not len(a):
        raise ValueError("Paired bootstrap requires equal non-empty arrays")
    differences = a - b
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(differences), len(differences))
        estimates[index] = float(np.mean(differences[chosen]))
    probability_non_positive = float(np.mean(estimates <= 0.0))
    probability_non_negative = float(np.mean(estimates >= 0.0))
    return {
        "difference": float(np.mean(differences)),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "p_value_two_sided": min(
            1.0,
            2.0
            * min(probability_non_positive, probability_non_negative),
        ),
    }


def holm_bonferroni(
    p_values: dict[str, float],
) -> dict[str, float]:
    """Return family-wise-error-controlled Holm adjusted p-values."""

    ordered = sorted(
        p_values.items(),
        key=lambda item: (float(item[1]), item[0]),
    )
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * float(p_value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def factorial_interaction(
    relevance_always: Sequence[float],
    mrv_always: Sequence[float],
    relevance_gated: Sequence[float],
    mrv_gated: Sequence[float],
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    ra = np.asarray(relevance_always, dtype=np.float64)
    ma = np.asarray(mrv_always, dtype=np.float64)
    rg = np.asarray(relevance_gated, dtype=np.float64)
    mg = np.asarray(mrv_gated, dtype=np.float64)
    if len({len(ra), len(ma), len(rg), len(mg)}) != 1 or not len(ra):
        raise ValueError("Factorial interaction requires equal non-empty arrays")
    per_query = (mg - ma) - (rg - ra)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        chosen = rng.integers(0, len(per_query), len(per_query))
        estimates[index] = float(np.mean(per_query[chosen]))
    return {
        "interaction": float(np.mean(per_query)),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }
