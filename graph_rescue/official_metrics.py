from __future__ import annotations

from collections import Counter
import re
import string
from typing import Iterable, Sequence


YES_NO = {"yes", "no", "noanswer"}


def normalize_answer(value: str) -> str:
    """HotpotQA-compatible answer normalization."""

    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def answer_scores(prediction: str, gold: str) -> dict[str, float]:
    predicted = normalize_answer(prediction)
    target = normalize_answer(gold)
    exact = float(predicted == target)
    if (
        (predicted in YES_NO or target in YES_NO)
        and predicted != target
    ):
        return {"em": exact, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    predicted_tokens = predicted.split()
    target_tokens = target.split()
    common = Counter(predicted_tokens) & Counter(target_tokens)
    same = sum(common.values())
    if same == 0:
        return {"em": exact, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    precision = same / max(1, len(predicted_tokens))
    recall = same / max(1, len(target_tokens))
    f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "em": exact,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def best_answer_scores(
    prediction: str, gold_answers: Sequence[str]
) -> dict[str, float]:
    if not gold_answers:
        return answer_scores(prediction, "")
    candidates = [answer_scores(prediction, gold) for gold in gold_answers]
    return max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["em"],
            item["precision"],
            item["recall"],
        ),
    )


def _normalized_support_fact(item: Sequence[object]) -> tuple[str, int]:
    return (str(item[0]).casefold().strip(), int(item[1]))


def support_fact_scores(
    predicted: Iterable[Sequence[object]],
    gold: Iterable[Sequence[object]],
) -> dict[str, float]:
    predicted_set = {_normalized_support_fact(item) for item in predicted}
    gold_set = {_normalized_support_fact(item) for item in gold}
    true_positive = len(predicted_set & gold_set)
    false_positive = len(predicted_set - gold_set)
    false_negative = len(gold_set - predicted_set)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "em": float(false_positive + false_negative == 0),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def _normalized_triple(item: Sequence[object]) -> tuple[str, str, str]:
    return tuple(normalize_answer(str(value)) for value in item[:3])  # type: ignore[return-value]


def evidence_scores(
    predicted: Iterable[Sequence[object]],
    gold: Iterable[Sequence[object]],
) -> dict[str, float]:
    predicted_set = {_normalized_triple(item) for item in predicted}
    gold_set = {_normalized_triple(item) for item in gold}
    true_positive = len(predicted_set & gold_set)
    false_positive = len(predicted_set - gold_set)
    false_negative = len(gold_set - predicted_set)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "em": float(false_positive + false_negative == 0),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def joint_scores(
    answer: dict[str, float],
    support: dict[str, float],
) -> dict[str, float]:
    precision = answer["precision"] * support["precision"]
    recall = answer["recall"] * support["recall"]
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "em": answer["em"] * support["em"],
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
