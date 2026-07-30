from __future__ import annotations
import math
import re
from collections import Counter
from typing import Iterable

TOKEN_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def overlap_ratio(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a))


def cosine(left, right) -> float:
    import numpy as np

    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def answer_f1(prediction: str, gold: str) -> float:
    pred = Counter(tokenize(prediction))
    target = Counter(tokenize(gold))
    common = sum((pred & target).values())
    if not pred or not target:
        return float(pred == target)
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(target.values())
    return 2 * precision * recall / (precision + recall)
