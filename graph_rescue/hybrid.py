from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .models import Passage, RetrievedPassage
from .ollama import HashingEmbedder, OllamaClient
from .text import overlap_ratio, tokenize


class TextEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class OllamaEmbedder:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.client.embed(self.model, texts)


class CachedEmbedder:
    def __init__(self, delegate: TextEmbedder, cache_dir: str | Path, namespace: str):
        self.delegate = delegate
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace.replace(":", "_").replace("/", "_")

    def _path(self, key: str) -> Path:
        import hashlib

        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{self.namespace}_{digest}.json"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for index, text in enumerate(texts):
            path = self._path(text)
            if path.exists():
                rows[index] = np.asarray(
                    json.loads(path.read_text(encoding="utf-8")), dtype=np.float32
                )
            else:
                missing_indices.append(index)
                missing_texts.append(text)

        batch_size = 32
        for start in range(0, len(missing_texts), batch_size):
            batch_texts = missing_texts[start : start + batch_size]
            batch_vectors = self.delegate.embed(batch_texts)
            for local_index, vector in enumerate(batch_vectors):
                original_index = missing_indices[start + local_index]
                rows[original_index] = vector
                self._path(texts[original_index]).write_text(
                    json.dumps(vector.tolist()), encoding="utf-8"
                )
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(rows)


class BM25Index:
    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.passages = list(passages)
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(item.full_text) for item in passages]
        self.term_frequencies = [Counter(tokens) for tokens in self.documents]
        self.document_lengths = np.asarray(
            [len(tokens) for tokens in self.documents], dtype=np.float32
        )
        self.avg_document_length = float(np.mean(self.document_lengths)) or 1.0
        document_frequency: Counter[str] = Counter()
        for tokens in self.documents:
            document_frequency.update(set(tokens))
        n = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_terms = tokenize(query)
        values = np.zeros(len(self.passages), dtype=np.float32)
        for index, frequencies in enumerate(self.term_frequencies):
            length_norm = self.k1 * (
                1.0
                - self.b
                + self.b * self.document_lengths[index] / self.avg_document_length
            )
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1.0) / (frequency + length_norm)
                    )
            values[index] = score
        return values


class HybridRetriever:
    def __init__(
        self,
        passages: Sequence[Passage],
        embedder: TextEmbedder,
        *,
        bm25_k: int = 100,
        dense_k: int = 100,
        rrf_k: int = 60,
        rerank_k: int = 50,
    ):
        self.passages = list(passages)
        self.passage_by_id = {item.id: item for item in passages}
        self.index_by_id = {item.id: index for index, item in enumerate(passages)}
        self.embedder = embedder
        self.bm25 = BM25Index(passages)
        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.rrf_k = rrf_k
        self.rerank_k = rerank_k
        self.passage_embeddings = embedder.embed(
            [f"passage: {item.full_text}" for item in passages]
        )
        self._query_embedding_cache: dict[str, np.ndarray] = {}

    def query_embedding(self, query: str) -> np.ndarray:
        if query not in self._query_embedding_cache:
            self._query_embedding_cache[query] = self.embedder.embed(
                [f"query: {query}"]
            )[0]
        return self._query_embedding_cache[query]

    @staticmethod
    def _top_indices(scores: np.ndarray, k: int) -> list[int]:
        if not len(scores):
            return []
        k = min(k, len(scores))
        indices = np.argpartition(-scores, k - 1)[:k]
        return sorted(indices.tolist(), key=lambda index: (-scores[index], index))

    @staticmethod
    def _minmax(values: list[float]) -> list[float]:
        if not values:
            return []
        low, high = min(values), max(values)
        if high <= low:
            return [0.0 for _ in values]
        return [(value - low) / (high - low) for value in values]

    def retrieve(self, query: str, k: int = 50) -> list[RetrievedPassage]:
        return self.retrieve_mode(query, k=k, mode="hybrid")

    def retrieve_mode(
        self,
        query: str,
        *,
        k: int = 50,
        mode: str = "hybrid",
    ) -> list[RetrievedPassage]:
        if mode not in {"bm25", "dense", "rrf_fusion", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        needs_bm25 = mode != "dense"
        needs_dense = mode != "bm25"
        bm25_scores = (
            self.bm25.scores(query)
            if needs_bm25
            else np.zeros(len(self.passages), dtype=np.float32)
        )
        dense_scores = (
            self.passage_embeddings @ self.query_embedding(query)
            if needs_dense
            else np.zeros(len(self.passages), dtype=np.float32)
        )
        bm25_order = (
            self._top_indices(bm25_scores, self.bm25_k)
            if needs_bm25
            else []
        )
        dense_order = (
            self._top_indices(dense_scores, self.dense_k)
            if needs_dense
            else []
        )

        if mode in {"bm25", "dense"}:
            order = bm25_order if mode == "bm25" else dense_order
            items = [
                RetrievedPassage(
                    passage_id=self.passages[index].id,
                    bm25_score=float(bm25_scores[index]),
                    dense_score=float(dense_scores[index]),
                    rerank_score=float(
                        bm25_scores[index]
                        if mode == "bm25"
                        else dense_scores[index]
                    ),
                )
                for index in order[:k]
            ]
            for rank, item in enumerate(items, start=1):
                item.rank = rank
            return items

        fused: defaultdict[int, float] = defaultdict(float)
        for rank, index in enumerate(bm25_order, start=1):
            fused[index] += 1.0 / (self.rrf_k + rank)
        for rank, index in enumerate(dense_order, start=1):
            fused[index] += 1.0 / (self.rrf_k + rank)

        candidates = sorted(
            fused, key=lambda index: (-fused[index], self.passages[index].id)
        )[: self.rerank_k]
        if mode == "rrf_fusion":
            items = [
                RetrievedPassage(
                    passage_id=self.passages[index].id,
                    bm25_score=float(bm25_scores[index]),
                    dense_score=float(dense_scores[index]),
                    rrf_score=float(fused[index]),
                    rerank_score=float(fused[index]),
                )
                for index in candidates[:k]
            ]
            for rank, item in enumerate(items, start=1):
                item.rank = rank
            return items
        rrf_values = self._minmax([fused[index] for index in candidates])
        dense_values = self._minmax([float(dense_scores[index]) for index in candidates])
        lexical_values = [
            overlap_ratio(tokenize(query), tokenize(self.passages[index].full_text))
            for index in candidates
        ]

        items: list[RetrievedPassage] = []
        for index, rrf_value, dense_value, lexical_value in zip(
            candidates, rrf_values, dense_values, lexical_values
        ):
            rerank_score = (
                0.50 * rrf_value + 0.30 * dense_value + 0.20 * lexical_value
            )
            items.append(
                RetrievedPassage(
                    passage_id=self.passages[index].id,
                    bm25_score=float(bm25_scores[index]),
                    dense_score=float(dense_scores[index]),
                    rrf_score=float(fused[index]),
                    rerank_score=rerank_score,
                )
            )
        items.sort(key=lambda item: (-item.rerank_score, item.passage_id))
        for rank, item in enumerate(items, start=1):
            item.rank = rank
        return items[:k]

    def dense_similarity(self, query: str, passage_id: str) -> float:
        return float(
            self.passage_embeddings[self.index_by_id[passage_id]]
            @ self.query_embedding(query)
        )


def make_embedder(
    *,
    base_url: str,
    model: str,
    timeout_seconds: int,
    cache_dir: str | Path,
    allow_hashing_fallback: bool = False,
) -> TextEmbedder:
    client = OllamaClient(base_url=base_url, timeout_seconds=timeout_seconds)
    try:
        models = client.models()
        if model not in models:
            raise RuntimeError(
                f"Embedding model {model!r} is not installed. Available: {models}"
            )
        delegate: TextEmbedder = OllamaEmbedder(client, model)
    except Exception:
        if not allow_hashing_fallback:
            raise
        delegate = HashingEmbedder()
        model = "hashing-test-only"
    return CachedEmbedder(delegate, cache_dir, model)
