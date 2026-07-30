from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .graph import KnowledgeGraph, passage_node
from .hybrid import HybridRetriever
from .models import CandidatePath, CandidateScore, Passage, RetrievedPassage
from .text import estimate_tokens, overlap_ratio, tokenize


CANDIDATE_FEATURE_NAMES = [
    "query_overlap",
    "dense_similarity",
    "base_rerank_score",
    "seed_rerank_score",
    "evidence_redundancy",
    "new_query_term_coverage",
    "path_confidence",
    "inverse_hubness",
    "one_over_hops",
    "token_cost",
    "entity_edge_fraction",
]

GATE_FEATURE_NAMES = [
    "top_seed_score",
    "seed_score_margin",
    "seed_score_mean",
    "seed_score_std",
    "cutoff_seed_score",
    "cutoff_next_margin",
    "dense_cutoff_margin",
    "bm25_cutoff_margin",
    "rrf_cutoff_margin",
    "ranking_score_entropy",
    "question_seed_overlap",
    "question_token_count_log",
    "comparison_cue",
    "evidence_size_log",
    "seed_degree_mean",
    "seed_degree_max",
    "seed_connected_fraction",
    "neighbor_degree_mean",
    "neighbor_confidence_mean",
    "evidence_token_fraction",
]

CONTINUE_GATE_FEATURE_NAMES = GATE_FEATURE_NAMES + [
    "previous_marginal_value",
    "previous_p_add_support",
    "previous_p_complete",
    "previous_p_reader_gain",
    "previous_p_harmful",
    "previous_relevance",
    "previous_path_confidence",
    "previous_inverse_hubness",
    "previous_one_over_hops",
    "previous_frontier_size_log",
    "previous_score_margin",
]


@dataclass
class CandidateFeatureExtractor:
    passages: dict[str, Passage]
    retriever: HybridRetriever
    token_budget: int

    def vector(
        self,
        query: str,
        evidence_ids: Sequence[str],
        path: CandidatePath,
        ranking: Sequence[RetrievedPassage],
    ) -> np.ndarray:
        target = self.passages[path.target_passage_id]
        query_tokens = tokenize(query)
        target_tokens = tokenize(target.full_text)
        evidence_tokens: set[str] = set()
        for passage_id in evidence_ids:
            evidence_tokens.update(tokenize(self.passages[passage_id].full_text))
        ranking_by_id = {item.passage_id: item for item in ranking}
        target_rank = ranking_by_id.get(path.target_passage_id)
        seed_rank = ranking_by_id.get(path.seed_passage_id)
        entity_edges = sum(kind == "mentions" for kind in path.edge_kinds)

        return np.asarray(
            [
                overlap_ratio(query_tokens, target_tokens),
                self.retriever.dense_similarity(query, target.id),
                target_rank.rerank_score if target_rank else 0.0,
                seed_rank.rerank_score if seed_rank else 0.0,
                overlap_ratio(target_tokens, evidence_tokens),
                len((set(target_tokens) & set(query_tokens)) - evidence_tokens)
                / max(1, len(set(query_tokens))),
                path.confidence,
                1.0 / max(1, path.max_hubness),
                1.0 / max(1, path.hop_count),
                estimate_tokens(target.full_text) / max(1, self.token_budget),
                entity_edges / max(1, len(path.edge_kinds)),
            ],
            dtype=np.float64,
        )

    def relevance(self, vector: np.ndarray) -> float:
        values = dict(zip(CANDIDATE_FEATURE_NAMES, vector))
        return (
            0.40 * values["dense_similarity"]
            + 0.25 * values["query_overlap"]
            + 0.20 * values["base_rerank_score"]
            + 0.10 * values["path_confidence"]
            + 0.05 * values["inverse_hubness"]
        )


def gate_vector(
    query: str,
    ranking: Sequence[RetrievedPassage],
    evidence_ids: Sequence[str],
    passages: dict[str, Passage],
    graph: KnowledgeGraph,
    *,
    seed_k: int,
    evidence_tokens: int,
    token_budget: int,
) -> np.ndarray:
    seeds = list(ranking[:seed_k])
    seed_scores = np.asarray(
        [item.rerank_score for item in seeds], dtype=np.float64
    )
    seed_degrees = np.asarray(
        [graph.degree(passage_node(item)) for item in evidence_ids],
        dtype=np.float64,
    )
    neighbor_degrees = np.asarray(
        [
            graph.degree(edge.target)
            for passage_id in evidence_ids
            for edge in graph.neighbors(passage_node(passage_id))
        ],
        dtype=np.float64,
    )
    neighbor_confidences = np.asarray(
        [
            edge.confidence
            for passage_id in evidence_ids
            for edge in graph.neighbors(passage_node(passage_id))
        ],
        dtype=np.float64,
    )
    query_tokens = tokenize(query)
    evidence_text_tokens: set[str] = set()
    for passage_id in evidence_ids:
        evidence_text_tokens.update(tokenize(passages[passage_id].full_text))

    def maximum(values: np.ndarray) -> float:
        return float(np.max(values)) if len(values) else 0.0

    def mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if len(values) else 0.0

    def std(values: np.ndarray) -> float:
        return float(np.std(values)) if len(values) else 0.0

    cutoff = seeds[-1] if seeds else None
    following = ranking[seed_k] if len(ranking) > seed_k else None
    cutoff_score = float(cutoff.rerank_score) if cutoff else 0.0
    next_score = float(following.rerank_score) if following else 0.0
    dense_margin = (
        float(cutoff.dense_score - following.dense_score)
        if cutoff and following
        else 0.0
    )
    bm25_margin = (
        float(cutoff.bm25_score - following.bm25_score)
        if cutoff and following
        else 0.0
    )
    rrf_margin = (
        float(cutoff.rrf_score - following.rrf_score)
        if cutoff and following
        else 0.0
    )
    top_scores = np.asarray(
        [max(0.0, item.rerank_score) for item in ranking[:10]],
        dtype=np.float64,
    )
    if len(top_scores) and float(np.sum(top_scores)) > 0.0:
        distribution = top_scores / np.sum(top_scores)
        entropy = -float(
            np.sum(distribution * np.log(np.maximum(distribution, 1e-12)))
        ) / max(1e-12, math.log(max(2, len(distribution))))
    else:
        entropy = 0.0
    comparison_terms = {
        "both",
        "compare",
        "compared",
        "earlier",
        "later",
        "older",
        "younger",
        "more",
        "less",
        "same",
        "which",
    }
    comparison_cue = float(bool(set(query_tokens) & comparison_terms))

    return np.asarray(
        [
            maximum(seed_scores),
            maximum(seed_scores) - (float(seed_scores[-1]) if len(seed_scores) else 0.0),
            mean(seed_scores),
            std(seed_scores),
            cutoff_score,
            cutoff_score - next_score,
            dense_margin,
            bm25_margin,
            rrf_margin,
            entropy,
            overlap_ratio(query_tokens, evidence_text_tokens),
            math.log1p(len(query_tokens)),
            comparison_cue,
            math.log1p(len(evidence_ids)),
            mean(seed_degrees),
            maximum(seed_degrees),
            float(np.mean(seed_degrees > 0.0)) if len(seed_degrees) else 0.0,
            mean(neighbor_degrees),
            mean(neighbor_confidences),
            evidence_tokens / max(1, token_budget),
        ],
        dtype=np.float64,
    )


def continue_gate_vector(
    query: str,
    ranking: Sequence[RetrievedPassage],
    evidence_ids: Sequence[str],
    passages: dict[str, Passage],
    graph: KnowledgeGraph,
    *,
    seed_k: int,
    evidence_tokens: int,
    token_budget: int,
    previous: CandidateScore,
    previous_frontier_size: int,
    previous_score_margin: float,
) -> np.ndarray:
    """Features available after one graph action but before the next traversal."""
    base = gate_vector(
        query,
        ranking,
        evidence_ids,
        passages,
        graph,
        seed_k=seed_k,
        evidence_tokens=evidence_tokens,
        token_budget=token_budget,
    )
    action = np.asarray(
        [
            previous.marginal_value,
            previous.p_add_support,
            previous.p_complete,
            previous.p_reader_gain,
            previous.p_harmful,
            previous.relevance,
            previous.path.confidence,
            1.0 / max(1, previous.path.max_hubness),
            1.0 / max(1, previous.path.hop_count),
            math.log1p(previous_frontier_size),
            previous_score_margin,
        ],
        dtype=np.float64,
    )
    return np.concatenate((base, action))
