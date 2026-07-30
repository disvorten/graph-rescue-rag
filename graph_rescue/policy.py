from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .features import CandidateFeatureExtractor, continue_gate_vector, gate_vector
from .graph import KnowledgeGraph
from .learning import GateModel, MRVModel
from .models import (
    CandidatePath,
    CandidateScore,
    Passage,
    QueryExample,
    RetrievalAction,
    RetrievedPassage,
    RetrievalTrace,
)
from .text import estimate_tokens


@dataclass
class PolicyConfig:
    seed_k: int = 5
    final_k: int = 10
    token_budget: int = 2048
    max_hops: int = 2
    frontier_cap: int = 200
    max_actions: int = 5
    token_cost_weight: float = 0.08
    hop_cost_weight: float = 0.05
    noise_cost_weight: float = 0.10
    reader_gain_weight: float = 0.50


class RescuePolicy:
    def __init__(
        self,
        *,
        name: str,
        passages: dict[str, Passage],
        graph: KnowledgeGraph,
        feature_extractor: CandidateFeatureExtractor,
        config: PolicyConfig,
        selector: str,
        mrv_model: MRVModel | None = None,
        preflight_gate_model: GateModel | None = None,
        continue_gate_model: GateModel | None = None,
    ):
        if selector not in {"none", "relevance", "mrv", "oracle"}:
            raise ValueError(f"Unknown selector: {selector}")
        if selector == "mrv" and mrv_model is None:
            raise ValueError("MRV selector requires an MRV model")
        self.name = name
        self.passages = passages
        self.graph = graph
        self.feature_extractor = feature_extractor
        self.config = config
        self.selector = selector
        self.mrv_model = mrv_model
        self.preflight_gate_model = preflight_gate_model
        self.continue_gate_model = continue_gate_model

    def _candidate_scores(
        self,
        example: QueryExample,
        evidence_ids: Sequence[str],
        candidates: Sequence[CandidatePath],
        ranking: Sequence[RetrievedPassage],
    ) -> list[CandidateScore]:
        if not candidates:
            return []
        matrix = np.vstack(
            [
                self.feature_extractor.vector(
                    example.question, evidence_ids, candidate, ranking
                )
                for candidate in candidates
            ]
        )
        relevance = np.asarray(
            [self.feature_extractor.relevance(row) for row in matrix]
        )
        if self.mrv_model:
            p_add, p_complete, p_reader_gain, p_harmful = (
                self.mrv_model.predict(matrix)
            )
        else:
            p_add = p_complete = np.zeros(len(candidates))
            p_reader_gain = np.zeros(len(candidates))
            p_harmful = np.zeros(len(candidates))

        scores: list[CandidateScore] = []
        for index, (candidate, vector) in enumerate(zip(candidates, matrix)):
            feature_map = dict(
                zip(self.mrv_model.feature_names if self.mrv_model else [], vector)
            )
            token_cost = estimate_tokens(
                self.passages[candidate.target_passage_id].full_text
            ) / max(1, self.config.token_budget)
            marginal = (
                float(p_complete[index])
                + 0.35 * float(p_add[index])
                + self.config.reader_gain_weight
                * float(p_reader_gain[index])
                - 0.25 * float(p_harmful[index])
                - self.config.token_cost_weight * token_cost
                - self.config.hop_cost_weight * candidate.hop_count
                - self.config.noise_cost_weight
                * min(1.0, candidate.max_hubness / 100.0)
            )
            scores.append(
                CandidateScore(
                    path=candidate,
                    relevance=float(relevance[index]),
                    p_add_support=float(p_add[index]),
                    p_complete=float(p_complete[index]),
                    p_reader_gain=float(p_reader_gain[index]),
                    p_harmful=float(p_harmful[index]),
                    marginal_value=marginal,
                    features=feature_map,
                )
            )
        return scores

    def _fill_context(
        self,
        selected: list[str],
        ranking: Sequence[RetrievedPassage],
    ) -> tuple[list[str], int]:
        final: list[str] = []
        tokens = 0
        order = selected + [item.passage_id for item in ranking]
        for passage_id in order:
            if passage_id in final:
                continue
            cost = estimate_tokens(self.passages[passage_id].full_text)
            if final and tokens + cost > self.config.token_budget:
                continue
            final.append(passage_id)
            tokens += cost
            if len(final) >= self.config.final_k:
                break
        return final, tokens

    def run(
        self,
        example: QueryExample,
        ranking: Sequence[RetrievedPassage],
    ) -> RetrievalTrace:
        start_time = time.perf_counter()
        seeds = [item.passage_id for item in ranking[: self.config.seed_k]]
        evidence = list(seeds)
        selected_graph: list[str] = []
        actions: list[RetrievalAction] = []
        graph_reads = 0
        scored = 0
        previous_selected: CandidateScore | None = None
        previous_frontier_size = 0
        previous_score_margin = 0.0

        if self.selector == "none":
            final, tokens = self._fill_context([], ranking)
            return RetrievalTrace(
                query_id=example.id,
                policy=self.name,
                seed_passage_ids=seeds,
                final_passage_ids=final,
                actions=[],
                latency_ms=(time.perf_counter() - start_time) * 1000,
                graph_reads=0,
                candidate_paths_scored=0,
                evidence_tokens=tokens,
            )

        for step in range(self.config.max_actions):
            evidence_before = list(evidence)
            evidence_tokens = sum(
                estimate_tokens(self.passages[item].full_text) for item in evidence
            )
            probability: float | None = None
            gate_stage: str | None = None
            active_gate: GateModel | None = None
            if step == 0 and self.preflight_gate_model is not None:
                gate_stage = "preflight"
                active_gate = self.preflight_gate_model
                vector = gate_vector(
                    example.question,
                    ranking,
                    evidence,
                    self.passages,
                    self.graph,
                    seed_k=self.config.seed_k,
                    evidence_tokens=evidence_tokens,
                    token_budget=self.config.token_budget,
                )
            elif (
                step > 0
                and self.continue_gate_model is not None
                and previous_selected is not None
            ):
                gate_stage = "continue"
                active_gate = self.continue_gate_model
                vector = continue_gate_vector(
                    example.question,
                    ranking,
                    evidence,
                    self.passages,
                    self.graph,
                    seed_k=self.config.seed_k,
                    evidence_tokens=evidence_tokens,
                    token_budget=self.config.token_budget,
                    previous=previous_selected,
                    previous_frontier_size=previous_frontier_size,
                    previous_score_margin=previous_score_margin,
                )
            if active_gate is not None:
                probability = float(active_gate.predict_proba(vector)[0])
                if probability < active_gate.threshold:
                    actions.append(
                        RetrievalAction(
                            step=step,
                            selected_path_id=None,
                            selected_passage_id=None,
                            score=None,
                            gate_probability=probability,
                            frontier_size=0,
                            stop_reason="gate_rejected_before_traversal",
                            gate_stage=gate_stage,
                            evidence_ids_before=evidence_before,
                        )
                    )
                    break

            candidates, reads = self.graph.candidate_paths(
                evidence,
                excluded_passage_ids=evidence,
                max_hops=self.config.max_hops,
                cap=self.config.frontier_cap,
            )
            graph_reads += reads
            candidate_scores = self._candidate_scores(
                example, evidence, candidates, ranking
            )
            scored += len(candidate_scores)

            if not candidate_scores:
                actions.append(
                    RetrievalAction(
                        step=step,
                        selected_path_id=None,
                        selected_passage_id=None,
                        score=None,
                        gate_probability=probability,
                        frontier_size=0,
                        stop_reason="empty_frontier",
                        gate_stage=gate_stage,
                        evidence_ids_before=evidence_before,
                    )
                )
                break

            if self.selector == "oracle":
                missing_support = (
                    set(example.supporting_passage_ids) - set(evidence)
                )
                oracle_candidates = [
                    item
                    for item in candidate_scores
                    if item.path.target_passage_id in missing_support
                ]
                if not oracle_candidates:
                    actions.append(
                        RetrievalAction(
                            step=step,
                            selected_path_id=None,
                            selected_passage_id=None,
                            score=None,
                            gate_probability=probability,
                            frontier_size=len(candidates),
                            stop_reason="oracle_no_reachable_support",
                            gate_stage=gate_stage,
                            evidence_ids_before=evidence_before,
                        )
                    )
                    break
                selected = min(
                    oracle_candidates,
                    key=lambda item: (
                        item.path.hop_count,
                        -item.path.confidence,
                        item.path.id,
                    ),
                )
                score = 1.0
            elif self.selector == "mrv":
                selected = max(
                    candidate_scores,
                    key=lambda item: (
                        item.marginal_value,
                        item.p_add_support,
                        item.path.confidence,
                        item.path.id,
                    ),
                )
                score = selected.marginal_value
                if score <= 0.0:
                    actions.append(
                        RetrievalAction(
                            step=step,
                            selected_path_id=None,
                            selected_passage_id=None,
                            score=score,
                            gate_probability=probability,
                            frontier_size=len(candidates),
                            stop_reason="non_positive_mrv",
                            gate_stage=gate_stage,
                            evidence_ids_before=evidence_before,
                        )
                    )
                    break
            else:
                selected = max(
                    candidate_scores,
                    key=lambda item: (
                        item.relevance,
                        item.path.confidence,
                        item.path.id,
                    ),
                )
                score = selected.relevance

            ordered_scores = sorted(
                (
                    item.marginal_value
                    if self.selector == "mrv"
                    else (
                        1.0
                        if self.selector == "oracle"
                        and item.path.target_passage_id
                        in (
                            set(example.supporting_passage_ids)
                            - set(evidence)
                        )
                        else item.relevance
                    )
                    for item in candidate_scores
                ),
                reverse=True,
            )
            score_margin = ordered_scores[0] - (
                ordered_scores[1] if len(ordered_scores) > 1 else 0.0
            )

            target_id = selected.path.target_passage_id
            target_cost = estimate_tokens(self.passages[target_id].full_text)
            if evidence_tokens + target_cost > self.config.token_budget:
                actions.append(
                    RetrievalAction(
                        step=step,
                        selected_path_id=selected.path.id,
                        selected_passage_id=None,
                        score=score,
                        gate_probability=probability,
                        frontier_size=len(candidates),
                        stop_reason="token_budget",
                        gate_stage=gate_stage,
                        evidence_ids_before=evidence_before,
                    )
                )
                break
            evidence.append(target_id)
            selected_graph.append(target_id)
            actions.append(
                RetrievalAction(
                    step=step,
                    selected_path_id=selected.path.id,
                    selected_passage_id=target_id,
                    score=score,
                    gate_probability=probability,
                    frontier_size=len(candidates),
                    gate_stage=gate_stage,
                    evidence_ids_before=evidence_before,
                )
            )
            previous_selected = selected
            previous_frontier_size = len(candidates)
            previous_score_margin = float(score_margin)

        final, tokens = self._fill_context(seeds + selected_graph, ranking)
        return RetrievalTrace(
            query_id=example.id,
            policy=self.name,
            seed_passage_ids=seeds,
            final_passage_ids=final,
            actions=actions,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            graph_reads=graph_reads,
            candidate_paths_scored=scored,
            evidence_tokens=tokens,
        )


class KG2RAGStylePolicy:
    """Equal-budget adaptation of KG²RAG's expand-and-organize pattern.

    This is deliberately labelled ``style`` rather than a reproduction.  The
    original system expands entity/relation triplets and reranks chunk groups;
    this project has a passage/entity graph instead.  We preserve the
    transferable algorithmic ingredients without using learned rescue models:

    * expand the initial semantic seeds through the graph;
    * reward candidates supported by several seeds;
    * combine propagated seed confidence with query relevance;
    * organize the final context as seed-centred evidence groups.
    """

    def __init__(
        self,
        *,
        name: str = "kg2rag_style_equal_budget",
        passages: dict[str, Passage],
        graph: KnowledgeGraph,
        feature_extractor: CandidateFeatureExtractor,
        config: PolicyConfig,
    ):
        self.name = name
        self.passages = passages
        self.graph = graph
        self.feature_extractor = feature_extractor
        self.config = config

    def _expanded_candidates(
        self,
        seeds: Sequence[str],
    ) -> tuple[dict[str, list[CandidatePath]], int]:
        by_target: dict[str, list[CandidatePath]] = {}
        graph_reads = 0
        per_seed_cap = max(
            1,
            int(np.ceil(self.config.frontier_cap / max(1, len(seeds)))),
        )
        for seed_id in seeds:
            candidates, reads = self.graph.candidate_paths(
                [seed_id],
                excluded_passage_ids=seeds,
                max_hops=self.config.max_hops,
                cap=per_seed_cap,
            )
            graph_reads += reads
            for candidate in candidates:
                by_target.setdefault(candidate.target_passage_id, []).append(
                    candidate
                )
        if len(by_target) > self.config.frontier_cap:
            ordered_targets = sorted(
                by_target,
                key=lambda target_id: (
                    min(
                        path.hop_count
                        for path in by_target[target_id]
                    ),
                    -max(
                        path.confidence
                        for path in by_target[target_id]
                    ),
                    target_id,
                ),
            )[: self.config.frontier_cap]
            by_target = {
                target_id: by_target[target_id]
                for target_id in ordered_targets
            }
        return by_target, graph_reads

    def _score_candidates(
        self,
        example: QueryExample,
        seeds: Sequence[str],
        candidates_by_target: dict[str, list[CandidatePath]],
        ranking: Sequence[RetrievedPassage],
    ) -> list[tuple[float, CandidatePath, float]]:
        ranking_by_id = {item.passage_id: item for item in ranking}
        seed_scores = {
            seed_id: max(
                0.0,
                ranking_by_id.get(
                    seed_id, RetrievedPassage(seed_id)
                ).rerank_score,
            )
            for seed_id in seeds
        }
        scored: list[tuple[float, CandidatePath, float]] = []
        for target_id, paths in candidates_by_target.items():
            best_path = min(
                paths,
                key=lambda path: (
                    path.hop_count,
                    -path.confidence,
                    path.id,
                ),
            )
            vector = self.feature_extractor.vector(
                example.question,
                seeds,
                best_path,
                ranking,
            )
            relevance = self.feature_extractor.relevance(vector)
            propagated = max(
                seed_scores.get(path.seed_passage_id, 0.0)
                * path.confidence
                / max(1, path.hop_count)
                for path in paths
            )
            seed_coverage = len(
                {path.seed_passage_id for path in paths}
            ) / max(1, len(seeds))
            path_quality = best_path.confidence / max(
                1, best_path.hop_count
            )
            score = (
                0.50 * relevance
                + 0.25 * propagated
                + 0.15 * seed_coverage
                + 0.10 * path_quality
            )
            scored.append((float(score), best_path, float(relevance)))
        return sorted(
            scored,
            key=lambda item: (
                -item[0],
                item[1].hop_count,
                -item[1].confidence,
                item[1].target_passage_id,
            ),
        )

    def _organized_context(
        self,
        seeds: Sequence[str],
        selected: Sequence[tuple[float, CandidatePath, float]],
        ranking: Sequence[RetrievedPassage],
    ) -> tuple[list[str], int]:
        grouped: dict[str, list[tuple[float, str]]] = {
            seed_id: [] for seed_id in seeds
        }
        for score, path, _ in selected:
            grouped.setdefault(path.seed_passage_id, []).append(
                (score, path.target_passage_id)
            )
        seed_order = sorted(
            seeds,
            key=lambda seed_id: (
                -max(
                    (score for score, _ in grouped.get(seed_id, [])),
                    default=-1.0,
                ),
                seeds.index(seed_id),
            ),
        )
        organized: list[str] = []
        for seed_id in seed_order:
            organized.append(seed_id)
            organized.extend(
                passage_id
                for _, passage_id in sorted(
                    grouped.get(seed_id, []),
                    key=lambda item: (-item[0], item[1]),
                )
            )

        final: list[str] = []
        tokens = 0
        order = organized + [item.passage_id for item in ranking]
        for passage_id in order:
            if passage_id in final:
                continue
            cost = estimate_tokens(self.passages[passage_id].full_text)
            if final and tokens + cost > self.config.token_budget:
                continue
            final.append(passage_id)
            tokens += cost
            if len(final) >= self.config.final_k:
                break
        return final, tokens

    def run(
        self,
        example: QueryExample,
        ranking: Sequence[RetrievedPassage],
    ) -> RetrievalTrace:
        start_time = time.perf_counter()
        seeds = [item.passage_id for item in ranking[: self.config.seed_k]]
        candidates_by_target, graph_reads = self._expanded_candidates(seeds)
        scored = self._score_candidates(
            example, seeds, candidates_by_target, ranking
        )

        selected: list[tuple[float, CandidatePath, float]] = []
        actions: list[RetrievalAction] = []
        evidence = list(seeds)
        evidence_tokens = sum(
            estimate_tokens(self.passages[item].full_text)
            for item in evidence
        )
        for score, path, _ in scored:
            if len(selected) >= self.config.max_actions:
                break
            target_id = path.target_passage_id
            target_cost = estimate_tokens(self.passages[target_id].full_text)
            if evidence and evidence_tokens + target_cost > self.config.token_budget:
                continue
            actions.append(
                RetrievalAction(
                    step=len(selected),
                    selected_path_id=path.id,
                    selected_passage_id=target_id,
                    score=score,
                    gate_probability=None,
                    frontier_size=len(scored),
                    gate_stage=None,
                    evidence_ids_before=list(evidence),
                )
            )
            selected.append((score, path, 0.0))
            evidence.append(target_id)
            evidence_tokens += target_cost

        if not selected:
            actions.append(
                RetrievalAction(
                    step=0,
                    selected_path_id=None,
                    selected_passage_id=None,
                    score=None,
                    gate_probability=None,
                    frontier_size=len(scored),
                    stop_reason=(
                        "empty_frontier" if not scored else "token_budget"
                    ),
                    evidence_ids_before=list(evidence),
                )
            )

        final, tokens = self._organized_context(seeds, selected, ranking)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return RetrievalTrace(
            query_id=example.id,
            policy=self.name,
            seed_passage_ids=seeds,
            final_passage_ids=final,
            actions=actions,
            latency_ms=latency_ms,
            graph_reads=graph_reads,
            candidate_paths_scored=len(scored),
            evidence_tokens=tokens,
            policy_latency_ms=latency_ms,
        )
