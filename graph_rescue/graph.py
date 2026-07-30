from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import CandidatePath, GraphEdge, Passage
from .text import normalize

CAPITALIZED_ENTITY = re.compile(
    r"\b(?:[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,3})\b", flags=re.UNICODE
)


def passage_node(passage_id: str) -> str:
    return f"p:{passage_id}"


def entity_node(entity: str) -> str:
    return f"e:{normalize(entity)}"


class EntityExtractor:
    def extract(self, passage: Passage) -> list[str]:
        if passage.entities:
            return sorted({normalize(item) for item in passage.entities if normalize(item)})
        values = CAPITALIZED_ENTITY.findall(passage.full_text)
        return sorted({normalize(item) for item in values if len(normalize(item)) > 2})


@dataclass
class GraphStats:
    passages: int
    entities: int
    edges: int
    filtered_entities: int
    dropped_edges: int = 0
    false_edges: int = 0
    entity_mode: str = "provided"


class KnowledgeGraph:
    def __init__(self):
        self.adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        self.nodes: set[str] = set()
        self.stats = GraphStats(0, 0, 0, 0)

    def add_edge(
        self, source: str, target: str, kind: str, confidence: float = 1.0
    ) -> None:
        self.nodes.update((source, target))
        self.adjacency[source].append(GraphEdge(source, target, kind, confidence))
        self.adjacency[target].append(GraphEdge(target, source, kind, confidence))

    def degree(self, node: str) -> int:
        return len(self.adjacency.get(node, ()))

    def neighbors(self, node: str) -> list[GraphEdge]:
        return self.adjacency.get(node, [])

    @classmethod
    def build(
        cls,
        passages: Sequence[Passage],
        *,
        extractor: EntityExtractor | None = None,
        min_entity_df: int = 2,
        max_entity_df_ratio: float = 0.20,
        entity_mode: str = "provided",
    ) -> "KnowledgeGraph":
        graph = cls()
        extractor = extractor or EntityExtractor()
        if entity_mode not in {"provided", "title_links", "explicit_only"}:
            raise ValueError(f"Unknown graph entity mode: {entity_mode}")
        entities_by_passage = {
            passage.id: extractor.extract(passage) for passage in passages
        }
        if entity_mode == "title_links":
            known_titles = {
                normalize(passage.title)
                for passage in passages
                if normalize(passage.title)
            }
            entities_by_passage = {
                passage.id: sorted(
                    (
                        set(entities_by_passage[passage.id]) & known_titles
                    )
                    | ({normalize(passage.title)} if normalize(passage.title) else set())
                )
                for passage in passages
            }
        elif entity_mode == "explicit_only":
            entities_by_passage = {passage.id: [] for passage in passages}
        entity_df: Counter[str] = Counter()
        for values in entities_by_passage.values():
            entity_df.update(set(values))

        max_df = max(2, math.ceil(len(passages) * max_entity_df_ratio))
        allowed = {
            entity
            for entity, count in entity_df.items()
            if min_entity_df <= count <= max_df
        }

        for passage in passages:
            p_node = passage_node(passage.id)
            graph.nodes.add(p_node)
            for entity in entities_by_passage[passage.id]:
                if entity not in allowed:
                    continue
                confidence = 1.0 / math.log2(2.0 + entity_df[entity])
                graph.add_edge(
                    p_node, entity_node(entity), "mentions", confidence=confidence
                )
            for linked_id in passage.links:
                if linked_id != passage.id:
                    graph.add_edge(
                        p_node, passage_node(linked_id), "explicit_link", confidence=1.0
                    )

        graph.stats = GraphStats(
            passages=len(passages),
            entities=len({node for node in graph.nodes if node.startswith("e:")}),
            edges=sum(len(edges) for edges in graph.adjacency.values()) // 2,
            filtered_entities=len(entity_df) - len(allowed),
            entity_mode=entity_mode,
        )
        return graph

    def corrupt(
        self,
        *,
        edge_dropout_rate: float = 0.0,
        false_edge_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Apply deterministic test-time edge deletion/addition in place."""
        if not 0.0 <= edge_dropout_rate < 1.0:
            raise ValueError("edge_dropout_rate must be in [0, 1)")
        if false_edge_ratio < 0.0:
            raise ValueError("false_edge_ratio must be non-negative")
        rng = random.Random(seed)
        original_nodes = set(self.nodes)
        unique: list[GraphEdge] = []
        seen: set[tuple[str, str, str]] = set()
        for edges in self.adjacency.values():
            for edge in edges:
                source, target = sorted((edge.source, edge.target))
                key = (source, target, edge.kind)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(
                    GraphEdge(source, target, edge.kind, edge.confidence)
                )

        kept: list[GraphEdge] = []
        dropped = 0
        for edge in unique:
            if rng.random() < edge_dropout_rate:
                dropped += 1
            else:
                kept.append(edge)

        self.adjacency = defaultdict(list)
        self.nodes = set(original_nodes)
        for edge in kept:
            self.add_edge(edge.source, edge.target, edge.kind, edge.confidence)

        original_count = len(unique)
        requested_false = int(round(original_count * false_edge_ratio))
        passage_nodes = sorted(node for node in self.nodes if node.startswith("p:"))
        existing_pairs = {
            tuple(sorted((edge.source, edge.target)))
            for edge in kept
        }
        added = 0
        attempts = 0
        max_attempts = max(100, requested_false * 30)
        while (
            added < requested_false
            and len(passage_nodes) >= 2
            and attempts < max_attempts
        ):
            attempts += 1
            source, target = rng.sample(passage_nodes, 2)
            pair = tuple(sorted((source, target)))
            if pair in existing_pairs:
                continue
            existing_pairs.add(pair)
            self.add_edge(
                source, target, "false_link", confidence=0.50
            )
            added += 1

        self.stats = GraphStats(
            passages=self.stats.passages,
            entities=len({node for node in self.nodes if node.startswith("e:")}),
            edges=sum(len(edges) for edges in self.adjacency.values()) // 2,
            filtered_entities=self.stats.filtered_entities,
            dropped_edges=dropped,
            false_edges=added,
            entity_mode=self.stats.entity_mode,
        )

    def candidate_paths(
        self,
        seed_passage_ids: Iterable[str],
        *,
        excluded_passage_ids: Iterable[str] = (),
        max_hops: int = 2,
        cap: int = 200,
    ) -> tuple[list[CandidatePath], int]:
        excluded = set(excluded_passage_ids) | set(seed_passage_ids)
        best_by_target: dict[str, CandidatePath] = {}
        graph_reads = 0

        for seed_id in seed_passage_ids:
            start = passage_node(seed_id)
            queue = deque([(start, (start,), (), 0, 1.0, self.degree(start))])
            visited: set[tuple[str, int]] = {(start, 0)}
            while queue and len(best_by_target) < cap * 3:
                node, path, kinds, hops, confidence, max_hubness = queue.popleft()
                for edge in self.neighbors(node):
                    graph_reads += 1
                    target = edge.target
                    if target in path:
                        continue
                    next_hops = hops
                    if target.startswith("p:") and target != start:
                        next_hops += 1
                    if next_hops > max_hops:
                        continue

                    next_path = path + (target,)
                    next_kinds = kinds + (edge.kind,)
                    next_confidence = confidence * edge.confidence
                    next_hubness = max(max_hubness, self.degree(target))
                    state = (target, next_hops)
                    if state not in visited:
                        visited.add(state)
                        queue.append(
                            (
                                target,
                                next_path,
                                next_kinds,
                                next_hops,
                                next_confidence,
                                next_hubness,
                            )
                        )

                    if not target.startswith("p:") or next_hops == 0:
                        continue
                    target_id = target[2:]
                    if target_id in excluded:
                        continue
                    candidate = CandidatePath(
                        seed_passage_id=seed_id,
                        target_passage_id=target_id,
                        nodes=next_path,
                        edge_kinds=next_kinds,
                        hop_count=next_hops,
                        confidence=next_confidence,
                        max_hubness=next_hubness,
                    )
                    previous = best_by_target.get(target_id)
                    if previous is None or (
                        candidate.hop_count,
                        -candidate.confidence,
                        candidate.id,
                    ) < (
                        previous.hop_count,
                        -previous.confidence,
                        previous.id,
                    ):
                        best_by_target[target_id] = candidate

        candidates = sorted(
            best_by_target.values(),
            key=lambda item: (
                item.hop_count,
                -item.confidence,
                item.max_hubness,
                item.target_passage_id,
            ),
        )[:cap]
        return candidates, graph_reads
