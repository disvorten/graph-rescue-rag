from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str
    entities: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    sentences: tuple[str, ...] = ()

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}".strip()

    @property
    def sentence_list(self) -> tuple[str, ...]:
        if self.sentences:
            return self.sentences
        values = tuple(
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+", self.text)
            if item.strip()
        )
        return values or (self.text,)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Passage":
        return cls(
            id=str(value["id"]),
            title=str(value.get("title", "")),
            text=str(value["text"]),
            entities=tuple(str(item) for item in value.get("entities", [])),
            links=tuple(str(item) for item in value.get("links", [])),
            sentences=tuple(str(item) for item in value.get("sentences", [])),
        )


@dataclass(frozen=True)
class QueryExample:
    id: str
    question: str
    answers: tuple[str, ...]
    supporting_passage_ids: tuple[str, ...]
    supporting_facts: tuple[tuple[str, int], ...] = ()
    evidence_triples: tuple[tuple[str, str, str], ...] = ()
    supporting_paragraph_indices: tuple[int, ...] = ()
    passage_indices: tuple[tuple[str, int], ...] = ()
    question_type: str = ""
    dataset: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QueryExample":
        answers = value.get("answers", value.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]
        return cls(
            id=str(value["id"]),
            question=str(value["question"]),
            answers=tuple(str(item) for item in answers),
            supporting_passage_ids=tuple(
                str(item) for item in value.get("supporting_passage_ids", [])
            ),
            supporting_facts=tuple(
                (str(item[0]), int(item[1]))
                for item in value.get("supporting_facts", [])
                if isinstance(item, (list, tuple)) and len(item) >= 2
            ),
            evidence_triples=tuple(
                (str(item[0]), str(item[1]), str(item[2]))
                for item in value.get(
                    "evidence_triples", value.get("evidences", [])
                )
                if isinstance(item, (list, tuple)) and len(item) >= 3
            ),
            supporting_paragraph_indices=tuple(
                int(item)
                for item in value.get("supporting_paragraph_indices", [])
            ),
            passage_indices=tuple(
                (str(item[0]), int(item[1]))
                for item in value.get("passage_indices", [])
                if isinstance(item, (list, tuple)) and len(item) >= 2
            ),
            question_type=str(value.get("question_type", value.get("type", ""))),
            dataset=str(value.get("dataset", "")),
        )


@dataclass(frozen=True)
class ReaderPrediction:
    answer: str
    supporting_facts: tuple[tuple[str, int], ...] = ()
    supporting_passage_ids: tuple[str, ...] = ()
    evidence_triples: tuple[tuple[str, str, str], ...] = ()


@dataclass
class RetrievedPassage:
    passage_id: str
    rank: int = 0
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: str
    confidence: float = 1.0


@dataclass(frozen=True)
class CandidatePath:
    seed_passage_id: str
    target_passage_id: str
    nodes: tuple[str, ...]
    edge_kinds: tuple[str, ...]
    hop_count: int
    confidence: float
    max_hubness: int

    @property
    def id(self) -> str:
        return "->".join(self.nodes)


@dataclass
class CandidateScore:
    path: CandidatePath
    relevance: float = 0.0
    p_add_support: float = 0.0
    p_complete: float = 0.0
    p_reader_gain: float = 0.0
    p_harmful: float = 0.0
    marginal_value: float = 0.0
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalAction:
    step: int
    selected_path_id: str | None
    selected_passage_id: str | None
    score: float | None
    gate_probability: float | None
    frontier_size: int
    stop_reason: str | None = None
    gate_stage: str | None = None
    evidence_ids_before: list[str] = field(default_factory=list)


@dataclass
class RetrievalTrace:
    query_id: str
    policy: str
    seed_passage_ids: list[str]
    final_passage_ids: list[str]
    actions: list[RetrievalAction]
    latency_ms: float
    graph_reads: int
    candidate_paths_scored: int
    evidence_tokens: int
    retrieval_latency_ms: float = 0.0
    policy_latency_ms: float = 0.0
    reader_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryResult:
    query_id: str
    policy: str
    retrieved_ids: list[str]
    seed_ids: list[str]
    supporting_ids: list[str]
    answer: str
    predicted_answer: str
    trace: RetrievalTrace
    metrics: dict[str, float]
