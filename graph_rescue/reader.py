from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
from typing import Sequence

from .models import Passage, QueryExample, ReaderPrediction
from .ollama import OllamaClient
from .text import normalize


class AnswerPresenceReader:
    """A deterministic context-answerability proxy for retrieval experiments."""

    name = "answer_presence"

    def predict(
        self,
        example: QueryExample,
        passage_ids: Sequence[str],
        passages: dict[str, Passage],
    ) -> ReaderPrediction:
        context = normalize(
            " ".join(passages[passage_id].full_text for passage_id in passage_ids)
        )
        for answer in example.answers:
            if normalize(answer) in context:
                return ReaderPrediction(answer=answer)
        return ReaderPrediction(answer="")

    def answer(
        self,
        example: QueryExample,
        passage_ids: Sequence[str],
        passages: dict[str, Passage],
    ) -> str:
        return self.predict(example, passage_ids, passages).answer


class OllamaReader:
    prompt_version = "grounded-reader-v4-citations"

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        *,
        cache_dir: str | Path | None = None,
    ):
        self.client = client
        self.model = model
        self.name = f"ollama:{model}"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hits = 0
        self.generation_calls = 0

    def _cache_path(
        self, example: QueryExample, passage_ids: Sequence[str]
    ) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps(
            {
                "version": self.prompt_version,
                "model": self.model,
                "query_id": example.id,
                "question": example.question,
                "passage_ids": list(passage_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        namespace = self.model.replace(":", "_").replace("/", "_")
        return self.cache_dir / f"{namespace}_{digest}.json"

    @staticmethod
    def _parse_supporting_facts(
        value: object,
        passage_ids: Sequence[str],
        passages: dict[str, Passage],
    ) -> tuple[tuple[str, int], ...]:
        result: set[tuple[str, int]] = set()
        if not isinstance(value, list):
            return ()
        by_title = {
            passages[passage_id].title.casefold(): passage_id
            for passage_id in passage_ids
        }
        for item in value:
            passage_ref: object | None = None
            sentence_ref: object | None = None
            if isinstance(item, dict):
                passage_ref = item.get(
                    "passage", item.get("passage_id", item.get("title"))
                )
                sentence_ref = item.get(
                    "sentence", item.get("sentence_id", item.get("sentence_index"))
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                passage_ref, sentence_ref = item[0], item[1]
            elif isinstance(item, str):
                match = re.fullmatch(
                    r"\[?P?(\d+)\s*[:;,/-]\s*S?(\d+)\]?",
                    item.strip(),
                    flags=re.IGNORECASE,
                )
                if match:
                    passage_ref, sentence_ref = match.groups()
            try:
                sentence_index = int(sentence_ref)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            passage_id: str | None = None
            try:
                numeric = int(passage_ref)  # type: ignore[arg-type]
                if 1 <= numeric <= len(passage_ids):
                    passage_id = passage_ids[numeric - 1]
            except (TypeError, ValueError):
                reference = str(passage_ref or "")
                if reference in passages and reference in passage_ids:
                    passage_id = reference
                else:
                    passage_id = by_title.get(reference.casefold())
            if passage_id is None:
                continue
            passage = passages[passage_id]
            if not 0 <= sentence_index < len(passage.sentence_list):
                continue
            result.add((passage.title, sentence_index))
        return tuple(sorted(result))

    @staticmethod
    def _parse_evidence_triples(
        value: object,
    ) -> tuple[tuple[str, str, str], ...]:
        if not isinstance(value, list):
            return ()
        result = {
            (str(item[0]), str(item[1]), str(item[2]))
            for item in value
            if isinstance(item, (list, tuple)) and len(item) >= 3
        }
        return tuple(sorted(result))

    def predict(
        self,
        example: QueryExample,
        passage_ids: Sequence[str],
        passages: dict[str, Passage],
    ) -> ReaderPrediction:
        cache_path = self._cache_path(example, passage_ids)
        if cache_path is not None and cache_path.exists():
            self.cache_hits += 1
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return ReaderPrediction(
                answer=str(cached.get("answer", "")),
                supporting_facts=tuple(
                    (str(item[0]), int(item[1]))
                    for item in cached.get("supporting_facts", [])
                ),
                supporting_passage_ids=tuple(
                    str(item)
                    for item in cached.get("supporting_passage_ids", [])
                ),
                evidence_triples=tuple(
                    (str(item[0]), str(item[1]), str(item[2]))
                    for item in cached.get("evidence_triples", [])
                ),
            )
        context_blocks: list[str] = []
        for passage_index, passage_id in enumerate(passage_ids, start=1):
            passage = passages[passage_id]
            lines = [f"[P{passage_index}] {passage.title}"]
            lines.extend(
                f"[P{passage_index}:S{sentence_index}] {sentence}"
                for sentence_index, sentence in enumerate(passage.sentence_list)
            )
            context_blocks.append("\n".join(lines))
        context = "\n\n".join(context_blocks)
        prompt = (
            "Answer the question using only the supplied evidence. "
            "Return one JSON object with fields: answer (string), "
            "supporting_facts (a list of objects with integer fields passage "
            "and sentence), and evidence_triples (a list of "
            "[subject, relation, object] lists when explicitly supported). "
            "Passage numbers are one-based and sentence numbers are zero-based. "
            "Cite only sentences necessary for the answer. If the evidence is "
            "insufficient, return an empty answer and empty lists. The answer "
            "must be the shortest answer span, not a full explanatory sentence. "
            "Follow this exact shape: {\"answer\":\"1960\","
            "\"supporting_facts\":[{\"passage\":1,\"sentence\":0}],"
            "\"evidence_triples\":[]}.\n\n"
            f"Question: {example.question}\n\nEvidence:\n{context}"
        )
        raw = self.client.generate(
            self.model, prompt, format_json=True, temperature=0.0
        )
        self.generation_calls += 1
        try:
            value = json.loads(raw)
            for key in ("answer", "result", "response"):
                if key in value:
                    answer = str(value[key]).strip()
                    break
            else:
                answer = ""
            supporting_facts = self._parse_supporting_facts(
                value.get("supporting_facts", []), passage_ids, passages
            )
            evidence_triples = self._parse_evidence_triples(
                value.get("evidence_triples", [])
            )
            cited_titles = {title.casefold() for title, _ in supporting_facts}
            supporting_passage_ids = tuple(
                passage_id
                for passage_id in passage_ids
                if passages[passage_id].title.casefold() in cited_titles
            )
        except json.JSONDecodeError:
            answer = raw.strip()
            supporting_facts = ()
            supporting_passage_ids = ()
            evidence_triples = ()
        prediction = ReaderPrediction(
            answer=answer,
            supporting_facts=supporting_facts,
            supporting_passage_ids=supporting_passage_ids,
            evidence_triples=evidence_triples,
        )
        if cache_path is not None:
            cache_path.write_text(
                json.dumps(
                    {
                        "answer": prediction.answer,
                        "supporting_facts": [
                            list(item) for item in prediction.supporting_facts
                        ],
                        "supporting_passage_ids": list(
                            prediction.supporting_passage_ids
                        ),
                        "evidence_triples": [
                            list(item) for item in prediction.evidence_triples
                        ],
                        "raw": raw,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return prediction

    def answer(
        self,
        example: QueryExample,
        passage_ids: Sequence[str],
        passages: dict[str, Passage],
    ) -> str:
        return self.predict(example, passage_ids, passages).answer

    def stats(self) -> dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "generation_calls": self.generation_calls,
        }
