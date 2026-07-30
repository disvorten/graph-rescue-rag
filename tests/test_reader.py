import tempfile
import unittest
from pathlib import Path

from graph_rescue.models import Passage, QueryExample
from graph_rescue.reader import OllamaReader


class FakeClient:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return (
            '{"answer": "1960", "supporting_facts": '
            '[{"passage": 1, "sentence": 0}], "evidence_triples": []}'
        )


class StringCitationClient:
    def generate(self, *args, **kwargs):
        return (
            '{"answer": "1960", "supporting_facts": ["P1:S0"], '
            '"evidence_triples": []}'
        )


class ReaderTests(unittest.TestCase):
    def test_ollama_reader_caches_identical_context(self):
        example = QueryExample(
            id="q1",
            question="When?",
            answers=("1960",),
            supporting_passage_ids=("p1",),
        )
        passages = {
            "p1": Passage(id="p1", title="Institute", text="Founded in 1960.")
        }
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            reader = OllamaReader(
                client, "test-model", cache_dir=Path(directory)
            )
            first = reader.answer(example, ["p1"], passages)
            second = reader.answer(example, ["p1"], passages)
            self.assertEqual(first, "1960")
            self.assertEqual(second, "1960")
            self.assertEqual(client.calls, 1)
            self.assertEqual(reader.stats()["cache_hits"], 1)

    def test_reader_returns_validated_citations(self):
        example = QueryExample(
            id="q2",
            question="When?",
            answers=("1960",),
            supporting_passage_ids=("p1",),
            supporting_facts=(("Institute", 0),),
        )
        passages = {
            "p1": Passage(
                id="p1",
                title="Institute",
                text="Founded in 1960.",
                sentences=("Founded in 1960.",),
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            reader = OllamaReader(
                FakeClient(), "test-model", cache_dir=Path(directory)
            )
            prediction = reader.predict(example, ["p1"], passages)
            self.assertEqual(
                prediction.supporting_facts,
                (("Institute", 0),),
            )

    def test_reader_accepts_compact_string_citation(self):
        example = QueryExample(
            id="q3",
            question="When?",
            answers=("1960",),
            supporting_passage_ids=("p1",),
        )
        passages = {
            "p1": Passage(
                id="p1",
                title="Institute",
                text="Founded in 1960.",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            reader = OllamaReader(
                StringCitationClient(),
                "test-model",
                cache_dir=Path(directory),
            )
            prediction = reader.predict(example, ["p1"], passages)
            self.assertEqual(prediction.supporting_facts, (("Institute", 0),))


if __name__ == "__main__":
    unittest.main()
