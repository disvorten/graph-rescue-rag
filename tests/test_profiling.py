from __future__ import annotations

import tempfile
import unittest
from collections import Counter
import math
from pathlib import Path

from graph_rescue.profiling import (
    directory_bytes,
    latency_summary,
    percentile,
    process_rss_bytes,
)
from graph_rescue.hybrid import BM25Index
from graph_rescue.models import Passage


class ProfilingTests(unittest.TestCase):
    def test_process_rss_is_positive_when_supported(self):
        value = process_rss_bytes()
        if value is not None:
            self.assertGreater(value, 0)

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.95), 9.5)

    def test_latency_summary_handles_empty_and_populated_inputs(self) -> None:
        self.assertEqual(latency_summary([])["count"], 0)
        value = latency_summary([1.0, 2.0, 9.0])
        self.assertEqual(value["count"], 3)
        self.assertEqual(value["median_ms"], 2.0)
        self.assertGreater(value["p95_ms"], value["median_ms"])

    def test_directory_bytes_sums_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"123")
            (root / "nested").mkdir()
            (root / "nested" / "b.bin").write_bytes(b"4567")
            self.assertEqual(directory_bytes(root), 7)

    def test_inverted_bm25_matches_direct_formula(self) -> None:
        passages = [
            Passage(id="a", title="Alpha", text="red red blue"),
            Passage(id="b", title="Beta", text="blue green"),
            Passage(id="c", title="Gamma", text="yellow"),
        ]
        index = BM25Index(passages)
        from graph_rescue.text import tokenize

        documents = [
            tokenize(f"{passage.title} {passage.text}") for passage in passages
        ]
        frequencies_by_document = [Counter(tokens) for tokens in documents]
        document_frequency = Counter()
        for frequencies in frequencies_by_document:
            document_frequency.update(frequencies.keys())
        expected_idf = {
            term: math.log(
                1.0
                + (len(documents) - count + 0.5) / (count + 0.5)
            )
            for term, count in document_frequency.items()
        }
        self.assertAlmostEqual(index.idf["red"], expected_idf["red"])
        self.assertEqual(document_frequency["red"], 1)

        for query in (
            "red",
            "red red",
            "red blue red",
            "blue green",
            "alpha red blue",
            "missing token",
        ):
            actual = index.scores(query)
            expected = []
            for document_index, frequencies in enumerate(frequencies_by_document):
                score = 0.0
                for term in tokenize(query):
                    frequency = frequencies.get(term, 0)
                    if frequency:
                        score += expected_idf[term] * (
                            frequency
                            * (index.k1 + 1.0)
                            / (frequency + index.length_norm[document_index])
                        )
                expected.append(score)
            for left, right in zip(actual, expected):
                self.assertAlmostEqual(float(left), float(right), places=5)


if __name__ == "__main__":
    unittest.main()
