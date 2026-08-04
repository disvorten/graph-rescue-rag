import unittest

from work.prepare_hipporag_benchmark_adapter import (
    build_passage_index,
    resolve_passage_id,
)


class HippoRAGAdapterTests(unittest.TestCase):
    def test_preserves_normalized_whitespace_collision(self) -> None:
        source = [
            {"title": "A", "text": "one two"},
            {"title": "A", "text": "one\u00a0two"},
        ]
        passages, raw_to_id, collisions = build_passage_index(source)

        self.assertEqual(len(passages), 2)
        self.assertEqual(collisions, 1)
        first = resolve_passage_id("A", "one two", raw_to_id)
        second = resolve_passage_id("A", "one\u00a0two", raw_to_id)
        self.assertNotEqual(first, second)
        self.assertIn(first, passages)
        self.assertIn(second, passages)

    def test_exact_duplicate_is_deduplicated(self) -> None:
        source = [
            {"title": "A", "text": "same"},
            {"title": "A", "text": "same"},
        ]
        passages, raw_to_id, collisions = build_passage_index(source)

        self.assertEqual(len(passages), 1)
        self.assertEqual(len(raw_to_id), 1)
        self.assertEqual(collisions, 0)


if __name__ == "__main__":
    unittest.main()
