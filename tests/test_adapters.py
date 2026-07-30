import json
import tempfile
import unittest
from pathlib import Path

from graph_rescue.adapters import convert_hotpot_or_2wiki
from graph_rescue.io import load_passages, load_queries


class AdapterTests(unittest.TestCase):
    def test_same_title_with_different_text_has_distinct_id(self):
        from graph_rescue.adapters import stable_passage_id

        self.assertNotEqual(
            stable_passage_id("Shared title", "First paragraph"),
            stable_passage_id("Shared title", "Second paragraph"),
        )

    def test_hotpot_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "_id": "q1",
                            "question": "Where?",
                            "answer": "Here",
                            "context": [
                                ["First", ["One sentence."]],
                                ["Second", ["Two sentence."]],
                            ],
                            "supporting_facts": [["First", 0], ["Second", 0]],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            corpus = root / "corpus.jsonl"
            queries = root / "queries.jsonl"
            report = convert_hotpot_or_2wiki(source, corpus, queries)
            self.assertEqual(report, {"passages": 2, "queries": 1})
            passages = load_passages(corpus)
            query = load_queries(queries)[0]
            self.assertEqual(len(passages), 2)
            self.assertEqual(passages[0].sentence_list, ("One sentence.",))
            self.assertEqual(len(query.supporting_passage_ids), 2)
            self.assertEqual(
                query.supporting_facts,
                (("First", 0), ("Second", 0)),
            )


if __name__ == "__main__":
    unittest.main()
