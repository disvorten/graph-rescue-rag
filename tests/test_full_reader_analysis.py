from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from work.analyze_full_reader import (
    index_support_scores,
    normalized_official_metrics,
    official_query_scores,
)


class FullReaderAnalysisTests(unittest.TestCase):
    def test_normalizes_2wiki_percentages(self) -> None:
        result = normalized_official_metrics(
            "2wiki",
            {
                "em": 20.0,
                "f1": 30.0,
                "sp_em": 40.0,
                "sp_f1": 50.0,
            },
        )
        self.assertAlmostEqual(result["answer_f1"], 0.3)
        self.assertAlmostEqual(result["support_f1"], 0.5)
        self.assertIsNone(result["joint_f1"])

    def test_keeps_hotpot_fractional_metrics(self) -> None:
        result = normalized_official_metrics(
            "hotpot",
            {
                "em": 0.2,
                "f1": 0.3,
                "sp_em": 0.4,
                "sp_f1": 0.5,
                "joint_f1": 0.1,
            },
        )
        self.assertAlmostEqual(result["answer_f1"], 0.3)
        self.assertAlmostEqual(result["joint_f1"], 0.1)

    def test_musique_empty_support_matches_official_scorer(self) -> None:
        result = index_support_scores([], [])
        self.assertEqual(result["em"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_2wiki_per_query_scores_include_answer_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.json"
            gold = root / "gold.json"
            aliases = root / "aliases.json"
            prediction.write_text(
                json.dumps(
                    {
                        "answer": {"q1": "alias answer"},
                        "sp": {"q1": [["Title", 1]]},
                        "evidence": {"q1": []},
                    }
                ),
                encoding="utf-8",
            )
            gold.write_text(
                json.dumps(
                    [
                        {
                            "_id": "q1",
                            "answer": "canonical answer",
                            "answer_id": "a1",
                            "supporting_facts": [["title", 1]],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            aliases.write_text(
                json.dumps(
                    {
                        "Q_id": "a1",
                        "aliases": ["alias answer"],
                        "demonyms": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            scores = official_query_scores(
                "2wiki",
                {
                    "prediction_path": str(prediction),
                    "gold_path": str(gold),
                    "alias_path": str(aliases),
                },
            )
        self.assertEqual(scores["answer_em"], [1.0])
        self.assertEqual(scores["answer_f1"], [1.0])
        self.assertEqual(scores["support_f1"], [1.0])


if __name__ == "__main__":
    unittest.main()
