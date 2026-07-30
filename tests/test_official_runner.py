import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from graph_rescue.official_runner import score_official_predictions


def _write_stub_evaluators(root: Path) -> Path:
    """Create tiny scorer fixtures so unit tests do not require downloaded tools."""
    evaluator_root = root / "official_evaluators"
    scripts = {
        "hotpot/hotpot_evaluate_v1.py": (
            "import json\n"
            "print(json.dumps({'joint_f1': 1.0}))\n"
        ),
        "2wiki/2wikimultihop_evaluate_v1.1.py": (
            "import json\n"
            "print(json.dumps({'f1': 100.0, 'sp_f1': 100.0}))\n"
        ),
        "musique/evaluate_v1.0.py": (
            "import json\n"
            "print(json.dumps({'answer_f1': 1.0, 'support_f1': 1.0}))\n"
        ),
    }
    for relative_path, source in scripts.items():
        path = evaluator_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return evaluator_root


class OfficialRunnerTests(unittest.TestCase):
    def test_official_hotpot_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator_root = _write_stub_evaluators(root)
            prediction = root / "prediction.json"
            gold = root / "gold.json"
            prediction.write_text(
                json.dumps(
                    {
                        "answer": {"q1": "Paris"},
                        "sp": {"q1": [["France", 0]]},
                    }
                ),
                encoding="utf-8",
            )
            gold.write_text(
                json.dumps(
                    [
                        {
                            "_id": "q1",
                            "answer": "Paris",
                            "supporting_facts": [["France", 0]],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with patch(
                "graph_rescue.official_runner.EVALUATOR_ROOT",
                evaluator_root,
            ):
                result = score_official_predictions(
                    dataset="hotpot",
                    prediction_path=prediction,
                    gold_path=gold,
                )
        self.assertEqual(result["metrics"]["joint_f1"], 1.0)

    def test_official_2wiki_scorer_with_json_shim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator_root = _write_stub_evaluators(root)
            prediction = root / "prediction.json"
            gold = root / "gold.json"
            aliases = root / "aliases.jsonl"
            prediction.write_text(
                json.dumps(
                    {
                        "answer": {"q1": "\u041f\u0430\u0440\u0438\u0436"},
                        "sp": {"q1": [["France", 0]]},
                        "evidence": {"q1": []},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            gold.write_text(
                json.dumps(
                    [
                        {
                            "_id": "q1",
                            "answer": "\u041f\u0430\u0440\u0438\u0436",
                            "answer_id": "Q90",
                            "supporting_facts": [["France", 0]],
                            "evidences": [],
                            "evidences_id": [],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            aliases.write_text(
                json.dumps(
                    {"Q_id": "Q90", "aliases": [], "demonyms": []}
                )
                + "\n",
                encoding="utf-8",
            )
            with patch(
                "graph_rescue.official_runner.EVALUATOR_ROOT",
                evaluator_root,
            ):
                result = score_official_predictions(
                    dataset="2wiki",
                    prediction_path=prediction,
                    gold_path=gold,
                    alias_path=aliases,
                )
        self.assertEqual(result["metrics"]["f1"], 100.0)
        self.assertEqual(result["metrics"]["sp_f1"], 100.0)

    def test_official_musique_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator_root = _write_stub_evaluators(root)
            prediction = root / "prediction.jsonl"
            gold = root / "gold.jsonl"
            prediction.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "predicted_answer": "Paris",
                        "predicted_support_idxs": [0],
                        "predicted_answerable": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            gold.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "answer": "Paris",
                        "answer_aliases": [],
                        "answerable": True,
                        "paragraphs": [
                            {"idx": 0, "is_supporting": True}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch(
                "graph_rescue.official_runner.EVALUATOR_ROOT",
                evaluator_root,
            ):
                result = score_official_predictions(
                    dataset="musique",
                    prediction_path=prediction,
                    gold_path=gold,
                )
        self.assertEqual(result["metrics"]["answer_f1"], 1.0)
        self.assertEqual(result["metrics"]["support_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
