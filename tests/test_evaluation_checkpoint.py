import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from graph_rescue.config import ExperimentConfig
from graph_rescue.experiment import Experiment


class EvaluationCheckpointTests(unittest.TestCase):
    def test_fingerprint_changes_when_input_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            train = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            model_dir = root / "models"
            model_dir.mkdir()
            corpus.write_text('{"id":"p1"}\n', encoding="utf-8")
            train.write_text("", encoding="utf-8")
            evaluation.write_text('{"id":"q1"}\n', encoding="utf-8")
            (model_dir / "gate.json").write_text("{}", encoding="utf-8")
            experiment = Experiment.__new__(Experiment)
            experiment.config = ExperimentConfig(
                corpus_path=str(corpus),
                train_queries_path=str(train),
                eval_queries_path=str(evaluation),
                model_dir=str(model_dir),
            )
            experiment.eval_queries = [SimpleNamespace(id="q1")]
            before = experiment._evaluation_fingerprint(["hybrid"])
            corpus.write_text('{"id":"p2"}\n', encoding="utf-8")
            after = experiment._evaluation_fingerprint(["hybrid"])

        self.assertNotEqual(before, after)

    def test_loads_complete_bundles_and_ignores_truncated_tail(self):
        policies = {"hybrid", "mrv_gated"}
        complete = {
            "query_id": "q1",
            "rows": [
                {"query_id": "q1", "policy": "hybrid"},
                {"query_id": "q1", "policy": "mrv_gated"},
            ],
            "traces": [
                {"query_id": "q1", "policy": "hybrid"},
                {"query_id": "q1", "policy": "mrv_gated"},
            ],
        }
        incomplete = {
            "query_id": "q2",
            "rows": [{"query_id": "q2", "policy": "hybrid"}],
            "traces": [{"query_id": "q2", "policy": "hybrid"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.jsonl"
            target.write_text(
                json.dumps(complete)
                + "\n"
                + json.dumps(incomplete)
                + "\n"
                + '{"query_id":',
                encoding="utf-8",
            )
            loaded = Experiment._load_evaluation_checkpoint(
                target, expected_policies=policies
            )
        self.assertEqual(set(loaded), {"q1"})

    def test_append_writes_one_valid_bundle_per_line(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.jsonl"
            Experiment._append_evaluation_bundle(
                target, {"query_id": "q1", "rows": [], "traces": []}
            )
            Experiment._append_evaluation_bundle(
                target, {"query_id": "q2", "rows": [], "traces": []}
            )
            rows = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual([item["query_id"] for item in rows], ["q1", "q2"])


if __name__ == "__main__":
    unittest.main()
