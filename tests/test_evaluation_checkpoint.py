import json
import tempfile
import unittest
from pathlib import Path

from graph_rescue.experiment import Experiment


class EvaluationCheckpointTests(unittest.TestCase):
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
