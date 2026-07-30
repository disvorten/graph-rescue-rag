import json
import tempfile
import unittest
from pathlib import Path

from graph_rescue.config import ExperimentConfig, OllamaConfig
from graph_rescue.reproducibility import freeze_protocol


class ReproducibilityTests(unittest.TestCase):
    def test_freeze_detects_disjoint_demo_protocol(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = ExperimentConfig(
                corpus_path=str(root / "examples/data/demo_corpus.jsonl"),
                train_queries_path=str(root / "examples/data/demo_train.jsonl"),
                eval_queries_path=str(root / "examples/data/demo_eval.jsonl"),
                ollama=OllamaConfig(
                    base_url="http://127.0.0.1:9",
                    timeout_seconds=1,
                ),
            )
            result = freeze_protocol(config, output)
            self.assertTrue(result["audit"]["passed"])
            self.assertEqual(len(result["protocol_id"]), 64)
            self.assertTrue((output / "protocol_manifest.json").exists())
            loaded = json.loads(
                (output / "protocol_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded["protocol_id"], result["protocol_id"])


if __name__ == "__main__":
    unittest.main()
