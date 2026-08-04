import tempfile
import unittest
from pathlib import Path

from graph_rescue.cli import (
    evaluation_model_artifacts,
    validate_evaluation_model_artifacts,
)
from graph_rescue.config import ExperimentConfig


class CliModelValidationTests(unittest.TestCase):
    @staticmethod
    def config(model_dir: Path) -> ExperimentConfig:
        return ExperimentConfig(
            corpus_path="corpus.jsonl",
            train_queries_path="train.jsonl",
            eval_queries_path="eval.jsonl",
            model_dir=str(model_dir),
        )

    def test_missing_models_fail_before_experiment_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory) / "wrong")
            with self.assertRaisesRegex(FileNotFoundError, "mrv_model.json"):
                validate_evaluation_model_artifacts(config)

    def test_legacy_gate_and_optional_continuation_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "mrv_model.json").write_text("{}", encoding="utf-8")
            (model_dir / "gate_model.json").write_text("{}", encoding="utf-8")
            config = self.config(model_dir)
            artifacts = validate_evaluation_model_artifacts(config)
            self.assertTrue(artifacts["preflight_gate"].endswith("gate_model.json"))
            self.assertIsNone(artifacts["continue_gate"])

    def test_preflight_gate_is_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "preflight_gate_model.json").write_text("{}", encoding="utf-8")
            config = self.config(model_dir)
            artifacts = evaluation_model_artifacts(config)
            self.assertTrue(
                artifacts["preflight_gate"].endswith("preflight_gate_model.json")
            )


if __name__ == "__main__":
    unittest.main()
