import tempfile
import unittest
from pathlib import Path

from graph_rescue.config import (
    EvaluationConfig,
    ExperimentConfig,
    GraphConfig,
    LearningConfig,
    OllamaConfig,
    RetrievalConfig,
)
from graph_rescue.experiment import Experiment


class EndToEndTests(unittest.TestCase):
    def test_demo_runs_without_external_models(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            config = ExperimentConfig(
                corpus_path=str(root / "examples/data/demo_corpus.jsonl"),
                train_queries_path=str(root / "examples/data/demo_train.jsonl"),
                eval_queries_path=str(root / "examples/data/demo_eval.jsonl"),
                output_dir=str(target / "output"),
                cache_dir=str(target / "cache"),
                model_dir=str(target / "models"),
                ollama=OllamaConfig(
                    base_url="http://127.0.0.1:9",
                    embedding_model="not-installed",
                    timeout_seconds=1,
                ),
                retrieval=RetrievalConfig(
                    bm25_k=20,
                    dense_k=20,
                    rerank_k=20,
                    seed_k=1,
                    final_k=2,
                    evidence_token_budget=512,
                ),
                graph=GraphConfig(
                    max_hops=2,
                    frontier_cap=50,
                    max_actions=1,
                    min_entity_df=1,
                    max_entity_df_ratio=1.0,
                ),
                learning=LearningConfig(
                    epochs=50,
                    calibration_fraction=0.25,
                    random_seed=42,
                ),
                evaluation=EvaluationConfig(
                    include_classic_baselines=True,
                    bootstrap_samples=100,
                ),
            )
            experiment = Experiment(config, allow_hashing_fallback=True)
            training, summary = experiment.run()
            self.assertGreater(training.candidate_examples, 0)
            self.assertEqual(
                set(summary["aggregate"]),
                {
                    "hybrid",
                    "relevance_always",
                    "mrv_always",
                    "relevance_gated",
                    "mrv_gated",
                    "bm25",
                    "dense",
                    "rrf_fusion",
                },
            )
            self.assertIn(
                "total_latency_ms_p95",
                summary["aggregate"]["mrv_gated"],
            )
            self.assertTrue((target / "output/summary.json").exists())

            config.output_dir = str(target / "filtered-output")
            config.evaluation.include_classic_baselines = False
            filtered = experiment.evaluate(
                policy_names_filter={"hybrid", "mrv_gated"},
                compute_slices=False,
            )
            self.assertEqual(
                set(filtered["aggregate"]),
                {"hybrid", "mrv_gated"},
            )
            self.assertEqual(filtered["factorial_interactions"], {})
            self.assertIn(
                "mrv_gated_vs_hybrid_full_evidence",
                filtered["comparisons"],
            )


if __name__ == "__main__":
    unittest.main()
