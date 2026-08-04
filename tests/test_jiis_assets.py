import unittest

from work.generate_jiis_extended_assets import (
    ROOT,
    gate_table,
    latency_table,
    official_table,
    portable_paths,
)


class JIISAssetTests(unittest.TestCase):
    def test_public_summary_paths_are_repo_relative(self) -> None:
        value = {"config": str(ROOT / "examples" / "demo_config.json")}
        self.assertEqual(
            portable_paths(value), {"config": "examples/demo_config.json"}
        )

    def test_official_tables_have_consistent_column_counts(self) -> None:
        systems = {
            name: {
                "full_evidence_at_7": 0.2,
                "support_recall_at_7": 0.4,
                "retrieval_latency": {"median_ms": 10.0, "p95_ms": 20.0},
            }
            for name in (
                "StandardRAG_official_code",
                "HippoRAG_official_code",
                "GraphRescue_hybrid",
                "GraphRescue_gated_MRV",
            )
        }
        comparisons = {
            key: {"difference": 0.1, "ci95_low": 0.05, "ci95_high": 0.15}
            for key in (
                "GraphRescue_gated_minus_HippoRAG_full_evidence_at_7",
                "GraphRescue_gated_minus_StandardRAG_full_evidence_at_7",
                "GraphRescue_gated_minus_GraphRescue_hybrid_full_evidence_at_7",
            )
        }
        text = official_table(
            {
                "systems": systems,
                "paired_full_evidence_comparisons": comparisons,
            }
        )
        for line in text.splitlines():
            if line.startswith(("StandardRAG", "HippoRAG", "Graph Rescue")):
                expected = 2 if "$-$" in line else 4
                self.assertEqual(line.count("&"), expected, line)

    def test_latency_rows_have_eight_columns(self) -> None:
        policy = {
            "online_total_latency": {"median_ms": 10.0, "p95_ms": 20.0},
            "mean_graph_actions": 1.0,
        }
        result = {
            "aggregate": {
                "hybrid": policy,
                "mrv_always": policy,
                "mrv_gated": policy,
            },
            "paired_latency_comparisons": {
                "mrv_gated_minus_mrv_always_online_total_ms": {
                    "difference": -1.0,
                    "ci95_low": -2.0,
                    "ci95_high": -0.1,
                }
            },
        }
        text = latency_table(
            {"hotpot": result, "2wiki": result, "musique": result}
        )
        for line in text.splitlines():
            if line.startswith(("HotpotQA", "2Wiki", "MuSiQue")):
                self.assertEqual(line.count("&"), 7, line)

    def test_tex_arrows_are_not_interpreted_as_tabs(self) -> None:
        gate_item = {
            "heldout_test_queries": 10,
            "frozen_gate_on_heldout": {"recall": 0.5, "ece": 0.2},
            "recalibrated_gate_on_heldout": {"recall": 0.8, "ece": 0.1},
            "preflight_only_policy_frozen": {
                "open_rate": 0.4,
                "graph_actions": 0.8,
                "full_evidence": 0.5,
            },
            "preflight_only_policy_recalibrated": {
                "open_rate": 0.7,
                "graph_actions": 1.2,
                "full_evidence": 0.6,
            },
        }
        gate_text = gate_table({"datasets": {name: gate_item for name in ("hotpot", "2wiki", "musique")}})
        self.assertIn(r"$\to$", gate_text)
        self.assertNotIn("\t", gate_text)

        policy = {
            "online_total_latency": {"median_ms": 10.0, "p95_ms": 20.0},
            "mean_graph_actions": 1.0,
        }
        latency_result = {
            "aggregate": {"hybrid": policy, "mrv_always": policy, "mrv_gated": policy},
            "paired_latency_comparisons": {
                "mrv_gated_minus_mrv_always_online_total_ms": {
                    "difference": -1.0,
                    "ci95_low": -2.0,
                    "ci95_high": -0.1,
                }
            },
        }
        latency_text = latency_table(
            {name: latency_result for name in ("hotpot", "2wiki", "musique")}
        )
        self.assertIn(r"$\to$", latency_text)
        self.assertNotIn("\t", latency_text)


if __name__ == "__main__":
    unittest.main()
