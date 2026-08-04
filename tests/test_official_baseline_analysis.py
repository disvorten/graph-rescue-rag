import unittest

from work.analyze_official_baselines import aggregate_graph, aggregate_official


class OfficialBaselineAnalysisTests(unittest.TestCase):
    def test_aggregates_aligned_official_and_graph_rows(self):
        ids = ["a", "b"]
        official = {
            "a": {
                "full_evidence_at_7": 1.0,
                "support_recall_at_7": 1.0,
                "latency_ms": 10.0,
            },
            "b": {
                "full_evidence_at_7": 0.0,
                "support_recall_at_7": 0.5,
                "latency_ms": 30.0,
            },
        }
        graph = {
            "a": {
                "metrics": {
                    "full_evidence": 0.0,
                    "support_recall": 0.5,
                    "total_latency_ms": 20.0,
                    "graph_actions": 0.0,
                }
            },
            "b": {
                "metrics": {
                    "full_evidence": 1.0,
                    "support_recall": 1.0,
                    "total_latency_ms": 40.0,
                    "graph_actions": 2.0,
                }
            },
        }
        official_result = aggregate_official(official, ids)
        graph_result = aggregate_graph(graph, ids)
        self.assertEqual(official_result["full_evidence_at_7"], 0.5)
        self.assertEqual(graph_result["full_evidence_at_7"], 0.5)
        self.assertEqual(official_result["support_recall_at_7"], 0.75)
        self.assertEqual(graph_result["support_recall_at_7"], 0.75)
        self.assertEqual(official_result["retrieval_latency"]["median_ms"], 20.0)
        self.assertEqual(graph_result["retrieval_latency"]["median_ms"], 30.0)
        self.assertEqual(graph_result["mean_graph_actions"], 1.0)
        self.assertEqual(graph_result["graph_open_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
