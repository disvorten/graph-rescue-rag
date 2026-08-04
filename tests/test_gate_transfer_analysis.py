import unittest

from work.analyze_gate_transfer import policy_summary, stable_order


class GateTransferAnalysisTests(unittest.TestCase):
    def test_stable_order_depends_on_seed_and_query(self):
        self.assertEqual(stable_order("q1", 7), stable_order("q1", 7))
        self.assertNotEqual(stable_order("q1", 7), stable_order("q1", 8))
        self.assertNotEqual(stable_order("q1", 7), stable_order("q2", 7))

    def test_policy_summary_switches_between_saved_traces(self):
        rows = [
            {
                "probability": 0.8,
                "hybrid": {
                    "full_evidence": 0.0,
                    "support_recall": 0.5,
                    "graph_actions": 0.0,
                },
                "mrv_always": {
                    "full_evidence": 1.0,
                    "support_recall": 1.0,
                    "graph_actions": 2.0,
                },
            },
            {
                "probability": 0.2,
                "hybrid": {
                    "full_evidence": 1.0,
                    "support_recall": 1.0,
                    "graph_actions": 0.0,
                },
                "mrv_always": {
                    "full_evidence": 0.0,
                    "support_recall": 0.5,
                    "graph_actions": 2.0,
                },
            },
        ]
        result = policy_summary(rows, threshold=0.5)
        self.assertEqual(result["open_rate"], 0.5)
        self.assertEqual(result["full_evidence"], 1.0)
        self.assertEqual(result["support_recall"], 1.0)
        self.assertEqual(result["graph_actions"], 1.0)


if __name__ == "__main__":
    unittest.main()
