from __future__ import annotations

import unittest

from work.analyze_multiseed_robustness import aggregate, diagnostics


class MultiseedRobustnessAnalysisTests(unittest.TestCase):
    def test_aggregates_seed_mean_and_dose_response(self) -> None:
        rows = []
        for condition, values in (
            ("dropout_10", (0.70, 0.68)),
            ("dropout_25", (0.61, 0.59)),
            ("dropout_50", (0.50, 0.48)),
        ):
            for seed, value in zip((101, 202), values):
                rows.append(
                    {
                        "dataset": "hotpot",
                        "condition": condition,
                        "corruption_seed": seed,
                        "policy": "mrv_gated",
                        "full_evidence": value,
                        "support_recall": value,
                        "graph_actions": 1.0,
                        "harmful_expansions": 0.1,
                        "policy_latency_ms": 5.0,
                    }
                )

        aggregated = aggregate(rows)
        first = next(
            row
            for row in aggregated
            if row["condition"] == "dropout_10"
        )
        self.assertAlmostEqual(first["full_evidence_mean"], 0.69)
        self.assertEqual(first["seeds"], 2)

        report = diagnostics(aggregated, raw_count=len(rows))
        check = next(
            row
            for row in report["dose_response_checks"]
            if row["family"] == "dropout"
        )
        self.assertTrue(check["nonincreasing_mean"])
        self.assertEqual(report["raw_rows"], 6)


if __name__ == "__main__":
    unittest.main()
