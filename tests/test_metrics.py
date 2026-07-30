import unittest

import numpy as np

from graph_rescue.metrics import (
    calibration_bins,
    factorial_interaction,
    holm_bonferroni,
    roc_auc,
)
from graph_rescue.official_metrics import (
    answer_scores,
    joint_scores,
    support_fact_scores,
)


class MetricTests(unittest.TestCase):
    def test_auc_ties_equal_random(self):
        probabilities = np.asarray([0.5, 0.5, 0.5, 0.5])
        labels = np.asarray([0, 1, 0, 1])
        self.assertAlmostEqual(roc_auc(probabilities, labels), 0.5)

    def test_factorial_interaction(self):
        result = factorial_interaction(
            [0.0, 0.0],
            [0.5, 0.5],
            [0.25, 0.25],
            [1.0, 1.0],
            samples=100,
        )
        self.assertAlmostEqual(result["interaction"], 0.25)

    def test_hotpot_answer_normalization(self):
        result = answer_scores("The United States.", "United States")
        self.assertEqual(result["em"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_yes_no_mismatch_is_zero(self):
        result = answer_scores("yes", "no")
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual(result["precision"], 0.0)

    def test_support_and_joint_metrics(self):
        support = support_fact_scores(
            [("Doc", 0), ("Other", 1)],
            [("Doc", 0), ("Missing", 2)],
        )
        self.assertAlmostEqual(support["precision"], 0.5)
        self.assertAlmostEqual(support["recall"], 0.5)
        answer = answer_scores("answer", "answer")
        joint = joint_scores(answer, support)
        self.assertAlmostEqual(joint["f1"], 0.5)

    def test_holm_adjustment_is_monotone_and_bounded(self):
        adjusted = holm_bonferroni(
            {"a": 0.01, "b": 0.03, "c": 0.50}
        )
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertGreaterEqual(adjusted["b"], adjusted["a"])
        self.assertLessEqual(adjusted["c"], 1.0)

    def test_calibration_bins_report_empirical_rate(self):
        bins = calibration_bins(
            [0.05, 0.15, 0.85, 0.95],
            [0, 1, 1, 1],
            bins=2,
        )
        self.assertEqual(bins[0]["count"], 2.0)
        self.assertAlmostEqual(bins[0]["empirical_rate"], 0.5)
        self.assertEqual(bins[1]["count"], 2.0)
        self.assertAlmostEqual(bins[1]["empirical_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
