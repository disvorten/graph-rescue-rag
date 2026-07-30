import unittest

import numpy as np

from graph_rescue.learning import (
    BinaryLogisticRegression,
    PlattCalibrator,
    IsotonicCalibrator,
    bootstrap_threshold_for_recall,
    select_calibration_method,
    threshold_for_recall,
)


class LearningTests(unittest.TestCase):
    def test_logistic_regression_learns_separable_data(self):
        x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.asarray([0, 0, 1, 1])
        model = BinaryLogisticRegression(
            epochs=800, learning_rate=0.1, l2=0.001
        ).fit(x, y)
        probabilities = model.predict_proba(x)
        self.assertLess(probabilities[0], 0.5)
        self.assertGreater(probabilities[-1], 0.5)

    def test_isotonic_output_is_monotonic(self):
        calibrator = IsotonicCalibrator().fit(
            [0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1]
        )
        transformed = calibrator.transform([0.1, 0.2, 0.3, 0.4])
        self.assertTrue(np.all(np.diff(transformed) >= -1e-12))

    def test_threshold_meets_recall(self):
        probabilities = [0.9, 0.7, 0.3, 0.1]
        labels = [1, 1, 0, 0]
        threshold = threshold_for_recall(probabilities, labels, 1.0)
        predicted = np.asarray(probabilities) >= threshold
        recall = np.sum(predicted & (np.asarray(labels) == 1)) / 2
        self.assertEqual(recall, 1.0)

    def test_platt_calibration_tracks_prevalence(self):
        calibrator = PlattCalibrator().fit(
            [0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
            [0, 0, 0, 0, 1, 1],
        )
        transformed = calibrator.transform([0.70, 0.90])
        self.assertLess(transformed[0], transformed[1])
        self.assertTrue(np.all((transformed > 0.0) & (transformed < 1.0)))

    def test_auto_calibration_returns_supported_method(self):
        method, scores = select_calibration_method(
            [0.05, 0.15, 0.30, 0.55, 0.70, 0.90],
            [0, 0, 0, 1, 1, 1],
            folds=3,
            seed=7,
        )
        self.assertIn(method, {"identity", "platt", "isotonic"})
        self.assertEqual(set(scores), {"identity", "platt", "isotonic"})

    def test_bootstrap_threshold_is_not_less_conservative(self):
        probabilities = [0.95, 0.80, 0.60, 0.35, 0.20]
        labels = [1, 1, 1, 0, 0]
        baseline = threshold_for_recall(probabilities, labels, 1.0)
        bootstrapped = bootstrap_threshold_for_recall(
            probabilities,
            labels,
            1.0,
            samples=100,
            quantile=0.10,
            seed=11,
        )
        self.assertLessEqual(bootstrapped, baseline)


if __name__ == "__main__":
    unittest.main()
