import unittest

from work.run_latency_benchmark import paired_query_means


class LatencyBenchmarkTests(unittest.TestCase):
    def test_repetitions_are_averaged_within_query(self) -> None:
        left = [
            {"query_id": "q1", "latency": 10.0},
            {"query_id": "q1", "latency": 14.0},
            {"query_id": "q2", "latency": 20.0},
            {"query_id": "q2", "latency": 24.0},
        ]
        right = [
            {"query_id": "q2", "latency": 10.0},
            {"query_id": "q1", "latency": 4.0},
            {"query_id": "q2", "latency": 14.0},
            {"query_id": "q1", "latency": 8.0},
        ]

        observed_left, observed_right = paired_query_means(
            left, right, "latency"
        )

        self.assertEqual(observed_left, [12.0, 22.0])
        self.assertEqual(observed_right, [6.0, 12.0])

    def test_requires_an_aligned_query(self) -> None:
        with self.assertRaises(ValueError):
            paired_query_means(
                [{"query_id": "q1", "latency": 1.0}],
                [{"query_id": "q2", "latency": 2.0}],
                "latency",
            )


if __name__ == "__main__":
    unittest.main()
