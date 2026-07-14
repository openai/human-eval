import unittest

from human_eval.evaluation import estimate_pass_at_k


class EstimatePassAtKTest(unittest.TestCase):
    def test_zero_correct_with_k_greater_than_n(self):
        # Regression test for https://github.com/openai/human-eval/issues/35
        # With no correct samples, pass@k is exactly 0 for any k. Previously the
        # ``n - c < k`` shortcut returned 1.0 when k exceeded the sample count.
        self.assertEqual(estimate_pass_at_k([2], [0], 5).tolist(), [0.0])

    def test_zero_correct_with_k_leq_n(self):
        # No correct samples must yield 0 regardless of k <= n.
        self.assertEqual(estimate_pass_at_k([10], [0], 1).tolist(), [0.0])
        self.assertEqual(estimate_pass_at_k([10], [0], 10).tolist(), [0.0])

    def test_all_correct(self):
        # Every sample correct must yield pass@k == 1.
        self.assertEqual(estimate_pass_at_k([5], [5], 1).tolist(), [1.0])
        self.assertEqual(estimate_pass_at_k([5], [5], 5).tolist(), [1.0])

    def test_fewer_incorrect_than_k_is_guaranteed_pass(self):
        # With at least one correct sample and n - c < k, a correct sample is
        # always drawn, so pass@k == 1.
        self.assertEqual(estimate_pass_at_k([2], [1], 5).tolist(), [1.0])

    def test_known_value(self):
        # 3 correct out of 6 -> pass@1 == 1 - C(3,1)/C(6,1) == 0.5.
        result = estimate_pass_at_k([6], [3], 1)
        self.assertAlmostEqual(result[0], 0.5)

    def test_scalar_num_samples(self):
        # num_samples may be a single int shared across all problems.
        result = estimate_pass_at_k(6, [0, 3, 6], 1).tolist()
        self.assertEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertEqual(result[2], 1.0)


if __name__ == "__main__":
    unittest.main()
