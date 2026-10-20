"""Tests for the drop-in ADEPT/SVoE surprisal benchmark."""

import math
import unittest

from svoe_calibration.surprisal_benchmark import (
    adept_surprisal_value,
    adept_surprisal_vectorized,
    empirical_tail_probability,
    equal_weights,
    svoe_surprisal_value,
    svoe_surprisal_vectorized,
)


class SurprisalBenchmarkTest(unittest.TestCase):
    def test_adept_surprisal_matches_weighted_likelihood(self):
        scores = (0.0, 2.0)
        weights = (0.25, 0.75)
        expected = -math.log(
            0.25 * math.exp(0.0) + 0.75 * math.exp(-2.0)
        )
        self.assertAlmostEqual(
            adept_surprisal_value(scores, weights),
            expected,
        )

    def test_vectorized_adept_matches_released_calculation(self):
        scores = tuple(float(value) / 7.0 for value in range(128))
        weights = equal_weights(128)
        self.assertAlmostEqual(
            adept_surprisal_vectorized(scores, weights),
            adept_surprisal_value(scores, weights),
            places=12,
        )

    def test_empirical_tail_probability_uses_add_one_smoothing(self):
        reference = (0.0, 1.0, 2.0, 3.0)
        self.assertAlmostEqual(
            empirical_tail_probability(4.0, reference),
            1.0 / 5.0,
        )

    def test_empirical_tail_probability_includes_ties(self):
        reference = (0.0, 1.0, 2.0, 2.0, 3.0)
        self.assertAlmostEqual(
            empirical_tail_probability(2.0, reference),
            4.0 / 6.0,
        )

    def test_svoe_single_particle_returns_shannon_bits(self):
        reference = (0.0, 1.0, 2.0, 3.0)
        self.assertAlmostEqual(
            svoe_surprisal_value((4.0,), (1.0,), reference),
            math.log(5.0, 2.0),
        )

    def test_svoe_combines_particle_probabilities_by_weight(self):
        reference = (0.0, 1.0, 2.0, 3.0)
        # Tail probabilities are 1.0 and 0.2; their equal mixture is 0.6.
        self.assertAlmostEqual(
            svoe_surprisal_value(
                (0.0, 4.0),
                (0.5, 0.5),
                reference,
            ),
            -math.log(0.6, 2.0),
        )

    def test_vectorized_svoe_matches_scalar_calculation(self):
        reference = tuple(float(value // 3) for value in range(300))
        scores = tuple(float(value) / 7.0 for value in range(128))
        weights = equal_weights(128)
        self.assertAlmostEqual(
            svoe_surprisal_vectorized(scores, weights, reference),
            svoe_surprisal_value(scores, weights, reference),
            places=12,
        )

    def test_equal_weights_are_normalized(self):
        weights = equal_weights(128)
        self.assertEqual(len(weights), 128)
        self.assertAlmostEqual(sum(weights), 1.0)


if __name__ == "__main__":
    unittest.main()
