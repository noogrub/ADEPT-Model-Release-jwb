import unittest

from svoe_calibration.calibration import (
    calibrate_score,
    calibrate_scores,
    prepare_reference,
    calibrate_value,
)


class CalibrationTests(unittest.TestCase):

    def test_score_above_reference_maximum(self):
        result = calibrate_scores([4.0], [1.0, 2.0, 3.0])[0]

        self.assertEqual(result["tail_count"], 0)
        self.assertEqual(result["tail_probability"], 0.25)
        self.assertEqual(result["svoe_bits"], 2.0)

    def test_equal_scores_are_included_in_tail(self):
        reference = prepare_reference([1.0, 2.0, 2.0, 3.0])
        result = calibrate_score(2.0, reference)

        self.assertEqual(result["tail_count"], 3)
        self.assertEqual(result["tail_probability"], 0.8)

    def test_larger_scores_produce_larger_surprisal(self):
        results = calibrate_scores(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        )

        bits = [result["svoe_bits"] for result in results]
        self.assertLess(bits[0], bits[1])
        self.assertLess(bits[1], bits[2])

    def test_empty_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_reference([])

    def test_calculation_only_values(self):
        reference = prepare_reference([1.0, 2.0, 3.0])

        tail_count, tail_probability, svoe_bits = calibrate_value(
            4.0,
            reference,
        )

        self.assertEqual(tail_count, 0)
        self.assertEqual(tail_probability, 0.25)
        self.assertEqual(svoe_bits, 2.0)

if __name__ == "__main__":
    unittest.main()
