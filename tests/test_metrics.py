"""Operating-point tests use known confusion counts, rather than reimplementing ROC."""

import unittest

import numpy as np

from ontoforensics.metrics import binary_metrics, choose_threshold, risk_coverage


class ThresholdTests(unittest.TestCase):
    def test_seconds_weighted_fpr_differs_from_frame_count(self):
        y = np.array([0, 0, 0, 1])
        p = np.array([0.1, 0.2, 0.9, 0.95])
        seconds = np.array([8.0, 1.0, 1.0, 20.0])
        threshold = choose_threshold(y, p, target_fpr=0.2, weight=seconds)
        self.assertGreater(threshold, 0.1)
        self.assertLess(threshold, 0.2)
        result = binary_metrics(y, p, threshold, weight=seconds)
        self.assertEqual(result["fp"], 2.0)
        self.assertEqual(result["tn"], 8.0)
        self.assertAlmostEqual(result["fpr"], 0.2)
        self.assertAlmostEqual(binary_metrics(y, p, threshold)["fpr"], 2 / 3)

    def test_tied_scores_cannot_be_partly_accepted(self):
        y = np.zeros(4, dtype=int)
        p = np.array([0.1, 0.8, 0.8, 0.95])
        seconds = np.array([7.0, 1.0, 1.0, 1.0])
        threshold = choose_threshold(y, p, target_fpr=0.2, weight=seconds)
        self.assertGreater(threshold, 0.8)
        result = binary_metrics(y, p, threshold, weight=seconds)
        self.assertAlmostEqual(result["fpr"], 0.1)
        self.assertLessEqual(result["fpr"], 0.2)

    def test_zero_fpr_excludes_even_probability_one(self):
        y, p = [0, 0, 1], [0.2, 1.0, 1.0]
        threshold = choose_threshold(y, p, target_fpr=0.0)
        self.assertGreater(threshold, 1.0)
        result = binary_metrics(y, p, threshold)
        self.assertEqual(result["fpr"], 0.0)
        self.assertEqual(result["recall"], 0.0)

    def test_threshold_uses_only_development_negatives(self):
        y = [0, 0, 1, 1]
        first = choose_threshold(y, [0.1, 0.8, 0.9, 0.95], target_fpr=0.5)
        second = choose_threshold(y, [0.1, 0.8, 0.01, 0.02], target_fpr=0.5)
        self.assertEqual(first, second)

    def test_missing_negatives_and_invalid_weights_fail(self):
        with self.assertRaisesRegex(ValueError, "bona fide"):
            choose_threshold([1, 1], [0.1, 0.2])
        for bad in (0.0, -1.0, float("nan")):
            with self.subTest(weight=bad), self.assertRaises(ValueError):
                choose_threshold([0, 1], [0.1, 0.9], weight=[bad, 1.0])


class BinaryMetricTests(unittest.TestCase):
    def test_known_weighted_confusion_matrix(self):
        result = binary_metrics(
            [0, 0, 1, 1], [0.9, 0.1, 0.9, 0.1], weight=[1, 3, 2, 4]
        )
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (2, 1, 3, 4))
        self.assertAlmostEqual(result["fpr"], 1 / 4)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertAlmostEqual(result["f1"], 4 / 9)
        self.assertAlmostEqual(result["brier"], 0.41)

    def test_eer_perfect_reversed_and_all_tied(self):
        y = [0, 0, 1, 1]
        examples = (
            ([0.1, 0.2, 0.8, 0.9], 0.0, 1.0),
            ([0.9, 0.8, 0.2, 0.1], 1.0, 0.0),
            ([0.5, 0.5, 0.5, 0.5], 0.5, 0.5),
        )
        for p, eer, auc in examples:
            with self.subTest(probabilities=p):
                result = binary_metrics(y, p)
                self.assertAlmostEqual(result["eer_interpolated"], eer)
                self.assertAlmostEqual(result["auroc"], auc)

    def test_single_class_has_no_invented_eer_or_auc(self):
        result = binary_metrics([0, 0], [0.1, 0.8])
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["eer_interpolated"])
        self.assertEqual(result["fpr"], 0.5)
        self.assertEqual(binary_metrics([], [])["n"], 0)


class RiskCoverageTests(unittest.TestCase):
    def test_confidence_follows_frozen_decision_and_includes_abstention_denominator(self):
        # At threshold .95 both predictions are negative. The .3 observation
        # has confidence .7 in that decision, and must rank above the .8 one.
        curve = risk_coverage([0, 1], [0.8, 0.3], threshold=0.95, n_total=4)
        self.assertEqual(curve[0]["n_selected"], 1)
        self.assertAlmostEqual(curve[0]["minimum_decision_confidence"], 0.7)
        self.assertEqual(curve[0]["classification_error_at_frozen_threshold"], 1.0)
        self.assertAlmostEqual(curve[-1]["coverage_of_all_recordings"], 0.5)
        self.assertEqual(curve[-1]["coverage_of_usable_recordings"], 1.0)
        self.assertEqual((curve[-1]["n_total"], curve[-1]["n_usable"]), (4, 2))
        self.assertEqual(curve[-1]["classification_error_at_frozen_threshold"], 0.5)
        conventional = risk_coverage([0, 1], [0.8, 0.3], threshold=0.5)
        self.assertEqual(conventional[-1]["classification_error_at_frozen_threshold"], 1.0)

    def test_no_usable_evidence_does_not_invent_a_risk_curve(self):
        self.assertEqual(risk_coverage([], [], threshold=0.95, n_total=20), [])


if __name__ == "__main__":
    unittest.main()
