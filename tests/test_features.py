import copy
import unittest
import numpy as np

from ontoforensics.features import build_features, fit_score_normalization, make_targets


class FeatureBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.record = {"recording_id": "u", "duration_sec": 1., "split": "train",
                       "label": 1, "fake_intervals": [[.2, .4]]}
        self.detector = {"detector_id": "a", "scope": "local", "family": "f",
                         "score_type": "probability", "higher_is_fake": False}
        self.observation = {"observation_id": "o", "recording_id": "u", "detector_id": "a",
                            "start_sec": 0., "end_sec": 1., "score": .8, "status": "valid", "quality": {}}

    def test_probability_direction_and_missingness_are_distinct(self):
        table = build_features([self.record], [self.observation], [self.detector], step_sec=.2)
        self.assertTrue(np.all(table.X[:, 0] < 0))  # 0.8 bona fide -> negative fake margin
        self.assertTrue(table.usable.all())
        missing = dict(self.observation, score=None, status="missing")
        absent = build_features([self.record], [missing], [self.detector], step_sec=.2)
        self.assertFalse(absent.usable.any())
        self.assertTrue(np.all(absent.X[:, 1] == 0))
        neutral = build_features([self.record], [dict(self.observation, score=.5)], [self.detector], step_sec=.2)
        self.assertTrue(np.all(neutral.X[:, 0] == 0))
        self.assertTrue(np.all(neutral.X[:, 1] == 1))

    def test_supervision_and_audit_metadata_do_not_enter_features(self):
        before = build_features([self.record], [self.observation], [self.detector], relational=True)
        changed = dict(self.record, label=None, fake_intervals=[], editor="hidden-editor",
                       condition="codec", dataset="different", speaker_id="secret", metadata={"oracle": 1})
        after = build_features([changed], [self.observation], [self.detector], relational=True)
        np.testing.assert_array_equal(before.X, after.X)

    def test_global_context_does_not_create_local_coverage(self):
        global_detector = dict(self.detector, scope="global")
        table = build_features([self.record], [self.observation], [global_detector], step_sec=.2)
        self.assertFalse(table.usable.any())
        self.assertTrue(np.all(table.X[:, 1] == 1))

    def test_native_unobserved_tail_is_not_filled(self):
        observation = dict(self.observation, start_sec=.1, end_sec=.8)
        table = build_features([self.record], [observation], [self.detector], step_sec=.1)
        self.assertFalse(table.usable[0])
        self.assertFalse(table.usable[-1])
        self.assertEqual(int(table.usable.sum()), 7)

    def test_coarse_grid_does_not_stretch_a_partial_native_tail(self):
        observation = dict(self.observation, end_sec=.98)
        table = build_features([self.record], [observation], [self.detector], step_sec=.04)
        self.assertTrue(table.usable[:-1].all())
        self.assertFalse(table.usable[-1])
        self.assertAlmostEqual(float(np.sum((table.ends-table.starts)*table.usable)), .96)

    def test_overlapping_windows_do_not_double_count_coverage(self):
        first = dict(self.observation, end_sec=.03)
        second = dict(first, observation_id="overlap", start_sec=.01)
        table = build_features([self.record], [first, second], [self.detector], step_sec=.04)
        self.assertFalse(table.usable[0])

    def test_targets_use_half_open_midpoint_and_ignore_mask(self):
        record = dict(self.record, fake_intervals=[[.1, .3]], ignore_intervals=[[.4, .6]])
        table = build_features([record], [self.observation], [self.detector], step_sec=.2)
        self.assertEqual(make_targets(table, [record]).tolist(), [1, 0, -1, 0, 0])

    def test_score_scaling_does_not_fit_test_scores(self):
        other = dict(self.record, recording_id="test", split="test")
        observations = [self.observation, dict(self.observation, observation_id="extreme", recording_id="test", score=.99999)]
        first = fit_score_normalization([self.record, other], observations, [self.detector])
        observations[-1]["score"] = .00001
        second = fit_score_normalization([self.record, other], observations, [self.detector])
        self.assertEqual(first, second)

    def test_disabling_relation_changes_executable_feature_layout(self):
        d2 = dict(self.detector, detector_id="b")
        o2 = dict(self.observation, observation_id="o2", detector_id="b", score=.4)
        full = build_features([self.record], [self.observation, o2], [self.detector, d2], relational=True)
        ablated = build_features([self.record], [self.observation, o2], [self.detector, d2], relational=True,
                                 relations={"same_family": False, "temporal_context": True, "quality_interaction": True})
        self.assertGreater(full.X.shape[1], ablated.X.shape[1])
        self.assertFalse(any("same_family" in name for name in ablated.names))

    def test_family_types_change_computation_not_only_names(self):
        detectors = [dict(self.detector, detector_id="a", family="shared", score_type="margin", higher_is_fake=True),
                     dict(self.detector, detector_id="b", family="shared", score_type="margin", higher_is_fake=True),
                     dict(self.detector, detector_id="c", family="independent", score_type="margin", higher_is_fake=True)]
        observations = [dict(self.observation, detector_id=did, observation_id=did, score=score)
                        for did, score in [("a", -2.), ("b", -1.), ("c", 3.)]]
        original = build_features([self.record], observations, detectors, relational=True)
        changed = copy.deepcopy(detectors)
        changed[0]["family"], changed[2]["family"] = "independent", "shared"
        altered = build_features([self.record], observations, changed, relational=True)
        self.assertEqual(original.names, altered.names)
        column = original.names.index("relation/same_family/joint_mean")
        self.assertFalse(np.array_equal(original.X[:, column], altered.X[:, column]))


if __name__ == "__main__":
    unittest.main()
