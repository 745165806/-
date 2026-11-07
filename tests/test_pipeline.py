from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from ontoforensics.demo import make_fixture
from ontoforensics.io import read_json
from ontoforensics.pipeline import train, predict, evaluate


class PipelineIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.records, cls.observations, cls.detectors = make_fixture(
            cls.root / "fixture", group_counts={"train": 6, "dev": 3, "calibration": 3, "test": 3})
        cls.model, _ = train(cls.records, cls.observations, cls.detectors, cls.root / "fit",
                             mode="relation_lr", step_sec=.2, context_sec=.2)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_json_roundtrip_and_unlabelled_prediction(self):
        loaded = read_json(self.root / "fit/model.json")
        records = deepcopy(self.records)
        for row in records:
            row["label"], row["fake_intervals"], row["ignore_intervals"] = None, [], []
            row["editor"], row["condition"], row["metadata"] = "unknown", "unknown", {"no_ground_truth": True}
        original = predict(self.model, self.records, self.observations)
        unlabelled = predict(loaded, records, self.observations)
        np.testing.assert_allclose(original["frame_probability"], unlabelled["frame_probability"], atol=0, rtol=0)
        np.testing.assert_allclose(original["utterance_probability"], unlabelled["utterance_probability"], atol=0, rtol=0)

    def test_verified_copies_do_not_change_prediction(self):
        extra = deepcopy(self.observations[0])
        extra["observation_id"] += ":zzcopy"
        extra["derived_from"] = self.observations[0]["observation_id"]
        original = predict(self.model, self.records, self.observations)
        duplicated = predict(self.model, self.records, self.observations + [extra])
        np.testing.assert_allclose(original["frame_probability"], duplicated["frame_probability"], atol=0, rtol=0)

    def test_no_local_observation_requires_review(self):
        result = predict(self.model, [self.records[0]], [])
        prediction = result["predictions"][0]
        self.assertEqual(prediction["decision"], "review")
        self.assertIsNone(prediction["p_edited"])
        self.assertEqual(prediction["candidate_intervals"], [])
        self.assertEqual(prediction["local_coverage"], 0)

    def test_external_test_cannot_reuse_fitted_speaker(self):
        test = deepcopy([r for r in self.records if r["split"] == "test"])
        test[0]["speaker_id"] = self.records[0]["speaker_id"]
        with self.assertRaisesRegex(ValueError, "speakers overlap"):
            evaluate(self.model, test, self.observations, self.root / "bad-test")

    def test_unknown_fitting_label_rejected(self):
        records = deepcopy(self.records)
        records[0]["label"], records[0]["fake_intervals"] = None, []
        with self.assertRaisesRegex(ValueError, "explicit labels"):
            train(records, self.observations, self.detectors, self.root / "bad-label", step_sec=.2)

    def test_missing_calibration_partition_rejected(self):
        records = [r for r in self.records if r["split"] != "calibration"]
        ids = {r["recording_id"] for r in records}
        observations = [o for o in self.observations if o["recording_id"] in ids]
        with self.assertRaisesRegex(ValueError, "calibration split"):
            train(records, observations, self.detectors, self.root / "bad-split", step_sec=.2)

    def test_test_label_and_score_changes_cannot_modify_fitted_model(self):
        records, observations = deepcopy(self.records), deepcopy(self.observations)
        test_ids = set()
        for record in records:
            if record["split"] == "test":
                record["label"], record["fake_intervals"] = None, []
                test_ids.add(record["recording_id"])
        for observation in observations:
            if observation["recording_id"] in test_ids and observation["status"] == "valid":
                observation["score"] += 10
        changed, _ = train(records, observations, self.detectors, self.root / "changed-test",
                           mode="relation_lr", step_sec=.2, context_sec=.2)
        before, after = deepcopy(self.model), deepcopy(changed)
        before.pop("created_utc"); after.pop("created_utc")
        self.assertEqual(before, after)

    def test_reject_confidence_tracks_frozen_decision(self):
        model = deepcopy(self.model)
        # Force p=0.8 at utterance level, with a deliberately higher decision threshold.
        model["utterance_calibrator"] = {"kind": "linear", "coef": [0.], "intercept": float(np.log(4.))}
        model["utterance_threshold"] = .95
        result = predict(model, [self.records[0]], [o for o in self.observations if o["recording_id"] == self.records[0]["recording_id"]])
        prediction = result["predictions"][0]
        self.assertFalse(prediction["threshold_decision"])
        self.assertAlmostEqual(prediction["decision_confidence"], .2)
        self.assertEqual(prediction["decision"], "review")


if __name__ == "__main__":
    unittest.main()
