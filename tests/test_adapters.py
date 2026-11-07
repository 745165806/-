"""Meaningful adapter checks: direction, timing, provenance and fail-closed IO.

These tests use small synthetic arrays to test plumbing. They do not test a
pretrained CFPRF/PartialSpoof detector, calibration, or detection performance.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from adapters import export_cfprf as cfprf
from adapters import import_frame_scores as importer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ontoforensics.ontology import deduplicate_observations, validate_observations


class ScoreSemanticsTests(unittest.TestCase):
    def test_two_class_margins_and_direction(self):
        raw = np.array([[0.8, 0.2], [-0.4, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(importer.canonical_scores(raw, "margin", "two-class"),
                                   [0.6, -0.9], atol=1e-7)
        np.testing.assert_allclose(importer.canonical_scores(np.array([0.2, -0.5]), "margin",
                                                            direction="higher-is-bonafide"),
                                   [-0.2, 0.5])
        with self.assertRaises(ValueError):
            importer.canonical_scores(raw, "margin", "scalar")
        with self.assertRaises(ValueError):
            importer.canonical_scores(raw, "margin", "two-class", "higher-is-bonafide")

    def test_probabilities_are_not_arbitrary_margins(self):
        np.testing.assert_allclose(importer.canonical_scores(np.array([0.9, 0.1]), "probability",
                                                            direction="higher-is-bonafide"), [0.1, 0.9])
        with self.assertRaises(ValueError):
            importer.canonical_scores(np.array([-0.3, 0.2]), "probability")
        with self.assertRaises(ValueError):
            importer.canonical_scores(np.array([[0.8, 0.8]]), "probability", "two-class")
        with self.assertRaises(ValueError):
            importer.canonical_scores(np.array([np.nan]), "margin")

    def test_native_grid_does_not_fill_unobserved_tail(self):
        intervals = importer.frame_intervals(49, duration_sec=1.0, hop_sec=0.02)
        self.assertEqual(intervals.shape, (49, 2))
        self.assertAlmostEqual(intervals[-1, 1], 0.98)
        with self.assertRaises(ValueError):
            importer.frame_intervals(51, duration_sec=1.0, hop_sec=0.02)
        with self.assertRaises(ValueError):
            importer.checked_intervals(np.array([[0, 0.04], [0.02, 0.06]]), 1.0)


class ImportIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="adapter-test-", dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.jsonl"
        self.checkpoint = self.root / "test-only-checkpoint.txt"
        self.checkpoint.write_bytes(b"Small unit-test fixture, not a trained network.")
        self.scores = self.root / "scores.npz"
        self.out = self.root / "observations.jsonl"
        self.registry = self.root / "registry.json"
        self.rows = [
            {"recording_id": "r1", "audio_path": "not-read-1.wav", "duration_sec": 0.1,
             "sample_rate": 16000, "split": "train", "label": 1,
             "fake_intervals": [[0.0, 0.03]]},
            {"recording_id": "r2", "audio_path": "not-read-2.wav", "duration_sec": 0.1,
             "sample_rate": 16000, "split": "test", "label": 0, "fake_intervals": []},
        ]
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def args(self, extra=()):
        return importer.build_parser().parse_args([
            "--manifest", str(self.manifest), "--scores", str(self.scores),
            "--out", str(self.out), "--registry", str(self.registry),
            "--checkpoint", str(self.checkpoint), "--detector-id", "test-detector",
            "--family", "test-family", "--run-id", "test-run",
            "--input-score-type", "margin", "--frame-hop-sec", "0.02", *extra,
        ])

    def observations(self):
        return [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]

    def test_npz_schema_direction_registry_and_label_independence(self):
        np.savez(self.scores, r1=np.array([[0.8, 0.1], [0.1, 0.9]]),
                 r2=np.array([[0.4, 0.3]]))
        args = self.args(("--layout", "two-class"))
        self.assertEqual(importer.run_import(args), 3)
        observations = self.observations()
        required = {"observation_id", "recording_id", "detector_id", "run_id",
                    "start_sec", "end_sec", "score", "status", "provenance_id"}
        self.assertTrue(all(required.issubset(row) for row in observations))
        self.assertTrue(all(row["status"] == "valid" for row in observations))
        self.assertTrue(all(len(row["provenance_id"]) == 64 for row in observations))
        self.assertEqual(len({row["provenance_id"] for row in observations}), len(observations))
        self.assertTrue(all(row["quality"] == {} for row in observations))
        np.testing.assert_allclose([row["score"] for row in observations], [0.7, -0.8, 0.1])
        self.assertAlmostEqual(observations[1]["start_sec"], 0.02)
        entry = json.loads(self.registry.read_text())["detectors"][0]
        self.assertEqual(entry["score_type"], "margin")
        self.assertTrue(entry["higher_is_fake"])
        self.assertFalse(entry["synthetic"])
        self.assertEqual(entry["checkpoint_sha256"], importer.sha256_file(self.checkpoint))
        before = self.out.read_bytes()
        for row in self.rows:
            row["label"] = {"deliberately": "unusable as a numerical feature"}
            row["fake_intervals"] = "not even an interval array"
        self.write_manifest()
        importer.run_import(args)
        self.assertEqual(before, self.out.read_bytes())

    def test_run_id_changes_observation_identity_but_not_provenance(self):
        np.savez(self.scores, r1=np.array([0.2]), r2=np.array([0.4]))
        args = self.args()
        importer.run_import(args)
        previous = self.observations()
        args.run_id = "second-run"
        importer.run_import(args)
        current = self.observations()
        self.assertEqual(previous[0]["provenance_id"], current[0]["provenance_id"])
        self.assertNotEqual(previous[0]["observation_id"], current[0]["observation_id"])

    def test_imported_frames_validate_and_deduplicate_through_real_ontology(self):
        np.savez(self.scores, r1=np.array([0.2, -0.1, 0.4]), r2=np.array([-0.3, 0.7]))
        args = self.args()
        self.assertEqual(importer.run_import(args), 5)
        observations = self.observations()
        registry = json.loads(self.registry.read_text())
        self.assertTrue(validate_observations(observations, self.rows, registry)["valid"])
        kept, report = deduplicate_observations(observations)
        self.assertEqual(len(kept), 5)
        self.assertEqual(report["removed_count"], 0)
        self.assertEqual(len({row["provenance_id"] for row in observations}), 5)
        args.run_id = "same-measurements-another-run"
        importer.run_import(args)
        repeated = self.observations()
        self.assertEqual([row["provenance_id"] for row in observations],
                         [row["provenance_id"] for row in repeated])
        self.assertTrue(validate_observations(observations + repeated, self.rows, registry)["valid"])
        kept, report = deduplicate_observations(observations + repeated)
        self.assertEqual(len(kept), 5)
        self.assertEqual(report["removed_count"], 5)

    def test_csv_partialspoof_style_columns_ignore_audit_column(self):
        self.scores = self.root / "scores.csv"
        self.scores.write_text("recording_id,frame_index,score0,score1,label\n"
                               "r1,0,0.7,0.1,DO_NOT_USE\n"
                               "r1,1,-0.4,0.2,DO_NOT_USE\n"
                               "r2,0,0.0,0.3,DO_NOT_USE\n", encoding="utf-8")
        importer.run_import(self.args(("--layout", "two-class")))
        np.testing.assert_allclose([row["score"] for row in self.observations()], [0.6, -0.6, -0.3])
        self.assertNotIn("label", self.out.read_text())

    def test_explicit_interval_csv_and_probability_inversion(self):
        self.scores = self.root / "intervals.csv"
        self.scores.write_text("recording_id,start_sec,end_sec,score\n"
                               "r1,0.01,0.04,0.8\n"
                               "r2,0.02,0.07,0.2\n", encoding="utf-8")
        args = self.args()
        args.time_mode = "intervals"
        args.frame_hop_sec = None
        args.input_score_type = "probability"
        args.direction = "higher-is-bonafide"
        importer.run_import(args)
        obs = self.observations()
        np.testing.assert_allclose([row["score"] for row in obs], [0.2, 0.8])
        self.assertEqual((obs[0]["start_sec"], obs[0]["end_sec"]), (0.01, 0.04))

    def test_missing_record_fails_without_publishing_partial_output(self):
        np.savez(self.scores, r1=np.array([0.4]))
        self.out.write_text("existing output must remain\n")
        with self.assertRaisesRegex(ValueError, "Missing score records"):
            importer.run_import(self.args())
        self.assertEqual(self.out.read_text(), "existing output must remain\n")
        self.assertFalse(self.registry.exists())
        self.assertFalse(list(self.root.glob(".frames-*")))

    def test_unknown_record_and_repeated_frame_indices_rejected(self):
        np.savez(self.scores, unknown=np.array([0.4]))
        with self.assertRaisesRegex(ValueError, "Unknown or duplicate"):
            importer.run_import(self.args())
        self.scores = self.root / "bad.csv"
        self.scores.write_text("recording_id,frame_index,score\nr1,0,0.1\nr1,0,0.2\n")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            importer.run_import(self.args())

    def test_single_npy_requires_explicit_recording_id_and_pickle_is_rejected(self):
        self.scores = self.root / "one.npy"
        np.save(self.scores, np.array([0.2, 0.5]))
        with self.assertRaisesRegex(ValueError, "recording-id"):
            importer.run_import(self.args())
        args = self.args(("--recording-id", "r1", "--allow-subset"))
        self.assertEqual(importer.run_import(args), 2)
        np.save(self.scores, np.array({"r1": [0.2]}, dtype=object))
        with self.assertRaises(ValueError):
            importer.run_import(args)

    def test_output_cannot_overwrite_input_or_conflicting_registry(self):
        np.savez(self.scores, r1=np.array([0.2]), r2=np.array([0.3]))
        args = self.args()
        args.out = self.scores
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            importer.run_import(args)
        args = self.args()
        importer.run_import(args)
        args.family = "changed-family"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            importer.run_import(args)

    def test_malformed_registry_is_reported_without_replacing_output(self):
        np.savez(self.scores, r1=np.array([0.2]), r2=np.array([0.3]))
        self.out.write_text("existing output\n")
        self.registry.write_text('{"detectors": [null]}')
        with self.assertRaisesRegex(ValueError, "nonempty detector_id"):
            importer.run_import(self.args())
        self.assertEqual(self.out.read_text(), "existing output\n")


class CFPRFContractTests(unittest.TestCase):
    def test_cfprf_frame_contract_validates_through_real_ontology(self):
        records = [importer.Recording("cfprf-r%d" % i, "not-read.wav", 1.0, 16000, "test")
                   for i in range(2)]
        raw = np.zeros((1, 49, 2), dtype=np.float32)
        raw[0, :, 0] = np.linspace(-0.3, 0.8, 49)
        scores = cfprf.fdn_margin(raw, 16000)
        observations = []
        for record in records:
            lineage = importer.stable_hash({"checkpoint_sha256": "a" * 64,
                                            "recording_id": record.recording_id,
                                            "score_semantics": "class0 minus class1 margin"})
            observations.extend(importer.frame_observations(
                record, scores, importer.frame_intervals(49, 1.0, 0.02),
                "cfprf-test-contract", "run1", lineage))
        registry = {"detectors": [importer.registry_entry(
            "cfprf-test-contract", "xlsr300m", "margin", "a" * 64)]}
        self.assertTrue(validate_observations(observations,
                        [record.__dict__ for record in records], registry)["valid"])
        kept, report = deduplicate_observations(observations)
        self.assertEqual(len(kept), 98)
        self.assertEqual(report["removed_count"], 0)

    def test_native_output_shape_and_raw_margin_without_torch(self):
        self.assertEqual(cfprf.expected_frame_count(400), 1)
        self.assertEqual(cfprf.expected_frame_count(16000), 49)
        self.assertEqual(cfprf.expected_frame_count(340000), 1062)
        raw = np.zeros((1, 49, 2), dtype=np.float32)
        raw[0, :, 0] = 0.9
        raw[0, :, 1] = 0.2
        np.testing.assert_allclose(cfprf.fdn_margin(raw, 16000), np.full(49, 0.7), atol=1e-7)
        with self.assertRaisesRegex(ValueError, "Unexpected FDN shape"):
            cfprf.fdn_margin(np.zeros((1, 50, 2)), 16000)
        with self.assertRaisesRegex(ValueError, "Unexpected FDN shape"):
            cfprf.fdn_margin(np.zeros((49, 2)), 16000)

    def test_audio_length_sample_rate_channels_and_duration_are_checked(self):
        record = importer.Recording("r1", "example.wav", 1.0, 16000, "test")
        cfprf.validate_audio_metadata(record, 16000, 16000, 1)
        with self.assertRaisesRegex(ValueError, "too short"):
            cfprf.expected_frame_count(399)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            cfprf.expected_frame_count(340001)
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            cfprf.validate_audio_metadata(record, 16000, 8000, 1)
        with self.assertRaisesRegex(ValueError, "mono"):
            cfprf.validate_audio_metadata(record, 16000, 16000, 2)
        with self.assertRaisesRegex(ValueError, "duration disagrees"):
            cfprf.validate_audio_metadata(record, 15000, 16000, 1)

    def test_cpu_rejected_before_importing_torch(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            cfprf._load_external_model(Path("missing"), Path("missing.pth"), "cpu")

    def test_missing_external_files_report_failure_without_output(self):
        with tempfile.TemporaryDirectory(prefix="cfprf-test-", dir=Path.cwd()) as directory:
            root = Path(directory)
            out, registry = root / "out.jsonl", root / "registry.json"
            with self.assertRaisesRegex(SystemExit, "CFPRF export failed: Required external file is missing"):
                cfprf.main([
                    "--repo", str(root / "missing-repo"),
                    "--checkpoint", str(root / "missing.pth"),
                    "--manifest", str(root / "manifest.jsonl"),
                    "--out", str(out), "--registry", str(registry), "--device", "cuda:0",
                ])
            self.assertFalse(out.exists())
            self.assertFalse(registry.exists())


if __name__ == "__main__":
    unittest.main()
