"""Behavioral tests of the evidence contract, independent of model backends."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ontoforensics.ontology import (  # noqa: E402
    NS, deduplicate_observations, export_graph, validate_observations, validate_rdf,
)


def fixture():
    records = [{"recording_id": "a", "duration_sec": 2.0, "sample_rate": 16000,
                "label": None, "editor": "SUPERVISION_MUST_NOT_APPEAR",
                "condition": "TRUE_CODEC_MUST_NOT_APPEAR",
                "fake_intervals": [[0.2, 0.4]]}]
    detectors = {"detectors": [
        {"detector_id": "d1", "family": "spectral", "score_type": "probability",
         "higher_is_fake": True, "scope": "local"},
        {"detector_id": "d2", "family": "spectral", "score_type": "margin",
         "higher_is_fake": False, "scope": "local"},
        {"detector_id": "global", "family": "reconstruction", "score_type": "logit",
         "higher_is_fake": True, "scope": "global"},
    ]}
    obs = [{"observation_id": "o1", "recording_id": "a", "detector_id": "d1",
            "run_id": "r", "start_sec": 0.0, "end_sec": 0.5, "score": 0.8,
            "status": "valid", "quality": {"rms_db": -18.0, "clipping_fraction": 0.0}}]
    return records, detectors, obs


class ObservationValidationTests(unittest.TestCase):
    def test_valid_zero_differs_from_missing(self):
        records, detectors, obs = fixture()
        obs[0]["score"] = 0.0
        self.assertTrue(validate_observations(obs, records, detectors)["valid"])
        obs[0]["status"] = "missing"
        with self.assertRaisesRegex(ValueError, "never zero"):
            validate_observations(obs, records, detectors)
        obs[0]["score"] = None
        self.assertEqual(validate_observations(obs, records, detectors)["status_counts"], {"missing": 1})

    def test_nan_infinity_and_boolean_are_not_scores(self):
        records, detectors, obs = fixture()
        for value in (math.nan, math.inf, -math.inf, True):
            with self.subTest(value=value):
                obs[0]["score"] = value
                with self.assertRaises(ValueError):
                    validate_observations(obs, records, detectors)

    def test_margin_is_not_restricted_to_probability_range(self):
        records, detectors, obs = fixture()
        obs[0]["detector_id"] = "d2"
        obs[0]["score"] = -4.7
        self.assertTrue(validate_observations(obs, records, detectors)["valid"])
        obs[0]["detector_id"] = "d1"
        with self.assertRaisesRegex(ValueError, "probability"):
            validate_observations(obs, records, detectors)

    def test_local_and_global_scope(self):
        records, detectors, obs = fixture()
        obs[0]["start_sec"] = None
        obs[0]["end_sec"] = None
        with self.assertRaisesRegex(ValueError, "local observations"):
            validate_observations(obs, records, detectors)
        obs[0]["detector_id"] = "global"
        with self.assertRaisesRegex(ValueError, "global observations"):
            validate_observations(obs, records, detectors)
        obs[0].update(start_sec=0.0, end_sec=2.0)
        self.assertTrue(validate_observations(obs, records, detectors)["valid"])
        obs[0]["end_sec"] = 1.0
        with self.assertRaisesRegex(ValueError, "whole recording"):
            validate_observations(obs, records, detectors)

    def test_out_of_bounds_or_unknown_recording_is_rejected(self):
        records, detectors, obs = fixture()
        obs[0]["end_sec"] = 3.0
        with self.assertRaisesRegex(ValueError, "duration"):
            validate_observations(obs, records, detectors)
        obs[0].update(end_sec=0.5, recording_id="unknown")
        with self.assertRaisesRegex(ValueError, "unknown recording_id"):
            validate_observations(obs, records, detectors)

    def test_predict_accepts_absent_labels(self):
        records, detectors, obs = fixture()
        del records[0]["label"]
        del records[0]["fake_intervals"]
        self.assertTrue(validate_observations(obs, records, detectors)["valid"])

    def test_only_measured_quality_allowlist_is_accepted(self):
        records, detectors, obs = fixture()
        obs[0]["quality"]["true_codec_id"] = 3
        with self.assertRaisesRegex(ValueError, "unsupported quality"):
            validate_observations(obs, records, detectors)
        del obs[0]["quality"]["true_codec_id"]
        obs[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "keep labels"):
            validate_observations(obs, records, detectors)

    def test_ambiguous_duplicate_cell_cannot_be_silently_overwritten(self):
        records, detectors, obs = fixture()
        other = dict(obs[0], observation_id="other", run_id="another-run", score=0.2)
        with self.assertRaisesRegex(ValueError, "multiple non-equivalent"):
            validate_observations(obs + [other], records, detectors)


class DeduplicationTests(unittest.TestCase):
    def test_copy_chain_keeps_original_and_does_not_mutate_input(self):
        _, _, obs = fixture()
        original = copy.deepcopy(obs)
        copy1 = dict(obs[0], observation_id="copy1", derived_from="o1", run_id="wrapper1")
        copy2 = dict(obs[0], observation_id="copy2", derived_from="copy1", run_id="wrapper2")
        kept, report = deduplicate_observations([copy2, copy1] + obs)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["observation_id"], "o1")
        self.assertEqual(report["removed_count"], 2)
        self.assertEqual(obs, original)

    def test_same_provenance_collapses_and_contradiction_fails(self):
        _, _, obs = fixture()
        obs[0]["provenance_id"] = "one-output"
        alias = dict(obs[0], observation_id="alias", run_id="wrapper")
        self.assertEqual(len(deduplicate_observations(obs + [alias])[0]), 1)
        alias["score"] = 0.1
        with self.assertRaisesRegex(ValueError, "contradictory copies"):
            deduplicate_observations(obs + [alias])

    def test_same_recording_provenance_cannot_accidentally_merge_windows(self):
        _, _, obs = fixture()
        obs[0]["provenance_id"] = "incorrectly-shared-WAV-id"
        other = dict(obs[0], observation_id="o2", start_sec=0.5, end_sec=1.0)
        with self.assertRaisesRegex(ValueError, "contradictory copies"):
            deduplicate_observations(obs + [other])

    def test_exact_detector_repeat_collapses_but_independent_detector_remains(self):
        _, _, obs = fixture()
        repeat = dict(obs[0], observation_id="repeat", run_id="run-two")
        other_detector = dict(obs[0], observation_id="other", detector_id="d2")
        self.assertEqual(len(deduplicate_observations(obs + [repeat, other_detector])[0]), 2)

    def test_cycles_and_unresolved_sources_fail(self):
        _, _, obs = fixture()
        for source in ("o1", "does-not-exist"):
            with self.subTest(source=source):
                candidate = dict(obs[0], derived_from=source)
                with self.assertRaises(ValueError):
                    deduplicate_observations([candidate])
        first = dict(obs[0], derived_from="o2")
        second = dict(obs[0], observation_id="o2", derived_from="o1")
        with self.assertRaisesRegex(ValueError, "cycle"):
            deduplicate_observations([first, second])

    def test_explicit_cross_detector_alias_requires_compatible_registry(self):
        records, detectors, obs = fixture()
        alias_detector = dict(detectors["detectors"][0], detector_id="alias-detector")
        detectors["detectors"].append(alias_detector)
        alias = dict(obs[0], observation_id="copy", detector_id="alias-detector", derived_from="o1")
        kept, _ = deduplicate_observations([alias] + obs)
        self.assertEqual(kept[0]["detector_id"], "d1")
        self.assertTrue(validate_observations(obs + [alias], records, detectors)["valid"])
        alias_detector["score_type"] = "margin"
        with self.assertRaisesRegex(ValueError, "detector semantics"):
            validate_observations(obs + [alias], records, detectors)

    def test_id_conflicts_fail(self):
        _, _, obs = fixture()
        with self.assertRaisesRegex(ValueError, "reuse observation_id"):
            deduplicate_observations(obs + [dict(obs[0], run_id="not-the-same-record")])


class RDFTests(unittest.TestCase):
    def test_export_has_real_relations_and_excludes_supervision(self):
        records, detectors, obs = fixture()
        obs += [dict(obs[0], observation_id="o2", start_sec=0.5, end_sec=1.0),
                dict(obs[0], observation_id="o3", detector_id="d2", score=-1.2),
                dict(obs[0], observation_id="copy", derived_from="o1"),
                dict(obs[0], observation_id="missing", start_sec=1.0, end_sec=1.5,
                     score=None, status="failed")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.ttl"
            report = export_graph(records, obs, detectors, path)
            ttl = path.read_text()
            self.assertEqual(report["same_family_edge_count"], 2)
            self.assertEqual(report["temporal_edge_count"], 2)
            self.assertEqual(report["duplicate_count"], 1)
            self.assertIn("knownDuplicateOf", ttl)
            self.assertIn("nextTemporalObservation", ttl)
            self.assertNotIn("SUPERVISION_MUST_NOT_APPEAR", ttl)
            self.assertNotIn("TRUE_CODEC_MUST_NOT_APPEAR", ttl)
            self.assertNotIn("fake_intervals", ttl)
            self.assertNotIn('"None"', ttl)
            if importlib.util.find_spec("rdflib"):
                import rdflib
                graph = rdflib.Graph().parse(path, format="turtle")
                self.assertEqual(len(list(graph.triples((None, rdflib.URIRef(NS + "score"), None)))), 4)

    def test_missing_optional_validation_dependencies_is_not_a_pass(self):
        with patch("ontoforensics.ontology.importlib.util.find_spec", return_value=None):
            report = validate_rdf("not-needed-when-dependencies-missing.ttl")
        self.assertFalse(report["validated"])
        self.assertIsNone(report["conforms"])
        self.assertIn("pip install", report["install_command"])

    @unittest.skipUnless(importlib.util.find_spec("rdflib") and importlib.util.find_spec("pyshacl"),
                         "rdflib and pyshacl are optional")
    def test_shacl_accepts_export_and_rejects_modified_invalid_score(self):
        import rdflib
        records, detectors, obs = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.ttl"
            export_graph(records, obs, detectors, path)
            self.assertTrue(validate_rdf(path)["conforms"])
            graph = rdflib.Graph().parse(path, format="turtle")
            predicate = rdflib.URIRef(NS + "score")
            subject = next(graph.subjects(predicate, None))
            graph.set((subject, predicate, rdflib.Literal(1.8, datatype=rdflib.XSD.double)))
            graph.serialize(path, format="turtle")
            with self.assertRaisesRegex(ValueError, "does not conform"):
                validate_rdf(path)

    def test_installed_package_resource_fallback(self):
        import ontoforensics.ontology as module
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "installed" / "ontoforensics"
            resources = package / "resources"
            resources.mkdir(parents=True)
            expected = resources / "forensics.ttl"
            expected.write_text("# resource fixture\n")
            with patch.object(module, "__file__", str(package / "ontology.py")):
                self.assertEqual(module._ontology_resource("forensics.ttl"), expected)

    def test_relation_configuration_is_machine_readable(self):
        path = Path(__file__).resolve().parents[1] / "ontology" / "relations.json"
        config = json.loads(path.read_text())
        self.assertEqual(set(config["relations"]), {"same_family", "temporal_context", "quality_interaction"})
        self.assertFalse(config["input_policy"]["known_duplicate_counts_are_features"])


if __name__ == "__main__":
    unittest.main()
