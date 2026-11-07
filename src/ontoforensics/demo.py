"""A deterministic cached-observation fixture, NOT a speech experiment.

No audio is generated, synthesized, downloaded, or run through a real expert.
The planted relationships intentionally exercise the numerical plumbing.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np

from .io import write_json, write_jsonl


def make_fixture(output, *, seed=17, group_counts=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    group_counts = group_counts or {"train": 24, "dev": 8, "calibration": 8, "test": 12}
    detectors = [
        {"detector_id": "sim_ssl_a", "family": "sim_shared_ssl", "score_type": "margin",
         "higher_is_fake": True, "scope": "local", "synthetic": True},
        {"detector_id": "sim_ssl_b", "family": "sim_shared_ssl", "score_type": "margin",
         "higher_is_fake": True, "scope": "local", "synthetic": True},
        {"detector_id": "sim_spectral", "family": "sim_spectral", "score_type": "margin",
         "higher_is_fake": True, "scope": "local", "synthetic": True},
    ]
    records, observations = [], []
    duration, step = 3.2, .1
    for split, count in group_counts.items():
        for group in range(count):
            source = f"simulation:{split}:{group:03d}"
            edit_start = round(float(rng.uniform(.5, 1.6)), 2)
            edit_end = min(2.8, round(edit_start + float(rng.uniform(.3, .9)), 2))
            speaker = f"simulation-speaker:{split}:{group:03d}"
            source_bias = rng.normal(0, .35)
            for label in (0, 1):
                for codec in (False, True):
                    condition = "sim_codec" if codec else "sim_clean"
                    rid = f"{split}_{group:03d}_{label}_{condition}"
                    record = {"recording_id": rid, "audio_path": str(output / "NO_AUDIO" / f"{rid}.wav"),
                              "source_recording_id": source, "speaker_id": speaker, "dataset": "simulation_fixture",
                              "split": split, "split_origin": "simulation-group-allocation", "duration_sec": duration,
                              "sample_rate": 16000, "label": label,
                              "fake_intervals": [[edit_start, edit_end]] if label else [], "ignore_intervals": [],
                              "editor": "simulated_edit" if label else "none", "condition": condition,
                              "metadata": {"simulation_only": True, "audio_exists": False}}
                    records.append(record)
                    for t in range(32):
                        a, b = round(t*step, 6), round((t+1)*step, 6)
                        local_fake = bool(label and edit_start <= (a+b)/2 < edit_end)
                        common = rng.normal(0, .65) + source_bias
                        latent = 1.3 if local_fake else -1.1
                        scores = [latent + common + (1.1 if codec else 0) + rng.normal(0, .25),
                                  .9*latent + common + (1.0 if codec else 0) + rng.normal(0, .35),
                                  .65*latent + rng.normal(0, 1.) - (.25 if codec else 0)]
                        # These covariates are simulated, not measurements from audio.
                        quality = {"rms_db": float((-26 if codec else -17) + rng.normal(0, 1.2)),
                                   "clipping_fraction": float(rng.uniform(0, .005))}
                        for j, detector in enumerate(detectors):
                            valid = bool(rng.uniform() > .035)
                            oid = f"{rid}:{detector['detector_id']}:{t}"
                            observation = {"observation_id": oid, "recording_id": rid,
                                           "detector_id": detector["detector_id"], "run_id": "simulation-seed-"+str(seed),
                                           "start_sec": a, "end_sec": b, "score": float(scores[j]) if valid else None,
                                           "status": "valid" if valid else "missing", "provenance_id": oid,
                                           "quality": quality}
                            observations.append(observation)
                            if group % 8 == 0 and t == 8 and j == 0:
                                duplicate = deepcopy(observation)
                                duplicate["observation_id"] += ":copy"
                                duplicate["derived_from"] = oid
                                observations.append(duplicate)
    write_jsonl(output / "manifest.jsonl", records)
    write_jsonl(output / "observations.jsonl", observations)
    write_json(output / "detectors.json", {"detectors": detectors})
    write_json(output / "SIMULATION_ONLY.json", {
        "purpose": "Engineering smoke test; cached numerical observations only; no real audio or models.",
        "seed": seed, "recordings": len(records), "observations": len(observations),
        "not_valid_for": ["benchmark scores", "claims of ontology benefit", "claims of generalization"],
    })
    return records, observations, detectors


def run_demo(output, *, seed=17, with_rdf=False):
    from .pipeline import train, evaluate, MODES
    output = Path(output)
    records, observations, detectors = make_fixture(output / "fixture", seed=seed)
    summary = {"simulation_only": True, "real_audio_inference_performed": False,
               "purpose": "The numbers below test code execution only; they are not research results.",
               "recordings": len(records), "observations": len(observations), "models": {}}
    for mode in MODES:
        model, report = train(records, observations, detectors, output / mode,
                              mode=mode, step_sec=.1, context_sec=.3, seed=seed)
        result = evaluate(model, records, observations, output / mode / "test", bootstrap=20)
        summary["models"][mode] = {"feature_count": report["feature_count"],
                                    "test_utterance": result["utterance"],
                                    "test_frame": result["frame"],
                                    "simulation_only": True}
    if with_rdf:
        from .ontology import export_graph, validate_rdf
        # A small graph keeps this smoke test fast; the full export CLI is separate.
        sample = records[:4]
        selected_ids = {r["recording_id"] for r in sample}
        graph = output / "evidence_sample.ttl"
        summary["rdf_export"] = export_graph(sample, [o for o in observations if o["recording_id"] in selected_ids], detectors, graph)
        summary["shacl"] = validate_rdf(graph)
    write_json(output / "demo_summary.json", summary)
    return summary
