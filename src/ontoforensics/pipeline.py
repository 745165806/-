"""Train/cache/predict workflow with separate train, dev, calibration and test.

Saved models are plain JSON numeric arrays, never pickle. The implemented
relational baseline is a feature model, NOT a trained CRF, GNN or SOTA claim.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .features import build_features, fit_score_normalization, make_targets, read_relations
from .io import write_json, write_jsonl
from .metrics import binary_metrics, bootstrap_utterance, choose_threshold, intervals_from_frames, risk_coverage

MODES = ("mean", "flat_lr", "flat_mlp", "relation_lr")


def _freeze(estimator):
    if isinstance(estimator, LogisticRegression):
        return {"kind": "linear", "coef": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0])}
    return {"kind": "mlp", "weights": [a.tolist() for a in estimator.coefs_],
            "biases": [a.tolist() for a in estimator.intercepts_]}


def _raw(model, X):
    X = np.asarray(X, dtype=float)
    if model["kind"] == "linear":
        return X @ np.asarray(model["coef"]) + model["intercept"]
    if model["kind"] == "mlp":
        value = X
        for i, (weights, bias) in enumerate(zip(model["weights"], model["biases"])):
            value = value @ np.asarray(weights) + np.asarray(bias)
            if i < len(model["weights"]) - 1:
                value = np.maximum(value, 0)
        return value[:, 0]
    raise ValueError(f"unknown model kind {model['kind']}")


def _fit_linear(X, y, C=1.0):
    if len(np.unique(y)) != 2:
        raise ValueError("fitting requires both bona fide and edited examples")
    estimator = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    estimator.fit(X, y)
    return _freeze(estimator)


def _normalize(model, X):
    return (X - np.asarray(model["scaler_mean"])) / np.asarray(model["scaler_scale"])


def _frame_raw(model, table):
    X = _normalize(model, table.X)
    if model["mode"] == "mean":
        mask = table.X[:, table.mask_columns]
        score = X[:, table.score_columns]
        return np.divide((score * mask).sum(axis=1), mask.sum(axis=1),
                         out=np.zeros(len(X)), where=mask.sum(axis=1) > 0)
    return _raw(model["frame_estimator"], X)


def _calibrated(calibrator, values):
    return expit(_raw(calibrator, np.asarray(values)[:, None]))


def _utterance_features(table, p, records):
    """Learned pooling rather than independent-window noisy-OR."""
    rows, ids, usable = [], [], []
    for record in records:
        idx = np.flatnonzero((table.recording_ids == record["recording_id"]) & table.usable)
        pp = p[idx]
        if len(pp):
            row = [float(pp.mean()), float(pp.std()), float(pp.max()), float(np.quantile(pp, .9)),
                   float(np.mean(pp >= .5)), float(np.log1p(record["duration_sec"])),
                   float(np.sum(table.ends[idx] - table.starts[idx]) / record["duration_sec"])]
        else:
            row = [0, 0, 0, 0, 0, float(np.log1p(record["duration_sec"])), 0]
        rows.append(row)
        ids.append(record["recording_id"])
        usable.append(bool(len(pp)))
    return np.asarray(rows), np.asarray(ids), np.asarray(usable)


def _prepare(records, observations, detectors):
    from .data import validate_manifest
    from .ontology import deduplicate_observations, validate_observations
    manifest_report = validate_manifest(records, check_audio=False)
    observation_report = validate_observations(observations, records, detectors)
    observations, dedup_report = deduplicate_observations(observations)
    return observations, {"manifest": manifest_report, "observations": observation_report, "deduplication": dedup_report}


def train(records, observations, detectors, output, *, mode="relation_lr", relations=None,
          step_sec=.04, context_sec=.4, target_fpr=.05, seed=17):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    observations, validation = _prepare(records, observations, detectors)
    # Validate the complete split manifest first, but do not load test labels
    # into the training target construction or use test observations in fits.
    records = [r for r in records if r["split"] in ("train", "dev", "calibration")]
    if any(r.get("label") not in (0, 1) for r in records):
        raise ValueError("train/dev/calibration records must all have explicit labels; unknown labels are for inference only")
    fit_ids = {r["recording_id"] for r in records}
    observations = [o for o in observations if o["recording_id"] in fit_ids]
    relations = relations or read_relations()
    score_normalization = fit_score_normalization(records, observations, detectors)
    table = build_features(records, observations, detectors, step_sec=step_sec,
                           context_sec=context_sec, relational=mode == "relation_lr", relations=relations,
                           score_normalization=score_normalization)
    targets = make_targets(table, records)
    split_by_id = {r["recording_id"]: r["split"] for r in records}
    splits = np.asarray([split_by_id[rid] for rid in table.recording_ids])
    masks = {key: (splits == key) & table.usable & (targets >= 0)
             for key in ("train", "dev", "calibration")}
    for key, mask in masks.items():
        if not mask.any() or len(np.unique(targets[mask])) != 2:
            raise ValueError(f"{key} split needs usable labelled frames of BOTH classes; no automatic split fallback")
    scaler = StandardScaler().fit(table.X[masks["train"]])
    model = {"format_version": 1, "package_version": "0.1.0", "mode": mode,
             "step_sec": step_sec, "context_sec": context_sec, "relations": relations,
             "detectors": detectors, "feature_names": table.names,
             "score_normalization": score_normalization,
             "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
             "target_dev_fpr": target_fpr, "seed": seed,
             "is_simulation": any(bool(d.get("synthetic")) for d in detectors),
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "implementation": "typed relation features + learned pooling; no CRF/GNN/OWL probability engine"}
    Xtrain = scaler.transform(table.X[masks["train"]])
    if mode in ("flat_lr", "relation_lr"):
        model["frame_estimator"] = _fit_linear(Xtrain, targets[masks["train"]])
    elif mode == "flat_mlp":
        estimator = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", alpha=.01,
                                  max_iter=250, random_state=seed, early_stopping=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(Xtrain, targets[masks["train"]])
        model["frame_estimator"] = _freeze(estimator)
        model["mlp_iterations"] = int(estimator.n_iter_)
        model["optimization_warnings"] = [str(w.message) for w in caught]
    raw = _frame_raw(model, table)
    model["frame_calibrator"] = _fit_linear(raw[masks["calibration"], None], targets[masks["calibration"]], C=10.)
    frame_p = _calibrated(model["frame_calibrator"], raw)
    duration_weights = table.ends - table.starts
    model["frame_threshold"] = choose_threshold(targets[masks["dev"]], frame_p[masks["dev"]], target_fpr,
                                               duration_weights[masks["dev"]])
    # Pooling learns on uncalibrated sigmoid outputs from the training split.
    # The frame calibrator is never used to train the utterance head.
    U, ids, usable = _utterance_features(table, expit(raw), records)
    labels = np.asarray([r.get("label") if r.get("label") is not None else -1 for r in records])
    usplits = np.asarray([r["split"] for r in records])
    umasks = {key: (usplits == key) & usable & (labels >= 0) for key in masks}
    for key, mask in umasks.items():
        if len(np.unique(labels[mask])) != 2:
            raise ValueError(f"{key} split needs both classes at recording level")
    model["utterance_estimator"] = _fit_linear(U[umasks["train"]], labels[umasks["train"]])
    uraw = _raw(model["utterance_estimator"], U)
    model["utterance_calibrator"] = _fit_linear(uraw[umasks["calibration"], None], labels[umasks["calibration"]], C=10.)
    up = _calibrated(model["utterance_calibrator"], uraw)
    model["utterance_threshold"] = choose_threshold(labels[umasks["dev"]], up[umasks["dev"]], target_fpr)
    fit_groups = {key: sorted({r["source_recording_id"] for r in records if r["split"] == key}) for key in masks}
    model["fit_source_groups"] = fit_groups
    model["fit_speaker_groups"] = {key: sorted({r["speaker_id"] for r in records if r["split"] == key}) for key in masks}
    model["fit_declared_groups"] = {key: sorted({r["group_id"] for r in records if r["split"] == key and r.get("group_id")}) for key in masks}
    report = {"mode": mode, "is_simulation": model["is_simulation"], "validation": validation,
              "feature_count": len(table.names), "feature_names": table.names,
              "split_counts": {key: {"frames": int(mask.sum()), "recordings": int(umasks[key].sum())} for key, mask in masks.items()},
              "frame_development": binary_metrics(targets[masks["dev"]], frame_p[masks["dev"]], model["frame_threshold"],
                                                  duration_weights[masks["dev"]]),
              "frame_metric_weight_unit": "seconds; midpoint target convention",
              "utterance_development": binary_metrics(labels[umasks["dev"]], up[umasks["dev"]], model["utterance_threshold"]),
              "threshold_source": "dev", "calibration_source": "calibration",
              "test_labels_used_for_fitting": False,
              "optimization_warnings": model.get("optimization_warnings", []),
              "scope_note": "Development numbers are not a final test; synthetic demo numbers are never benchmark evidence."}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "model.json", model)
    write_json(output / "train_report.json", report)
    return model, report


def predict(model, records, observations, *, output=None, rejection_confidence=.6):
    if not .5 <= rejection_confidence <= 1:
        raise ValueError("rejection_confidence must be in [.5,1]")
    observations, validation = _prepare(records, observations, model["detectors"])
    table = build_features(records, observations, model["detectors"], step_sec=model["step_sec"],
                           context_sec=model["context_sec"], relational=model["mode"] == "relation_lr",
                           relations=model["relations"], score_normalization=model["score_normalization"])
    if table.names != model["feature_names"]:
        raise ValueError("feature layout differs from the saved model")
    raw = _frame_raw(model, table)
    p = _calibrated(model["frame_calibrator"], raw)
    U, ids, usable = _utterance_features(table, expit(raw), records)
    up = _calibrated(model["utterance_calibrator"], _raw(model["utterance_estimator"], U))
    predictions = []
    for i, record in enumerate(records):
        idx = np.flatnonzero(table.recording_ids == record["recording_id"])
        active = (p[idx] >= model["frame_threshold"]) & table.usable[idx]
        evidence_idx = idx[np.argmax(np.where(table.usable[idx], p[idx], -1))]
        top_contributions = []
        if usable[i] and model.get("frame_estimator", {}).get("kind") == "linear":
            contribution = _normalize(model, table.X[evidence_idx]) * np.asarray(model["frame_estimator"]["coef"])
            order = np.argsort(-np.abs(contribution))[:6]
            top_contributions = [{"feature": table.names[j], "raw_margin_contribution": float(contribution[j])} for j in order]
        probability = float(up[i]) if usable[i] else None
        decision_confidence = float(up[i] if up[i] >= model["utterance_threshold"] else 1-up[i]) if usable[i] else None
        review = not usable[i] or decision_confidence < rejection_confidence
        predictions.append({"recording_id": record["recording_id"], "p_edited": probability,
                            "decision": "review" if review else ("suspected_edit" if up[i] >= model["utterance_threshold"] else "no_target_evidence"),
                            "threshold_decision": bool(up[i] >= model["utterance_threshold"]) if usable[i] else None,
                            "decision_confidence": decision_confidence,
                            "candidate_intervals": intervals_from_frames(table.starts[idx], table.ends[idx], active),
                            "local_coverage": float(np.average(table.usable[idx], weights=table.ends[idx]-table.starts[idx])),
                            "evidence_ids": table.evidence_ids[evidence_idx] if usable[i] else [],
                            "largest_local_score_time": [float(table.starts[evidence_idx]), float(table.ends[evidence_idx])] if usable[i] else None,
                            "local_linear_contributions": top_contributions,
                            "explanation_scope": "Computed feature contributions are not causal proof; no original wording is reconstructed.",
                            "is_simulation": bool(model["is_simulation"])})
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        write_jsonl(output / "predictions.jsonl", predictions)
        frame_rows = ({"recording_id": rid, "start_sec": float(a), "end_sec": float(b),
                       "p_edited": float(pp) if valid else None, "usable": bool(valid)}
                      for rid, a, b, pp, valid in zip(table.recording_ids, table.starts, table.ends, p, table.usable))
        write_jsonl(output / "frame_predictions.jsonl", frame_rows)
        write_json(output / "validation.json", validation)
    return {"predictions": predictions, "table": table, "frame_probability": p,
            "utterance_probability": up, "utterance_usable": usable, "validation": validation}


def evaluate(model, records, observations, output, *, split="test", bootstrap=0):
    from .data import validate_manifest
    validate_manifest(records)
    selected = [r for r in records if r["split"] == split]
    if not selected:
        raise ValueError(f"no records in split {split}")
    if any(r.get("label") not in (0, 1) for r in selected):
        raise ValueError("evaluation requires explicit labels")
    if split == "test":
        fit_groups = set().union(*(set(v) for v in model["fit_source_groups"].values()))
        overlap = fit_groups & {r["source_recording_id"] for r in selected}
        if overlap:
            raise ValueError(f"test source groups overlap training/development/calibration: {sorted(overlap)[:5]}")
        fit_speakers = set().union(*(set(v) for v in model.get("fit_speaker_groups", {}).values()))
        speaker_overlap = fit_speakers & {r["speaker_id"] for r in selected}
        if speaker_overlap:
            raise ValueError(f"test speakers overlap training/development/calibration: {sorted(speaker_overlap)[:5]}")
        fit_declared = set().union(*(set(v) for v in model.get("fit_declared_groups", {}).values()))
        declared_overlap = fit_declared & {r["group_id"] for r in selected if r.get("group_id")}
        if declared_overlap:
            raise ValueError(f"test declared groups overlap fitted splits: {sorted(declared_overlap)[:5]}")
    selected_ids = {r["recording_id"] for r in selected}
    obs = [o for o in observations if o["recording_id"] in selected_ids]
    result = predict(model, selected, obs, output=output)
    table, p = result["table"], result["frame_probability"]
    y = make_targets(table, selected)
    mask = (y >= 0) & table.usable
    labels = np.asarray([r["label"] for r in selected])
    umask, up = result["utterance_usable"], result["utterance_probability"]
    durations = table.ends - table.starts
    report = {"split": split, "mode": model["mode"], "is_simulation": model["is_simulation"],
              "frame": binary_metrics(y[mask], p[mask], model["frame_threshold"], durations[mask]),
              "frame_metric_weight_unit": "seconds; midpoint target convention",
              "utterance": binary_metrics(labels[umask], up[umask], model["utterance_threshold"]),
              "n_recordings": len(selected), "usable_recordings": int(umask.sum()),
              "labelled_frame_coverage": float(durations[mask].sum()/durations[y >= 0].sum()) if (y >= 0).any() else 0.,
              "risk_coverage": risk_coverage(labels[umask], up[umask], model["utterance_threshold"], len(selected)),
              "thresholds_frozen_from_dev": True, "test_recalibration": False,
              "per_condition": {}, "bootstrap": bootstrap_utterance(labels[umask], up[umask],
                    np.asarray([r["source_recording_id"] for r in selected])[umask], model["utterance_threshold"], bootstrap),
              "scope_note": "Simulation validates plumbing only. This baseline has no claim of ontology superiority or real-world accuracy."}
    conditions = np.asarray([str(r.get("condition", "unknown")) for r in selected])
    for condition in sorted(set(conditions)):
        keep = (conditions == condition) & umask
        report["per_condition"][condition] = binary_metrics(labels[keep], up[keep], model["utterance_threshold"])
    write_json(Path(output) / "metrics.json", report)
    return report
