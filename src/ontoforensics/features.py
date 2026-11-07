"""Executable typed-relation features. Only measured observations enter X.

No labels, editor IDs, dataset IDs, speaker IDs or hidden processing labels are
read by build_features. Ground truth enters only make_targets, separately.
This is feature-based relational fusion, not OWL probabilistic inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np

QUALITY_KEYS = ("rms_db", "clipping_fraction", "estimated_snr_db")


@dataclass
class FrameTable:
    X: np.ndarray
    names: list[str]
    recording_ids: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    usable: np.ndarray
    score_columns: list[int]
    mask_columns: list[int]
    evidence_ids: list[list[str]]


def read_relations(path=None):
    if path:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        value = {"same_family": True, "temporal_context": True, "quality_interaction": True}
    # Accept the documented compact switch map, or metadata-wrapped config.
    if "feature_switches" in value:
        value = value["feature_switches"]
    elif "relations" in value and isinstance(value["relations"], dict):
        value = value["relations"]
    result = {}
    for key in ("same_family", "temporal_context", "quality_interaction"):
        raw = value.get(key, True)
        result[key] = bool(raw.get("enabled", True) if isinstance(raw, dict) else raw)
    return result


def margin(score, detector):
    score = float(score)
    if detector["score_type"] == "probability":
        score = np.clip(score, 1e-6, 1 - 1e-6)
        score = np.log(score / (1 - score))
    if not detector["higher_is_fake"]:
        score = -score
    return float(np.clip(score, -20, 20))


def _context(values, valid, radius):
    """Mask-aware moving mean/std; missing is not a genuine zero score."""
    kernel = np.ones(2 * radius + 1)
    def convolve(a):
        return np.convolve(np.pad(a, (radius, radius)), kernel, mode="valid")
    count = convolve(valid.astype(float))
    total = convolve(values * valid)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    second = np.divide(convolve(values ** 2 * valid), count, out=np.zeros_like(total), where=count > 0)
    return mean, np.sqrt(np.maximum(second - mean ** 2, 0))


def fit_score_normalization(records, observations, detectors):
    """Fit one time-weighted location/scale per detector, using train only.

    This is a scale transformation of raw scores, not probability calibration.
    Deduplicate observations before calling this function.
    """
    train_ids = {r["recording_id"] for r in records if r["split"] == "train"}
    duration_by_id = {r["recording_id"]: r["duration_sec"] for r in records}
    result = {}
    for detector in detectors:
        selected = [o for o in observations if o["recording_id"] in train_ids
                    and o["detector_id"] == detector["detector_id"] and o["status"] == "valid"]
        if not selected:
            raise ValueError(f"detector {detector['detector_id']} has no valid training observations")
        values = np.asarray([margin(o["score"], detector) for o in selected])
        weights = np.asarray([(o["end_sec"]-o["start_sec"]) if o.get("start_sec") is not None
                              else duration_by_id[o["recording_id"]] for o in selected])
        mean = float(np.average(values, weights=weights))
        scale = float(np.sqrt(np.average((values-mean)**2, weights=weights)))
        result[detector["detector_id"]] = {"mean": mean, "scale": scale if scale > 1e-8 else 1.}
    return result


def build_features(records, observations, detectors, *, step_sec=0.04, context_sec=0.4,
                   relational=False, relations=None, score_normalization=None):
    if not np.isfinite(step_sec) or step_sec <= 0:
        raise ValueError("step_sec must be positive")
    if not np.isfinite(context_sec) or context_sec < 0:
        raise ValueError("context_sec must be nonnegative")
    relations = relations or read_relations()
    by_record = {}
    for observation in observations:
        by_record.setdefault(observation["recording_id"], []).append(observation)
    detector_ids = [d["detector_id"] for d in detectors]
    if len(set(detector_ids)) != len(detector_ids):
        raise ValueError("duplicate detector IDs")
    detector_index = {key: i for i, key in enumerate(detector_ids)}
    chunks, rids, starts_all, ends_all, usable_all, evidence_all = [], [], [], [], [], []
    names = None
    radius = max(1, round(context_sec / step_sec)) if context_sec else 0
    for record in records:
        rid, duration = record["recording_id"], float(record["duration_sec"])
        if duration <= 0 or not np.isfinite(duration):
            raise ValueError(f"{rid}: invalid duration")
        n = int(np.ceil(duration / step_sec - 1e-10))
        starts = np.arange(n) * step_sec
        ends = np.minimum(starts + step_sec, duration)
        total = np.zeros((n, len(detectors)))
        weight = np.zeros_like(total)
        valid_spans = [[] for _ in detectors]
        qtotal, qweight = np.zeros((n, len(QUALITY_KEYS))), np.zeros((n, len(QUALITY_KEYS)))
        evidence = [[] for _ in range(n)]
        for obs in by_record.get(rid, []):
            if obs["status"] != "valid":
                continue
            j = detector_index[obs["detector_id"]]
            detector = detectors[j]
            if detector["scope"] == "global" and obs.get("start_sec") is None:
                a, b = 0., duration
            else:
                a, b = float(obs["start_sec"]), float(obs["end_sec"])
            # Global observations are context features; they do not count as
            # independent local evidence or make local frames usable.
            overlap = np.maximum(0, np.minimum(ends, b) - np.maximum(starts, a))
            idx = np.flatnonzero(overlap > 1e-10)
            if not len(idx):
                continue
            w = overlap[idx]
            valid_spans[j].append((a, b))
            score = margin(obs["score"], detector)
            if score_normalization:
                transform = score_normalization[detector["detector_id"]]
                score = (score - transform["mean"]) / transform["scale"]
            total[idx, j] += score * w
            weight[idx, j] += w
            for k, key in enumerate(QUALITY_KEYS):
                value = obs.get("quality", {}).get(key)
                if value is not None and np.isfinite(float(value)):
                    qtotal[idx, k] += float(value) * w
                    qweight[idx, k] += w
            for t in idx:
                evidence[t].append(obs["observation_id"])
        # A partly observed grid cell must not extend a native prediction into
        # an unseen tail. Merge native spans to measure union coverage, so
        # overlapping windows cannot falsely count the same milliseconds twice.
        coverage = np.zeros_like(weight)
        for j, spans in enumerate(valid_spans):
            merged = []
            for a, b in sorted(spans):
                if merged and a <= merged[-1][1] + 1e-10:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            for a, b in merged:
                coverage[:, j] += np.maximum(0, np.minimum(ends, b)-np.maximum(starts, a))
        valid = (weight > 0) & (coverage >= (ends-starts)[:, None]-1e-8)
        scores = np.divide(total, weight, out=np.zeros_like(total), where=valid)
        qvalid = qweight > 0
        quality = np.divide(qtotal, qweight, out=np.zeros_like(qtotal), where=qvalid)
        columns, current_names = [], []
        contexts = []
        for j, detector in enumerate(detectors):
            did = detector["detector_id"]
            mu, sd = _context(scores[:, j], valid[:, j], radius)
            contexts.append((mu, sd))
            for suffix, value in [("score", scores[:, j]), ("valid", valid[:, j].astype(float)),
                                  ("context_mean", mu), ("context_std", sd)]:
                columns.append(value)
                current_names.append(f"detector/{did}/{suffix}")
        for k, key in enumerate(QUALITY_KEYS):
            columns.extend([quality[:, k], qvalid[:, k].astype(float)])
            current_names.extend([f"quality/{key}", f"quality/{key}/valid"])
        columns.append(np.full(n, np.log1p(duration)))
        current_names.append("context/log_duration")
        if relational:
            # Type-pooled pair relations tie coefficients by ontology edge type.
            # Relabeling family membership changes these aggregates, not just
            # column names. Products are learned covariates, not probability
            # multiplication or an assertion of conditional independence.
            pair_groups = {"same_family": [], "different_family": []}
            for i, j in combinations(range(len(detectors)), 2):
                pair_valid = (valid[:, i] & valid[:, j]).astype(float)
                same = detectors[i]["family"] == detectors[j]["family"]
                if same and not relations["same_family"]:
                    continue
                relation = "same_family" if same else "different_family"
                pair_groups[relation].append((scores[:, i] * scores[:, j] * pair_valid,
                                              np.abs(scores[:, i] - scores[:, j]) * pair_valid,
                                              pair_valid))
            for relation, pairs in pair_groups.items():
                if not pairs:
                    continue
                total_joint = np.sum([pair[0] for pair in pairs], axis=0)
                total_disagreement = np.sum([pair[1] for pair in pairs], axis=0)
                valid_pairs = np.sum([pair[2] for pair in pairs], axis=0)
                mean_joint = np.divide(total_joint, valid_pairs, out=np.zeros(n), where=valid_pairs > 0)
                mean_disagreement = np.divide(total_disagreement, valid_pairs, out=np.zeros(n), where=valid_pairs > 0)
                prefix = f"relation/{relation}"
                for suffix, value in [
                    ("joint_mean", mean_joint),
                    ("disagreement_mean", mean_disagreement),
                    ("available_fraction", valid_pairs / len(pairs))]:
                    columns.append(value)
                    current_names.append(f"{prefix}/{suffix}")
            if relations["temporal_context"]:
                for j, did in enumerate(detector_ids):
                    residual = np.abs(scores[:, j] - contexts[j][0]) * valid[:, j]
                    columns.append(residual)
                    current_names.append(f"relation/temporal_abs_deviation/{did}")
            if relations["quality_interaction"]:
                # Center/scale constants are physical units, not fitted using labels.
                scales = np.array([40., 1., 30.])
                q = np.clip(quality / scales, -5, 5)
                for j, did in enumerate(detector_ids):
                    for k, key in enumerate(QUALITY_KEYS):
                        columns.append(scores[:, j] * q[:, k] * valid[:, j] * qvalid[:, k])
                        current_names.append(f"relation/condition/{did}/{key}")
        matrix = np.column_stack(columns)
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError("inconsistent feature layout")
        local_indices = [j for j, d in enumerate(detectors) if d["scope"] == "local"]
        usable = valid[:, local_indices].any(axis=1) if local_indices else np.zeros(n, dtype=bool)
        chunks.append(matrix)
        rids.extend([rid] * n)
        starts_all.extend(starts)
        ends_all.extend(ends)
        usable_all.extend(usable)
        evidence_all.extend([sorted(set(ids)) for ids in evidence])
    if not chunks:
        raise ValueError("empty manifest")
    names = names or []
    return FrameTable(np.concatenate(chunks), names, np.asarray(rids), np.asarray(starts_all),
                      np.asarray(ends_all), np.asarray(usable_all),
                      [names.index(f"detector/{d}/score") for d in detector_ids],
                      [names.index(f"detector/{d}/valid") for d in detector_ids], evidence_all)


def make_targets(table, records):
    """Midpoint labeling is a declared grid convention, separate from inference."""
    by_id = {r["recording_id"]: r for r in records}
    targets = np.full(len(table.starts), -1, dtype=int)
    for rid in np.unique(table.recording_ids):
        record = by_id[rid]
        if record.get("label") is None:
            continue
        idx = np.flatnonzero(table.recording_ids == rid)
        mid = (table.starts[idx] + table.ends[idx]) / 2
        labels = np.zeros(len(idx), dtype=int)
        for a, b in record.get("fake_intervals", []):
            labels[(mid >= a) & (mid < b)] = 1
        if record["label"] == 1 and not record.get("fake_intervals"):
            raise ValueError(f"{rid}: local task requires positive interval annotations")
        for a, b in record.get("ignore_intervals", []):
            labels[(mid >= a) & (mid < b)] = -1
        targets[idx] = labels
    return targets
