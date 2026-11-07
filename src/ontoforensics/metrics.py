"""Explicit fake-positive metrics. Thresholds are selected on development only."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def choose_threshold(y, probability, target_fpr=0.05, weight=None):
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0,1)")
    negative = np.asarray(y) == 0
    negatives = np.asarray(probability)[negative]
    if not len(negatives):
        raise ValueError("development set needs bona fide examples for threshold selection")
    weights = np.ones(len(negatives)) if weight is None else np.asarray(weight)[negative]
    if not np.isfinite(negatives).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("threshold selection needs finite scores and positive weights")
    order = np.argsort(negatives)
    cutoff = np.searchsorted(np.cumsum(weights[order]), (1-target_fpr)*weights.sum(), side="left")
    cutoff = min(len(order)-1, int(cutoff))
    return float(np.nextafter(negatives[order[cutoff]], np.inf))


def binary_metrics(y, probability, threshold=0.5, weight=None):
    y, p = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if not len(y):
        return {"n": 0, "reason": "no labelled usable observations"}
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("metrics require finite probabilities in [0,1]")
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    pred = p >= threshold
    tp = float(w[(y == 1) & pred].sum())
    fp = float(w[(y == 0) & pred].sum())
    tn = float(w[(y == 0) & ~pred].sum())
    fn = float(w[(y == 1) & ~pred].sum())
    div = lambda a, b: float(a / b) if b > 0 else None
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    result = {"n": int(len(y)), "threshold": float(threshold),
              "positive_class": "edited_or_synthetic", "tp": tp, "fp": fp, "tn": tn, "fn": fn,
              "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
              "fpr": div(fp, fp + tn), "f1": div(2 * tp, 2 * tp + fp + fn),
              "accuracy": div(tp + tn, w.sum()),
              "brier": float(np.average((p - y) ** 2, weights=w)),
              "nll": float(np.average(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)), weights=w))}
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, p, sample_weight=w))
        result["average_precision"] = float(average_precision_score(y, p, sample_weight=w))
        fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
        fnr = 1 - tpr
        delta = fpr - fnr
        k = int(np.flatnonzero(delta >= 0)[0])
        if k == 0 or delta[k] == delta[k-1]:
            eer = (fpr[k] + fnr[k]) / 2
        else:
            alpha = -delta[k-1] / (delta[k] - delta[k-1])
            eer = fpr[k-1] + alpha * (fpr[k] - fpr[k-1])
        result["eer_interpolated"] = float(eer)
    else:
        result.update(auroc=None, average_precision=None, eer_interpolated=None)
    return result


def risk_coverage(y, p, threshold=.5, n_total=None):
    y, p = np.asarray(y), np.asarray(p)
    if not len(y):
        return []
    n_total = len(y) if n_total is None else int(n_total)
    decisions = p >= threshold
    confidence = np.where(decisions, p, 1-p)
    order = np.argsort(-confidence, kind="stable")
    result = []
    for fraction in (0.25, 0.5, 0.75, 0.9, 1.0):
        n = max(1, int(np.ceil(len(y) * fraction)))
        idx = order[:n]
        result.append({"coverage_of_all_recordings": n / n_total,
                       "coverage_of_usable_recordings": n / len(y),
                       "n_selected": n, "n_usable": len(y), "n_total": n_total,
                       "classification_error_at_frozen_threshold": float(np.mean(decisions[idx] != y[idx])),
                       "threshold": float(threshold), "minimum_decision_confidence": float(confidence[idx].min())})
    return result


def intervals_from_frames(starts, ends, positive):
    intervals = []
    for start, end, active in zip(starts, ends, positive):
        if not active:
            continue
        if intervals and abs(float(start) - intervals[-1][1]) < 1e-7:
            intervals[-1][1] = float(end)
        else:
            intervals.append([float(start), float(end)])
    return intervals


def bootstrap_utterance(y, p, groups, threshold, iterations=200, seed=17):
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2 or iterations <= 0:
        return {"iterations": 0, "reason": "insufficient groups or disabled"}
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("auroc", "fpr", "recall", "brier")}
    for _ in range(iterations):
        selected = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in selected])
        result = binary_metrics(y[idx], p[idx], threshold)
        for key in values:
            if result[key] is not None:
                values[key].append(result[key])
    return {"iterations": int(iterations), "n_groups": int(len(unique)), "resampling_unit": "source_recording_id",
            "percentile_95_ci": {key: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if v else None
                                 for key, v in values.items()}}
