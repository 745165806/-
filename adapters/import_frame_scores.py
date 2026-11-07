#!/usr/bin/env python3
"""Import real, native frame scores without consulting supervision fields.

Accepted inputs: numeric NPY (one recording), numeric NPZ (one named array per
recording), or a long CSV. Pickled/object arrays are deliberately unsupported.
This module also supplies small, label-free helpers used by export_cfprf.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


ADAPTER_VERSION = "2"
SCORE_TYPES = ("logit", "probability", "margin")
_FORBIDDEN_SCORE_COLUMNS = {
    "label", "fake_intervals", "target", "target_label", "class_label", "is_fake"
}


@dataclass(frozen=True)
class Recording:
    recording_id: str
    audio_path: str
    duration_sec: float
    sample_rate: int
    split: str


def read_manifest(path: Path) -> Dict[str, Recording]:
    """Whitelist inference metadata; do not access audit/supervision values."""
    recordings = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Manifest line %d must be a JSON object" % lineno)
            rid = row.get("recording_id")
            audio = row.get("audio_path")
            duration = row.get("duration_sec")
            rate = row.get("sample_rate")
            split = row.get("split")
            if not isinstance(rid, str) or not rid.strip() or rid in recordings:
                raise ValueError("Missing or duplicate recording_id at line %d" % lineno)
            if not isinstance(audio, str) or not audio.strip():
                raise ValueError("Missing audio_path for %s" % rid)
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                raise ValueError("duration_sec must be numeric for %s" % rid)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration_sec must be finite and positive for %s" % rid)
            if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
                raise ValueError("sample_rate must be a positive integer for %s" % rid)
            if not isinstance(split, str) or not split:
                raise ValueError("Missing split for %s" % rid)
            recordings[rid] = Recording(rid, audio, float(duration), rate, split)
    if not recordings:
        raise ValueError("Manifest contains no recordings")
    return recordings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checked_sha256(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("checkpoint_sha256 must be a 64-character SHA-256 hex digest")
    return value.lower()


def canonical_scores(values: np.ndarray, score_type: str, layout: str = "scalar",
                     direction: str = "higher-is-fake", fake_class_index: int = 0) -> np.ndarray:
    """Orient scores toward fake; never infer calibration or score semantics."""
    values = np.asarray(values)
    if values.dtype.kind not in "fiu" or values.size == 0:
        raise ValueError("Scores must be a nonempty numeric array, not labels/objects/strings")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Scores contain NaN or infinity")
    if score_type not in SCORE_TYPES:
        raise ValueError("Unsupported score type: %s" % score_type)
    if direction not in ("higher-is-fake", "higher-is-bonafide"):
        raise ValueError("Unknown score direction")
    if score_type == "probability" and ((values < 0).any() or (values > 1).any()):
        raise ValueError("Probability scores must lie in [0, 1]")
    if layout == "scalar":
        if values.ndim != 1:
            raise ValueError("Scalar layout requires shape [T]; do not flatten multi-class arrays")
        if direction == "higher-is-bonafide":
            return 1.0 - values if score_type == "probability" else -values
        return values
    if layout != "two-class" or values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Two-class layout requires shape [T, 2]")
    if fake_class_index not in (0, 1):
        raise ValueError("fake_class_index must be 0 or 1")
    if direction != "higher-is-fake":
        raise ValueError("For two-class scores, specify the fake class index, not direction reversal")
    if score_type == "probability":
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5, rtol=0):
            raise ValueError("Two-class probabilities must sum to one in every frame")
        return values[:, fake_class_index]
    return values[:, fake_class_index] - values[:, 1 - fake_class_index]


def checked_intervals(intervals: np.ndarray, duration_sec: float) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.ndim != 2 or intervals.shape[1] != 2 or len(intervals) == 0:
        raise ValueError("Intervals must have nonempty shape [T, 2]")
    if not np.isfinite(intervals).all():
        raise ValueError("Intervals contain NaN or infinity")
    starts, ends = intervals[:, 0], intervals[:, 1]
    if (starts < 0).any() or (ends <= starts).any():
        raise ValueError("Every interval must satisfy 0 <= start_sec < end_sec")
    if (ends > duration_sec + 1e-9).any():
        raise ValueError("Frame grid exceeds recording duration; no automatic clipping or stretching")
    if len(intervals) > 1 and (starts[1:] < ends[:-1] - 1e-9).any():
        raise ValueError("Frame intervals must be ordered and non-overlapping")
    return intervals


def frame_intervals(count: int, duration_sec: float, hop_sec: float,
                    start_sec: float = 0.0) -> np.ndarray:
    if count <= 0 or not math.isfinite(hop_sec) or hop_sec <= 0:
        raise ValueError("Frame count and frame hop must be positive")
    if not math.isfinite(start_sec) or start_sec < 0:
        raise ValueError("Frame start must be finite and nonnegative")
    starts = start_sec + np.arange(count, dtype=np.float64) * hop_sec
    return checked_intervals(np.column_stack((starts, starts + hop_sec)), duration_sec)


def frame_observations(recording: Recording, scores: np.ndarray, intervals: np.ndarray,
                       detector_id: str, run_id: str, input_lineage_id: str) -> Iterator[dict]:
    """One provenance identity per native measurement, stable across run IDs.

    The caller supplies the hash of input lineage and detector semantics. It is
    deliberately not used as an output provenance ID: separate frames must not
    be mistaken for copies of the same measurement by ontology deduplication.
    """
    if len(scores) != len(intervals):
        raise ValueError("Score and interval counts differ")
    for index, (score, interval) in enumerate(zip(scores, intervals)):
        start, end = (float(interval[0]), float(interval[1]))
        provenance_id = stable_hash({
            "measurement_type": "native_frame_score", "adapter_contract_version": ADAPTER_VERSION,
            "input_lineage_id": input_lineage_id,
            "recording_id": recording.recording_id, "detector_id": detector_id,
            "frame_index": index, "start_sec": start, "end_sec": end,
        })
        identity = {"run_id": run_id, "provenance_id": provenance_id}
        yield {
            "observation_id": "obs_" + stable_hash(identity)[:32],
            "recording_id": recording.recording_id,
            "detector_id": detector_id, "run_id": run_id,
            "start_sec": start, "end_sec": end, "score": float(score),
            "status": "valid", "provenance_id": provenance_id,
            "quality": {},
        }


def registry_entry(detector_id: str, family: str, score_type: str,
                   checkpoint_sha256: str, **extra: object) -> dict:
    if not detector_id.strip() or not family.strip() or score_type not in SCORE_TYPES:
        raise ValueError("A nonempty detector_id, family, and supported score type are required")
    return {"detector_id": detector_id, "family": family, "score_type": score_type,
            "higher_is_fake": True, "scope": "local",
            "checkpoint_sha256": checked_sha256(checkpoint_sha256),
            "synthetic": False, **extra}


def protect_input_paths(outputs: Sequence[Path], inputs: Sequence[Path]) -> None:
    resolved_outputs = [Path(p).resolve() for p in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Output observations and registry must be different files")
    if set(resolved_outputs).intersection(Path(p).resolve() for p in inputs):
        raise ValueError("An output path would overwrite an input")


def write_bundle(observations: Iterable[dict], entry: dict, out: Path, registry: Path) -> int:
    """Stage both files; inference/input failure does not publish partial results."""
    out, registry = Path(out), Path(registry)
    protect_input_paths((out, registry), ())
    payload = {"detectors": []}
    if registry.exists():
        payload = json.loads(registry.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("detectors"), list):
            raise ValueError("Existing registry must contain a detectors list")
        if any(not isinstance(item, dict) or not isinstance(item.get("detector_id"), str)
               or not item["detector_id"].strip() for item in payload["detectors"]):
            raise ValueError("Every existing registry entry must have a nonempty detector_id")
        ids = [item.get("detector_id") for item in payload["detectors"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Existing registry contains duplicate detector IDs")
    found = False
    for existing in payload["detectors"]:
        if existing.get("detector_id") == entry["detector_id"]:
            if existing != entry:
                raise ValueError("Registry entry conflicts with this detector; use a new detector_id")
            found = True
    if not found:
        payload["detectors"].append(entry)
    out.parent.mkdir(parents=True, exist_ok=True)
    registry.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent,
                                         prefix=".frames-", delete=False) as stream:
            temporary.append(Path(stream.name))
            count = 0
            for observation in observations:
                stream.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
            if count == 0:
                raise ValueError("No frame observations were produced")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=registry.parent,
                                         prefix=".registry-", delete=False) as stream:
            temporary.append(Path(stream.name))
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary[0], out)
        os.replace(temporary[1], registry)
        return count
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()


def _csv_batches(path: Path, layout: str, time_mode: str, score_column: str,
                 class_columns: Sequence[str]) -> Iterator[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
    columns = [score_column] if layout == "scalar" else list(class_columns)
    if len(set(columns)) != len(columns) or any(c.lower() in _FORBIDDEN_SCORE_COLUMNS for c in columns):
        raise ValueError("Score columns must be distinct detector outputs, never supervision columns")
    grouped = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"recording_id", *columns}
        required.update({"frame_index"} if time_mode == "grid" else {"start_sec", "end_sec"})
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV requires columns: %s" % ", ".join(sorted(required)))
        for line_number, row in enumerate(reader, 2):
            rid = row["recording_id"]
            if not rid:
                raise ValueError("CSV recording_id is empty at line %d" % line_number)
            scores, times = grouped.setdefault(rid, ([], []))
            values = [float(row[column]) for column in columns]
            scores.append(values[0] if layout == "scalar" else values)
            if time_mode == "grid":
                try:
                    index = int(row["frame_index"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("CSV frame_index must be a zero-based integer") from exc
                if index != len(scores) - 1:
                    raise ValueError("Frame indices for %s must start at 0 and be contiguous in file order" % rid)
            else:
                times.append([float(row["start_sec"]), float(row["end_sec"])])
    for rid, (scores, times) in grouped.items():
        yield rid, np.asarray(scores), np.asarray(times) if time_mode == "intervals" else None


def score_batches(path: Path, recording_id: Optional[str], layout: str, time_mode: str,
                  score_column: str, class_columns: Sequence[str]) -> Iterator[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        if recording_id is not None:
            raise ValueError("CSV already contains recording_id; omit --recording-id")
        yield from _csv_batches(path, layout, time_mode, score_column, class_columns)
        return
    if time_mode != "grid":
        raise ValueError("Explicit intervals are supported only in CSV")
    if suffix == ".npy":
        if not recording_id:
            raise ValueError("A single NPY array requires --recording-id")
        yield recording_id, np.load(path, allow_pickle=False), None
    elif suffix == ".npz":
        if recording_id is not None:
            raise ValueError("NPZ keys are recording IDs; omit --recording-id")
        with np.load(path, allow_pickle=False) as arrays:
            if len(set(arrays.files)) != len(arrays.files):
                raise ValueError("NPZ contains duplicate member names")
            for rid in arrays.files:
                yield rid, arrays[rid], None
    else:
        raise ValueError("Supported score files are numeric .npy/.npz and .csv; pickle is not loaded")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--detector-id", required=True)
    parser.add_argument("--family", required=True, help="Shared encoder/source family for dependence modeling")
    parser.add_argument("--run-id", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint", type=Path)
    checkpoint.add_argument("--checkpoint-sha256")
    parser.add_argument("--input-score-type", choices=SCORE_TYPES, required=True)
    parser.add_argument("--direction", choices=("higher-is-fake", "higher-is-bonafide"), default="higher-is-fake")
    parser.add_argument("--layout", choices=("scalar", "two-class"), default="scalar")
    parser.add_argument("--fake-class-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--recording-id", help="Only for a single-recording NPY file")
    parser.add_argument("--time-mode", choices=("grid", "intervals"), default="grid")
    parser.add_argument("--frame-hop-sec", type=float, help="Native model hop; required in grid mode")
    parser.add_argument("--frame-start-sec", type=float, default=0.0)
    parser.add_argument("--score-column", default="score", help="Scalar CSV output column")
    parser.add_argument("--class-columns", nargs=2, default=("score0", "score1"))
    parser.add_argument("--allow-subset", action="store_true", help="Explicitly allow recordings absent from scores")
    return parser


def run_import(args: argparse.Namespace) -> int:
    if not args.run_id.strip():
        raise ValueError("run_id must be nonempty")
    if args.time_mode == "grid" and args.frame_hop_sec is None:
        raise ValueError("--frame-hop-sec is required; native frame timing is never inferred from labels")
    if args.time_mode == "intervals" and (args.frame_hop_sec is not None or args.frame_start_sec != 0):
        raise ValueError("Explicit interval CSV must not also specify a grid")
    inputs = [args.manifest, args.scores] + ([args.checkpoint] if args.checkpoint else [])
    protect_input_paths((args.out, args.registry), inputs)
    recordings = read_manifest(args.manifest)
    checkpoint_hash = sha256_file(args.checkpoint) if args.checkpoint else checked_sha256(args.checkpoint_sha256)
    artifact_hash = sha256_file(args.scores)
    transform = {"layout": args.layout, "direction": args.direction,
                 "input_score_type": args.input_score_type, "fake_class_index": args.fake_class_index,
                 "time_mode": args.time_mode, "frame_hop_sec": args.frame_hop_sec,
                 "frame_start_sec": args.frame_start_sec,
                 "score_column": args.score_column, "class_columns": list(args.class_columns)}
    entry = registry_entry(args.detector_id, args.family, args.input_score_type, checkpoint_hash,
                           adapter="import_frame_scores", adapter_version=ADAPTER_VERSION,
                           input_score_semantics=transform)

    def observations() -> Iterator[dict]:
        seen = set()
        for rid, raw, times in score_batches(args.scores, args.recording_id, args.layout,
                                             args.time_mode, args.score_column, args.class_columns):
            if rid not in recordings or rid in seen:
                raise ValueError("Unknown or duplicate recording ID in scores: %s" % rid)
            seen.add(rid)
            record = recordings[rid]
            scores = canonical_scores(raw, args.input_score_type, args.layout,
                                      args.direction, args.fake_class_index)
            if times is None:
                times = frame_intervals(len(scores), record.duration_sec, args.frame_hop_sec, args.frame_start_sec)
            else:
                times = checked_intervals(times, record.duration_sec)
            input_lineage = stable_hash({"adapter": "import_frame_scores", "version": ADAPTER_VERSION,
                                      "artifact_sha256": artifact_hash, "recording_id": rid,
                                      "detector_id": args.detector_id, "checkpoint_sha256": checkpoint_hash,
                                      "transform": transform})
            yield from frame_observations(record, scores, times, args.detector_id, args.run_id, input_lineage)
        if not args.allow_subset and seen != set(recordings):
            missing = sorted(set(recordings) - seen)
            raise ValueError("Missing score records: %s; use --allow-subset only intentionally" % ", ".join(missing[:10]))

    return write_bundle(observations(), entry, args.out, args.registry)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        count = run_import(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit("Frame-score import failed: %s" % exc) from exc
    print("Imported %d native frame observations into %s" % (count, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
