"""Validated recording manifests and the public PartialEdit data adapter.

Labels, editing systems and processing conditions are supervision/audit data.
They are not inference-time features. This module never downloads audio or models.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tempfile
from typing import Any, Iterable, Mapping
import warnings
import zipfile


SPLITS = frozenset({"train", "dev", "calibration", "test", "unspecified"})
PARTIALEDIT_SUBSETS = ("E1", "E1-Codec", "E2", "E2-Codec")
_SPEAKER_RE = re.compile(r"(?:p\d+|s\d+)\Z")
_EDITED_RE = re.compile(r"((p\d+|s\d+)_\d+)_edited_partial_16k\.wav\Z")
_BONA_FIDE_RE = re.compile(
    r"(?P<source>(?P<speaker>p\d+|s\d+)_\d+)"
    r"(?:_(?P<mic>mic[12]))?(?:_16k)?\.(?:wav|flac)\Z", re.IGNORECASE
)
_DURATION_TOLERANCE_SEC = 0.025  # Published annotations are quantized in time.
_ANNOTATION_VERSIONS = {
    "e667aa0bfd728cf26618f6dd6cc683d0": "1.0",
    "3ff74c4e5cab69013d1d9c4a30f59cae": "1.1",
}


class DataImportWarning(UserWarning):
    """Some explicitly non-strict import items were rejected; inspect the report."""


class ImportResult(list):
    """A list of manifest records, with an auditable ``report`` dictionary.

    In non-strict mode, callers must preserve ``report['failures']`` when saving
    a manifest. The project CLI writes the report alongside its output manifest.
    """

    def __init__(self, records: Iterable[dict], report: dict):
        super().__init__(records)
        self.report = report


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _intervals(value: Any, duration: float, name: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of [start, end] intervals")
    result: list[list[float]] = []
    previous_end = -1.0
    for index, pair in enumerate(value):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly start and end")
        start = _finite_number(pair[0], f"{name}[{index}].start")
        end = _finite_number(pair[1], f"{name}[{index}].end")
        if not 0 <= start < end <= duration:
            raise ValueError(
                f"{name}[{index}] must satisfy 0 <= start < end <= duration "
                f"({duration}); got [{start}, {end}]"
            )
        if start < previous_end:
            raise ValueError(f"{name} must be sorted and non-overlapping")
        result.append([start, end])
        previous_end = end
    return result


def inspect_audio(path: str | Path) -> dict:
    """Read WAV/FLAC header metadata without decoding the entire recording.

    Supports ordinary PCM, IEEE-float and extensible PCM/float WAV, and FLAC
    with a nonzero STREAMINFO sample count. This is a container/header check,
    not a guarantee that every encoded audio frame is decodable.
    """
    audio_path = Path(path).expanduser()
    try:
        size = audio_path.stat().st_size
        with audio_path.open("rb") as handle:
            magic = handle.read(12)
            if magic[:4] == b"fLaC":
                handle.seek(4)
                while True:
                    header = handle.read(4)
                    if len(header) != 4:
                        raise ValueError("truncated FLAC metadata")
                    block_type = header[0] & 0x7F
                    is_last = bool(header[0] & 0x80)
                    length = int.from_bytes(header[1:], "big")
                    if handle.tell() + length > size:
                        raise ValueError("FLAC metadata extends beyond the file")
                    if block_type == 0:
                        if length != 34:
                            raise ValueError("FLAC STREAMINFO must contain 34 bytes")
                        stream_info = handle.read(34)
                        packed = int.from_bytes(stream_info[10:18], "big")
                        sample_rate = packed >> 44
                        channels = ((packed >> 41) & 7) + 1
                        frames = packed & ((1 << 36) - 1)
                        if sample_rate <= 0 or frames <= 0:
                            raise ValueError("FLAC has no usable sample rate/sample count")
                        return {
                            "duration_sec": frames / sample_rate,
                            "sample_rate": sample_rate,
                            "channels": channels,
                            "format": "FLAC",
                            "frames": frames,
                        }
                    handle.seek(length, 1)
                    if is_last:
                        raise ValueError("FLAC has no STREAMINFO block")

            if magic[:4] not in (b"RIFF", b"RIFX") or magic[8:] != b"WAVE":
                raise ValueError("supported containers are PCM/float WAV and FLAC")
            byte_order = "<" if magic[:4] == b"RIFF" else ">"
            fmt: tuple[int, int, int, int] | None = None
            data_bytes = 0
            while handle.tell() + 8 <= size:
                header = handle.read(8)
                chunk_id = header[:4]
                chunk_size = struct.unpack(byte_order + "I", header[4:])[0]
                chunk_start = handle.tell()
                if chunk_start + chunk_size > size:
                    raise ValueError("WAV chunk extends beyond the file (truncated audio)")
                if chunk_id == b"fmt ":
                    raw = handle.read(min(chunk_size, 64))
                    if len(raw) < 16:
                        raise ValueError("invalid WAV fmt chunk")
                    encoding, channels, sample_rate, _, block_align, bits = struct.unpack(
                        byte_order + "HHIIHH", raw[:16]
                    )
                    if encoding == 0xFFFE:
                        if len(raw) < 40:
                            raise ValueError("truncated extensible WAV fmt chunk")
                        encoding = struct.unpack(byte_order + "H", raw[24:26])[0]
                    if encoding not in (1, 3):
                        raise ValueError("compressed WAV is unsupported; decode it first")
                    if channels <= 0 or sample_rate <= 0 or block_align <= 0 or bits <= 0:
                        raise ValueError("invalid WAV audio dimensions")
                    if block_align != channels * ((bits + 7) // 8):
                        raise ValueError("WAV block alignment is inconsistent")
                    fmt = (encoding, channels, sample_rate, block_align)
                elif chunk_id == b"data":
                    data_bytes += chunk_size
                handle.seek(chunk_start + chunk_size + chunk_size % 2)
            if fmt is None or data_bytes <= 0:
                raise ValueError("WAV needs nonempty fmt and data chunks")
            encoding, channels, sample_rate, block_align = fmt
            if data_bytes % block_align:
                raise ValueError("WAV data ends in an incomplete audio frame")
            frames = data_bytes // block_align
            return {
                "duration_sec": frames / sample_rate,
                "sample_rate": sample_rate,
                "channels": channels,
                "format": "WAV_PCM" if encoding == 1 else "WAV_FLOAT",
                "frames": frames,
            }
    except (OSError, struct.error, ValueError) as exc:
        raise ValueError(f"Cannot inspect audio {audio_path}: {exc}") from exc


def validate_manifest(records: Iterable[dict], check_audio: bool = False) -> dict:
    """Validate supervision, time ranges and source/speaker split separation.

    Unknown labels (omitted or null) are allowed for inference, with no fake
    intervals. Training code must additionally require labeled examples.
    All explicitly declared ``group_id``, source and speaker IDs are checked
    across every split, including ``unspecified``. No IDs are inferred from
    model predictions or speaker embeddings.
    """
    ids: set[str] = set()
    paths: dict[str, str] = {}
    assigned: dict[str, dict[str, str]] = {
        "group_id": {}, "source_recording_id": {}, "speaker_id": {}
    }
    counts_by_split: dict[str, int] = {}
    counts_by_label = {"0": 0, "1": 0, "unknown": 0}
    required_strings = (
        "recording_id", "audio_path", "source_recording_id", "speaker_id", "dataset"
    )
    count = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a JSON object")
        for field in required_strings:
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Record {index}: {field} must be a nonempty string")
        rid = record["recording_id"]
        if rid in ids:
            raise ValueError(f"Duplicate recording_id: {rid}")
        ids.add(rid)
        # Reusing one path as two labeled recordings can duplicate evidence.
        path_key = str(Path(record["audio_path"]).expanduser().resolve())
        if path_key in paths:
            raise ValueError(f"Duplicate audio_path for {paths[path_key]} and {rid}")
        paths[path_key] = rid
        split = record.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"{rid}: split must be one of {sorted(SPLITS)}")
        duration = _finite_number(record.get("duration_sec"), f"{rid}.duration_sec")
        if duration <= 0:
            raise ValueError(f"{rid}: duration_sec must be positive")
        sample_rate = record.get("sample_rate")
        if sample_rate is not None and (
            isinstance(sample_rate, bool) or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ValueError(f"{rid}: sample_rate must be a positive integer or null")
        label = record.get("label")
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError(f"{rid}: label must be 0, 1 or null")
        fake = _intervals(record.get("fake_intervals", []), duration, f"{rid}.fake_intervals")
        _intervals(record.get("ignore_intervals", []), duration, f"{rid}.ignore_intervals")
        if label in (None, 0) and fake:
            raise ValueError(f"{rid}: unlabeled/bona fide audio cannot have fake_intervals")
        if label == 1 and not fake:
            raise ValueError(f"{rid}: edited audio needs at least one fake interval")
        if "metadata" in record and not isinstance(record["metadata"], dict):
            raise ValueError(f"{rid}: metadata must be a JSON object")
        for field, registry in assigned.items():
            key = record.get(field)
            if field == "group_id" and key is None:
                continue
            if not isinstance(key, str) or not key:
                raise ValueError(f"{rid}: {field} must be a nonempty string")
            previous = registry.setdefault(key, split)
            if previous != split:
                raise ValueError(
                    f"Split leakage: {field}={key!r} appears in {previous!r} and {split!r}"
                )
        if check_audio:
            info = inspect_audio(record["audio_path"])
            if abs(info["duration_sec"] - duration) > _DURATION_TOLERANCE_SEC:
                raise ValueError(f"{rid}: manifest duration differs from the audio header")
            if sample_rate is not None and sample_rate != info["sample_rate"]:
                raise ValueError(f"{rid}: manifest sample_rate differs from the audio header")
            # The tolerance above is for rounded duration metadata, not labels
            # extending outside the recording.
            _intervals(fake, info["duration_sec"], f"{rid}.fake_intervals")
            _intervals(record.get("ignore_intervals", []), info["duration_sec"],
                       f"{rid}.ignore_intervals")
        counts_by_split[split] = counts_by_split.get(split, 0) + 1
        counts_by_label["unknown" if label is None else str(label)] += 1
        count += 1
    return {
        "recordings": count,
        "source_groups": len(assigned["source_recording_id"]),
        "speakers": len(assigned["speaker_id"]),
        "counts_by_split": counts_by_split,
        "counts_by_label": counts_by_label,
        "audio_checked": bool(check_audio),
        "warnings": ["The manifest is empty."] if not count else [],
    }


def load_manifest(path: str | Path) -> list[dict]:
    """Load JSONL (or a JSON list), resolving relative audio paths at the manifest."""
    manifest_path = Path(path).expanduser().resolve()
    try:
        text = manifest_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"Cannot read manifest {manifest_path}: {exc}") from exc
    records: list[dict] = []
    try:
        if text.lstrip().startswith("["):
            records = json.loads(text)
            if not isinstance(records, list):
                raise ValueError("JSON manifest must contain a list")
        else:
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on manifest line {line_number}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid manifest JSON: {exc}") from exc
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("audio_path"), str):
            audio_path = Path(record["audio_path"]).expanduser()
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            record["audio_path"] = str(audio_path.resolve())
    validate_manifest(records)
    return records


def write_manifest(path: str | Path, records: Iterable[dict]) -> None:
    """Atomically save a validated JSONL manifest. Paths are preserved as supplied."""
    materialized = list(records)
    validate_manifest(materialized)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = handle.name
            for record in materialized:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


def _speaker_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Speaker ID must be a string")
    value = value.removeprefix("vctk:").strip()
    if not _SPEAKER_RE.fullmatch(value):
        raise ValueError(f"Invalid VCTK speaker ID: {value!r}")
    return value


def _split_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Speaker split must be a string")
    value = {"eval": "test", "validation": "dev"}.get(value, value)
    if value not in SPLITS or value == "unspecified":
        raise ValueError(f"Invalid speaker split: {value!r}")
    return value


def _speaker_protocol(
    value: str | Path | Mapping | None, root: Path
) -> tuple[dict[str, str], str]:
    """Read the top-level PartialEdit split files, not the ASVspoof copies."""
    if value is None:
        for candidate in (root / "PartialEdit_spk.zip", root / "PartialEdit_spk", root):
            if candidate.is_file() and candidate.suffix == ".zip":
                value = candidate
                break
            if candidate.is_dir() and (candidate / "train" / "train.spk").is_file():
                value = candidate
                break
        if value is None:
            return {}, "unspecified"
    result: dict[str, str] = {}

    def assign(speaker: str, split: str) -> None:
        speaker, split = _speaker_name(speaker), _split_name(split)
        if speaker in result:
            raise ValueError(f"Speaker protocol lists {speaker} more than once")
        result[speaker] = split

    if isinstance(value, Mapping):
        if not value:
            raise ValueError("Speaker protocol contains no speakers")
        if all(isinstance(item, str) for item in value.values()):
            for speaker, split in value.items():
                assign(speaker, split)
        else:
            for split, speakers in value.items():
                if not isinstance(speakers, (list, tuple, set)):
                    raise ValueError("Speaker protocol must map splits to speaker lists")
                for speaker in speakers:
                    assign(speaker, split)
        if not result:
            raise ValueError("Speaker protocol contains no speakers")
        return result, "provided-speaker-mapping"
    protocol_path = Path(value).expanduser().resolve()
    texts: dict[str, str] = {}
    expected = {split: f"{split}/{split}.spk" for split in ("train", "dev", "eval")}
    try:
        if protocol_path.is_file() and protocol_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(protocol_path) as archive:
                names = archive.namelist()
                for split, member in expected.items():
                    if names.count(member) != 1:
                        raise ValueError(f"Expected exactly one top-level {member} in split ZIP")
                    if archive.getinfo(member).file_size > 1_000_000:
                        raise ValueError("Speaker protocol member is unexpectedly large")
                    texts[split] = archive.read(member).decode("utf-8-sig")
        elif protocol_path.is_dir():
            for split, member in expected.items():
                texts[split] = (protocol_path / member).read_text(encoding="utf-8-sig")
        else:
            raise ValueError("speaker_splits must be a split ZIP, extracted directory or mapping")
    except (OSError, zipfile.BadZipFile, UnicodeError) as exc:
        raise ValueError(f"Cannot read speaker protocol {protocol_path}: {exc}") from exc
    for split, text in texts.items():
        for line in text.splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                assign(line.strip(), split)
    if not result:
        raise ValueError("Speaker protocol contains no speakers")
    return result, "provided-partialedit-speaker-protocol"


def _safe_audio_path(root: Path, relative_path: PurePosixPath) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Audio path must stay inside its supplied root: {relative_path}")
    resolved = root.joinpath(*relative_path.parts).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Audio path escapes its supplied root: {relative_path}")
    return resolved


def import_partialedit(
    root: str | Path,
    csv_path: str | Path,
    bona_fide_root: str | Path | None = None,
    speaker_splits: str | Path | Mapping | None = None,
    mic: str | None = None,
    include_codec: bool = True,
    strict: bool = True,
    subsets: Iterable[str] | str | None = None,
) -> ImportResult:
    """Import locally downloaded PartialEdit audio and optional VCTK originals.

    CSV format is headerless ``filename,(start,end)*N,duration``. The v1.1
    release actually contains some three-interval rows. E1/E2-Codec inherit
    the corresponding editor's intervals only under the published naming
    mapping and after a duration/bounds check; no time shift is guessed.

    ``subsets=None`` discovers available subset directories. In non-strict
    mode invalid/missing items are listed in ``result.report['failures']``.
    """
    root = Path(root).expanduser().resolve()
    csv_path = Path(csv_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"PartialEdit root does not exist: {root}")
    if mic not in (None, "mic1", "mic2"):
        raise ValueError("mic must be mic1, mic2 or None")
    if subsets is None:
        selected = [
            name for name in PARTIALEDIT_SUBSETS
            if (root / name).is_dir() and (include_codec or not name.endswith("-Codec"))
        ]
    else:
        selected = [item.strip() for item in subsets.split(",")] if isinstance(subsets, str) else list(subsets)
        if any(name not in PARTIALEDIT_SUBSETS for name in selected):
            raise ValueError(f"subsets must be chosen from {PARTIALEDIT_SUBSETS}")
        if len(set(selected)) != len(selected):
            raise ValueError("subsets must not contain duplicates")
        if not include_codec and any(name.endswith("-Codec") for name in selected):
            raise ValueError("Codec subsets conflict with include_codec=False")
        for name in selected:
            if not (root / name).is_dir():
                raise ValueError(f"Selected subset directory does not exist: {root / name}")
        selected = [name for name in PARTIALEDIT_SUBSETS if name in selected]
    if not selected:
        raise ValueError("No requested PartialEdit audio subset directory is present")
    speaker_map, split_origin = _speaker_protocol(speaker_splits, root)
    try:
        annotation_bytes = csv_path.read_bytes()
        csv_text = annotation_bytes.decode("utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read annotation CSV {csv_path}: {exc}") from exc
    annotation_hash = hashlib.sha256(annotation_bytes).hexdigest()
    annotation_version = _ANNOTATION_VERSIONS.get(hashlib.md5(annotation_bytes).hexdigest(), "unverified")
    report: dict[str, Any] = {
        "adapter": "partialedit",
        "annotation_file": str(csv_path),
        "annotation_sha256": annotation_hash,
        "annotation_version": annotation_version,
        "selected_subsets": selected,
        "split_origin": split_origin,
        "failures": [],
        "skipped_unselected_annotations": 0,
        "counts_by_subset": {},
        "warnings": [],
        "codec_annotation_policy": "published filename mapping; equal duration within 25 ms; no timestamp shift",
    }
    if split_origin == "unspecified":
        report["warnings"].append("No speaker protocol found: every record remains unspecified.")
    if annotation_version == "1.0":
        report["warnings"].append("PartialEdit v1.0 timestamps are superseded by the corrected v1.1 release.")
    records: list[dict] = []
    successful_sources: dict[str, str] = {}
    seen_csv_paths: set[str] = set()

    def reject(stage: str, path: str, reason: str, line: int | None = None) -> None:
        failure = {"stage": stage, "path": path, "reason": reason}
        if line is not None:
            failure["csv_line"] = line
        report["failures"].append(failure)
        if strict:
            location = f" on CSV line {line}" if line is not None else ""
            raise ValueError(f"PartialEdit {stage} error{location} ({path}): {reason}")

    def split_for(speaker: str) -> str:
        if speaker_map and speaker not in speaker_map:
            raise ValueError(f"Speaker {speaker} is absent from the supplied speaker protocol")
        return speaker_map.get(speaker, "unspecified")

    for line_number, row in enumerate(csv.reader(csv_text.splitlines()), 1):
        if not row or not any(item.strip() for item in row):
            continue
        row = [item.strip() for item in row]
        if line_number == 1 and row[0].lower() == "filename":
            continue
        relative_name = row[0]
        try:
            relative = PurePosixPath(relative_name)
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3:
                raise ValueError("Expected a relative E1|E2/speaker/filename path")
            base_subset, speaker, filename = relative.parts
            if base_subset not in ("E1", "E2"):
                raise ValueError("Official annotations must name E1 or E2, not a Codec variant")
            applicable = [name for name in selected if name.removesuffix("-Codec") == base_subset]
            if not applicable:
                report["skipped_unselected_annotations"] += 1
                continue
            if relative_name in seen_csv_paths:
                raise ValueError("Duplicate annotation path")
            seen_csv_paths.add(relative_name)
            match = _EDITED_RE.fullmatch(filename)
            if match is None or match.group(2) != speaker:
                raise ValueError("Filename/source speaker does not match the published naming convention")
            source_id = match.group(1)
            if len(row) < 4 or len(row) % 2:
                raise ValueError("Expected filename,(start,end)*N,duration with N >= 1")
            values = [float(item) for item in row[1:]]
            annotation_duration = _finite_number(values[-1], "annotation duration")
            if annotation_duration <= 0:
                raise ValueError("Annotation duration must be positive")
            intervals = _intervals(
                [values[i:i + 2] for i in range(0, len(values) - 1, 2)],
                annotation_duration, "annotation intervals",
            )
            split = split_for(speaker)
        except (ValueError, TypeError) as exc:
            reject("annotation", relative_name, str(exc), line_number)
            continue
        for subset in applicable:
            codec = subset.endswith("-Codec")
            target_name = filename.replace("_edited_partial_16k.wav", "_edited_16k.wav") if codec else filename
            target_relative = PurePosixPath(subset, speaker, target_name)
            try:
                audio_path = _safe_audio_path(root, target_relative)
                info = inspect_audio(audio_path)
                if abs(info["duration_sec"] - annotation_duration) > _DURATION_TOLERANCE_SEC:
                    raise ValueError(
                        f"Audio duration {info['duration_sec']:.6f}s differs from annotation "
                        f"{annotation_duration:.6f}s by more than 25 ms; timestamps were not transferred"
                    )
                checked_intervals = _intervals(intervals, info["duration_sec"], "fake intervals")
                record = {
                    "recording_id": f"partialedit:{subset}:{source_id}",
                    "audio_path": str(audio_path),
                    "source_recording_id": f"vctk:{source_id}",
                    "speaker_id": f"vctk:{speaker}",
                    "dataset": "PartialEdit",
                    "split": split,
                    "split_origin": split_origin,
                    "duration_sec": info["duration_sec"],
                    "sample_rate": info["sample_rate"],
                    "label": 1,
                    "fake_intervals": checked_intervals,
                    "ignore_intervals": [],
                    "editor": "VoiceCraft" if base_subset == "E1" else "SSR-Speech",
                    "condition": "codec_full_decode" if codec else "local_edit",
                    "metadata": {
                        "subset": subset,
                        "annotation_csv": str(csv_path),
                        "annotation_csv_line": line_number,
                        "annotation_sha256": annotation_hash,
                        "annotation_version": annotation_version,
                        "annotation_duration_sec": annotation_duration,
                        "annotation_transfer": "editor_csv_name_and_duration_checked" if codec else "direct_csv",
                        "audio_format": info["format"],
                        "channels": info["channels"],
                    },
                }
                records.append(record)
                successful_sources[source_id] = speaker
                report["counts_by_subset"][subset] = report["counts_by_subset"].get(subset, 0) + 1
            except (ValueError, OSError) as exc:
                reject("audio", str(target_relative), str(exc), line_number)

    if bona_fide_root is not None:
        bona_root = Path(bona_fide_root).expanduser().resolve()
        if not bona_root.is_dir():
            raise ValueError(f"Bona fide directory does not exist: {bona_root}")
        candidates: dict[str, list[tuple[Path, str | None]]] = {}
        for candidate in sorted(bona_root.rglob("*")):
            if not candidate.is_file():
                continue
            match = _BONA_FIDE_RE.fullmatch(candidate.name)
            if match is None or match.group("source") not in successful_sources:
                continue
            candidate_mic = match.group("mic")
            if mic is not None and candidate_mic not in (None, mic):
                continue
            candidates.setdefault(match.group("source"), []).append((candidate, candidate_mic))
        for source_id, speaker in sorted(successful_sources.items()):
            try:
                matches = candidates.get(source_id, [])
                if not matches:
                    raise ValueError("No matching bona fide recording for the requested microphone/root")
                if len(matches) != 1:
                    raise ValueError("Ambiguous bona fide recording; specify mic1/mic2 or a curated audio root")
                candidate, candidate_mic = matches[0]
                audio_path = _safe_audio_path(bona_root, PurePosixPath(candidate.relative_to(bona_root).as_posix()))
                info = inspect_audio(audio_path)
                records.append({
                    "recording_id": f"vctk:bonafide:{source_id}" + (f":{candidate_mic}" if candidate_mic else ""),
                    "audio_path": str(audio_path),
                    "source_recording_id": f"vctk:{source_id}",
                    "speaker_id": f"vctk:{speaker}",
                    "dataset": "VCTK",
                    "split": split_for(speaker),
                    "split_origin": split_origin,
                    "duration_sec": info["duration_sec"],
                    "sample_rate": info["sample_rate"],
                    "label": 0,
                    "fake_intervals": [],
                    "ignore_intervals": [],
                    "editor": "none",
                    "condition": "bona_fide_original",
                    "metadata": {"mic": candidate_mic, "channels": info["channels"], "audio_format": info["format"]},
                })
                report["counts_by_subset"]["bona_fide"] = report["counts_by_subset"].get("bona_fide", 0) + 1
            except (ValueError, OSError) as exc:
                reject("bona_fide", source_id, str(exc))
    records.sort(key=lambda record: record["recording_id"])
    report["validation"] = validate_manifest(records)
    report["imported_count"] = len(records)
    report["failure_count"] = len(report["failures"])
    if not records:
        report["warnings"].append("No valid recordings were imported.")
        if strict:
            raise ValueError("No valid recordings were imported; check selected subsets and annotations")
    if report["failures"]:
        warnings.warn(
            f"PartialEdit rejected {len(report['failures'])} item(s). "
            "The complete failure list is in result.report['failures']; save it with the manifest.",
            DataImportWarning, stacklevel=2,
        )
    return ImportResult(records, report)


def split_development(
    records: Iterable[dict], calibration_fraction: float = 0.5, seed: int = 17
) -> list[dict]:
    """Divide the existing dev speakers into dev and calibration, reproducibly.

    This is a project development protocol, not an official benchmark split.
    Train/test are untouched, original objects are not mutated, and both new
    partitions must contain at least one speaker.
    """
    fraction = _finite_number(calibration_fraction, "calibration_fraction")
    if not 0 < fraction < 1:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    result = copy.deepcopy(list(records))
    validate_manifest(result)
    if any(record["split"] == "calibration" for record in result):
        raise ValueError("A calibration split already exists; do not silently repartition it")
    speakers = {record["speaker_id"] for record in result if record["split"] == "dev"}
    if len(speakers) < 2:
        raise ValueError("At least two dev speakers are needed for separate dev/calibration splits")
    ordered = sorted(speakers, key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest())
    count = max(1, min(len(ordered) - 1, int(math.floor(len(ordered) * fraction + 0.5))))
    calibration_speakers = set(ordered[:count])
    for record in result:
        if record["split"] == "dev":
            record.setdefault("metadata", {})["original_split"] = "dev"
            record["metadata"]["development_split_seed"] = seed
            record["metadata"]["calibration_fraction"] = fraction
            record["metadata"]["upstream_split_origin"] = record.get("split_origin", "unspecified")
            record["split"] = "calibration" if record["speaker_id"] in calibration_speakers else "dev"
            record["split_origin"] = "project-development-speaker-holdout"
    validate_manifest(result)
    return result
