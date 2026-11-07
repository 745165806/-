#!/usr/bin/env python3
"""Export native CFPRF FDN frame margins from a separately installed upstream repo.

No upstream network implementation or weights are bundled. This adapter performs
label-free inference only, without upstream evaluation or target-alignment code.
The upstream CUDA/fairseq inference path requires external validation: it was not
run with real checkpoints in the environment that produced this baseline.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import inspect
import os
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

if __package__:
    from .import_frame_scores import (
        Recording, canonical_scores, frame_intervals, frame_observations,
        protect_input_paths, read_manifest, registry_entry, sha256_file,
        stable_hash, write_bundle,
    )
else:
    from import_frame_scores import (
        Recording, canonical_scores, frame_intervals, frame_observations,
        protect_input_paths, read_manifest, registry_entry, sha256_file,
        stable_hash, write_bundle,
    )


UPSTREAM_URL = "https://github.com/ItzJuny/CFPRF"
ADAPTER_VERSION = "2"
SAMPLE_RATE = 16000
HOP_SAMPLES = 320
RECEPTIVE_FIELD_SAMPLES = 400
FRAME_HOP_SEC = HOP_SAMPLES / SAMPLE_RATE
MAX_INPUT_SAMPLES = 340000
SEQUENCE_CAPACITY = 1070


def expected_frame_count(num_samples: int) -> int:
    """Native XLS-R valid-convolution count, never derived from a target vector."""
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise ValueError("num_samples must be an integer")
    if num_samples < RECEPTIVE_FIELD_SAMPLES:
        raise ValueError("CFPRF input is too short: at least 400 samples (25 ms) are required")
    if num_samples > MAX_INPUT_SAMPLES:
        raise ValueError("CFPRF input exceeds 340000 samples (21.25 s); no silent skipping or chunking")
    return (num_samples - RECEPTIVE_FIELD_SAMPLES) // HOP_SAMPLES + 1


def validate_audio_metadata(record: Recording, num_samples: int, sample_rate: int,
                            channels: int) -> None:
    if sample_rate != SAMPLE_RATE or record.sample_rate != SAMPLE_RATE:
        raise ValueError("%s must be 16000 Hz in both file and manifest; resample explicitly before export" % record.recording_id)
    if channels != 1:
        raise ValueError("%s must be mono; no implicit channel mixing" % record.recording_id)
    try:
        expected_frame_count(num_samples)
    except ValueError as exc:
        raise ValueError("%s: %s" % (record.recording_id, exc)) from exc
    if abs(num_samples / sample_rate - record.duration_sec) > 1.0 / sample_rate + 1e-9:
        raise ValueError("Audio duration disagrees with manifest for %s" % record.recording_id)


def fdn_margin(raw_scores: np.ndarray, num_samples: int) -> np.ndarray:
    """Class 0 is spoof, class 1 bona fide. Their difference is a raw margin."""
    raw_scores = np.asarray(raw_scores)
    expected = expected_frame_count(num_samples)
    if raw_scores.shape != (1, expected, 2):
        raise ValueError("Unexpected FDN shape %r; expected (1, %d, 2). Check upstream/checkpoint compatibility."
                         % (raw_scores.shape, expected))
    return canonical_scores(raw_scores[0], "margin", "two-class", fake_class_index=0)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_external_model(repo: Path, checkpoint: Path, requested_device: str):
    if requested_device != "cuda" and not requested_device.startswith("cuda:"):
        raise ValueError("Unmodified upstream CFPRF requires CUDA; CPU is not supported by this adapter")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing. Use the separate upstream CFPRF/fairseq environment; see docs/EXPERTS.md") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; upstream CFPRF allocates positional encoding directly on CUDA")
    device = (torch.device("cuda:%d" % torch.cuda.current_device())
              if requested_device == "cuda" else torch.device(requested_device))
    torch.cuda.set_device(device)
    source_path = repo / "models" / "FDN.py"
    module_name = "_external_cfprf_fdn_" + sha256_file(source_path)[:12]
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load upstream models/FDN.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # ASRModel's constructor resolves ./pretrain_models/xlsr2_300m.pt against cwd.
    with _working_directory(repo):
        model = module.CFPRF_FDN(seq_len=SEQUENCE_CAPACITY, gmlp_layers=1)
    model = model.to(device)
    load_options = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = True
    state = torch.load(str(checkpoint), **load_options)
    if not isinstance(state, dict):
        raise ValueError("Expected an upstream FDN state_dict checkpoint")
    model.load_state_dict(state, strict=True)
    model.eval()
    return torch, model, device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="External ItzJuny/CFPRF checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="External 1FDN_PS.pth or compatible FDN state_dict")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-id", default="cfprf_fdn")
    parser.add_argument("--family", default="xlsr300m")
    parser.add_argument("--run-id", default="cfprf-fdn-v1")
    return parser


def run_export(args: argparse.Namespace) -> int:
    repo, checkpoint = args.repo.resolve(), args.checkpoint.resolve()
    source_path = repo / "models" / "FDN.py"
    encoder_checkpoint = repo / "pretrain_models" / "xlsr2_300m.pt"
    for path in (source_path, checkpoint, encoder_checkpoint):
        if not path.is_file():
            raise ValueError("Required external file is missing: %s. See docs/EXPERTS.md; no automatic downloads are performed." % path)
    if args.device != "cuda" and not args.device.startswith("cuda:"):
        raise ValueError("Unmodified CFPRF requires CUDA; --device cpu is unsupported")
    if not args.run_id.strip():
        raise ValueError("run_id must be nonempty")
    recordings = read_manifest(args.manifest)
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required in the external inference environment") from exc
    prepared = []
    for record in recordings.values():
        audio_path = Path(record.audio_path).expanduser()
        if not audio_path.is_absolute():
            audio_path = args.manifest.resolve().parent / audio_path
        audio_path = audio_path.resolve()
        info = sf.info(str(audio_path))
        validate_audio_metadata(record, int(info.frames), info.samplerate, info.channels)
        prepared.append((record, audio_path, int(info.frames)))
    protect_input_paths((args.out, args.registry),
                        [args.manifest, checkpoint, encoder_checkpoint, source_path]
                        + [path for _, path, _ in prepared])
    checkpoint_hash = sha256_file(checkpoint)
    encoder_hash = sha256_file(encoder_checkpoint)
    source_hash = sha256_file(source_path)
    entry = registry_entry(
        args.detector_id, args.family, "margin", checkpoint_hash,
        adapter="export_cfprf", adapter_version=ADAPTER_VERSION,
        upstream_repo=UPSTREAM_URL, upstream_fdn_sha256=source_hash,
        ssl_checkpoint_sha256=encoder_hash, sample_rate=SAMPLE_RATE,
        frame_hop_sec=FRAME_HOP_SEC, max_input_samples=MAX_INPUT_SAMPLES,
        input_score_semantics="FDN class0 minus class1; uncalibrated linear-head margin",
        frame_alignment="native 20-ms bins from t=0; uncovered tail remains unobserved",
    )
    torch, model, device = _load_external_model(repo, checkpoint, args.device)

    def observations() -> Iterator[dict]:
        for record, audio_path, expected_samples in prepared:
            wave, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
            validate_audio_metadata(record, len(wave), sample_rate, wave.shape[1])
            if len(wave) != expected_samples or not np.isfinite(wave).all():
                raise ValueError("Audio changed after preflight or contains non-finite samples: %s" % record.recording_id)
            waveform = torch.as_tensor(wave[:, 0], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                outputs = model(waveform)
                if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
                    raise ValueError("FDN must return seg_scores, bd_scores, emb_T, F_BA")
                raw = outputs[0].detach().cpu().numpy()
            scores = fdn_margin(raw, len(wave))
            intervals = frame_intervals(len(scores), record.duration_sec, FRAME_HOP_SEC)
            input_lineage = stable_hash({
                "adapter": "export_cfprf", "version": ADAPTER_VERSION,
                "recording_id": record.recording_id, "audio_sha256": sha256_file(audio_path),
                "detector_id": args.detector_id, "checkpoint_sha256": checkpoint_hash,
                "ssl_checkpoint_sha256": encoder_hash, "upstream_fdn_sha256": source_hash,
                "preprocessing": "mono 16kHz float32; no normalization, padding, resampling, or truncation",
                "frame_hop_sec": FRAME_HOP_SEC,
                "score_semantics": "class0 minus class1; raw margin; higher is fake",
            })
            yield from frame_observations(record, scores, intervals, args.detector_id, args.run_id, input_lineage)

    return write_bundle(observations(), entry, args.out, args.registry)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        count = run_export(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        raise SystemExit("CFPRF export failed: %s" % exc) from exc
    print("Exported %d native CFPRF frame margins into %s" % (count, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
