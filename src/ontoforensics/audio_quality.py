"""Measured local quality covariates. No clean reference or SNR is invented."""
from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
import math
import hashlib
from pathlib import Path
import numpy as np


def prepare_audio(records, output_dir, *, sample_rate=16000):
    """Make a common mono PCM16 representation, preserving the time origin.

    Originals are not edited. Resampling changes total duration by at most one
    output sample. There is no VAD trimming, loudness normalization or denoising.
    """
    try:
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError('Install the audio extra: pip install ".[audio]"') from exc
    from .data import validate_manifest
    validate_manifest(records, check_audio=True)
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = deepcopy(records)
    for record in result:
        source_path = Path(record["audio_path"]).resolve()
        identifier = hashlib.sha256(record["recording_id"].encode()).hexdigest()[:24]
        destination = output_dir / f"{identifier}.wav"
        if destination.exists() or destination == source_path:
            raise ValueError(f"Refusing to overwrite audio: {destination}; use a new output directory")
        x, rate = sf.read(source_path, dtype="float32", always_2d=True)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError(f"{source_path}: empty/nonfinite waveform")
        original_channels = x.shape[1]
        x = x.mean(axis=1, dtype=np.float64)
        if rate != sample_rate:
            divisor = math.gcd(rate, sample_rate)
            x = resample_poly(x, sample_rate//divisor, rate//divisor)
        duration = len(x)/sample_rate
        # Polyphase resampling uses ceil(N*out_rate/in_rate), so no true label
        # boundary should be silently clipped to a shorter recording.
        if any(end > duration+1e-8 for _, end in record.get("fake_intervals", []) + record.get("ignore_intervals", [])):
            raise ValueError(f"{record['recording_id']}: preprocessing would invalidate an interval")
        before_quantization_clip = float(np.mean(np.abs(x) > 1.))
        sf.write(destination, np.clip(x, -1., 1.), sample_rate, subtype="PCM_16")
        record["audio_path"] = str(destination)
        record["sample_rate"], record["duration_sec"] = int(sample_rate), duration
        record.setdefault("metadata", {})["preprocessing"] = {
            "original_audio_path": str(source_path), "original_sample_rate": rate,
            "original_channels": original_channels, "mono_policy": "arithmetic_mean",
            "resampling": "scipy.signal.resample_poly", "output_subtype": "PCM_16",
            "pre_quantization_out_of_range_fraction": before_quantization_clip,
            "time_origin_unchanged": True,
        }
    validate_manifest(result, check_audio=True)
    return result


def annotate_quality(records, observations):
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError('Install the audio extra: pip install ".[audio]"') from exc
    by_record = defaultdict(list)
    result = deepcopy(observations)
    for row in result:
        by_record[row["recording_id"]].append(row)
    index = {r["recording_id"]: r for r in records}
    for rid, rows in by_record.items():
        record = index[rid]
        waveform, sr = sf.read(record["audio_path"], dtype="float32", always_2d=True)
        if not np.isfinite(waveform).all() or not len(waveform):
            raise ValueError(f"{rid}: waveform is empty or contains nonfinite samples")
        if abs(len(waveform)/sr - record["duration_sec"]) > .05:
            raise ValueError(f"{rid}: waveform duration differs from the manifest")
        cache = {}
        for observation in rows:
            if observation["status"] != "valid":
                continue
            a = 0. if observation.get("start_sec") is None else observation["start_sec"]
            b = record["duration_sec"] if observation.get("end_sec") is None else observation["end_sec"]
            span = (int(math.floor(a*sr)), min(len(waveform), int(math.ceil(b*sr))))
            if span not in cache:
                x = waveform[span[0]:span[1]].astype(np.float64)
                if not len(x):
                    raise ValueError(f"{rid}: quality interval has no samples: {span}")
                # RMS over all channels avoids cancellation from stereo averaging.
                rms = float(np.sqrt(np.mean(x*x)))
                cache[span] = {"rms_db": float(20*np.log10(max(rms, 1e-12))),
                               "clipping_fraction": float(np.mean(np.abs(x) >= .999))}
            existing = observation.setdefault("quality", {})
            existing.update(cache[span])
    return result
