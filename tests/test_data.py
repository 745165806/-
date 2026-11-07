"""Tests for protocol integrity and real public annotation/file conventions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import warnings
import wave
import zipfile

from ontoforensics.data import (
    DataImportWarning,
    import_partialedit,
    inspect_audio,
    load_manifest,
    split_development,
    validate_manifest,
    write_manifest,
)


def wav(path: Path, duration: float = 1.0, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * round(duration * sample_rate))


def record(rid="a", speaker="vctk:p225", split="dev", source=None):
    return {
        "recording_id": rid,
        "audio_path": f"/not-read/{rid}.wav",
        "source_recording_id": source or f"vctk:{rid}",
        "speaker_id": speaker,
        "dataset": "test",
        "split": split,
        "duration_sec": 1.0,
        "sample_rate": 8000,
        "label": 0,
        "fake_intervals": [],
        "ignore_intervals": [],
    }


class AudioHeaderTests(unittest.TestCase):
    def test_pcm_and_float_wav_headers_without_soundfile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wav(root / "pcm.wav", duration=0.25)
            self.assertEqual(inspect_audio(root / "pcm.wav")["frames"], 2000)
            samples = b"\x00" * 4 * 2400
            fmt = struct.pack("<HHIIHH", 3, 1, 24000, 96000, 4, 32)
            chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
            chunks += b"data" + struct.pack("<I", len(samples)) + samples
            (root / "float.wav").write_bytes(b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks)
            info = inspect_audio(root / "float.wav")
            self.assertEqual(info["format"], "WAV_FLOAT")
            self.assertAlmostEqual(info["duration_sec"], 0.1)

    def test_flac_streaminfo_and_truncated_wav(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packed = (48000 << 44) | (1 << 41) | (15 << 36) | 96000
            stream_info = b"\x00" * 10 + packed.to_bytes(8, "big") + b"\x00" * 16
            (root / "metadata.flac").write_bytes(b"fLaC" + b"\x80\x00\x00\x22" + stream_info)
            info = inspect_audio(root / "metadata.flac")
            self.assertEqual((info["sample_rate"], info["channels"]), (48000, 2))
            self.assertEqual(info["duration_sec"], 2.0)
            wav(root / "broken.wav")
            (root / "broken.wav").write_bytes((root / "broken.wav").read_bytes()[:-9])
            with self.assertRaisesRegex(ValueError, "truncated audio"):
                inspect_audio(root / "broken.wav")


class ManifestTests(unittest.TestCase):
    def test_invalid_split_type_raises_actionable_value_error(self):
        for split in ([], {}, None):
            item = record()
            item["split"] = split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "split must be"):
                validate_manifest([item])

    def test_group_source_and_speaker_leakage(self):
        for field in ("source_recording_id", "speaker_id", "group_id"):
            left = record("a", speaker="vctk:p225", split="train")
            right = record("b", speaker="vctk:p226", split="test")
            left[field] = right[field] = "shared"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"Split leakage: {field}"):
                validate_manifest([left, right])

    def test_unlabeled_inference_and_supervision_consistency(self):
        item = record()
        del item["label"]
        del item["fake_intervals"]
        self.assertEqual(validate_manifest([item])["counts_by_label"]["unknown"], 1)
        item["label"] = None
        item["fake_intervals"] = [[0.1, 0.2]]
        with self.assertRaisesRegex(ValueError, "unlabeled/bona fide"):
            validate_manifest([item])
        item["label"] = 1
        validate_manifest([item])

    def test_interval_bounds_order_and_ignore_validation(self):
        for intervals in ([[0.1, 1.1]], [[0.3, 0.5], [0.2, 0.4]], [[float("nan"), 0.2]]):
            item = record()
            item["ignore_intervals"] = intervals
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                validate_manifest([item])

    def test_duplicate_audio_is_rejected(self):
        left, right = record("a"), record("b")
        right["audio_path"] = left["audio_path"]
        with self.assertRaisesRegex(ValueError, "Duplicate audio_path"):
            validate_manifest([left, right])

    def test_relative_path_round_trip_and_header_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wav(root / "one.wav")
            item = record()
            item["audio_path"] = "one.wav"
            manifest = root / "records.jsonl"
            write_manifest(manifest, [item])
            loaded = load_manifest(manifest)
            self.assertEqual(loaded[0]["audio_path"], str(root / "one.wav"))
            validate_manifest(loaded, check_audio=True)
            loaded[0]["duration_sec"] = 2.0
            with self.assertRaisesRegex(ValueError, "duration differs"):
                validate_manifest(loaded, check_audio=True)


class PartialEditTests(unittest.TestCase):
    def make_root(self, temporary, csv_text, files):
        root = Path(temporary)
        annotation = root / "PartialEdit_E1E2.csv"
        annotation.write_text(csv_text, encoding="utf-8")
        for name, duration in files.items():
            wav(root / name, duration)
        return root, annotation

    def test_headerless_arbitrary_intervals_s5_and_subset_discovery(self):
        # The public v1.1 CSV has a few 8-column (three-region) rows and s5.
        name = "E1/s5/s5_001_edited_partial_16k.wav"
        other = "E2/s5/s5_001_edited_partial_16k.wav"
        with tempfile.TemporaryDirectory() as temporary:
            root, annotation = self.make_root(
                temporary,
                f"{name},0.1,0.2,0.4,0.5,0.7,0.8,1.0\n{other},0.2,0.4,1.0\n",
                {name: 1.0},
            )
            result = import_partialedit(root, annotation, speaker_splits={"s5": "eval"})
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["source_recording_id"], "vctk:s5_001")
            self.assertEqual(result[0]["split"], "test")
            self.assertEqual(len(result[0]["fake_intervals"]), 3)
            self.assertEqual(result.report["skipped_unselected_annotations"], 1)

    def test_codec_published_name_and_content_mask(self):
        base = "E1/p237/p237_189_edited_partial_16k.wav"
        codec = "E1-Codec/p237/p237_189_edited_16k.wav"
        with tempfile.TemporaryDirectory() as temporary:
            root, annotation = self.make_root(temporary, f"{base},0.2,0.5,1.0\n", {base: 1.0, codec: 1.0})
            result = import_partialedit(root, annotation)
            self.assertEqual(len(result), 2)
            self.assertEqual({item["source_recording_id"] for item in result}, {"vctk:p237_189"})
            self.assertTrue(all(item["fake_intervals"] == [[0.2, 0.5]] for item in result))
            codec_record = next(item for item in result if item["condition"] == "codec_full_decode")
            self.assertTrue(codec_record["audio_path"].endswith("_edited_16k.wav"))
            wav(root / codec, 1.5)
            with self.assertRaisesRegex(ValueError, "timestamps were not transferred"):
                import_partialedit(root, annotation)

    def test_missing_files_fail_or_have_complete_report(self):
        name = "E1/p237/p237_189_edited_partial_16k.wav"
        missing = "E1/p237/p237_190_edited_partial_16k.wav"
        with tempfile.TemporaryDirectory() as temporary:
            root, annotation = self.make_root(
                temporary, f"{name},0.1,0.3,1\n{missing},0.1,0.4,1\n", {name: 1.0}
            )
            with self.assertRaises(ValueError):
                import_partialedit(root, annotation)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = import_partialedit(root, annotation, strict=False)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.report["failure_count"], 1)
            self.assertEqual(result.report["failures"][0]["path"], missing)
            self.assertTrue(any(issubclass(item.category, DataImportWarning) for item in caught))

    def test_original_mic_ambiguity_and_source_grouping(self):
        name = "E1/p225/p225_001_edited_partial_16k.wav"
        with tempfile.TemporaryDirectory() as temporary:
            root, annotation = self.make_root(temporary, f"{name},0.1,0.3,1\n", {name: 1.0})
            bona = root / "originals"
            wav(bona / "p225" / "p225_001_mic1.wav", 1.2)
            wav(bona / "p225" / "p225_001_mic2.wav", 1.2)
            with self.assertRaisesRegex(ValueError, "Ambiguous bona fide"):
                import_partialedit(root, annotation, bona_fide_root=bona)
            result = import_partialedit(root, annotation, bona_fide_root=bona, mic="mic1")
            self.assertEqual(len(result), 2)
            self.assertEqual({item["source_recording_id"] for item in result}, {"vctk:p225_001"})
            original = next(item for item in result if item["label"] == 0)
            self.assertEqual(original["fake_intervals"], [])
            self.assertEqual(original["duration_sec"], 1.2)  # Source and edited duration can differ.

    def test_official_zip_top_level_splits_and_auto_discovery(self):
        name = "E1/p225/p225_001_edited_partial_16k.wav"
        with tempfile.TemporaryDirectory() as temporary:
            root, annotation = self.make_root(temporary, f"{name},0.1,0.3,1\n", {name: 1.0})
            with zipfile.ZipFile(root / "PartialEdit_spk.zip", "w") as archive:
                archive.writestr("train/train.spk", "p240\n")
                archive.writestr("dev/dev.spk", "p293\n")
                archive.writestr("eval/eval.spk", "p225\ns5\n")
                archive.writestr("asvspoof2019/train/train.spk", "p225\n")
            result = import_partialedit(root, annotation)
            self.assertEqual(result[0]["split"], "test")

    def test_bad_csv_bounds_and_paths_are_not_silently_repaired(self):
        name = "E1/p225/p225_001_edited_partial_16k.wav"
        for row in (f"{name},0.8,0.2,1", f"{name},0.2,1.2,1", f"../{name},0.1,0.2,1"):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as temporary:
                root, annotation = self.make_root(temporary, row + "\n", {name: 1.0})
                with self.assertRaises(ValueError):
                    import_partialedit(root, annotation)


class DevelopmentSplitTests(unittest.TestCase):
    def test_speaker_groups_determinism_and_no_mutation(self):
        records = [
            record("a", speaker="vctk:p225"),
            record("a-codec", speaker="vctk:p225", source="vctk:a"),
            record("b", speaker="vctk:p226"),
            record("c", speaker="vctk:p227"),
            record("d", speaker="vctk:p228"),
            record("train", speaker="vctk:p229", split="train"),
            record("test", speaker="vctk:p230", split="test"),
        ]
        original = copy.deepcopy(records)
        first = split_development(records)
        second = split_development(reversed(records))
        mapping = lambda rows: {item["recording_id"]: item["split"] for item in rows}
        self.assertEqual(mapping(first), mapping(second))
        self.assertEqual(records, original)
        self.assertEqual(mapping(first)["a"], mapping(first)["a-codec"])
        self.assertEqual(mapping(first)["train"], "train")
        self.assertEqual(mapping(first)["test"], "test")
        self.assertEqual({item["split"] for item in first if item["split"] not in ("train", "test")}, {"dev", "calibration"})
        validate_manifest(first)

    def test_too_few_speakers_and_existing_calibration_fail(self):
        with self.assertRaisesRegex(ValueError, "At least two"):
            split_development([record()])
        rows = [record("a"), record("b", speaker="vctk:p226", split="calibration")]
        with self.assertRaisesRegex(ValueError, "already exists"):
            split_development(rows)


if __name__ == "__main__":
    unittest.main()
