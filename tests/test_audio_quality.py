import tempfile
from pathlib import Path
import unittest
import wave
import numpy as np

from ontoforensics.audio_quality import annotate_quality, prepare_audio


class QualityMeasurementTests(unittest.TestCase):
    def test_preparation_resamples_without_changing_group_or_labels(self):
        try:
            import soundfile
        except ImportError:
            self.skipTest("optional soundfile not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(8000)
                handle.writeframes(np.zeros(8000, dtype="<i2").tobytes())
            record = {"recording_id": "u", "source_recording_id": "original", "speaker_id": "s",
                      "audio_path": str(source), "duration_sec": 1., "sample_rate": 8000,
                      "dataset": "unit-test", "split": "train", "label": 1, "fake_intervals": [[.4, .6]]}
            result = prepare_audio([record], root / "prepared")
            self.assertEqual(result[0]["sample_rate"], 16000)
            self.assertEqual(result[0]["fake_intervals"], [[.4, .6]])
            self.assertEqual(result[0]["source_recording_id"], "original")
            self.assertEqual(result[0]["duration_sec"], 1.)
            self.assertEqual(record["sample_rate"], 8000)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                prepare_audio([record], root / "prepared")

    def test_local_quality_uses_samples_and_does_not_invent_snr(self):
        try:
            import soundfile
        except ImportError:
            self.skipTest("optional soundfile not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
                handle.writeframes(np.concatenate([np.zeros(8000), np.full(8000, 32767)]).astype("<i2").tobytes())
            records = [{"recording_id": "u", "audio_path": str(path), "duration_sec": 1.}]
            observations = [{"recording_id": "u", "start_sec": a, "end_sec": b, "status": "valid"}
                            for a, b in [(0., .5), (.5, 1.)]]
            measured = annotate_quality(records, observations)
            self.assertEqual(measured[0]["quality"]["clipping_fraction"], 0.)
            self.assertEqual(measured[1]["quality"]["clipping_fraction"], 1.)
            self.assertLess(measured[0]["quality"]["rms_db"], -100)
            self.assertNotIn("estimated_snr_db", measured[1]["quality"])
            self.assertNotIn("quality", observations[0])


if __name__ == "__main__":
    unittest.main()
