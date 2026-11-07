from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from ontoforensics.cli import build_parser, dispatch
from ontoforensics.demo import make_fixture
from ontoforensics.io import read_json, write_jsonl
from ontoforensics.data import load_manifest


class CommandContractTests(unittest.TestCase):
    def test_module_reports_nonzero_exit_on_input_error(self):
        result = subprocess.run([sys.executable, "-m", "ontoforensics", "split-dev",
                                 "--manifest", "/definitely-not-a-file.jsonl", "--output", "/unused.jsonl"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("ERROR:", result.stderr)

    def test_pilot_length_filter_excludes_whole_source_with_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, _, _ = make_fixture(root / "fixture", group_counts={"train": 3})
            disallowed_source = records[0]["source_recording_id"]
            records[0]["duration_sec"] = 25.
            path = root / "input.jsonl"
            output = root / "pilot.jsonl"
            write_jsonl(path, records)
            args = build_parser().parse_args(["select-pilot", "--manifest", str(path), "--output", str(output),
                                              "--groups-per-split", "10", "--max-duration-sec", "21.25", "--require-paired"])
            dispatch(args)
            selected = load_manifest(output)
            self.assertEqual(len(selected), 8)
            self.assertNotIn(disallowed_source, {r["source_recording_id"] for r in selected})
            report = read_json(output.with_suffix(".selection_report.json"))
            self.assertIn(disallowed_source, report["excluded_source_groups"])


if __name__ == "__main__":
    unittest.main()
