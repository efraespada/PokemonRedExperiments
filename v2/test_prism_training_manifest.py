import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prism_training_manifest import (
    finish_training_manifest,
    start_training_manifest,
)


class PrismTrainingManifestTest(unittest.TestCase):
    def test_records_completed_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_manifest.json"
            with patch(
                "prism_training_manifest.utc_now",
                side_effect=("start", "finish"),
            ), patch(
                "prism_training_manifest.time.monotonic",
                side_effect=(10.0, 12.5),
            ):
                manifest, started = start_training_manifest(path, {"seed": 7})
                finish_training_manifest(
                    path,
                    manifest,
                    started,
                    "completed",
                    checkpoints=("prism_32_steps.zip",),
                )

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["duration_seconds"], 2.5)
            self.assertEqual(saved["configuration"]["seed"], 7)
            self.assertEqual(saved["checkpoints"], ["prism_32_steps.zip"])

    def test_records_failed_training_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_manifest.json"
            with patch(
                "prism_training_manifest.utc_now",
                side_effect=("start", "finish"),
            ), patch(
                "prism_training_manifest.time.monotonic",
                side_effect=(10.0, 11.0),
            ):
                manifest, started = start_training_manifest(path, {})
                finish_training_manifest(
                    path,
                    manifest,
                    started,
                    "failed",
                    error=ValueError("invalid setup"),
                )

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["error"], "ValueError: invalid setup")


if __name__ == "__main__":
    unittest.main()
