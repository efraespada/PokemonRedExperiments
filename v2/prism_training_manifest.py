import json
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def start_training_manifest(path, configuration):
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "configuration": configuration,
    }
    write_manifest(path, manifest)
    return manifest, time.monotonic()


def finish_training_manifest(
    path, manifest, started_monotonic, status, checkpoints=(), error=None
):
    result = dict(manifest)
    result.update(
        {
            "status": status,
            "finished_at": utc_now(),
            "duration_seconds": round(time.monotonic() - started_monotonic, 3),
            "checkpoints": [str(Path(checkpoint)) for checkpoint in checkpoints],
        }
    )
    if error is not None:
        result["error"] = f"{type(error).__name__}: {error}"
    write_manifest(path, result)
    return result
