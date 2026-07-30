import argparse
import hashlib
import json
import shutil
from pathlib import Path

from stable_baselines3 import PPO


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Package compatible Prism navigation and battle policies."
    )
    parser.add_argument("--navigation", type=Path, required=True)
    parser.add_argument("--battle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for label, path in (("navigation", args.navigation), ("battle", args.battle)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")
        PPO.load(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "navigation": args.output_dir / "navigation.zip",
        "battle": args.output_dir / "battle.zip",
    }
    shutil.copy2(args.navigation, outputs["navigation"])
    shutil.copy2(args.battle, outputs["battle"])

    manifest = {
        "format": 1,
        "navigation": {
            "file": outputs["navigation"].name,
            "sha256": sha256(outputs["navigation"]),
            "source": str(args.navigation),
        },
        "battle": {
            "file": outputs["battle"].name,
            "sha256": sha256(outputs["battle"]),
            "source": str(args.battle),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
