import argparse
import json
from pathlib import Path

import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent


INPUTS = {
    "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
    "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    "start": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
    "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
    "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
    "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
    "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
}


def parse_script(script):
    steps = []
    if not script:
        return steps
    for raw in script.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            name, ticks = raw.split(":", 1)
            steps.append((name.strip().lower(), int(ticks.strip())))
        else:
            steps.append((raw.lower(), 24))
    return steps


def snapshot_memory(pyboy):
    ranges = {
        "wram0": np.array([pyboy.memory[a] for a in range(0xC000, 0xD000)], dtype=np.uint8),
        "wramx": np.array([pyboy.memory[a] for a in range(0xD000, 0xE000)], dtype=np.uint8),
        "hram": np.array([pyboy.memory[a] for a in range(0xFF80, 0xFFFF)], dtype=np.uint8),
    }
    summary = {
        "coord_candidates": {
            "d35d": int(pyboy.memory[0xD35D]),
            "d35e": int(pyboy.memory[0xD35E]),
            "d361": int(pyboy.memory[0xD361]),
            "d362": int(pyboy.memory[0xD362]),
        },
        "badge_candidates": {
            "d356": int(pyboy.memory[0xD356]),
            "d57c": int(pyboy.memory[0xD57C]),
            "d857": int(pyboy.memory[0xD857]),
        },
    }
    return ranges, summary


def run_script(pyboy, steps):
    for name, ticks in steps:
        if name == "wait":
            pyboy.tick(ticks, False)
            continue
        press, release = INPUTS[name]
        pyboy.send_input(press)
        pyboy.tick(8, False)
        pyboy.send_input(release)
        pyboy.tick(max(ticks - 8, 1), False)


def main():
    parser = argparse.ArgumentParser(description="Capture Prism memory snapshots.")
    parser.add_argument("--rom", required=True)
    parser.add_argument("--state")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", default="memory")
    parser.add_argument(
        "--script",
        default="",
        help="Comma-separated actions like 'a,start,wait:120,right'.",
    )
    parser.add_argument("--boot-ticks", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pyboy = PyBoy(str(Path(args.rom).resolve()), window="null")
    try:
        if args.state:
            with open(Path(args.state).resolve(), "rb") as f:
                pyboy.load_state(f)
        elif args.boot_ticks:
            pyboy.tick(args.boot_ticks, False)

        run_script(pyboy, parse_script(args.script))
        ranges, summary = snapshot_memory(pyboy)

        base = output_dir / args.label
        np.savez_compressed(
            base.with_suffix(".npz"),
            wram0=ranges["wram0"],
            wramx=ranges["wramx"],
            hram=ranges["hram"],
        )
        base.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        print(base.with_suffix(".npz"))
        print(base.with_suffix(".json"))
    finally:
        pyboy.stop()


if __name__ == "__main__":
    main()
