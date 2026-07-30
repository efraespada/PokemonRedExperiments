import argparse
import json
from pathlib import Path

from PIL import Image
from pyboy import PyBoy
from pyboy.utils import WindowEvent


BUTTONS = {
    "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
    "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    "start": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
    "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
    "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
    "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
    "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
}


PRESETS = {
    "title": [("wait", 1200)],
    "new_game_menu": [("wait", 1200), ("start", 24), ("wait", 180)],
    "calendar": [
        ("wait", 1200),
        ("start", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
    ],
    "intro_text": [
        ("wait", 1200),
        ("start", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
    ],
    "name_selection": [
        ("wait", 1200),
        ("start", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 240),
    ]
    + [("a", 24), ("wait", 180)] * 40,
    "name_adam": [
        ("wait", 1200),
        ("start", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 240),
    ]
    + [("a", 24), ("wait", 180)] * 40
    + [("down", 24), ("wait", 60), ("a", 24), ("wait", 240)],
    "map_ready_adam": [
        ("wait", 1200),
        ("start", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 180),
        ("a", 24),
        ("wait", 240),
    ]
    + [("a", 24), ("wait", 180)] * 40
    + [("down", 24), ("wait", 60), ("a", 24), ("wait", 240)]
    + [("a", 24), ("wait", 180)] * 8,
}


def run_step(pyboy, action, ticks):
    if action == "wait":
        if ticks > 1:
            pyboy.tick(ticks - 1, False)
        pyboy.tick(1, True)
        return

    press, release = BUTTONS[action]
    pyboy.send_input(press)
    pyboy.tick(8, True)
    pyboy.send_input(release)
    if ticks - 8 > 1:
        pyboy.tick(ticks - 8 - 1, False)
    pyboy.tick(1, True)


def snapshot(pyboy):
    return {
        "d35d": int(pyboy.memory[0xD35D]),
        "d35e": int(pyboy.memory[0xD35E]),
        "d356": int(pyboy.memory[0xD356]),
        "d361": int(pyboy.memory[0xD361]),
        "d362": int(pyboy.memory[0xD362]),
        "dcb5": int(pyboy.memory[0xDCB5]),
        "dcb6": int(pyboy.memory[0xDCB6]),
        "dcb7": int(pyboy.memory[0xDCB7]),
        "dcb8": int(pyboy.memory[0xDCB8]),
        "dcd7": int(pyboy.memory[0xDCD7]),
        "dcd8": int(pyboy.memory[0xDCD8]),
        "dcd9": int(pyboy.memory[0xDCD9]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible Pokemon Prism onboarding states."
    )
    parser.add_argument("--rom", required=True)
    parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(PRESETS.keys()),
        help="Named onboarding checkpoint to build.",
    )
    parser.add_argument(
        "--output-dir",
        default="bootstrap_states",
        help="Directory where state, screenshot and metadata will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pyboy = PyBoy(str(Path(args.rom).resolve()), window="null")
    try:
        for action, ticks in PRESETS[args.preset]:
            run_step(pyboy, action, ticks)

        stem = output_dir / args.preset
        with open(stem.with_suffix(".state"), "wb") as f:
            pyboy.save_state(f)
        Image.fromarray(pyboy.screen.ndarray[:, :, :3], "RGB").save(
            stem.with_suffix(".png")
        )
        stem.with_suffix(".json").write_text(
            json.dumps(
                {
                    "preset": args.preset,
                    "steps": PRESETS[args.preset],
                    "memory": snapshot(pyboy),
                },
                indent=2,
            )
        )
        print(stem.with_suffix(".state"))
        print(stem.with_suffix(".png"))
        print(stem.with_suffix(".json"))
    finally:
        pyboy.stop()


if __name__ == "__main__":
    main()
