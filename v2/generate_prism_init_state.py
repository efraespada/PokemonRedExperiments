import argparse
from pathlib import Path

from pyboy import PyBoy


def main():
    parser = argparse.ArgumentParser(
        description="Generate a deterministic initial PyBoy state for Pokemon Prism."
    )
    parser.add_argument("--rom", required=True, help="Path to the Pokemon Prism ROM.")
    parser.add_argument(
        "--output",
        default="../prism_init.state",
        help="Where to write the generated state.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=1200,
        help="How many emulator ticks to advance before saving the state.",
    )
    args = parser.parse_args()

    rom_path = Path(args.rom).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    pyboy = PyBoy(str(rom_path), window="null")
    try:
        for _ in range(args.ticks):
            pyboy.tick(1, False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pyboy.save_state(f)
    finally:
        pyboy.stop()

    print(output_path)


if __name__ == "__main__":
    main()
