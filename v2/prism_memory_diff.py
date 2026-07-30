import argparse
import json
from pathlib import Path

import numpy as np


def diff_section(name, before, after, base_addr):
    changed = np.where(before != after)[0]
    rows = []
    for idx in changed[:512]:
        addr = base_addr + int(idx)
        rows.append(
            {
                "addr": f"0x{addr:04X}",
                "before": int(before[idx]),
                "after": int(after[idx]),
            }
        )
    return {"section": name, "changed_count": int(changed.size), "changes": rows}


def main():
    parser = argparse.ArgumentParser(description="Diff two Prism memory snapshots.")
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    before = np.load(Path(args.before).resolve())
    after = np.load(Path(args.after).resolve())

    report = {
        "wram0": diff_section("wram0", before["wram0"], after["wram0"], 0xC000),
        "wramx": diff_section("wramx", before["wramx"], after["wramx"], 0xD000),
        "hram": diff_section("hram", before["hram"], after["hram"], 0xFF80),
    }

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
