import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from prism_gym_env_v2 import PrismGymEnv
from prism_memory_scan import snapshot_memory


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def save_snapshot(output_dir, label, ranges, metadata):
    base = output_dir / label
    np.savez_compressed(base.with_suffix(".npz"), **ranges)
    base.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def snapshot(env):
    ranges, summary = snapshot_memory(env.pyboy)
    return ranges, summary


def main():
    parser = argparse.ArgumentParser(
        description="Trace Prism WRAM at PPO battle entry and exit transitions."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rom", type=Path, default=REPO_ROOT / "PokemonPrism.gbc")
    parser.add_argument(
        "--state",
        type=Path,
        default=SCRIPT_DIR / "bootstrap_states/larvitar_ready_adam.state",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "memory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "headless": True,
        "save_final_state": False,
        "action_freq": 24,
        "init_state": str(args.state),
        "max_steps": args.steps,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": args.output_dir,
        "gb_path": str(args.rom),
    }
    env = PrismGymEnv(config)
    model = PPO.load(args.checkpoint)
    model.set_random_seed(args.seed)
    transitions = []
    try:
        obs, _ = env.reset(seed=args.seed)
        was_in_battle = env.is_in_battle()
        before_ranges, before_summary = snapshot(env)
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, _ = env.step(int(action))
            in_battle = env.is_in_battle()
            if in_battle != was_in_battle:
                transition = "entry" if in_battle else "exit"
                prefix = f"{step:06d}_{transition}"
                common = {
                    "step": step,
                    "action": int(action),
                    "transition": transition,
                    "health": env.read_hp_fraction(),
                    "party_levels": env.read_party_levels(),
                    "pokedex": env.get_pokedex_counts(),
                    "coordinates": env.coord_key(),
                }
                save_snapshot(
                    args.output_dir,
                    f"{prefix}_before",
                    before_ranges,
                    {**common, "side": "before", "memory": before_summary},
                )
                after_ranges, after_summary = snapshot(env)
                save_snapshot(
                    args.output_dir,
                    f"{prefix}_after",
                    after_ranges,
                    {**common, "side": "after", "memory": after_summary},
                )
                with open(args.output_dir / f"{prefix}_after.state", "wb") as state_file:
                    env.pyboy.save_state(state_file)
                transitions.append(common)
            before_ranges, before_summary = snapshot(env)
            was_in_battle = in_battle
            if terminated or truncated:
                break
    finally:
        env.close()

    report = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "steps": env.step_count,
        "transitions": transitions,
    }
    (args.output_dir / "battle_trace.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
