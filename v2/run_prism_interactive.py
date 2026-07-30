import argparse
import os
import uuid
from pathlib import Path

from stable_baselines3 import PPO

from prism_gym_env_v2 import PrismGymEnv


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def select_policy(env, navigation_model, battle_model=None):
    if battle_model is not None and env.is_in_battle():
        return battle_model
    return navigation_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a navigation policy with an optional Prism battle specialist."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=os.getenv("PRISM_NAV_CHECKPOINT"),
        required=os.getenv("PRISM_NAV_CHECKPOINT") is None,
    )
    parser.add_argument(
        "--battle-checkpoint",
        type=Path,
        default=os.getenv("PRISM_BATTLE_CHECKPOINT"),
    )
    parser.add_argument(
        "--rom",
        type=Path,
        default=Path(os.getenv("PRISM_ROM", REPO_ROOT / "PokemonPrism.gbc")),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(
            os.getenv(
                "PRISM_INIT_STATE",
                SCRIPT_DIR / "bootstrap_states/larvitar_ready_adam.state",
            )
        ),
    )
    parser.add_argument("--max-steps", type=int, default=2**20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--always-on", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for label, path in (
        ("ROM", args.rom),
        ("state", args.state),
        ("navigation checkpoint", args.checkpoint),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Prism {label} not found: {path}")
    if args.battle_checkpoint and not args.battle_checkpoint.is_file():
        raise FileNotFoundError(
            f"Prism battle checkpoint not found: {args.battle_checkpoint}"
        )

    session_path = SCRIPT_DIR / f"session_{str(uuid.uuid4())[:8]}"
    env = PrismGymEnv(
        {
            "headless": args.headless,
            "save_final_state": True,
            "action_freq": 24,
            "init_state": str(args.state),
            "max_steps": args.max_steps,
            "print_rewards": True,
            "save_video": False,
            "fast_video": True,
            "session_path": session_path,
            "gb_path": str(args.rom),
            "reward_scale": 1.0,
            "screen_explore_weight": 0.005,
            "coord_explore_weight": 0.50,
            "map_explore_weight": 5.0,
            "pokedex_seen_weight": 1.0,
            "pokedex_caught_weight": 2.0,
            "opponent_weight": 5.0,
            "experience_weight": 0.25,
            "damage_weight": 5.0,
            "death_penalty_weight": 5.0,
            "stuck_penalty_weight": 0.05,
        }
    )
    navigation_model = PPO.load(args.checkpoint)
    battle_model = (
        PPO.load(args.battle_checkpoint) if args.battle_checkpoint else None
    )
    print(f"navigation policy: {args.checkpoint}")
    print(f"battle policy: {args.battle_checkpoint or 'navigation policy'}")

    try:
        obs, _ = env.reset()
        truncated = False
        while not truncated:
            agent_enabled = args.always_on
            if not agent_enabled:
                try:
                    agent_enabled = (
                        SCRIPT_DIR / "agent_enabled.txt"
                    ).read_text().startswith("yes")
                except OSError:
                    agent_enabled = False

            if agent_enabled:
                model = select_policy(env, navigation_model, battle_model)
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, _, _, truncated, _ = env.step(int(action))
            else:
                env.pyboy.tick(1, True)
                obs = env._get_obs()
                truncated = env.step_count >= env.max_steps - 1
            env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
