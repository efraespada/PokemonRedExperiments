import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from prism_gym_env_v2 import PrismGymEnv


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def build_env_config(args, output_dir):
    return {
        "headless": True,
        "save_final_state": False,
        "action_freq": args.action_freq,
        "init_state": str(args.state),
        "max_steps": args.steps,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": output_dir,
        "gb_path": str(args.rom),
        "reward_scale": 1.0,
        "screen_explore_weight": 0.005,
        "coord_explore_weight": 0.50,
        "map_explore_weight": 5.0,
        "pokedex_seen_weight": 0.25,
        "pokedex_caught_weight": 2.0,
        "level_weight": 0.5,
        "heal_weight": 0.25,
        "death_penalty_weight": 5.0,
        "stuck_penalty_weight": 0.05,
    }


def evaluate(env, model, episodes, seed, deterministic=True):
    rng = np.random.default_rng(seed)
    results = []
    for episode in range(episodes):
        episode_seed = seed + episode
        obs, _ = env.reset(seed=episode_seed)
        if model is not None:
            model.set_random_seed(episode_seed)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            if model is None:
                action = int(rng.integers(env.action_space.n))
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
                action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

        final = env.agent_stats[-1]
        visited_maps = {
            (step["map_group"], step["map"]) for step in env.agent_stats
        }
        results.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": env.step_count,
                "coordinates": final["coord_count"],
                "screens": final["screen_count"],
                "level_sum": final["levels_sum"],
                "max_level_sum": max(step["levels_sum"] for step in env.agent_stats),
                "party_count": final["pcount"],
                "maps": len(visited_maps),
                "battle_steps": sum(step["battle"] for step in env.agent_stats),
                "min_health": min(step["hp"] for step in env.agent_stats),
                "pokedex_seen": final["pokedex_seen"],
                "pokedex_caught": final["pokedex_caught"],
                "badges": final["badge"],
                "deaths": final["deaths"],
            }
        )
    return results


def summarize(results):
    metric_names = (
        "reward",
        "coordinates",
        "screens",
        "level_sum",
        "max_level_sum",
        "party_count",
        "maps",
        "battle_steps",
        "min_health",
        "pokedex_seen",
        "pokedex_caught",
        "badges",
        "deaths",
    )
    return {
        name: {
            "mean": float(np.mean([episode[name] for episode in results])),
            "min": float(np.min([episode[name] for episode in results])),
            "max": float(np.max([episode[name] for episode in results])),
        }
        for name in metric_names
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Prism PPO checkpoint or a random-policy baseline."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the PPO action distribution instead of taking its mode.",
    )
    parser.add_argument("--action-freq", type=int, default=24)
    parser.add_argument("--rom", type=Path, default=REPO_ROOT / "PokemonPrism.gbc")
    parser.add_argument(
        "--state",
        type=Path,
        default=SCRIPT_DIR / "bootstrap_states/larvitar_ready_adam.state",
    )
    parser.add_argument("--output", type=Path, default=Path("prism_evaluation.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes < 1 or args.steps < 1:
        raise ValueError("--episodes and --steps must be positive")
    if not args.rom.is_file():
        raise FileNotFoundError(f"Prism ROM not found: {args.rom}")
    if not args.state.is_file():
        raise FileNotFoundError(f"Prism state not found: {args.state}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = PrismGymEnv(build_env_config(args, args.output.parent))
    try:
        model = PPO.load(args.checkpoint) if args.checkpoint else None
        episodes = evaluate(
            env,
            model,
            args.episodes,
            args.seed,
            deterministic=not args.stochastic,
        )
    finally:
        env.close()

    report = {
        "policy": str(args.checkpoint) if args.checkpoint else "random",
        "deterministic": model is not None and not args.stochastic,
        "seed": args.seed,
        "episodes": episodes,
        "summary": summarize(episodes),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
