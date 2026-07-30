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
        "pokedex_seen_weight": 1.0,
        "pokedex_caught_weight": 2.0,
        "event_weight": 1.0,
        "level_weight": 0.5,
        "heal_weight": 0.25,
        "death_penalty_weight": 5.0,
        "opponent_weight": 5.0,
        "experience_weight": 0.25,
        "damage_weight": 5.0,
        "stuck_penalty_weight": 0.05,
    }


def evaluate(env, model, episodes, seed, deterministic=True, battle_model=None):
    rng = np.random.default_rng(seed)
    results = []
    for episode in range(episodes):
        episode_seed = seed + episode
        obs, _ = env.reset(seed=episode_seed)
        if model is not None:
            model.set_random_seed(episode_seed)
        if battle_model is not None:
            battle_model.set_random_seed(episode_seed)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            if model is None:
                action = int(rng.integers(env.action_space.n))
            else:
                active_model = (
                    battle_model
                    if battle_model is not None and env.is_in_battle()
                    else model
                )
                action, _ = active_model.predict(obs, deterministic=deterministic)
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
                "experience_gained": max(
                    step["experience_gained"] for step in env.agent_stats
                ),
                "damage": max(step["damage"] for step in env.agent_stats),
                "party_count": final["pcount"],
                "party_species": final["party_species_count"],
                "maps": len(visited_maps),
                "battle_steps": sum(step["battle"] for step in env.agent_stats),
                "encounters": final["encounters"],
                "victories": final["victories"],
                "battle_defeats": final["battle_defeats"],
                "other_battle_exits": final["other_battle_exits"],
                "opponents": max(step["opponent_count"] for step in env.agent_stats),
                "max_enemy_level": max(
                    step["enemy_level"] for step in env.agent_stats
                ),
                "min_health": min(step["hp"] for step in env.agent_stats),
                "pokedex_seen": final["pokedex_seen"],
                "pokedex_caught": final["pokedex_caught"],
                "pokedex_seen_progress": final["pokedex_seen_progress"],
                "pokedex_caught_progress": final["pokedex_caught_progress"],
                "event_count": final["event_count"],
                "event_progress": final["event_progress"],
                "item_slots": final["item_slots"],
                "item_quantity": final["item_quantity"],
                "key_items": final["key_items"],
                "ball_slots": final["ball_slots"],
                "ball_quantity": final["ball_quantity"],
                "badges": final["badge"],
                "badges_naljo": final["badges_naljo"],
                "badges_rijon": final["badges_rijon"],
                "badges_other": final["badges_other"],
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
        "experience_gained",
        "damage",
        "party_count",
        "party_species",
        "maps",
        "battle_steps",
        "encounters",
        "victories",
        "battle_defeats",
        "other_battle_exits",
        "opponents",
        "max_enemy_level",
        "min_health",
        "pokedex_seen",
        "pokedex_caught",
        "pokedex_seen_progress",
        "pokedex_caught_progress",
        "event_count",
        "event_progress",
        "item_slots",
        "item_quantity",
        "key_items",
        "ball_slots",
        "ball_quantity",
        "badges",
        "badges_naljo",
        "badges_rijon",
        "badges_other",
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


def success_rates(results):
    episode_count = len(results)
    return {
        "battle": sum(episode["encounters"] > 0 for episode in results) / episode_count,
        "victory": sum(episode["victories"] > 0 for episode in results)
        / episode_count,
        "battle_defeat": sum(episode["battle_defeats"] > 0 for episode in results)
        / episode_count,
        "map_transition": sum(episode["maps"] > 1 for episode in results)
        / episode_count,
        "death": sum(episode["deaths"] > 0 for episode in results)
        / episode_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Prism PPO checkpoint or a random-policy baseline."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--battle-checkpoint",
        type=Path,
        help="Optional specialist policy used whenever Prism is in battle.",
    )
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
        battle_model = (
            PPO.load(args.battle_checkpoint) if args.battle_checkpoint else None
        )
        if battle_model is not None and model is None:
            raise ValueError("--battle-checkpoint requires --checkpoint")
        episodes = evaluate(
            env,
            model,
            args.episodes,
            args.seed,
            deterministic=not args.stochastic,
            battle_model=battle_model,
        )
    finally:
        env.close()

    report = {
        "policy": str(args.checkpoint) if args.checkpoint else "random",
        "battle_policy": (
            str(args.battle_checkpoint) if args.battle_checkpoint else None
        ),
        "deterministic": model is not None and not args.stochastic,
        "seed": args.seed,
        "episodes": episodes,
        "summary": summarize(episodes),
        "success_rates": success_rates(episodes),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps({"success_rates": report["success_rates"]}, indent=2))


if __name__ == "__main__":
    main()
