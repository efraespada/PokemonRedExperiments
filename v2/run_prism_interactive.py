import glob
import os
import time
import uuid
from pathlib import Path

from prism_gym_env_v2 import PrismGymEnv
from stable_baselines3 import PPO


def get_most_recent_zip_with_age(folder_path):
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    if not zip_files:
        return None, None

    most_recent_zip = max(zip_files, key=os.path.getmtime)
    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600
    return most_recent_zip, age_in_hours


if __name__ == "__main__":
    sess_path = Path(f"session_{str(uuid.uuid4())[:8]}")
    ep_length = 2**20

    env_config = {
        "headless": False,
        "save_final_state": True,
        "action_freq": 24,
        "init_state": "../prism_init.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "../PokemonPrism.gbc",
        "reward_scale": 1.0,
        "screen_explore_weight": 0.05,
        "coord_explore_weight": 0.10,
        "stuck_penalty_weight": 0.05,
    }

    env = PrismGymEnv(env_config)
    most_recent_checkpoint, time_since = get_most_recent_zip_with_age("runs_prism")
    if most_recent_checkpoint is None:
        raise FileNotFoundError("No se encontro ningun checkpoint en runs_prism/")

    print(f"using checkpoint: {most_recent_checkpoint}, age: {time_since:.2f}h")
    model = PPO.load(
        most_recent_checkpoint, env=env, custom_objects={"lr_schedule": 0, "clip_range": 0}
    )

    obs, info = env.reset()
    while True:
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except Exception:
            agent_enabled = False

        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, terminated, truncated, info = env.step(action)
        else:
            env.pyboy.tick(1, True)
            obs = env._get_obs()
            truncated = env.step_count >= env.max_steps - 1

        env.render()
        if truncated:
            break

    env.close()
