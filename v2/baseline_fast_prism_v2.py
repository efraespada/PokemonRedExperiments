import sys
from os.path import exists
from pathlib import Path

from prism_gym_env_v2 import PrismGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from tensorboard_callback import TensorboardCallback


def make_env(rank, env_conf, seed=0):
    def _init():
        env = PrismGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    ep_length = 2048 * 80
    sess_id = "runs_prism"
    sess_path = Path(sess_id)

    env_config = {
        "headless": True,
        "save_final_state": False,
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

    print(env_config)

    num_cpu = 8
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length // 2, save_path=sess_path, name_prefix="prism"
    )
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip()

    train_steps_batch = ep_length // 64

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=512,
            n_epochs=1,
            gamma=0.997,
            ent_coef=0.01,
            tensorboard_log=sess_path,
        )

    print(model.policy)
    model.learn(
        total_timesteps=(ep_length) * num_cpu * 10000,
        callback=CallbackList(callbacks),
        tb_log_name="prism_ppo",
    )
