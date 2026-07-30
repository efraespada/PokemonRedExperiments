import sys
import os
from os.path import exists
from pathlib import Path

from prism_gym_env_v2 import PrismGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tensorboard_callback import TensorboardCallback


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def make_env(rank, env_conf, seed=0):
    def _init():
        env = PrismGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    ep_length = int(os.getenv("PRISM_EP_LENGTH", 2048 * 80))
    sess_id = os.getenv("PRISM_SESSION_DIR", "runs_prism")
    sess_path = Path(sess_id)

    init_state = Path(
        os.getenv(
            "PRISM_INIT_STATE",
            REPO_ROOT / "v2/bootstrap_states/larvitar_ready_adam.state",
        )
    )
    if not init_state.is_file():
        raise FileNotFoundError(
            f"Playable Prism state not found at {init_state}. "
            "Generate it with: python prism_bootstrap.py --rom ../PokemonPrism.gbc "
            "--preset larvitar_ready_adam"
        )
    init_states = [
        Path(value.strip()).resolve()
        for value in os.getenv("PRISM_INIT_STATES", str(init_state)).split(",")
        if value.strip()
    ]
    missing_states = [state for state in init_states if not state.is_file()]
    if missing_states:
        raise FileNotFoundError(f"Prism curriculum states not found: {missing_states}")

    env_config = {
        "headless": True,
        "save_final_state": False,
        "action_freq": 24,
        "init_state": str(init_state),
        "init_states": tuple(str(state) for state in init_states),
        "max_steps": ep_length,
        "print_rewards": os.getenv("PRISM_PRINT_REWARDS", "0") == "1",
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": str(
            Path(os.getenv("PRISM_ROM", REPO_ROOT / "PokemonPrism.gbc"))
        ),
        "reward_scale": 1.0,
        "screen_explore_weight": float(
            os.getenv("PRISM_SCREEN_EXPLORE_WEIGHT", "0.005")
        ),
        "coord_explore_weight": float(
            os.getenv("PRISM_COORD_EXPLORE_WEIGHT", "0.50")
        ),
        "map_explore_weight": float(
            os.getenv("PRISM_MAP_EXPLORE_WEIGHT", "5.0")
        ),
        "pokedex_seen_weight": float(
            os.getenv("PRISM_POKEDEX_SEEN_WEIGHT", "1.0")
        ),
        "pokedex_caught_weight": 2.0,
        "level_weight": 0.5,
        "heal_weight": 0.25,
        "death_penalty_weight": float(
            os.getenv("PRISM_DEATH_PENALTY_WEIGHT", "5.0")
        ),
        "opponent_weight": float(os.getenv("PRISM_OPPONENT_WEIGHT", "5.0")),
        "experience_weight": float(
            os.getenv("PRISM_EXPERIENCE_WEIGHT", "0.25")
        ),
        "damage_weight": float(os.getenv("PRISM_DAMAGE_WEIGHT", "5.0")),
        "stuck_penalty_weight": 0.05,
    }

    print(env_config)

    num_cpu = int(os.getenv("PRISM_NUM_CPU", 8))
    vec_env = os.getenv("PRISM_VEC_ENV", "subproc").lower()
    env_factories = [make_env(i, env_config) for i in range(num_cpu)]
    if vec_env == "dummy":
        env = DummyVecEnv(env_factories)
    elif vec_env == "subproc":
        env = SubprocVecEnv(env_factories)
    else:
        raise ValueError(
            f"Unsupported PRISM_VEC_ENV={vec_env!r}; expected 'subproc' or 'dummy'"
        )

    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip()

    train_steps_batch = int(os.getenv("PRISM_N_STEPS", ep_length // 64))
    if train_steps_batch < 2:
        raise ValueError("PRISM_N_STEPS must be at least 2")

    batch_size = int(
        os.getenv("PRISM_BATCH_SIZE", min(512, train_steps_batch * num_cpu))
    )
    n_epochs = int(os.getenv("PRISM_N_EPOCHS", 4))
    learning_rate = float(os.getenv("PRISM_LEARNING_RATE", 0.0003))
    ent_coef = float(os.getenv("PRISM_ENT_COEF", 0.01))
    seed = int(os.getenv("PRISM_SEED", 0))
    checkpoint_freq = int(
        os.getenv("PRISM_CHECKPOINT_FREQ", max(1, ep_length // 4))
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq, save_path=sess_path, name_prefix="prism"
    )
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.batch_size = batch_size
        model.n_epochs = n_epochs
        model.ent_coef = ent_coef
        model.learning_rate = learning_rate
        model._setup_lr_schedule()
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.997,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            seed=seed,
            tensorboard_log=sess_path,
        )

    print(model.policy)
    total_timesteps = int(
        os.getenv("PRISM_TOTAL_TIMESTEPS", (ep_length) * num_cpu * 10000)
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        tb_log_name="prism_ppo",
    )
