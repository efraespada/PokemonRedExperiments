import hashlib
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from einops import repeat
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from skimage.transform import downscale_local_mean

from prism_memory import (
    BADGES,
    BATTLE_MODE,
    ENEMY_LEVEL,
    ENEMY_HP,
    ENEMY_MAX_HP,
    ENEMY_SPECIES,
    PARTY_COUNT,
    PARTY_EXP,
    PARTY_HP,
    PARTY_LEVELS,
    PARTY_MAX_HP,
    POKEDEX_BYTES,
    POKEDEX_CAUGHT,
    POKEDEX_SEEN,
    PRISM_WRAM_BANK,
    active_party_values,
    classify_battle_outcome,
    count_bits,
    read_u16_be,
    read_u24_be,
)


class PrismGymEnv(Env):
    def __init__(self, config=None):
        config = config or {}
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.init_states = tuple(config.get("init_states", (self.init_state,)))
        if not self.init_states:
            raise ValueError("init_states must contain at least one state")
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = config.get("frame_stacks", 3)
        self.reward_scale = config.get("reward_scale", 1.0)
        self.screen_explore_weight = config.get("screen_explore_weight", 0.05)
        self.coord_explore_weight = config.get("coord_explore_weight", 0.10)
        self.map_explore_weight = config.get("map_explore_weight", 5.0)
        self.pokedex_seen_weight = config.get("pokedex_seen_weight", 1.0)
        self.pokedex_caught_weight = config.get("pokedex_caught_weight", 2.0)
        self.level_weight = config.get("level_weight", 0.5)
        self.heal_weight = config.get("heal_weight", 0.25)
        self.death_penalty_weight = config.get("death_penalty_weight", 5.0)
        self.opponent_weight = config.get("opponent_weight", 5.0)
        self.experience_weight = config.get("experience_weight", 0.25)
        self.damage_weight = config.get("damage_weight", 5.0)
        self.stuck_penalty_weight = config.get("stuck_penalty_weight", 0.05)
        self.stuck_threshold = config.get("stuck_threshold", 600)
        self.instance_id = config.get("instance_id", str(uuid.uuid4())[:8])
        self.coord_addrs = config.get(
            "coord_addrs",
            {
                "x": 0xDCB8,
                "y": 0xDCB7,
                "map": 0xDCB6,
                "map_group": 0xDCB5,
                "battle": BATTLE_MODE,
            },
        )
        self.level_addrs = config.get("level_addrs", PARTY_LEVELS)
        self.hp_addrs = config.get("hp_addrs", PARTY_HP)
        self.max_hp_addrs = config.get("max_hp_addrs", PARTY_MAX_HP)
        self.badge_addrs = config.get("badge_addrs", BADGES)
        self.party_count_addr = config.get("party_count_addr", PARTY_COUNT)
        self.pokedex_caught_addr = config.get("pokedex_caught_addr", POKEDEX_CAUGHT)
        self.pokedex_seen_addr = config.get("pokedex_seen_addr", POKEDEX_SEEN)
        self.pokedex_bytes = config.get("pokedex_bytes", POKEDEX_BYTES)
        self.event_obs_length = config.get("event_obs_length", 8)

        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        self.metadata = {"render.modes": []}
        self.reward_range = (-1000, 10000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12
        self.enc_freqs = 8

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(
                    low=0, high=255, shape=self.output_shape, dtype=np.uint8
                ),
                "health": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "level": spaces.Box(
                    low=-1, high=1, shape=(self.enc_freqs,), dtype=np.float32
                ),
                "badges": spaces.MultiBinary(max(1, len(self.badge_addrs) * 8)),
                "battle": spaces.MultiDiscrete([2, 101, 257]),
                "pokedex": spaces.MultiDiscrete([257, 257]),
                "events": spaces.MultiBinary(self.event_obs_length),
                "map": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.coords_pad * 4, self.coords_pad * 4, 1),
                    dtype=np.uint8,
                ),
                "recent_actions": spaces.MultiDiscrete(
                    [len(self.valid_actions)] * self.frame_stacks
                ),
            }
        )

        head = "null" if self.headless else "SDL2"
        self.pyboy = PyBoy(config["gb_path"], window=head)
        if not self.headless:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed = seed
        self.init_state_index = int(self.np_random.integers(len(self.init_states)))
        selected_state = self.init_states[self.init_state_index]
        with open(selected_state, "rb") as f:
            self.pyboy.load_state(f)

        self.agent_stats = []
        self.explore_map = np.zeros(
            (self.coords_pad * 4, self.coords_pad * 4), dtype=np.uint8
        )
        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)
        self.seen_coords = {}
        self.seen_maps = set()
        self.seen_opponents = set()
        self.seen_screen_hashes = set()
        self.current_event_flags_set = {}
        self.screen_explore_count = 0
        self.coord_explore_count = 0
        self.stuck_penalty_count = 0
        self.last_health = self.read_hp_fraction()
        self.party_size = self.read_party_count()
        self.max_level_sum = self.read_level_sum()
        self.initial_experience = self.read_experience_sum()
        self.max_experience = self.initial_experience
        self.total_healing = 0.0
        self.total_damage = 0.0
        self.last_enemy_health = self.read_enemy_hp_fraction()
        self.last_in_battle = self.is_in_battle()
        self.battle_start_experience = self.read_experience_sum()
        self.encounter_count = int(self.last_in_battle)
        self.victory_count = 0
        self.defeat_count = 0
        self.other_battle_exit_count = 0
        self.died_count = 0
        self.step_count = 0

        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum(self.progress_reward.values())
        self.reset_count += 1
        return self._get_obs(), {}

    def render(self, reduce_res=True):
        game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1]
        if reduce_res:
            game_pixels_render = downscale_local_mean(
                game_pixels_render, (2, 2, 1)
            ).astype(np.uint8)
        return game_pixels_render

    def _get_obs(self):
        screen = self.render()
        self.update_recent_screens(screen)
        level_sum = 0.02 * self.read_level_sum()

        pokedex_seen, pokedex_caught = self.get_pokedex_counts()
        return {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()], dtype=np.float32),
            "level": self.fourier_encode(level_sum).astype(np.float32),
            "badges": np.array(self.read_badges_bits(), dtype=np.int8),
            "battle": np.array(
                [
                    int(self.is_in_battle()),
                    min(100, self.read_enemy_level()),
                    self.read_enemy_species(),
                ],
                dtype=np.int64,
            ),
            "pokedex": np.array([pokedex_seen, pokedex_caught], dtype=np.int64),
            "events": self.read_battle_bits(),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
        }

    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.update_recent_actions(action)
        self.update_screen_exploration()
        self.update_seen_coords()
        self.update_battle_progress()
        self.update_explore_map()
        self.update_heal_reward()
        self.append_agent_stats(action)

        new_reward = self.update_reward()
        self.last_health = self.read_hp_fraction()
        step_limit_reached = self.check_if_done()
        obs = self._get_obs()
        self.step_count += 1
        return obs, new_reward, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        render_screen = self.save_video or not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = self.read_party_levels()
        pokedex_seen, pokedex_caught = self.get_pokedex_counts()
        self.agent_stats.append(
            {
                "step": self.step_count,
                "init_state": self.init_state_index,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "map_group": self.get_map_group(),
                "battle": int(self.is_in_battle()),
                "enemy_species": self.read_enemy_species(),
                "enemy_level": self.read_enemy_level(),
                "enemy_health": self.read_enemy_hp_fraction(),
                "damage": self.total_damage,
                "encounters": self.encounter_count,
                "victories": self.victory_count,
                "battle_defeats": self.defeat_count,
                "other_battle_exits": self.other_battle_exit_count,
                "opponent_count": len(self.seen_opponents),
                "last_action": int(action),
                "pcount": self.read_party_count(),
                "levels_sum": sum(levels),
                "experience": self.read_experience_sum(),
                "experience_gained": self.get_experience_gained(),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "map_count": len(self.seen_maps),
                "screen_count": len(self.seen_screen_hashes),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "pokedex_seen": pokedex_seen,
                "pokedex_caught": pokedex_caught,
                "stuck": self.stuck_penalty_count,
            }
        )

    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60,
            input_format="gray",
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer.__enter__()
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False)[:, :, 0])
        self.model_frame_writer.add_image(self.render(reduce_res=True)[:, :, 0])
        self.map_frame_writer.add_image(self.get_explore_map())

    def get_game_coords(self):
        return (
            self.read_m(self.coord_addrs["x"]),
            self.read_m(self.coord_addrs["y"]),
            self.read_m(self.coord_addrs["map"]),
        )

    def get_map_group(self):
        addr = self.coord_addrs.get("map_group")
        return self.read_m(addr) if addr is not None else 0

    def is_in_battle(self):
        battle_addr = self.coord_addrs.get("battle")
        if battle_addr is None:
            return False
        return self.read_m(battle_addr) != 0

    def read_enemy_species(self):
        return self.read_m(ENEMY_SPECIES) if self.is_in_battle() else 0

    def read_enemy_level(self):
        return self.read_m(ENEMY_LEVEL) if self.is_in_battle() else 0

    def read_enemy_hp_fraction(self):
        if not self.is_in_battle():
            return 0.0
        current_hp = read_u16_be(self.read_m, ENEMY_HP)
        max_hp = max(1, read_u16_be(self.read_m, ENEMY_MAX_HP))
        return min(1.0, current_hp / max_hp)

    def read_battle_bits(self):
        health = round(self.read_enemy_hp_fraction() * 127)
        bits = [int(bit) for bit in f"{health:07b}"]
        return np.array([int(self.is_in_battle()), *bits], dtype=np.int8)

    def update_battle_progress(self):
        in_battle = self.is_in_battle()
        if in_battle and not self.last_in_battle:
            self.encounter_count += 1
            self.battle_start_experience = self.read_experience_sum()
        if in_battle:
            species = self.read_enemy_species()
            if species:
                self.seen_opponents.add(species)
            health = self.read_enemy_hp_fraction()
            if self.last_in_battle and health < self.last_enemy_health:
                self.total_damage += self.last_enemy_health - health
            self.last_enemy_health = health
        else:
            if self.last_in_battle:
                outcome = classify_battle_outcome(
                    self.battle_start_experience,
                    self.read_experience_sum(),
                    self.read_hp_fraction(),
                )
                if outcome == "victory":
                    self.victory_count += 1
                elif outcome == "defeat":
                    self.defeat_count += 1
                else:
                    self.other_battle_exit_count += 1
            self.last_enemy_health = 0.0
        self.last_in_battle = in_battle

    def coord_key(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        return (int(self.get_map_group()), int(map_n), int(x_pos), int(y_pos))

    def update_seen_coords(self):
        if self.is_in_battle():
            return
        key = self.coord_key()
        self.seen_maps.add(key[:2])
        count = self.seen_coords.get(key, 0) + 1
        self.seen_coords[key] = count
        if count == 1:
            self.coord_explore_count += 1
        if count == self.stuck_threshold:
            self.stuck_penalty_count += 1

    def get_current_coord_count_reward(self):
        return int(self.seen_coords.get(self.coord_key(), 0) >= self.stuck_threshold)

    def update_explore_map(self):
        self.explore_map = self.render_local_explore_map()

    def render_local_explore_map(self):
        out = np.zeros((self.coords_pad * 2, self.coords_pad * 2), dtype=np.uint8)
        map_group, map_n, x_pos, y_pos = self.coord_key()
        for (coord_group, coord_map, coord_x, coord_y), count in self.seen_coords.items():
            if coord_group != map_group or coord_map != map_n:
                continue
            dx = coord_x - x_pos + self.coords_pad
            dy = coord_y - y_pos + self.coords_pad
            if 0 <= dx < self.coords_pad * 2 and 0 <= dy < self.coords_pad * 2:
                out[dy, dx] = min(255, 32 + count * 8)
        return repeat(out, "h w -> (h h2) (w w2)", h2=2, w2=2)

    def get_explore_map(self):
        return self.explore_map

    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:, :, 0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def update_screen_exploration(self):
        screen = self.render(reduce_res=True)[:, :, 0]
        digest = hashlib.sha1(screen.tobytes()).digest()
        if digest not in self.seen_screen_hashes:
            self.seen_screen_hashes.add(digest)
            self.screen_explore_count += 1

    def update_reward(self):
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(self.progress_reward.values())
        new_step = new_total - self.total_reward
        self.total_reward = new_total
        return new_step

    def check_if_done(self):
        return self.step_count >= self.max_steps - 1

    def save_and_print_info(self, done, obs):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:, :, 0],
            )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

    def read_m(self, addr):
        if 0xD000 <= addr < 0xE000:
            return self.pyboy.memory[PRISM_WRAM_BANK, addr]
        return self.pyboy.memory[addr]

    def read_badges_bits(self):
        bits = []
        for addr in self.badge_addrs:
            bits.extend(int(bit) for bit in f"{self.read_m(addr):08b}")
        if not bits:
            bits = [0]
        return bits

    def get_badges(self):
        return sum(self.read_badges_bits())

    def read_party_count(self):
        if self.party_count_addr is None:
            return 0
        return self.read_m(self.party_count_addr)

    def get_pokedex_counts(self):
        caught = count_bits(self.read_m, self.pokedex_caught_addr, self.pokedex_bytes)
        seen = count_bits(self.read_m, self.pokedex_seen_addr, self.pokedex_bytes)
        return seen, caught

    def read_party_levels(self):
        return active_party_values(
            self.read_m, self.level_addrs, self.read_party_count()
        )

    def read_level_sum(self):
        return sum(self.read_party_levels())

    def read_experience_sum(self):
        values = active_party_values(
            self.read_m, PARTY_EXP, self.read_party_count(), read_u24_be
        )
        return sum(values)

    def get_experience_gained(self):
        self.max_experience = max(self.max_experience, self.read_experience_sum())
        return max(0, self.max_experience - self.initial_experience)

    def get_game_state_reward(self):
        pokedex_seen, pokedex_caught = self.get_pokedex_counts()
        self.max_level_sum = max(self.max_level_sum, self.read_level_sum())
        return {
            "screen": self.reward_scale
            * self.screen_explore_weight
            * self.screen_explore_count,
            "explore": self.reward_scale
            * self.coord_explore_weight
            * self.coord_explore_count,
            "map": self.reward_scale
            * self.map_explore_weight
            * max(0, len(self.seen_maps) - 1),
            "badge": self.reward_scale * self.get_badges() * 5,
            "pokedex_seen": self.reward_scale
            * self.pokedex_seen_weight
            * pokedex_seen,
            "pokedex_caught": self.reward_scale
            * self.pokedex_caught_weight
            * pokedex_caught,
            "level": self.reward_scale * self.level_weight * self.max_level_sum,
            "heal": self.reward_scale * self.heal_weight * self.total_healing,
            "death": self.reward_scale
            * self.death_penalty_weight
            * self.died_count
            * -1,
            "opponent": self.reward_scale
            * self.opponent_weight
            * len(self.seen_opponents),
            "experience": self.reward_scale
            * self.experience_weight
            * self.get_experience_gained(),
            "damage": self.reward_scale * self.damage_weight * self.total_damage,
            "stuck": self.reward_scale
            * self.stuck_penalty_weight
            * self.stuck_penalty_count
            * -1,
        }

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health and self.read_party_count() == self.party_size:
            if self.last_health <= 0:
                self.died_count += 1
            else:
                self.total_healing += cur_health - self.last_health

    def read_hp_fraction(self):
        party_count = self.read_party_count()
        if party_count <= 0 or not self.hp_addrs or not self.max_hp_addrs:
            return 0.0
        hp_values = active_party_values(
            self.read_m, self.hp_addrs, party_count, read_u16_be
        )
        max_hp_values = active_party_values(
            self.read_m, self.max_hp_addrs, party_count, read_u16_be
        )
        hp_sum = sum(hp_values)
        max_hp_sum = sum(max_hp_values)
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return read_u16_be(self.read_m, start)

    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def close(self):
        if self.pyboy is not None:
            self.pyboy.stop()
