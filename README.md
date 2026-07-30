# Train RL agents to play Pokemon Red

### New 10-19-24! Updated & Simplified V2 Training Script - See V2 below
### New 1-29-24! - [Multiplayer Live Training Broadcast](https://github.com/pwhiddy/pokerl-map-viz/)  🎦 🔴 [View Here](https://pwhiddy.github.io/pokerl-map-viz/)
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py)  

See how in [Training Broadcast](#training-broadcast) section
  
## Watch the Video on Youtube! 

<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

## Join the discord server
[![Join the Discord server!](https://invidget.switchblade.xyz/RvadteZk4G)](http://discord.gg/RvadteZk4G)
  
## Running the Pretrained Model Interactively 🎮  
🐍 Python 3.10+ is recommended. Other versions may work but have not been tested.   
You also need to install ffmpeg and have it available in the command line.

### Windows Setup
Refer to this [Windows Setup Guide](windows-setup-guide.md)

### For AMD GPUs
Follow this [guide to install pytorch with ROCm support](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/howto_wsl.html)

### Linux / MacOS

V2 is now recommended over the original version. You may follow all steps below but replace `baselines` with `v2`.

1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`. 
2. Move into the `baselines/` directory:  
 ```cd baselines```  
3. Install dependencies:  
```pip install -r requirements.txt```  
It may be necessary in some cases to separately install the SDL libraries.
For V2 MacOS users should use ```macos_requirements.txt``` instead of ```requirements.txt```
4. Run:  
```python run_pretrained_interactive.py```
  
Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).  
You can pause the AI's input during the game by editing `agent_enabled.txt`

Note: the Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.

## Training the Model 🏋️ 

<img src="/assets/grid.png?raw=true" height="156">


### V2

- Trains faster and with less memory
- Reaches Cerulean
- Streams to map by default
- Other improvements

Replaces the frame KNN with a coordinate based exploration reward, as well as some other tweaks.
1. Previous steps but in the `v2` directory instead of `baselines`
2. Run:
```python baseline_fast_v2.py```

### Pokemon Prism (experimental)

This fork also includes an experimental `Pokemon Prism` environment under `v2/`.
Unlike the original `Pokemon Red` setup, the Prism variant avoids hard dependencies on Red-specific event flags and the global map, so it can boot from a Prism ROM and train with screen and coordinate exploration rewards.

Suggested setup:

1. Place your legally obtained Prism ROM at the repo root as `PokemonPrism.gbc`.
2. Move into `v2/`.
3. Generate a reproducible training state with the first Larvitar:
```python prism_bootstrap.py --rom ../PokemonPrism.gbc --preset larvitar_ready_adam```
4. Train with:
```python baseline_fast_prism_v2.py```

The trainer uses `bootstrap_states/larvitar_ready_adam.state` by default,
avoiding the title screen and onboarding dialogue and starting with the first
Larvitar in the party. You can override the ROM and state paths with `PRISM_ROM`
and `PRISM_INIT_STATE`. Restricted environments that cannot create subprocesses
can use `PRISM_VEC_ENV=dummy PRISM_NUM_CPU=1`; normal training uses the faster
`subproc` vector environment by default. Short experimental episodes should set
`PRISM_N_STEPS` explicitly (for example, `256`) so PPO does not learn from tiny
rollout batches. The Prism defaults prioritize coordinate exploration over
screen changes, use four PPO optimization epochs, and can be tuned with the
`PRISM_*` environment variables in `baseline_fast_prism_v2.py`. Resumed models
reapply those optimizer settings, and `PRISM_CHECKPOINT_FREQ` controls how often
intermediate policies are preserved for evaluation.

For curriculum training, `PRISM_INIT_STATES` accepts a comma-separated list of
local PyBoy states. The environment samples one state reproducibly per reset,
allowing mixed overworld and battle practice.

Evaluate a checkpoint deterministically and compare it with a seeded random
baseline:
```bash
python prism_evaluate.py --checkpoint runs_prism/prism_4096_steps.zip
python prism_evaluate.py --checkpoint runs_prism/prism_4096_steps.zip --stochastic
python prism_evaluate.py --checkpoint runs_prism/navigation.zip \
  --battle-checkpoint runs_prism/battle.zip --stochastic
python prism_evaluate.py --seed 0
```
Both commands write machine-readable episode metrics to
`prism_evaluation.json`; use `--output` to preserve multiple reports.

Run the hierarchical agent interactively (or headless for a smoke test):

```bash
python run_prism_interactive.py \
  --checkpoint runs_prism/navigation.zip \
  --battle-checkpoint runs_prism/battle.zip
```

Use `--always-on` to ignore `agent_enabled.txt`. The same paths can be provided
through `PRISM_NAV_CHECKPOINT` and `PRISM_BATTLE_CHECKPOINT`.

Notes:
- `prism_init.state` is intentionally not committed because it is generated from your ROM.
- Bootstrap states are also local, generated assets and are not committed.
- The reward shaping is currently conservative and generic; it is meant to provide a working starting point for Prism-specific iteration rather than feature parity with the Red environment.

## Tracking Training Progress 📈

### Training Broadcast
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py) on your environment like this:
```python
env = StreamWrapper(
            env, 
            stream_metadata = { # All of this is part is optional
                "user": "super-cool-user", # choose your own username
                "env_id": id, # environment identifier
                "color": "#0033ff", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
```

Hack on the broadcast viewing client or set up your own local stream with this repo:  
  
https://github.com/pwhiddy/pokerl-map-viz/

### Local Metrics
The current state of each game is rendered to images in the session directory.   
You can track the progress in tensorboard by moving into the session directory and running:  
```tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view metrics.  
To enable wandb integration, change `use_wandb_logging` in the training script to `True`.

## Static Visualization 🐜
Map visualization code can be found in `visualization/` directory.

## Follow up work  
 
Check out our follow up projects & papers!  
  
### [Pokemon Red via Reinforcement Learning 🔗](https://arxiv.org/abs/2502.19920)
```  
  @misc{pleines2025pokemon,
    title={Pokemon Red via Reinforcement Learning},
    author={Marco Pleines and Daniel Addis and David Rubinstein and Frank Zimmer and Mike Preuss and Peter Whidden},
    year={2025},
    eprint={2502.19920},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
  }
```
### [Pokemon RL Edition 🔗](https://drubinstein.github.io/pokerl/)
### [PokeGym 🔗](https://github.com/PufferAI/pokegym)

## Supporting Libraries
Check out these awesome projects!
### [PyBoy](https://github.com/Baekalfen/PyBoy)
<a href="https://github.com/Baekalfen/PyBoy">
  <img src="/assets/pyboy.svg" height="64">
</a>

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
<a href="https://github.com/DLR-RM/stable-baselines3">
  <img src="/assets/sblogo.png" height="64">
</a>
