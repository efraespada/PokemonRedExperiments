# Prism experiment pipeline

`prism_experiment_pipeline.py` runs a Prism experiment as a gated sequence:

1. train a candidate with `baseline_fast_prism_v2.py`;
2. wait for completion or stop it after a timeout;
3. read `training_manifest.json` and select the checkpoint with the highest
   numeric step count;
4. evaluate that checkpoint stochastically;
5. classify it as `ACCEPT` or `DISCARD` using configurable success-rate gates;
6. save the complete report and command logs.

The script invokes Python subprocesses directly, without a shell. This keeps
the checkpoint hand-off deterministic and prevents a stale or incomplete run
from being accepted accidentally.

## Run with eight environments

From `v2/`, a battle-specialist continuation can be run with:

```bash
../.venv/bin/python prism_experiment_pipeline.py \
  --state memory/curriculum_seed91/000624_entry_after.state \
  --output-dir /tmp/prism-pipeline-battle \
  --resume /tmp/prism-battle-specialist-seed337/prism_32768_steps.zip \
  --num-cpu 8 \
  --total-timesteps 32768 \
  --episodes 20 \
  --evaluation-steps 256 \
  --min-victory-rate 0.25 \
  --max-defeat-rate 0
```

When `--num-cpu 8` and the default `--vec-env subproc` are used, the command
must run in an environment where subprocess creation is permitted. In a
restricted smoke test, use `--vec-env dummy --num-cpu 1`; that validates the
orchestration but is not an eight-environment performance result.

## Sequential candidates

Repeat `--state` to process candidates one after another. Each candidate gets
its own `candidate-NN` directory and is fully trained and evaluated before the
next state starts:

```bash
../.venv/bin/python prism_experiment_pipeline.py \
  --state memory/curriculum_seed91/000624_entry_after.state \
  --state memory/curriculum_seed91/000905_entry_after.state \
  --output-dir /tmp/prism-pipeline-battle-candidates \
  --resume /tmp/prism-battle-specialist-seed337/prism_32768_steps.zip \
  --num-cpu 8 --total-timesteps 32768 \
  --min-victory-rate 0.25 --max-defeat-rate 0
```

This is intentionally sequential. It avoids mixing candidate metrics and
makes a blocked state visible as a failed candidate instead of contaminating
the next experiment.

## Acceptance gates

The default gates are deliberately permissive for smoke tests. For meaningful
experiments, set the thresholds explicitly:

- `--min-victory-rate`: required `success_rates.victory`;
- `--min-story-rate`: required `success_rates.story_event`;
- `--min-party-rate`: required `success_rates.party_growth`;
- `--max-defeat-rate`: upper bound for `success_rates.battle_defeat`;
- `--timeout`: maximum seconds for each training or evaluation subprocess.

A candidate is discarded when training is not `completed`, the evaluator does
not return successfully, the JSON report is missing, or any configured gate
fails. A zero-exit pipeline means at least one candidate was accepted; a
non-zero exit means every candidate was discarded.

## Outputs

For each candidate, the output directory contains:

- `training.log` and `evaluation.log`;
- `training_manifest.json` from the trainer;
- the generated `prism_*_steps.zip` checkpoints;
- `evaluation.json` when evaluation completes;
- `training_timeout.txt` or `evaluation_timeout.txt` when a timeout occurs.

The root `pipeline_report.json` records the configuration, each candidate's
decision and reasons, and the accepted/discarded candidate labels. The report
is the source of truth for whether a run is usable.

## Interpreting improvements

An isolated battle improvement is not automatically a project-wide
improvement. Compare every candidate with the currently accepted reference and
record whether it improved, regressed, or remained inconclusive. A candidate
must also receive an integrated navigation-plus-battle evaluation before it
can replace the current hierarchical policy pair.

In particular, a state that blocks `SubprocVecEnv` is a failed execution, not a
bad policy result. Preserve its logs and checkpoint, mark it `DISCARD`, and
repair or isolate the state reset/termination behavior before retrying.
