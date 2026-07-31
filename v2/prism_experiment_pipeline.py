"""Run Prism training/evaluation experiments sequentially and gate results.

The trainer reads a resume checkpoint from stdin, so this module invokes it
without a shell, waits for completion, then evaluates the newest checkpoint.
Every candidate is classified as ACCEPT or DISCARD from its manifest and
evaluation success rates.  A failed, timed-out, or incomplete run can never
be accepted accidentally.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


CHECKPOINT_RE = re.compile(r"prism_(\d+)_steps\.zip$")


def checkpoint_step(path):
    match = CHECKPOINT_RE.search(Path(path).name)
    return int(match.group(1)) if match else -1


def newest_checkpoint(directory):
    checkpoints = list(Path(directory).glob("prism_*_steps.zip"))
    return max(checkpoints, key=checkpoint_step, default=None)


def classify_result(
    manifest,
    evaluation,
    *,
    min_victory_rate=0.0,
    min_story_rate=0.0,
    min_party_rate=0.0,
    max_defeat_rate=0.0,
):
    reasons = []
    if manifest.get("status") != "completed":
        reasons.append(f"training status is {manifest.get('status', 'missing')}")
    if not evaluation:
        reasons.append("evaluation report is missing")

    rates = (evaluation or {}).get("success_rates", {})
    checks = (
        ("victory", min_victory_rate, rates.get("victory", 0.0), "minimum victory rate"),
        ("story_event", min_story_rate, rates.get("story_event", 0.0), "minimum story rate"),
        ("party_growth", min_party_rate, rates.get("party_growth", 0.0), "minimum party-growth rate"),
    )
    for name, minimum, actual, label in checks:
        if actual < minimum:
            reasons.append(f"{label}: {actual:.3f} < {minimum:.3f}")

    defeat_rate = rates.get("battle_defeat", 0.0)
    if defeat_rate > max_defeat_rate:
        reasons.append(f"maximum defeat rate: {defeat_rate:.3f} > {max_defeat_rate:.3f}")

    return {
        "decision": "ACCEPT" if not reasons else "DISCARD",
        "reasons": reasons,
        "success_rates": rates,
    }


def run_command(command, *, cwd, env, stdin=None, timeout=None, log_path=None):
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if log_path is not None:
        Path(log_path).write_text(completed.stdout, encoding="utf-8")
    return completed


def run_experiment(args, state, label):
    session = args.output_dir / label
    session.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PRISM_NUM_CPU": str(args.num_cpu),
            "PRISM_VEC_ENV": args.vec_env,
            "PRISM_EP_LENGTH": str(args.episode_length),
            "PRISM_N_STEPS": str(args.rollout_steps),
            "PRISM_BATCH_SIZE": str(args.batch_size),
            "PRISM_TOTAL_TIMESTEPS": str(args.total_timesteps),
            "PRISM_SEED": str(args.seed),
            "PRISM_INIT_STATE": str(state),
            "PRISM_SESSION_DIR": str(session),
        }
    )
    if args.target_coords:
        env["PRISM_TARGET_COORDS"] = args.target_coords
        env["PRISM_TARGET_WEIGHT"] = str(args.target_weight)

    trainer = [
        sys.executable,
        str(args.trainer),
    ]
    if args.resume:
        trainer_input = f"{args.resume}\n"
    else:
        trainer_input = ""
    try:
        training = run_command(
            trainer,
            cwd=args.repo_root,
            env=env,
            stdin=trainer_input,
            timeout=args.timeout,
            log_path=session / "training.log",
        )
    except subprocess.TimeoutExpired as error:
        training = None
        (session / "training_timeout.txt").write_text(str(error), encoding="utf-8")

    manifest_path = session / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    checkpoint = newest_checkpoint(session)
    evaluation = None
    evaluation_returncode = None
    if training is not None and training.returncode == 0 and checkpoint is not None:
        report_path = session / "evaluation.json"
        evaluator = [
            sys.executable,
            str(args.evaluator),
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            str(args.episodes),
            "--steps",
            str(args.evaluation_steps),
            "--seed",
            str(args.evaluation_seed),
            "--stochastic",
            "--state",
            str(state),
            "--output",
            str(report_path),
        ]
        try:
            evaluation_process = run_command(
                evaluator,
                cwd=args.repo_root,
                env=env,
                timeout=args.timeout,
                log_path=session / "evaluation.log",
            )
            evaluation_returncode = evaluation_process.returncode
        except subprocess.TimeoutExpired as error:
            (session / "evaluation_timeout.txt").write_text(str(error), encoding="utf-8")
        if evaluation_returncode == 0 and report_path.is_file():
            evaluation = json.loads(report_path.read_text())

    decision = classify_result(
        manifest,
        evaluation,
        min_victory_rate=args.min_victory_rate,
        min_story_rate=args.min_story_rate,
        min_party_rate=args.min_party_rate,
        max_defeat_rate=args.max_defeat_rate,
    )
    return {
        "label": label,
        "state": str(state),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "training_returncode": training.returncode if training is not None else None,
        "evaluation_returncode": evaluation_returncode,
        "manifest": manifest,
        "decision": decision,
    }


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", required=True, help="Battle or curriculum state; repeat for sequential candidates.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--trainer", type=Path, default=root / "baseline_fast_prism_v2.py")
    parser.add_argument("--evaluator", type=Path, default=root / "prism_evaluate.py")
    parser.add_argument("--repo-root", type=Path, default=root.parent)
    parser.add_argument("--num-cpu", type=int, default=8)
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--total-timesteps", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--evaluation-steps", type=int, default=256)
    parser.add_argument("--evaluation-seed", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--target-coords")
    parser.add_argument("--target-weight", type=float, default=3.0)
    parser.add_argument("--min-victory-rate", type=float, default=0.0)
    parser.add_argument("--min-story-rate", type=float, default=0.0)
    parser.add_argument("--min-party-rate", type=float, default=0.0)
    parser.add_argument("--max-defeat-rate", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, raw_state in enumerate(args.state):
        state = Path(raw_state).resolve()
        if not state.is_file():
            raise FileNotFoundError(f"Prism state not found: {state}")
        label = f"candidate-{index + 1:02d}"
        results.append(run_experiment(args, state, label))

    report = {
        "schema_version": 1,
        "configuration": {
            "num_cpu": args.num_cpu,
            "vector_environment": args.vec_env,
            "total_timesteps": args.total_timesteps,
            "episodes": args.episodes,
        },
        "results": results,
        "accepted": [result["label"] for result in results if result["decision"]["decision"] == "ACCEPT"],
        "discarded": [result["label"] for result in results if result["decision"]["decision"] == "DISCARD"],
    }
    report_path = args.output_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "accepted": report["accepted"], "discarded": report["discarded"]}, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
