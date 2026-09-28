"""Score saved predictions without running the vision model again.

Usage: python score_saved.py score PATH_TO_RUN_OR_PREDICTIONS_JSONL
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from evaluate_run import BERTSCORE_MODEL
from run import dump_json, utc_now


def run_scoring(path, batch_size=1, device="cpu", model_type=BERTSCORE_MODEL):
    path = Path(path).expanduser().resolve()
    run_dir = path.parent if path.is_file() and path.name == "predictions.jsonl" else path
    for name in ("manifest.json", "predictions.jsonl"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Expected {run_dir / name}; provide a run directory or its predictions.jsonl")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    status_path = run_dir / "status.json"
    prior = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    dump_json(status_path, {**prior, "state": "evaluating", "updated_utc": utc_now()})
    command = [sys.executable, str(Path(__file__).with_name("evaluate_run.py")),
               str(run_dir), "--batch-size", str(batch_size), "--model-type", model_type]
    if device:
        command += ["--device", device]
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    print(f"Scoring {run_dir / 'predictions.jsonl'} with {model_type} ({device or 'auto'}); this does not run inference.", flush=True)
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode:
        explanation = (f"scorer terminated by signal {-completed.returncode}"
                       if completed.returncode < 0 else f"scorer exited with code {completed.returncode}")
        dump_json(status_path, {**prior, "state": "evaluation_error", "error": explanation,
                                "updated_utc": utc_now()})
        raise RuntimeError(f"{explanation}. Predictions remain at {run_dir / 'predictions.jsonl'}")
    return run_dir / "evaluation.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score",))
    parser.add_argument("path", help="Run directory or its predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-type", default=BERTSCORE_MODEL)
    args = parser.parse_args()
    print(f"Saved {run_scoring(args.path, args.batch_size, args.device, args.model_type)}", flush=True)


if __name__ == "__main__":
    main()
