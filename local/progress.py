"""Show progress of a local run, including while its main process is still running.

Usage: python local/progress.py [run_directory] [--watch 15]
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def latest_run():
    runs = list((PROJECT_ROOT / "local" / "runs").glob("*/manifest.json"))
    if not runs:
        raise FileNotFoundError("No local run found. Pass a run directory explicitly.")
    return max(runs, key=lambda path: path.stat().st_mtime).parent


def expected_requests(manifest):
    limit = manifest.get("max_rows_per_track")
    if limit is None:
        return None
    tasks = manifest.get("tasks", ["qa"])
    total = 0
    for split in manifest["splits"]:
        variants = [v for v in manifest["variants"]
                    if v == "original" or split in manifest["robustness_splits"]]
        total += len(manifest["tracks"]) * limit * len(variants) * len(tasks)
    return total


def snapshot(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    records = {}
    recent_latencies = []
    path = run_dir / "predictions.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (row["track"], row["split"], row["id"], row["variant"], row.get("task", "qa"))
                records[key] = row
                if row["status"] == "ok":
                    recent_latencies.append(row["latency_seconds"])
                    if len(recent_latencies) > 50:
                        recent_latencies.pop(0)
    counts = Counter(row["status"] for row in records.values())
    done = counts["ok"]
    expected = expected_requests(manifest)
    last = max(records.values(), key=lambda row: row["completed_utc"]) if records else None
    mean_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else None
    remaining = max(0, expected - done) if expected is not None else None
    eta_hours = remaining * mean_latency / 3600 if remaining is not None and mean_latency else None
    return {"state": status["state"], "done": done, "errors": counts["error"],
            "expected": expected, "last": last, "mean_latency": mean_latency,
            "eta_hours": eta_hours}


def show(run_dir):
    value = snapshot(run_dir)
    total = f"/{value['expected']}" if value["expected"] is not None else ""
    percent = (f" ({100 * value['done'] / value['expected']:.1f}%)"
               if value["expected"] else "")
    print(f"{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}  {run_dir.name}")
    print(f"  Status: {value['state']} | successful requests: {value['done']}{total}{percent}"
          f" | errors: {value['errors']}")
    if value["last"]:
        row = value["last"]
        print(f"  Last saved: {row['track']}/{row['split']} {row['task']}"
              f" at {row['completed_utc']}")
    if value["mean_latency"]:
        print(f"  Recent mean: {value['mean_latency']:.1f} s/request", end="")
        if value["eta_hours"] is not None:
            print(f" | rough remaining time: {value['eta_hours']:.1f} hours")
        else:
            print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    args = parser.parse_args()
    if args.watch is not None and args.watch < 1:
        parser.error("--watch must be at least 1 second")
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run()
    while True:
        show(run_dir)
        if args.watch is None:
            return
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            print()
            return


if __name__ == "__main__":
    main()
