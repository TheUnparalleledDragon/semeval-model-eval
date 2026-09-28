"""Compare completed two-task runs on shared successful original-dev examples.

Usage: python compare.py local/runs/<run-a> local/runs/<run-b> --output comparison.csv
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from evaluate_run import parse_visual


def mean(values):
    return round(statistics.mean(values), 6) if values else None


def load_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version", 0) < 2:
        raise ValueError(f"{path} is an older QA-only run; use new two-task runs for comparison.")
    latest = {}
    with (path / "predictions.jsonl").open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["split"] == "dev" and row["variant"] == "original":
                latest[(row["track"], row["id"], row["task"])] = row
    scores = {}
    with (path / "qa_scores.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            scores[(row["track"], row["id"])] = row
    reviews = {}
    with (path / "visual_review.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            reviews[(row["track"], row["id"])] = row
    return manifest, latest, scores, reviews, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Completed run directories")
    parser.add_argument("--output", default="comparison.csv")
    args = parser.parse_args()
    runs = [load_run(path) for path in args.runs]
    signatures = {(m["dataset_revision"], m["max_rows_per_track"], m["prompt_suffix"],
                   m["visual_prompt"], m["max_new_tokens"], m["visual_max_new_tokens"])
                  for m, _, _, _, _ in runs}
    if len(signatures) != 1:
        raise ValueError("Dataset, sampling, prompts, or generation limits differ between runs.")
    metric_signatures = {(e["metric"]["model_type"], e["metric"].get("idf"),
                          e["metric"].get("rescale_with_baseline"),
                          e["metric"].get("use_fast_tokenizer"))
                         for _, _, _, _, e in runs}
    if len(metric_signatures) != 1:
        raise ValueError("BERTScore settings differ between runs.")
    names = [m["model_key"] for m, _, _, _, _ in runs]
    if len(names) != len(set(names)):
        raise ValueError("Supply only one run per model key.")

    tracks = sorted(set.intersection(*[
        {key[0] for key in latest} for _, latest, _, _, _ in runs]))
    output_rows = []
    for track in tracks:
        common = {}
        for task in ("qa", "visual"):
            common[task] = sorted(set.intersection(*[
                {key[1] for key, row in latest.items() if key[0] == track and key[2] == task
                 and row["status"] == "ok"}
                for _, latest, _, _, _ in runs]))
        review_fields = ("detail_accuracy_0_3", "cultural_clue_accuracy_0_3",
                         "unsupported_claims_0_3")
        shared_review_ids = {
            field: [row_id for row_id in common["visual"] if all(
                reviews.get((track, row_id), {}).get(field, "").strip()
                for _, _, _, reviews, _ in runs)]
            for field in review_fields
        }
        for manifest, latest, scores, reviews, _ in runs:
            qa_ids = common["qa"]
            visual_ids = common["visual"]
            qa_rows = [latest[(track, row_id, "qa")] for row_id in qa_ids]
            visual_rows = [latest[(track, row_id, "visual")] for row_id in visual_ids]
            visual_parsed = [value for row in visual_rows
                             if (value := parse_visual(row["prediction"])) is not None]
            qa_scores = [scores[(track, row_id)] for row_id in qa_ids]
            def review_mean(field):
                values = [float(reviews[(track, row_id)][field])
                          for row_id in shared_review_ids[field]]
                return mean(values)

            output_rows.append({
                "model_key": manifest["model_key"], "track": track,
                "qa_successful_dev_original": sum(
                    row["status"] == "ok" for key, row in latest.items()
                    if key[0] == track and key[2] == "qa"),
                "qa_shared_scored_rows": len(qa_ids),
                "qa_bertscore_f1_shared_mean": mean([r["bertscore_f1"] for r in qa_scores]),
                "qa_exact_match_shared": mean([int(r["exact_match"]) for r in qa_scores]),
                "qa_mean_latency_seconds": mean([r["latency_seconds"] for r in qa_rows]),
                "visual_successful_dev_original": sum(
                    row["status"] == "ok" for key, row in latest.items()
                    if key[0] == track and key[2] == "visual"),
                "visual_shared_rows": len(visual_ids),
                "visual_valid_json_rate": mean([int(parse_visual(r["prediction"]) is not None)
                                                 for r in visual_rows]),
                "visual_mean_visible_details_count": mean([
                    len(r["visible_details"]) for r in visual_parsed]),
                "visual_mean_location_clues_count": mean([
                    len(r["location_clues"]) for r in visual_parsed]),
                "visual_mean_latency_seconds": mean([r["latency_seconds"] for r in visual_rows]),
                "visual_shared_reviewed_rows": len(shared_review_ids["detail_accuracy_0_3"]),
                "visual_review_detail_accuracy_0_3": review_mean("detail_accuracy_0_3"),
                "visual_review_cultural_clue_accuracy_0_3": review_mean("cultural_clue_accuracy_0_3"),
                "visual_review_unsupported_claims_0_3": review_mean("unsupported_claims_0_3"),
            })
    if not output_rows:
        raise ValueError("No shared original-dev tracks found.")
    with Path(args.output).open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved {args.output} ({len(output_rows)} model/track results)")


if __name__ == "__main__":
    main()
