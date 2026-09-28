"""Score a completed two-task run. Run directly to retry evaluation without inference.

The organizers name BERTScore F1 but have not published scorer settings in the
current dataset repository. The multilingual BERT score here is a recorded local
estimate, not a claim of leaderboard equivalence.
"""

import argparse
import csv
import importlib.metadata
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from models import has_answer_text


VISUAL_FIELDS = ("visible_details", "text_in_image", "location_clues", "uncertain_inferences")
BERTSCORE_MODEL = "bert-base-multilingual-cased"


def latest_records(path):
    latest = {}
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row["track"], row["split"], row["id"], row["variant"], row.get("task", "qa"))
            if row.get("status") == "ok" and not has_answer_text(row.get("prediction")):
                row = {**row, "status": "error", "error": {
                    "type": "InvalidPrediction", "message": "Empty or special-token-only prediction"}}
            latest[key] = row
    return latest


def parse_visual(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or any(field not in value for field in VISUAL_FIELDS):
        return None
    if any(not isinstance(value[field], list) for field in VISUAL_FIELDS):
        return None
    if any(not isinstance(item, str) for field in VISUAL_FIELDS[:3] for item in value[field]):
        return None
    for item in value["uncertain_inferences"]:
        if isinstance(item, str):
            continue
        if not isinstance(item, dict) or not isinstance(item.get("possible_identity"), str) or \
                not isinstance(item.get("visible_evidence"), str):
            return None
    return value


def mean(values):
    return round(statistics.mean(values), 6) if values else None


def summarize_group(rows, task):
    ok = [r for r in rows if r["status"] == "ok"]
    result = {"attempted": len(rows), "successful": len(ok),
              "errors": len(rows) - len(ok),
              "success_rate": round(len(ok) / len(rows), 6) if rows else None,
              "mean_latency_seconds": mean([r["latency_seconds"] for r in rows]),
              "median_latency_seconds": round(statistics.median(
                  r["latency_seconds"] for r in rows), 6) if rows else None}
    if task == "visual":
        parsed = [value for row in ok if (value := parse_visual(row["prediction"])) is not None]
        result.update({"valid_json": len(parsed),
                       "valid_json_rate": round(len(parsed) / len(ok), 6) if ok else None,
                       "mean_output_chars": mean([len(r["prediction"]) for r in ok])})
        for field in VISUAL_FIELDS:
            result[f"mean_{field}_count"] = mean([len(value[field]) for value in parsed])
    return result


def existing_reviews(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as source:
        return {(r["track"], r["id"]): r for r in csv.DictReader(source)}


def save_reviews(path, visual_rows):
    """Preserve human ratings when regenerating the review sheet."""
    prior = existing_reviews(path)
    headers = ["track", "id", "image", "prediction", "detail_accuracy_0_3",
               "cultural_clue_accuracy_0_3", "unsupported_claims_0_3", "reviewer_notes"]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=headers)
        writer.writeheader()
        for row in visual_rows:
            old = prior.get((row["track"], row["id"]), {})
            if old.get("prediction") != row["prediction"]:
                old = {}
            writer.writerow({"track": row["track"], "id": row["id"],
                             "image": row["image"], "prediction": row["prediction"],
                             **{key: old.get(key, "") for key in headers[4:]}})


def manual_review_stats(path):
    reviews = existing_reviews(path)
    dimensions = ("detail_accuracy_0_3", "cultural_clue_accuracy_0_3", "unsupported_claims_0_3")
    values = defaultdict(list)
    for row in reviews.values():
        for dimension in dimensions:
            raw = row.get(dimension, "").strip()
            if not raw:
                continue
            score = float(raw)
            if not 0 <= score <= 3:
                raise ValueError(f"{dimension} must be between 0 and 3: {raw}")
            values[dimension].append(score)
    return {"reviewed_rows": sum(bool(row.get("detail_accuracy_0_3", "").strip())
                                 for row in reviews.values()),
            **{f"mean_{key}": mean(values[key]) for key in dimensions}}


def bertscore_predictions(candidates, references, *, model_type, batch_size, device,
                          verbose, idf, rescale_with_baseline, use_fast_tokenizer):
    """Run BERTScore with private CPU weights on macOS.

    On this Mac, Accelerate raises SIGBUS when multiplying directly from the
    memory-mapped safetensors checkpoint. Cloning the loaded parameter tensors
    into ordinary RAM avoids that native crash without changing their values.
    """
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type=model_type, batch_size=batch_size, device=device,
                        idf=idf, rescale_with_baseline=rescale_with_baseline,
                        use_fast_tokenizer=use_fast_tokenizer)
    if sys.platform == "darwin" and scorer.device == "cpu":
        for parameter in scorer._model.parameters():
            parameter.data = parameter.data.clone()
    return scorer.score(candidates, references, batch_size=batch_size, verbose=verbose)


def evaluate_run(run_dir, batch_size=8, device=None, score_function=None,
                 model_type=BERTSCORE_MODEL):
    """Write per-row QA metrics and per-track summaries after inference."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    latest = latest_records(run_dir / "predictions.jsonl")
    grouped = defaultdict(list)
    for row in latest.values():
        grouped[(row.get("task", "qa"), row["track"], row["split"], row["variant"])].append(row)

    visual_review_rows = sorted((r for r in latest.values() if r.get("task") == "visual"
                                 and r["split"] == "dev" and r["variant"] == "original"
                                 and r["status"] == "ok"), key=lambda r: (r["track"], r["id"]))
    review_path = run_dir / "visual_review.csv"
    save_reviews(review_path, visual_review_rows)
    visual_by_track = {}
    for (task, track, split, variant), rows in sorted(grouped.items()):
        if task == "visual" and split == "dev" and variant == "original":
            visual_by_track[track] = summarize_group(rows, "visual")
    for track in manifest["tracks"]:
        visual_by_track.setdefault(track, summarize_group([], "visual"))
    from run import dump_json
    dump_json(run_dir / "visual_summary.json", {"run_id": manifest["run_id"],
                                               "visual_by_track": visual_by_track,
                                               "manual_review": manual_review_stats(review_path)})

    qa = sorted((row for row in latest.values() if row.get("task", "qa") == "qa"
                 and row["split"] == "dev" and row["variant"] == "original"
                 and row["status"] == "ok" and row.get("reference") is not None),
                key=lambda r: (r["track"], r["id"]))
    if qa:
        if score_function is None:
            score_function = bertscore_predictions
        candidates = [r["prediction"] for r in qa]
        references = [r["reference"] for r in qa]
        precision, recall, f1 = score_function(
            candidates, references, model_type=model_type,
            batch_size=batch_size, device=device, verbose=False,
            idf=False, rescale_with_baseline=False, use_fast_tokenizer=False)
        qa_scores = [{"track": row["track"], "split": "dev", "id": row["id"],
                      "variant": "original", "prediction": row["prediction"],
                      "reference": row["reference"],
                      "exact_match": row["prediction"].strip().casefold() == row["reference"].strip().casefold(),
                      "bertscore_precision": float(precision[i]),
                      "bertscore_recall": float(recall[i]),
                      "bertscore_f1": float(f1[i])} for i, row in enumerate(qa)]
    else:
        qa_scores = []

    with (run_dir / "qa_scores.jsonl").open("w", encoding="utf-8") as target:
        for row in qa_scores:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    score_groups = defaultdict(list)
    for item in qa_scores:
        score_groups[item["track"]].append(item)
    qa_by_track = {}
    for track, rows in sorted(score_groups.items()):
        qa_by_track[track] = {"scored_rows": len(rows),
                              "bertscore_precision_mean": mean([r["bertscore_precision"] for r in rows]),
                              "bertscore_recall_mean": mean([r["bertscore_recall"] for r in rows]),
                              "bertscore_f1_mean": mean([r["bertscore_f1"] for r in rows]),
                              "bertscore_f1_median": round(statistics.median(
                                  r["bertscore_f1"] for r in rows), 6),
                              "exact_match_rate": mean([int(r["exact_match"]) for r in rows])}
    for track in manifest["tracks"]:
        qa_by_track.setdefault(track, {"scored_rows": 0, "bertscore_f1_mean": None})

    result = {"run_id": manifest["run_id"], "dataset_revision": manifest["dataset_revision"],
              "metric": {"name": "BERTScore F1 local estimate", "model_type": model_type,
                         "official_scorer_settings_published": False,
                         "bert_score_package_version": importlib.metadata.version("bert-score")
                         if score_function is bertscore_predictions or
                         (score_function is not None and score_function.__module__.startswith("bert_score"))
                         else None,
                         "idf": False, "rescale_with_baseline": False,
                         "use_fast_tokenizer": False,
                         "batch_size": batch_size, "device": device,
                         "scored_split": "dev", "scored_variant": "original"},
              "qa_by_track": qa_by_track,
              "qa_macro_f1": mean([v["bertscore_f1_mean"] for v in qa_by_track.values()
                                   if v["bertscore_f1_mean"] is not None]),
              "qa_micro_f1": mean([r["bertscore_f1"] for r in qa_scores]),
              "visual_by_track": visual_by_track,
              "visual_manual_review": manual_review_stats(review_path),
              "all_groups": {"/".join(key): summarize_group(rows, key[0])
                             for key, rows in sorted(grouped.items())}}
    dump_json(run_dir / "evaluation.json", result)
    return result


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-type", default=BERTSCORE_MODEL)
    args = parser.parse_args()
    result = evaluate_run(args.run_dir, args.batch_size, args.device,
                          model_type=args.model_type)
    from run import dump_json, summarize_records, utc_now
    summary_path = Path(args.run_dir) / "summary.json"
    if summary_path.exists():
        counts = json.loads(summary_path.read_text(encoding="utf-8"))["counts_total"]
        dump_json(Path(args.run_dir) / "status.json",
                  {"run_id": result["run_id"],
                   "state": "finished_with_errors" if counts.get("error", 0) else "finished",
                   "counts_total": counts, "updated_utc": utc_now()})
    else:
        counts, _ = summarize_records(Path(args.run_dir) / "predictions.jsonl")
        dump_json(Path(args.run_dir) / "status.json",
                  {"run_id": result["run_id"], "state": "scored_partial",
                   "counts_total": counts, "updated_utc": utc_now()})
    print(f"Saved {Path(args.run_dir) / 'evaluation.json'}; QA macro F1: {result['qa_macro_f1']}")


if __name__ == "__main__":
    main()
