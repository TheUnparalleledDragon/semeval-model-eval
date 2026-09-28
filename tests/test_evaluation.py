import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import compare
from evaluate_run import evaluate_run, parse_visual
from score_saved import run_scoring


class EvaluationTests(unittest.TestCase):
    def test_saved_scoring_marks_native_child_failure_and_keeps_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            predictions = root / "predictions.jsonl"
            predictions.write_text('{"prediction":"saved"}\n', encoding="utf-8")
            (root / "status.json").write_text(json.dumps({"run_id": "x", "counts_total": {"ok": 1}}))
            with patch("score_saved.subprocess.run", return_value=Mock(returncode=-10)) as child:
                with self.assertRaisesRegex(RuntimeError, "signal 10"):
                    run_scoring(predictions)
            self.assertIn("--model-type", child.call_args.args[0])
            self.assertEqual(predictions.read_text(encoding="utf-8"), '{"prediction":"saved"}\n')
            self.assertEqual(json.loads((root / "status.json").read_text())["state"], "evaluation_error")

    def test_separate_qa_scoring_and_visual_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({
                "run_id": "x", "dataset_revision": "snapshot", "tracks": ["qa_mena_en"]
            }), encoding="utf-8")
            base = {"track": "qa_mena_en", "split": "dev", "variant": "original",
                    "id": "image-a", "image": "images/image-a.jpg", "latency_seconds": 2.0}
            rows = [
                {**base, "task": "qa", "status": "ok", "prediction": "Lantern",
                 "reference": "lantern"},
                {**base, "task": "visual", "status": "ok", "prediction": json.dumps({
                    "visible_details": ["two hanging lanterns"], "text_in_image": [],
                    "location_clues": [], "uncertain_inferences": []})},
            ]
            (root / "predictions.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            calls = []

            def fake_score(candidates, references, **options):
                calls.append((candidates, references, options))
                return [0.7], [0.8], [0.75]

            result = evaluate_run(root, score_function=fake_score)
            self.assertEqual(calls[0][0], ["Lantern"])
            self.assertEqual(calls[0][1], ["lantern"])
            self.assertEqual(result["qa_by_track"]["qa_mena_en"]["bertscore_f1_mean"], 0.75)
            self.assertEqual(result["visual_by_track"]["qa_mena_en"]["valid_json_rate"], 1.0)
            self.assertEqual(len((root / "qa_scores.jsonl").read_text().splitlines()), 1)
            self.assertTrue((root / "visual_summary.json").exists())
            with (root / "visual_review.csv").open(newline="") as source:
                review = next(csv.DictReader(source))
            self.assertEqual(review["prediction"], rows[1]["prediction"])
            review["detail_accuracy_0_3"] = "2"
            with (root / "visual_review.csv").open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=list(review))
                writer.writeheader()
                writer.writerow(review)
            repeated = evaluate_run(root, score_function=fake_score)
            self.assertEqual(repeated["visual_manual_review"]["mean_detail_accuracy_0_3"], 2)

    def test_visual_parser_requires_expected_lists(self):
        self.assertIsNone(parse_visual("not JSON"))
        self.assertIsNone(parse_visual('{"visible_details": []}'))
        self.assertIsNotNone(parse_visual(json.dumps({
            "visible_details": [], "text_in_image": [], "location_clues": [],
            "uncertain_inferences": [{"possible_identity": "souk",
                                      "visible_evidence": "market stalls"}]})))

    def test_comparison_uses_shared_qa_ids_and_keeps_visual_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dirs = [root / "one", root / "two"]
            visual = json.dumps({"visible_details": ["flag"], "text_in_image": [],
                                 "location_clues": ["red flag"], "uncertain_inferences": []})
            for index, path in enumerate(dirs):
                path.mkdir()
                (path / "manifest.json").write_text(json.dumps({
                    "schema_version": 2, "model_key": f"model-{index}",
                    "dataset_revision": "same", "max_rows_per_track": 2,
                    "prompt_suffix": "same", "visual_prompt": "same",
                    "max_new_tokens": 128, "visual_max_new_tokens": 512,
                }), encoding="utf-8")
                base = {"track": "qa_mena_en", "split": "dev", "variant": "original",
                        "id": "shared", "status": "ok", "latency_seconds": 1.0}
                records = [{**base, "task": "qa", "prediction": "a flag"},
                           {**base, "task": "visual", "prediction": visual}]
                (path / "predictions.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
                (path / "qa_scores.jsonl").write_text(json.dumps({
                    "track": "qa_mena_en", "id": "shared", "bertscore_f1": 0.8 + index * 0.1,
                    "exact_match": False}) + "\n", encoding="utf-8")
                (path / "evaluation.json").write_text(json.dumps({
                    "metric": {"model_type": "xlm-roberta-large", "idf": False,
                               "rescale_with_baseline": False, "use_fast_tokenizer": False}
                }), encoding="utf-8")
                (path / "visual_review.csv").write_text(
                    "track,id,detail_accuracy_0_3,cultural_clue_accuracy_0_3,unsupported_claims_0_3\n"
                    "qa_mena_en,shared,2,3,0\n", encoding="utf-8")
            output = root / "comparison.csv"
            with patch("sys.argv", ["compare.py", *(str(p) for p in dirs),
                                    "--output", str(output)]):
                compare.main()
            with output.open(newline="") as source:
                results = list(csv.DictReader(source))
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["qa_shared_scored_rows"], "1")
            self.assertEqual(results[0]["visual_valid_json_rate"], "1")
            self.assertEqual(results[1]["qa_bertscore_f1_shared_mean"], "0.9")


if __name__ == "__main__":
    unittest.main()
