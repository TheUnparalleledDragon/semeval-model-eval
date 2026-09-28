import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from judge_predictions import OpenRouterJudge, evaluate_saved, validate_rating, resolve_api_key, JudgeResponseError


def rating(verdict="correct"):
    return {"accuracy": 9, "faithfulness": 9, "relevance": 10, "helpfulness": 9,
            "verdict": verdict, "explanation": "The core identity matches.", "issues": []}


class JudgeTests(unittest.TestCase):
    def test_dotenv_quotes_comments_bom_and_shell_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('\ufeffexport OPENROUTER_API_KEY = " file-test-key " # comment\n', encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True), patch("judge_predictions.__file__", str(path.with_name("judge_predictions.py"))):
                self.assertEqual(resolve_api_key(), ("file-test-key", str(path.resolve())))
                with patch.dict("os.environ", {"OPENROUTER_API_KEY": "shell-test-key"}):
                    self.assertEqual(resolve_api_key()[0], "shell-test-key")
                    self.assertEqual(resolve_api_key(path)[0], "file-test-key")
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "Bearer test-key"}, clear=True):
                self.assertEqual(resolve_api_key()[0], "test-key")
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "bad key"}, clear=True):
                with self.assertRaisesRegex(ValueError, "whitespace"):
                    resolve_api_key()

    def test_auth_check_uses_get_only_and_redacts_key_in_errors(self):
        session = Mock()
        success = Mock(status_code=200)
        success.json.return_value = {"data": {"limit": None}}
        session.get.return_value = success
        judge = OpenRouterJudge("secret-test-key", "test-model", session=session)
        self.assertTrue(judge.check_auth())
        session.post.assert_not_called()
        self.assertEqual(session.get.call_args.kwargs["headers"]["Authorization"], "Bearer secret-test-key")
        rejected = Mock(status_code=401)
        rejected.json.return_value = {"error": {"message": "Rejected secret-test-key sk-or-v1-otherkey"}}
        session.get.return_value = rejected
        with self.assertRaises(RuntimeError) as context:
            judge.check_auth()
        self.assertNotIn("secret-test-key", str(context.exception))
        self.assertNotIn("sk-or-v1-otherkey", str(context.exception))

    def fixture(self, root):
        rows = [{"track": track, "split": "dev", "id": str(i), "variant": "original",
                 "task": "qa", "status": "ok", "question": "Which landmark?",
                 "reference": "Eiffel Tower", "prediction": "The Eiffel Tower",
                 "latency_seconds": 1} for i, track in enumerate(("qa_mena_en", "qa_mena_en", "qa_mena_msa"))]
        rows += [{**rows[0], "task": "visual", "prediction": "Visual details"}]
        path = root / "predictions.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        scores = [{**r, "bertscore_precision": .8, "bertscore_recall": .9,
                   "bertscore_f1": .85, "exact_match": False} for r in rows if r["task"] == "qa"]
        (root / "qa_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
        return path

    def test_wrong_core_cannot_receive_high_final_rating(self):
        wrong = {**rating(), "verdict": "incorrect", "accuracy": 2, "faithfulness": 3,
                 "helpfulness": 2}
        self.assertLessEqual(validate_rating(wrong)["final_rating"], 3)
        adjusted = validate_rating({**wrong, "accuracy": 9})
        self.assertEqual(adjusted["accuracy"], 3)
        self.assertEqual(adjusted["raw_judge_scores"]["accuracy"], 9)
        with self.assertRaises(ValueError):
            validate_rating({**rating(), "accuracy": True})
        self.assertIsNone(validate_rating(rating("unjudgeable"))["final_rating"])

    def test_partial_verdict_seven_normalizes_without_increasing_score(self):
        value = {**rating(), "verdict": "partially_correct", "accuracy": 7}
        result = validate_rating(value)
        self.assertEqual(result["accuracy"], 6)
        self.assertEqual(result["verdict"], "partially_correct")
        self.assertLessEqual(result["final_rating"], 6)
        self.assertEqual(result["raw_judge_scores"]["accuracy"], 7)
        downgraded = validate_rating({**rating(), "accuracy": 2})
        self.assertEqual(downgraded["verdict"], "incorrect")
        self.assertLessEqual(downgraded["final_rating"], 3)

    def test_two_outputs_english_only_resume_and_all_track_averages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.fixture(root)
            judge = Mock()
            judge.usage = {}
            judge.rate.return_value = (validate_rating(rating()), {"usage": {"cost": .001}})
            summary = evaluate_saved(path, judge=judge)
            self.assertEqual(judge.rate.call_count, 2)
            self.assertEqual(summary["bertscore"]["all_tracks_micro"]["scored_rows"], 3)
            self.assertEqual(summary["bertscore"]["all_tracks_macro_f1"], .85)
            self.assertEqual(summary["llm_english"]["overall"]["accuracy_mean"], 9)
            outputs = list(root.glob("llm_english_scores.jsonl")) + list(root.glob("final_summary.json"))
            self.assertEqual(len(outputs), 2)
            rows = [json.loads(line) for line in outputs[0].read_text().splitlines()]
            self.assertTrue(all(r["track"].endswith("_en") and r["task"] == "qa" for r in rows))
            evaluate_saved(path, judge=judge)
            self.assertEqual(judge.rate.call_count, 2)
            with self.assertRaisesRegex(ValueError, "another model/rubric"):
                evaluate_saved(path, model="another-judge", judge=judge)

    def test_saved_scores_must_match_source_and_dry_run_has_no_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.fixture(root)
            plan = evaluate_saved(path, dry_run=True)
            self.assertEqual(plan["pending"], 2)
            self.assertFalse((root / "final_summary.json").exists())
            scores_path = root / "qa_scores.jsonl"
            scores_path.write_text(scores_path.read_text().replace("The Eiffel Tower", "Wrong tower"))
            with self.assertRaisesRegex(ValueError, "do not match"):
                evaluate_saved(path, dry_run=True)

    def test_openrouter_payload_labels_and_structured_output(self):
        session = Mock()
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(rating())}}], "usage": {"prompt_tokens": 100, "cost": .001}}
        session.post.return_value = response
        judge = OpenRouterJudge("test-key", "test-model", session=session)
        scores, metadata = judge.rate({"question": "Q", "reference": "Gold", "prediction": "Candidate"})
        payload = session.post.call_args.kwargs["json"]
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user["ORIGINAL_REFERENCE_ANSWER"], "Gold")
        self.assertEqual(user["PREDICTED_MODEL_ANSWER"], "Candidate")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["provider"]["require_parameters"])
        self.assertEqual(judge.usage["cost"], .001)
        self.assertEqual(scores["accuracy"], 9)

    def test_retry_invalid_json_and_stop_on_auth_error(self):
        session = Mock()
        invalid = Mock(status_code=200)
        invalid.json.return_value = {"choices": [{"message": {"content": "not JSON"}}]}
        valid = Mock(status_code=200)
        valid.json.return_value = {"choices": [{"message": {"content": json.dumps(rating())}}]}
        session.post.side_effect = [invalid, valid]
        judge = OpenRouterJudge("test", "test", session=session)
        row = {"question": "Q", "reference": "A", "prediction": "A"}
        with patch("judge_predictions.time.sleep"):
            judge.rate(row)
        self.assertEqual(session.post.call_count, 2)
        session.post.side_effect = None
        session.post.return_value = Mock(status_code=401)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            judge.rate(row)

    def test_verified_key_user_lookup_failure_retries_but_remains_bounded(self):
        session = Mock()
        rejected = Mock(status_code=401)
        rejected.json.return_value = {"error": {"message": "User not found."}}
        session.post.return_value = rejected
        judge = OpenRouterJudge("test", "test", session=session)
        judge.auth_verified = True
        with patch("judge_predictions.time.sleep"), self.assertRaisesRegex(RuntimeError, "accepted this key"):
            judge.rate({"question": "Q", "reference": "A", "prediction": "A"})
        self.assertEqual(session.post.call_count, 3)

    def test_failure_saves_progress_summary_and_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.fixture(root)
            judge = Mock()
            judge.usage = {}
            result = (validate_rating(rating()), {})
            judge.rate.side_effect = [result, RuntimeError("OpenRouter HTTP 402")]
            with self.assertRaisesRegex(RuntimeError, "resume"):
                evaluate_saved(path, judge=judge)
            summary = json.loads((root / "final_summary.json").read_text())
            self.assertEqual(summary["state"], "judge_error")
            self.assertEqual(summary["selection"]["pending_judgments"], 1)
            judge.rate.side_effect = None
            judge.rate.return_value = result
            summary = evaluate_saved(path, judge=judge)
            self.assertEqual(summary["llm_english"]["overall"]["successful_judgments"], 2)
            self.assertEqual(judge.rate.call_count, 3)

    def test_isolated_invalid_response_does_not_block_other_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.fixture(root)
            judge = Mock()
            judge.usage = {}
            judge.rate.side_effect = [JudgeResponseError("Invalid JSON", [{"error": "Invalid JSON"}]),
                                      (validate_rating(rating()), {})]
            with self.assertRaisesRegex(RuntimeError, "finished with failed judgments"):
                evaluate_saved(path, judge=judge)
            self.assertEqual(judge.rate.call_count, 2)
            summary = json.loads((root / "final_summary.json").read_text())
            self.assertEqual(summary["state"], "finished_with_judge_errors")
            self.assertEqual(summary["selection"]["pending_judgments"], 1)

    def test_truncated_response_retry_has_more_space_and_validation_feedback(self):
        session = Mock()
        truncated = Mock(status_code=200)
        truncated.json.return_value = {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]}
        valid = Mock(status_code=200)
        valid.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(rating())}}]}
        session.post.side_effect = [truncated, valid]
        judge = OpenRouterJudge("test", "test", session=session)
        with patch("judge_predictions.time.sleep"):
            result, metadata = judge.rate({"question": "Q", "reference": "A", "prediction": "A"})
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 1200)
        self.assertIn("truncated", payload["messages"][-1]["content"])
        self.assertEqual(metadata["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
