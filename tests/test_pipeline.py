import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from importlib import metadata

from PIL import Image

import run
from models import clean_answer, validate_quantization_dependency, validate_transformers_dependency


class PipelineContractTests(unittest.TestCase):
    def test_transformers_preflight_rejects_old_and_missing_gemma_support(self):
        old = SimpleNamespace(__version__="4.57.0", __file__="/test/transformers/__init__.py")
        with patch.dict("sys.modules", {"transformers": old}):
            with self.assertRaisesRegex(RuntimeError, "requires Transformers>=5.13.0"):
                validate_transformers_dependency("gemma_4_12b")
        from unittest.mock import Mock
        current = SimpleNamespace(__version__="5.13.0", __file__="/test/transformers/__init__.py",
                                  AutoConfig=Mock())
        current.AutoConfig.for_model.side_effect = ValueError("unknown architecture")
        with patch.dict("sys.modules", {"transformers": current}):
            with self.assertRaisesRegex(RuntimeError, "gemma4_unified architecture"):
                validate_transformers_dependency("gemma_4_12b")
        current.AutoConfig.for_model.side_effect = None
        with patch.dict("sys.modules", {"transformers": current}):
            validate_transformers_dependency("gemma_4_12b")

    def test_quantization_dependency_rejects_missing_and_old_versions(self):
        with patch("models.metadata.version", return_value="0.45.5"):
            with self.assertRaisesRegex(RuntimeError, "bitsandbytes>=0.46.1"):
                validate_quantization_dependency()
        with patch("models.metadata.version", side_effect=metadata.PackageNotFoundError("bitsandbytes")):
            with self.assertRaisesRegex(RuntimeError, "installed: missing"):
                validate_quantization_dependency()

    def test_quantization_dependency_accepts_required_version(self):
        with patch("models.metadata.version", return_value="0.46.1"):
            validate_quantization_dependency()

    def test_discovers_only_text_qa_tracks(self):
        files = [SimpleNamespace(rfilename=name) for name in (
            "qa/mena/train_en.parquet", "qa/mena/dev_en.parquet",
            "qa/mena/train_arz.parquet", "sqa/mena/train_en.parquet",
            "demo/mena_en_qa_train.parquet",
        )]
        self.assertEqual(run.discover_tracks(files), ["qa_mena_arz", "qa_mena_en"])

    def test_variant_keeps_image_size_and_changes_pixels(self):
        image = Image.new("RGB", (40, 40))
        for x in range(40):
            for y in range(40):
                image.putpixel((x, y), (x * 6, y * 6, (x + y) * 3))
        variant = run.image_for_variant(image, "jpeg_85")
        self.assertEqual(variant.size, image.size)
        self.assertNotEqual(variant.tobytes(), image.tobytes())

    def test_resume_skips_only_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = [
                {"track": "qa_mena_en", "split": "dev", "id": "a", "variant": "original", "status": "ok", "prediction": "answer"},
                {"track": "qa_mena_en", "split": "dev", "id": "b", "variant": "original", "status": "error"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            self.assertEqual(run.completed_keys(path), {("qa_mena_en", "dev", "a", "original", "qa")})
            totals, by_track = run.summarize_records(path)
            self.assertEqual(totals, {"ok": 1, "error": 1})
            self.assertEqual(by_track["qa/qa_mena_en/dev/original"], totals)

    def test_visual_resume_is_independent_of_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            base = {"track": "qa_mena_en", "split": "dev", "id": "a", "variant": "original"}
            path.write_text(json.dumps({**base, "task": "qa", "status": "ok", "prediction": "answer"}) + "\n",
                            encoding="utf-8")
            self.assertIn(run.row_key("qa_mena_en", "dev", "a", "original", "qa"),
                          run.completed_keys(path))
            self.assertNotIn(run.row_key("qa_mena_en", "dev", "a", "original", "visual"),
                             run.completed_keys(path))
            self.assertNotIn("question", run.VISUAL_PROMPT.lower().split("do not answer")[0])

    def test_generated_text_normalization(self):
        result = [{"generated_text": [{"role": "assistant", "content": [{"type": "text", "text": "أهلاً"}]}]}]
        self.assertEqual(clean_answer(result), "أهلاً")


if __name__ == "__main__":
    unittest.main()
