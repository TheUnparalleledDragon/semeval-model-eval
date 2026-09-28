"""Precision regressions use real CPU tensors; loader contracts never download weights."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

import models
import run


class ModelTests(unittest.TestCase):
    def test_native_precision_considers_every_gpu(self):
        for capabilities, expected in [([(7, 5)], torch.float16),
                                       ([(8, 0)], torch.bfloat16),
                                       ([(8, 0), (7, 5)], torch.float16)]:
            with self.subTest(capabilities=capabilities), \
                 patch("torch.cuda.device_count", return_value=len(capabilities)), \
                 patch("torch.cuda.get_device_capability", side_effect=capabilities):
                self.assertEqual(models.cuda_compute_dtype(), expected)

    def test_every_model_load_route_in_both_precisions_and_quantization_modes(self):
        for dtype in (torch.float16, torch.bfloat16):
            for four_bit in (False, True):
                for key, spec in models.MODELS.items():
                    with self.subTest(key=key, dtype=dtype, four_bit=four_bit):
                        model = Mock()
                        model.eval.return_value = model
                        auto = Mock()
                        auto.from_pretrained.return_value = model
                        tokenizer, processor = Mock(), Mock()
                        bnb = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
                        tf = SimpleNamespace(__version__="5.17.0", __file__="/test/transformers",
                            AutoConfig=Mock(), AutoModel=auto, AutoModelForMultimodalLM=auto,
                            AutoTokenizer=tokenizer, AutoProcessor=processor, BitsAndBytesConfig=bnb)
                        builder = Mock(return_value=(tokenizer, model, processor, 4096))
                        llava = {"llava": SimpleNamespace(), "llava.model": SimpleNamespace(),
                                 "llava.model.builder": SimpleNamespace(load_pretrained_model=builder)}
                        with patch.dict("sys.modules", {"transformers": tf, **llava}), \
                             patch("torch.cuda.is_available", return_value=True), \
                             patch("models.cuda_compute_dtype", return_value=dtype), \
                             patch("models.validate_quantization_dependency"), patch("builtins.print"):
                            runner = models.load_model(key, four_bit)
                        self.assertEqual(runner.dtype, dtype)
                        if spec.backend == "pangea":
                            kwargs = builder.call_args.kwargs
                            self.assertNotIn("load_4bit", kwargs)
                            self.assertEqual(kwargs["torch_dtype"], str(dtype).split(".")[-1])
                            self.assertIsInstance(runner, models.PangeaRunner)
                        else:
                            self.assertEqual(auto.from_pretrained.call_args.args[0], spec.repo)
                            kwargs = auto.from_pretrained.call_args.kwargs
                            self.assertEqual(kwargs.get("dtype", kwargs.get("torch_dtype")), dtype)
                        if spec.backend == "minicpm_o":
                            self.assertFalse(kwargs["init_audio"])
                            self.assertFalse(kwargs["init_tts"])
                        if four_bit:
                            quant = kwargs["quantization_config"]
                            self.assertEqual(quant.bnb_4bit_compute_dtype, dtype)
                            from transformers.quantizers.quantizers_utils import should_convert_module
                            for name in models.VISION_MODULES[key]:
                                for full_name in (f"{name}.patch_dense", f"model.{name}.patch_dense"):
                                    self.assertFalse(should_convert_module(full_name, quant.llm_int8_skip_modules))
                            self.assertFalse(should_convert_module("llm.lm_head", quant.llm_int8_skip_modules))
                            self.assertTrue(should_convert_module("model.language_model.layers.0.q_proj",
                                                                 quant.llm_int8_skip_modules))
                        else:
                            bnb.assert_not_called()

    def test_gemma_real_vision_dtype_regression_and_integer_inputs(self):
        from transformers.models.gemma4_unified.configuration_gemma4_unified import (
            Gemma4UnifiedVisionConfig, Gemma4UnifiedTextConfig)
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder

        class PackedLinear(torch.nn.Module):
            """Emulate packed weight dtype and floating compute output, without CUDA/bnb."""
            def __init__(self, dense):
                super().__init__()
                self.dense = dense
                self.register_buffer("weight", torch.zeros(1, dtype=torch.uint8))

            def forward(self, value):
                return self.dense(value.to(self.dense.weight.dtype))

        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                vision = Gemma4UnifiedVisionEmbedder(
                    Gemma4UnifiedVisionConfig(patch_size=2, pooling_kernel_size=1,
                        mm_embed_dim=8, mm_posemb_size=4, output_proj_dims=8),
                    Gemma4UnifiedTextConfig(hidden_size=8)).to(dtype).eval()
                vision.patch_dense = PackedLinear(vision.patch_dense)
                pixels = torch.randn(1, 2, 12)
                positions = torch.tensor([[[0, 0], [1, 1]]])
                # The old runner left these pixels FP32. Gemma cannot infer a
                # floating input dtype from the uint8 packed patch_dense weight.
                with self.assertRaises(RuntimeError):
                    vision(pixels, positions, return_dict=True)
                processor = Mock()
                processor.apply_chat_template.side_effect = lambda *a, **kw: BatchFeature({
                    "input_ids": torch.tensor([[1, 2]]), "pixel_values": pixels.clone(),
                    "image_position_ids": positions.clone(), "attention_mask": torch.ones(1, 2, dtype=torch.long)})
                def generate(**inputs):
                    self.assertEqual(inputs["pixel_values"].dtype, dtype)
                    self.assertEqual(inputs["input_ids"].dtype, torch.long)
                    self.assertEqual(inputs["image_position_ids"].dtype, torch.long)
                    self.assertEqual(inputs["attention_mask"].dtype, torch.long)
                    output = vision(inputs["pixel_values"], inputs["image_position_ids"], return_dict=True)
                    self.assertTrue(torch.isfinite(output.pooler_output).all())
                    self.assertEqual(output.pooler_output.dtype, dtype)
                    return torch.tensor([[1, 2, 3]])
                model = SimpleNamespace(device=torch.device("cpu"), generate=generate)
                processor.decode.return_value = "answer"
                processor.parse_response.return_value = {"content": "answer"}
                runner = models.MultimodalRunner(model, processor, True, dtype)
                with patch("torch.cuda.is_available", return_value=False):
                    self.assertEqual(runner.answer(Image.new("RGB", (4, 4)), "question", 16), "answer")

    def test_minicpm_uses_backend_sampling_switch_and_fresh_history(self):
        for backend in ("minicpm_v", "minicpm_o"):
            model = Mock()
            model.chat.return_value = "answer"
            runner = models.MiniCPMRunner(model, Mock(), backend, torch.float16)
            with patch("torch.cuda.is_available", return_value=False):
                runner.answer(Image.new("RGB", (4, 4)), "qa prompt", 128)
                runner.answer(Image.new("RGB", (4, 4)), "visual prompt", 512)
            calls = model.chat.call_args_list
            self.assertEqual(calls[0].kwargs["msgs"][0]["content"][1], "qa prompt")
            self.assertEqual(calls[1].kwargs["msgs"][0]["content"][1], "visual prompt")
            for call in calls:
                self.assertEqual(len(call.kwargs["msgs"]), 1)
                self.assertFalse(call.kwargs["enable_thinking"])
                if backend == "minicpm_v":
                    self.assertFalse(call.kwargs["sampling"])
                    self.assertEqual(call.kwargs["num_beams"], 1)
                else:
                    self.assertFalse(call.kwargs["do_sample"])
                    self.assertFalse(call.kwargs["generate_audio"])

    def test_pangea_uses_checkpoint_preprocessing_and_vision_dtype(self):
        for use_list in (False, True):
            vision = SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16)
            model = Mock(device=torch.device("cpu"))
            model.get_vision_tower.return_value = vision
            model.generate.return_value = torch.tensor([[3, 4]])
            tokenizer = Mock()
            tokenizer.batch_decode.return_value = ["answer"]
            pixels = torch.randn(1, 3, 4, 4)
            preprocess = Mock(return_value=[pixels] if use_list else pixels)
            modules = {"llava": SimpleNamespace(),
                "llava.constants": SimpleNamespace(IMAGE_TOKEN_INDEX=-200),
                "llava.mm_utils": SimpleNamespace(process_images=preprocess,
                    tokenizer_image_token=Mock(return_value=torch.tensor([1, 2])))}
            runner = models.PangeaRunner(tokenizer, model, Mock(), torch.float16)
            image = Image.new("RGB", (4, 4))
            with patch.dict("sys.modules", modules), patch("torch.cuda.is_available", return_value=False):
                self.assertEqual(runner.answer(image, "question", 16), "answer")
            preprocess.assert_called_once_with([image], runner.image_processor, model.config)
            passed = model.generate.call_args.kwargs["images"]
            self.assertEqual((passed[0] if use_list else passed).dtype, vision.dtype)

    def test_kaggle_stops_repeated_errors_with_saved_records_and_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            Image.new("RGB", (4, 4)).save(root / "image.png")
            rows = [{"id": str(i), "image": "image.png", "question": "question", "answer": "gold"}
                    for i in range(3)]
            dataset = SimpleNamespace(__version__="test", load_dataset=Mock(return_value=rows))
            api = Mock()
            api.dataset_info.return_value = SimpleNamespace(sha="dataset", siblings=[
                SimpleNamespace(rfilename="qa/mena/dev_en.parquet")])
            api.model_info.return_value = SimpleNamespace(sha="model")
            runner = Mock()
            runner.answer.side_effect = RuntimeError("expected scalar type Float but found BFloat16")
            settings = dict(OUTPUT_DIR=str(root / "runs"), MODEL_KEY="gemma_4_12b", TRACKS="all",
                SPLITS=("dev",), VARIANTS=("original",), MAX_ROWS_PER_TRACK=3)
            with patch.dict("sys.modules", {"datasets": dataset,
                "huggingface_hub": SimpleNamespace(HfApi=Mock(return_value=api))}), \
                 patch.multiple(run.config, **settings), patch("run.media_root", return_value=root), \
                 patch("run.ensure_images"), patch("run.load_model", return_value=runner):
                with self.assertRaisesRegex(RuntimeError, "Three consecutive"):
                    run.main()
            out = next((root / "runs").iterdir())
            records = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertEqual(json.loads((out / "status.json").read_text())["state"], "inference_error")
            self.assertIn("Traceback", records[-1]["error"]["traceback"])
            self.assertEqual(run.completed_keys(out / "predictions.jsonl"), set())


if __name__ == "__main__":
    unittest.main()
