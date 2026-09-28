"""Exercise real GPU loading plus three independent image requests before a run."""

import argparse
import gc
import json
import traceback
from pathlib import Path

from PIL import Image, ImageDraw

import config
from models import MODELS, MODEL_ADAPTER_VERSION, load_model, has_answer_text
from run import VISUAL_PROMPT, prompt_for, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", choices=MODELS, default=None)
    selection.add_argument("--all", action="store_true", help="Download/load every model sequentially")
    parser.add_argument("--image", type=Path, help="Optional real image; otherwise use a synthetic color/shape image")
    parser.add_argument("--no-four-bit", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("model_smoke_results.json"))
    args = parser.parse_args()
    import torch
    import transformers

    if args.image:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
    else:
        image = Image.new("RGB", (448, 448), "white")
        ImageDraw.Draw(image).rectangle((100, 100, 348, 348), fill="red")
    checks = [
        ("qa_en", prompt_for("Describe the main object in this image."), config.MAX_NEW_TOKENS),
        ("qa_ar", prompt_for("ما العنصر الرئيسي الظاهر في هذه الصورة؟"), config.MAX_NEW_TOKENS),
        ("visual", VISUAL_PROMPT, config.VISUAL_MAX_NEW_TOKENS),
    ]
    report = {"created_utc": utc_now(), "model_adapter_version": MODEL_ADAPTER_VERSION,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "four_bit": not args.no_four_bit, "image": str(args.image) if args.image else "synthetic_red_square",
              "models": []}
    failed = False
    for key in (list(MODELS) if args.all else [args.model or config.MODEL_KEY]):
        result = {"model_key": key, "model_repo": MODELS[key].repo, "checks": []}
        runner = None
        try:
            runner = load_model(key, four_bit=not args.no_four_bit)
            result["compute_dtype"] = str(runner.dtype)
            for task, prompt, cap in checks:
                prediction = runner.answer(image.copy(), prompt, cap)
                if not has_answer_text(prediction):
                    raise ValueError(f"{task}: empty or special-token-only answer")
                result["checks"].append({"task": task, "prediction": prediction, "status": "ok"})
                print(f"{key}/{task}: {prediction}", flush=True)
            result["status"] = "ok"
        except Exception as exc:
            failed = True
            result.update(status="error", error=str(exc), traceback=traceback.format_exc())
            print(f"FAILED {key}: {exc}", flush=True)
        finally:
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["models"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output.resolve()}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
