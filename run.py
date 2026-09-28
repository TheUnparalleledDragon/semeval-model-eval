"""Run every released MMCultureQA Task 2 image+text row with one model.

Kaggle: enable a GPU and Internet, install requirements, edit MODEL_KEY in
config.py, then run ``python run.py``. Results are checkpointed after each row.
"""

import hashlib
import gc
import io
import json
import random
import re
import time
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import config
from models import MODELS, MODEL_ADAPTER_VERSION, load_model, has_answer_text


REPO = "QCRI/MMCQA-SemEval27"
PROMPT_SUFFIX = "\nAnswer in the same language as the question. Give a concise answer based on the image."
VISUAL_PROMPT = ("Describe only what you can see in this image. Return a JSON object with four arrays of short strings: "
                 '"visible_details" (objects, clothing, food, buildings, symbols, colors, and actions), '
                 '"text_in_image" (only clearly readable words), "location_clues" (flags, scripts, signs, '
                 "landmarks, or other visible clues that could help identify a place or culture), and "
                 '"uncertain_inferences" (possible identities or places, each with its visible evidence). '
                 "Be specific, distinguish observation from inference, and use empty arrays where evidence is absent. "
                 "Do not guess a country from appearance alone. Do not answer any separate question. Output JSON only.")
TASKS = ("qa", "visual")
SCHEMA_VERSION = 2


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def discover_tracks(siblings):
    pattern = re.compile(r"^qa/([^/]+)/(?:train|dev)_([^/.]+)\.parquet$")
    tracks = set()
    for sibling in siblings:
        name = sibling.rfilename if hasattr(sibling, "rfilename") else str(sibling)
        match = pattern.match(name)
        if match:
            tracks.add(f"qa_{match.group(1)}_{match.group(2)}")
    return sorted(tracks)


def media_root():
    root = Path(config.DATA_DIR).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_images(region, split, revision, root):
    """Get organizer images once per region/split; reject unsafe zip paths."""
    from huggingface_hub import hf_hub_download

    archive_name = f"archives/images_{region}_{split}.zip"
    marker = root / f".images_{region}_{split}_{revision}.complete"
    if marker.exists():
        return
    archive = hf_hub_download(REPO, archive_name, repo_type="dataset", revision=revision)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts or member_path.parts[0] != "images":
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.is_dir():
                continue
            destination = root / member_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.stat().st_size != member.file_size:
                with zf.open(member) as source, destination.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
    marker.write_text(archive_name + "\n", encoding="utf-8")


def image_for_variant(image, variant):
    if variant == "original":
        return image
    if variant == "jpeg_85":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    raise ValueError(f"Unknown variant: {variant}")


def selected_variants(split):
    return tuple(v for v in config.VARIANTS if v == "original" or split in config.ROBUSTNESS_SPLITS)


def prompt_for(question):
    return str(question).strip() + PROMPT_SUFFIX


def row_key(track, split, row_id, variant, task="qa"):
    return (track, split, str(row_id), variant, task)


def completed_keys(path):
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # A partially written final line can be retried.
        if record.get("status") == "ok" and has_answer_text(record.get("prediction")):
            done.add(row_key(record["track"], record["split"], record["id"], record["variant"], record.get("task", "qa")))
    return done


def summarize_records(path):
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row_key(record["track"], record["split"], record["id"], record["variant"], record.get("task", "qa"))
        latest[key] = ("error" if record["status"] == "ok"
                       and not has_answer_text(record.get("prediction")) else record["status"])
    totals = Counter(latest.values())
    by_track = {}
    for (track, split, _, variant, task), status in latest.items():
        label = f"{task}/{track}/{split}/{variant}"
        by_track.setdefault(label, Counter())[status] += 1
    return dict(totals), {key: dict(value) for key, value in sorted(by_track.items())}


def make_run_id(dataset_revision, model_revision):
    material = {
        "schema": SCHEMA_VERSION, "dataset": dataset_revision, "model": model_revision,
        "model_key": config.MODEL_KEY, "splits": config.SPLITS, "tracks": config.TRACKS,
        "variants": config.VARIANTS, "robustness_splits": config.ROBUSTNESS_SPLITS,
        "max_rows": config.MAX_ROWS_PER_TRACK, "max_new_tokens": config.MAX_NEW_TOKENS,
        "four_bit": config.LOAD_IN_4BIT, "seed": config.SEED, "prompt": PROMPT_SUFFIX,
        "visual_prompt": VISUAL_PROMPT, "tasks": TASKS,
        "visual_max_new_tokens": config.VISUAL_MAX_NEW_TOKENS,
        "model_adapter_version": MODEL_ADAPTER_VERSION,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:12]


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    import torch
    import transformers
    import datasets

    if config.MODEL_KEY not in MODELS:
        raise ValueError(f"Choose MODEL_KEY from {', '.join(MODELS)}")
    if config.MAX_ROWS_PER_TRACK is not None and config.MAX_ROWS_PER_TRACK < 1:
        raise ValueError("MAX_ROWS_PER_TRACK must be positive or None")
    if "original" not in config.VARIANTS:
        raise ValueError("VARIANTS must include 'original'")
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    api = HfApi()
    dataset_info = api.dataset_info(REPO, files_metadata=False)
    model_info = api.model_info(MODELS[config.MODEL_KEY].repo)
    revision = dataset_info.sha
    available = discover_tracks(dataset_info.siblings)
    tracks = available if config.TRACKS == "all" else list(config.TRACKS)
    unknown = set(tracks) - set(available)
    if unknown or not tracks:
        raise ValueError(f"Tracks unavailable: {sorted(unknown)}. Available: {available}")

    run_id = make_run_id(revision, model_info.sha)
    out = Path(config.OUTPUT_DIR).expanduser().resolve() / f"{config.MODEL_KEY}_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "predictions.jsonl"
    manifest_path = out / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "task": "SemEval 2027 MMCultureQA Task 2",
        "backend": "huggingface_transformers",
        "input_fields": {"qa": ["image", "question"], "visual": ["image"]},
        "dataset_repo": REPO, "dataset_revision": revision,
        "model_key": config.MODEL_KEY, "model_repo": MODELS[config.MODEL_KEY].repo,
        "model_revision": model_info.sha, "tracks": tracks, "splits": list(config.SPLITS),
        "model_adapter_version": MODEL_ADAPTER_VERSION,
        "precision_policy": "native_bf16_else_fp16; gemma_bf16_else_fp32; language_nf4_if_enabled; vision_unquantized",
        "variants": list(config.VARIANTS), "robustness_splits": list(config.ROBUSTNESS_SPLITS),
        "max_rows_per_track": config.MAX_ROWS_PER_TRACK,
        "max_new_tokens": config.MAX_NEW_TOKENS, "load_in_4bit": config.LOAD_IN_4BIT,
        "seed": config.SEED, "prompt_suffix": PROMPT_SUFFIX,
        "visual_prompt": VISUAL_PROMPT, "tasks": list(TASKS),
        "visual_max_new_tokens": config.VISUAL_MAX_NEW_TOKENS,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "datasets": datasets.__version__},
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "created_utc": utc_now(),
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["run_id"] != run_id or previous["dataset_revision"] != revision:
            raise RuntimeError("Existing run has different inputs. Choose another output directory.")
        manifest["created_utc"] = previous["created_utc"]
    dump_json(manifest_path, manifest)
    status_path = out / "status.json"
    dump_json(status_path, {"run_id": run_id, "state": "preparing", "updated_utc": utc_now()})
    done = completed_keys(records_path)
    root = media_root()
    print(f"Run: {out}\nTracks: {', '.join(tracks)}\nCompleted rows: {len(done)}", flush=True)

    # Download images before loading the checkpoint, leaving more GPU memory for inference.
    for split in config.SPLITS:
        for region in sorted({t.split("_")[1] for t in tracks}):
            print(f"Checking images: {region}/{split}", flush=True)
            ensure_images(region, split, revision, root)

    try:
        runner = load_model(config.MODEL_KEY, config.LOAD_IN_4BIT)
    except Exception as exc:
        dump_json(status_path, {"run_id": run_id, "state": "load_error",
                                "error_type": type(exc).__name__, "error": str(exc),
                                "updated_utc": utc_now()})
        raise
    manifest["compute_dtype"] = str(runner.dtype)
    dump_json(manifest_path, manifest)
    dump_json(status_path, {"run_id": run_id, "state": "running", "updated_utc": utc_now()})
    counts = Counter()
    consecutive_errors = 0
    with records_path.open("a", encoding="utf-8") as sink:
        for track in tracks:
            for split in config.SPLITS:
                print(f"Running {track}/{split}", flush=True)
                dataset = load_dataset(REPO, track, split=split, revision=revision, streaming=True)
                for index, row in enumerate(dataset):
                    if config.MAX_ROWS_PER_TRACK is not None and index >= config.MAX_ROWS_PER_TRACK:
                        break
                    row_id = str(row["id"])
                    variants = selected_variants(split)
                    pending = [(v, task) for v in variants for task in TASKS
                               if row_key(track, split, row_id, v, task) not in done]
                    if not pending:
                        counts["skipped"] += len(variants) * len(TASKS)
                        continue
                    image_path = (root / row["image"]).resolve()
                    if not image_path.is_relative_to(root):
                        raise ValueError(f"Image path escapes data directory: {row['image']}")
                    for variant, task in pending:
                        start = time.monotonic()
                        prediction, error, status, out_of_memory = "", None, "ok", False
                        try:
                            with Image.open(image_path) as source:
                                image = image_for_variant(source.convert("RGB"), variant)
                            with torch.inference_mode():
                                prediction = runner.answer(
                                    image, prompt_for(row["question"]) if task == "qa" else VISUAL_PROMPT,
                                    config.MAX_NEW_TOKENS if task == "qa" else config.VISUAL_MAX_NEW_TOKENS)
                            if not has_answer_text(prediction):
                                raise ValueError("Model returned empty or special-token-only answer text")
                        except Exception as exc:
                            status = "error"
                            out_of_memory = isinstance(exc, torch.cuda.OutOfMemoryError)
                            error = {"type": type(exc).__name__, "message": str(exc),
                                     "traceback": traceback.format_exc()}
                        record = {
                            "schema_version": SCHEMA_VERSION, "run_id": run_id,
                            "dataset_revision": revision, "model_key": config.MODEL_KEY,
                            "model_repo": MODELS[config.MODEL_KEY].repo,
                            "model_revision": model_info.sha,
                            "track": track, "split": split, "id": row_id, "variant": variant,
                            "task": task,
                            "question": row["question"], "reference": row.get("answer"),
                            "country": row.get("country"), "category": row.get("category"),
                            "subcategory": row.get("subcategory"),
                            "image": row["image"], "prediction": prediction,
                            "status": status, "error": error,
                            "latency_seconds": round(time.monotonic() - start, 3),
                            "completed_utc": utc_now(),
                        }
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                        sink.flush()
                        counts[status] += 1
                        if status == "ok":
                            consecutive_errors = 0
                            done.add(row_key(track, split, row_id, variant, task))
                        if status == "error":
                            consecutive_errors += 1
                            print(f"ERROR {track}/{split}/{row_id}/{variant}/{task}: {error['message']}", flush=True)
                            if out_of_memory or consecutive_errors >= 3:
                                dump_json(status_path, {"run_id": run_id, "state": "inference_error",
                                    "counts_this_session": dict(counts), "error": error,
                                    "updated_utc": utc_now()})
                                message = ("GPU out of memory; use a larger GPU or smaller model."
                                           if out_of_memory else "Three consecutive prediction failures; check status.json for the full traceback.")
                                raise RuntimeError(message + " Progress is saved; rerun to resume successful tasks.")
                    if (index + 1) % 100 == 0:
                        print(f"{track}/{split}: {index + 1} rows, {dict(counts)}", flush=True)
    total_counts, by_track = summarize_records(records_path)
    state = "finished_with_errors" if total_counts.get("error", 0) else "finished"
    dump_json(out / "summary.json", {"run_id": run_id, "counts_this_session": dict(counts),
                                      "counts_total": total_counts, "by_track": by_track,
                                      "finished_utc": utc_now(), "predictions": str(records_path)})
    dump_json(status_path, {"run_id": run_id, "state": "evaluating",
                            "counts_total": total_counts, "updated_utc": utc_now()})
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from evaluate_run import evaluate_run
    try:
        evaluate_run(out, batch_size=config.BERTSCORE_BATCH_SIZE,
                     device=config.BERTSCORE_DEVICE, model_type=config.BERTSCORE_MODEL)
    except BaseException as exc:
        dump_json(status_path, {"run_id": run_id, "state": "evaluation_error",
                                "counts_total": total_counts, "error_type": type(exc).__name__,
                                "error": str(exc) or "Interrupted", "updated_utc": utc_now()})
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeError(f"Predictions saved at {out}, but evaluation failed: {exc}") from exc
    dump_json(status_path, {"run_id": run_id, "state": state,
                            "counts_total": total_counts, "updated_utc": utc_now()})
    print(f"Finished. Outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
