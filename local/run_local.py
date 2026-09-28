"""Run MMCultureQA Task 2 through an already running LM Studio server."""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

# Large scorer weights can stall on the Xet transport on some Mac setups.
# Use regular HTTPS downloads before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local import config as settings
from local.lmstudio import LMStudioClient
from run import (REPO, PROMPT_SUFFIX, VISUAL_PROMPT, TASKS, SCHEMA_VERSION, completed_keys,
                 discover_tracks, dump_json, ensure_images, image_for_variant,
                 prompt_for, row_key, summarize_records, utc_now)


def local_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def selected_variants(split):
    return tuple(v for v in settings.VARIANTS if v == "original" or split in settings.ROBUSTNESS_SPLITS)


def model_key(model_id):
    slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")[:48]
    return "lmstudio_" + slug + "_" + hashlib.sha256(model_id.encode()).hexdigest()[:8]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def model_identity(model_id, metadata):
    if not metadata:
        return {"model_id": model_id}
    return {name: metadata.get(name) for name in
            ("key", "format", "size_bytes", "selected_variant", "quantization")}


def validate_settings():
    if not settings.MODEL_ID.strip():
        raise ValueError("Set MODEL_ID in local/config.py. List IDs with: python local/run_local.py --list-models")
    if settings.MAX_ROWS_PER_TRACK is not None and settings.MAX_ROWS_PER_TRACK < 1:
        raise ValueError("MAX_ROWS_PER_TRACK must be positive or None")
    if "original" not in settings.VARIANTS:
        raise ValueError("VARIANTS must include 'original'")
    if settings.REQUEST_TIMEOUT_SECONDS <= 0:
        raise ValueError("REQUEST_TIMEOUT_SECONDS must be positive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true", help="Print IDs visible to LM Studio and exit")
    args = parser.parse_args()
    client = LMStudioClient(settings.SERVER_URL, settings.REQUEST_TIMEOUT_SECONDS)
    available_models = client.available_models()
    if args.list_models:
        print("Models visible to LM Studio:")
        for item in available_models:
            metadata = client.model_metadata(item)
            vision = metadata.get("capabilities", {}).get("vision") if metadata else None
            reasoning = (metadata or {}).get("capabilities", {}).get("reasoning") or {}
            tags = []
            if vision is not None:
                tags.append("vision" if vision else "no vision")
            if "off" in reasoning.get("allowed_options", []):
                tags.append("reasoning off supported")
            print(f"  {item}" + (f" [{', '.join(tags)}]" if tags else ""))
        return

    from datasets import load_dataset
    from huggingface_hub import HfApi
    import datasets

    validate_settings()
    if settings.MODEL_ID not in available_models:
        raise ValueError(f"MODEL_ID {settings.MODEL_ID!r} is unavailable. Run --list-models for exact IDs.")
    metadata = client.model_metadata(settings.MODEL_ID)
    if metadata and metadata.get("capabilities", {}).get("vision") is False:
        raise ValueError("The selected LM Studio model reports no vision support. Choose an image-capable model.")
    reasoning = (metadata or {}).get("capabilities", {}).get("reasoning") or {}
    reasoning_off = "off" in reasoning.get("allowed_options", [])
    if reasoning and not reasoning_off and reasoning.get("default") != "off":
        raise ValueError("This LM Studio model reports that reasoning cannot be disabled. Use a model with reasoning='off' support for the 128-token benchmark.")
    random.seed(settings.SEED)

    dataset_info = HfApi().dataset_info(REPO, files_metadata=False)
    revision = dataset_info.sha
    available_tracks = discover_tracks(dataset_info.siblings)
    tracks = available_tracks if settings.TRACKS == "all" else list(settings.TRACKS)
    unknown = set(tracks) - set(available_tracks)
    if unknown or not tracks:
        raise ValueError(f"Tracks unavailable: {sorted(unknown)}. Available: {available_tracks}")

    key = model_key(settings.MODEL_ID)
    model_revision = fingerprint(model_identity(settings.MODEL_ID, metadata))
    run_id = fingerprint({
        "schema": SCHEMA_VERSION, "dataset": revision, "model_id": settings.MODEL_ID,
        "model_revision": model_revision, "splits": settings.SPLITS, "tracks": settings.TRACKS,
        "variants": settings.VARIANTS, "robustness_splits": settings.ROBUSTNESS_SPLITS,
        "max_rows": settings.MAX_ROWS_PER_TRACK, "max_new_tokens": settings.MAX_NEW_TOKENS,
        "seed": None if reasoning_off else settings.SEED, "prompt": PROMPT_SUFFIX,
        "visual_prompt": VISUAL_PROMPT, "tasks": TASKS,
        "visual_max_new_tokens": settings.VISUAL_MAX_NEW_TOKENS,
        "reasoning_off": reasoning_off,
    })
    out = local_path(settings.OUTPUT_DIR) / f"{key}_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "predictions.jsonl"
    manifest_path = out / "manifest.json"
    status_path = out / "status.json"
    manifest = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "task": "SemEval 2027 MMCultureQA Task 2", "backend": "lmstudio",
        "input_fields": {"qa": ["image", "question"], "visual": ["image"]},
        "dataset_repo": REPO, "dataset_revision": revision,
        "model_key": key, "model_repo": settings.MODEL_ID,
        "model_revision": model_revision, "lmstudio_model_metadata": metadata,
        "tracks": tracks, "splits": list(settings.SPLITS),
        "variants": list(settings.VARIANTS),
        "robustness_splits": list(settings.ROBUSTNESS_SPLITS),
        "max_rows_per_track": settings.MAX_ROWS_PER_TRACK,
        "max_new_tokens": settings.MAX_NEW_TOKENS,
        "reasoning_mode": "off" if reasoning_off else "model_default",
        "inference_endpoint": "/api/v1/chat" if reasoning_off else "/v1/chat/completions",
        "seed": None if reasoning_off else settings.SEED,
        "temperature": 0, "prompt_suffix": PROMPT_SUFFIX,
        "visual_prompt": VISUAL_PROMPT, "tasks": list(TASKS),
        "visual_max_new_tokens": settings.VISUAL_MAX_NEW_TOKENS,
        "versions": {"datasets": datasets.__version__},
        "created_utc": utc_now(),
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["created_utc"] = previous["created_utc"]
    dump_json(manifest_path, manifest)
    dump_json(status_path, {"run_id": run_id, "state": "preparing", "updated_utc": utc_now()})
    done = completed_keys(records_path)
    root = local_path(settings.DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Run: {out}\nModel: {settings.MODEL_ID}\nTracks: {', '.join(tracks)}", flush=True)

    for split in settings.SPLITS:
        for region in sorted({t.split("_")[1] for t in tracks}):
            print(f"Checking images: {region}/{split}", flush=True)
            ensure_images(region, split, revision, root)

    dump_json(status_path, {"run_id": run_id, "state": "running", "updated_utc": utc_now()})
    counts = Counter()
    consecutive_errors = 0
    with records_path.open("a", encoding="utf-8") as sink:
        for track in tracks:
            for split in settings.SPLITS:
                print(f"Running {track}/{split}", flush=True)
                dataset = load_dataset(REPO, track, split=split, revision=revision, streaming=True)
                for index, row in enumerate(dataset):
                    if settings.MAX_ROWS_PER_TRACK is not None and index >= settings.MAX_ROWS_PER_TRACK:
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
                        prediction, error, status, fatal = "", None, "ok", False
                        try:
                            with Image.open(image_path) as source:
                                image = image_for_variant(source.convert("RGB"), variant)
                            prediction = client.answer(
                                image, prompt_for(row["question"]) if task == "qa" else VISUAL_PROMPT,
                                settings.MODEL_ID,
                                settings.MAX_NEW_TOKENS if task == "qa" else settings.VISUAL_MAX_NEW_TOKENS,
                                settings.SEED,
                                reasoning_off=reasoning_off,
                            )
                            if not prediction:
                                raise ValueError("Model returned an empty answer")
                        except Exception as exc:
                            status = "error"
                            fatal = isinstance(exc, (requests.ConnectionError, requests.Timeout, requests.HTTPError))
                            error = {"type": type(exc).__name__, "message": str(exc),
                                     "traceback": traceback.format_exc(limit=5)}
                        record = {
                            "schema_version": SCHEMA_VERSION, "run_id": run_id,
                            "dataset_revision": revision, "model_key": key,
                            "model_repo": settings.MODEL_ID, "model_revision": model_revision,
                            "track": track, "split": split, "id": row_id,
                            "variant": variant, "task": task, "question": row["question"],
                            "reference": row.get("answer"), "country": row.get("country"),
                            "category": row.get("category"), "subcategory": row.get("subcategory"),
                            "image": row["image"], "prediction": prediction,
                            "status": status, "error": error,
                            "latency_seconds": round(time.monotonic() - start, 3),
                            "completed_utc": utc_now(),
                        }
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                        sink.flush()
                        counts[status] += 1
                        if status == "ok":
                            done.add(row_key(track, split, row_id, variant, task))
                            consecutive_errors = 0
                        else:
                            consecutive_errors += 1
                            print(f"ERROR {track}/{split}/{row_id}/{variant}/{task}: {error['message']}", flush=True)
                            if fatal or consecutive_errors >= 3:
                                dump_json(status_path, {"run_id": run_id,
                                                        "state": "connection_error" if fatal else "run_error",
                                                        "error": error, "updated_utc": utc_now()})
                                raise RuntimeError("LM Studio failed repeatedly; progress is saved. Check the model/server and rerun.")
                    if (index + 1) % 10 == 0:
                        print(f"{track}/{split}: {index + 1} rows, {dict(counts)}", flush=True)
    total_counts, by_track = summarize_records(records_path)
    state = "finished_with_errors" if total_counts.get("error", 0) else "finished"
    dump_json(out / "summary.json", {"run_id": run_id, "counts_this_session": dict(counts),
                                      "counts_total": total_counts, "by_track": by_track,
                                      "finished_utc": utc_now(), "predictions": str(records_path)})
    dump_json(status_path, {"run_id": run_id, "state": "evaluating",
                            "counts_total": total_counts, "updated_utc": utc_now()})
    from score_saved import run_scoring
    try:
        if settings.UNLOAD_MODEL_BEFORE_SCORING:
            unloaded = client.unload_model(settings.MODEL_ID)
            print(f"Unloaded {unloaded} LM Studio model instance(s) before BERTScore.", flush=True)
        run_scoring(out, batch_size=settings.BERTSCORE_BATCH_SIZE,
                    device=settings.BERTSCORE_DEVICE,
                    model_type=settings.BERTSCORE_MODEL)
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
