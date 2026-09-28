# Run the same Task 2 benchmark with LM Studio on a Mac

This local path sends the **original SemEval dataset images and questions** to a vision model already downloaded in LM Studio. For each image variant it makes two separate, stateless requests: `qa` receives image + dataset question; `visual` receives image + the fixed visual-evidence prompt, with no dataset question. Neither request receives country, category, subcategory, or reference answer. It writes the same result files as the Kaggle runner, so the project root's `compare.py` can compare both. Its Python requirements now include PyTorch and BERTScore **for evaluation only**; the VLM still runs in LM Studio without CUDA or bitsandbytes.

LM Studio's [OpenAI-compatible server](https://lmstudio.ai/docs/developer/openai-compat) supports image chat at `/v1/chat/completions`. Its [model listing endpoint](https://lmstudio.ai/docs/developer/openai-compat/models) gives the exact ID to paste into the config. The [native model listing](https://lmstudio.ai/docs/developer/rest/list) can report whether a model supports vision. The runner checks that flag when LM Studio provides it, and the first request exposes unsupported image input on older versions.

## Start from scratch

1. In LM Studio, download and **load an image-capable model**. A text-only model cannot complete this task. On a 16 GB Mac, choose a quantization that fits alongside the vision encoder and context. A 27B model may not fit; LM Studio's displayed memory estimate is the useful check. Start the local server in the **Developer** tab, usually on port `1234`, or run `lms server start --port 1234`. Keep LM Studio running.
2. From this project's root, create a Python environment and install only the Mac requirements:

   ```bash
   python3 -m venv .venv-local
   source .venv-local/bin/activate
   python -m pip install -r local/requirements.txt
   ```

3. List the model IDs visible to the server:

   ```bash
   python local/run_local.py --list-models
   ```

   Copy the exact ID of a vision model into the **one model line** in [`local/config.py`](config.py):

   ```python
   MODEL_ID = "paste-id-from-list-here"
   ```

4. For a smoke run, temporarily set `MAX_ROWS_PER_TRACK = 2` in `local/config.py`. Then:

   ```bash
   python local/run_local.py
   ```

   The first run downloads the dataset's **dev image archive** and streams the original dev questions from Hugging Face, so Internet access is needed for data. Inference itself stays on your Mac. No image or question is sent to a hosted model API. Open `local/runs/<model_key>_<run_id>/predictions.jsonl`: `qa` should reflect image and question, while `visual` should describe image evidence without answering the question. After inference, the runner automatically downloads the BERTScore backbone if needed and saves `qa_scores.jsonl`, `evaluation.json`, and `visual_review.csv`.

5. Set `MAX_ROWS_PER_TRACK = None` for the full configured run. With the current dev/original settings and all four released language tracks, this produces **8,000 requests**: 4,000 images × two independent tasks. The checked-in `MAX_ROWS_PER_TRACK = 100` instead produces **800 requests**; use `2` for a 16-request smoke run. Add train to `SPLITS` or `jpeg_85` to `VARIANTS` only for later analysis; adding both produces 96,000 requests under the current release. The runner saves after every request. Re-running with the same settings skips successful task requests. A different row limit creates a different run folder.

6. To try a different downloaded model, change only `MODEL_ID` and rerun. Models and quantizations have different LM Studio IDs; keep each run directory. Use the same `SPLITS`, `VARIANTS`, row limit, and both token caps if you want directly matched results.

## Compare with Kaggle runs

For additional factual judging of English QA answers using OpenRouter, use the project root's [`judge_predictions.py`](../judge_predictions.py). Run `python judge_predictions.py local/runs/<run_directory>/predictions.jsonl --dry-run` first. The [root README](../README.md#additional-english-evaluation-with-an-llm-judge) explains API-key setup, costs, ratings, resumable execution, and the two new outputs. This is a separate optional step and does not change local inference or BERTScore scoring.

The judge automatically reads `OPENROUTER_API_KEY` from the project root's `.env` when no shell key is set. Install `python-dotenv` through the requirements file. Use `python judge_predictions.py --check-auth --env-file .env` to check the file's credential without paid model calls or printing the key.

From the project root:

```bash
python compare.py local/runs/* --output local_comparison.csv
```

You can include Kaggle run directories in the same command. `compare.py` matches **original dev** items by track and ID and checks that the dataset revision, both prompts, row limit, and both output caps match. If a Kaggle run used a two-row smoke limit, compare it with a local two-row smoke run; compare full runs with full runs. BERTScore F1 is computed automatically per run, then compared on IDs answered successfully by every model. Visual JSON validity, detail counts, coverage, latency, and completed manual ratings are shown separately. Old schema-1 run folders lack these two-task outputs and cannot be compared with new ones.

## Settings and output

[`config.py`](config.py) keeps local settings separate from Kaggle's `config.py`. `MODEL_ID` is the only model selection setting. `SERVER_URL` defaults to `http://127.0.0.1:1234`; change it if you chose another port. `DATA_DIR` and `OUTPUT_DIR` default to `local/data` and `local/runs` under this project. These directories can become large; they are ignored by Git. QA uses `MAX_NEW_TOKENS = 128`; visual JSON uses `VISUAL_MAX_NEW_TOKENS = 512`. BERTScore uses CPU, batch size one, and `bert-base-multilingual-cased` by default, matching Kaggle's scorer. After inference the runner unloads its LM Studio model instances before launching scoring in a separate process. Set `UNLOAD_MODEL_BEFORE_SCORING = False` only if you need the model to stay loaded and have sufficient RAM. The same `id`, `track`, `split`, `variant`, `task`, `status`, `prediction`, and reference fields are saved as Kaggle output. The manifest identifies the LM Studio model and its quantization metadata when the server exposes it. Original images are transported as lossless PNG data URLs; the optional JPEG variant is recompressed **before** that transport.

The organizers name BERTScore F1 as the official ranking metric but have not exposed the scorer's exact settings in the current public dataset files. The automatic `bert-base-multilingual-cased` score is therefore a **local estimate**, recorded in `evaluation.json`; use the organizer's scorer when published. The runner uses ordinary HTTPS for Hugging Face downloads on Mac; setting an `HF_TOKEN` can help with unauthenticated rate limits. Predictions and `visual_summary.json` remain saved if scoring fails, and `status.json` reports `evaluation_error`. The dataset has no gold visual-description labels. Review the **same image IDs across models** and fill the 0–3 detail accuracy, cultural-clue accuracy, and unsupported-claim columns in `visual_review.csv` before selecting a visual specialist. The review sheet's `image` path is relative to `local/data/`. Higher is better for the first two ratings; lower is better for unsupported claims. Format validity and number of details alone cannot establish visual accuracy. To rerun evaluation without VLM inference, use `python score_saved.py score local/runs/<run_directory>` from the project root. You may pass `predictions.jsonl` instead of its directory; no LM Studio server is needed. Per-track and overall scores are in `evaluation.json`, and per-item scores are in `qa_scores.jsonl`.

The server must remain running while inference proceeds. A connection error or three consecutive failed answers stops the run after saving the failing record. Restart the server or adjust the model and rerun to retry failures. If LM Studio uses Just-in-Time loading, its `/v1/models` list may show downloaded models that are not in memory yet; explicitly loading the selected model first makes a smoke run easier to diagnose.

For a live run, open another terminal and use `python local/progress.py --watch 15`. It reads saved predictions, so it works without interrupting inference. It shows successful requests, errors, the latest saved item, recent request time, and a rough remaining-time estimate when `MAX_ROWS_PER_TRACK` is set. New runner processes print progress every 10 rows; a process started before that change keeps its original 100-row logging interval until restarted. A full 1,000-row-per-track dev run makes 8,000 model requests and can take many hours on a Mac.

### Gemma or another reasoning model returns empty answers

If `reasoning_content` uses the entire 128-token allowance, the model never reaches its final answer. The runner now checks LM Studio's reported reasoning capabilities and uses its [native chat endpoint](https://lmstudio.ai/docs/developer/rest/chat) with `reasoning: "off"` when supported. This is the correct API control for Gemma 4; simply omitting reasoning from the output would not solve the token-budget problem. A model that reports reasoning enabled but does not allow `off` is rejected before the dataset run. The inference endpoint and reasoning mode are recorded in the manifest. The native endpoint uses temperature zero but has no documented seed parameter, so its manifest records `seed: null`. Changing this setting creates a new run directory, so old failed smoke records remain available while the corrected run starts cleanly. If your LM Studio version does not expose capability metadata, turn off **Enable Thinking** for that model in LM Studio before running.

Model performance on a 16 GB Mac depends on the **specific quantization, vision support, context size, and LM Studio engine**. We cannot guarantee that every original checkpoint in the Kaggle list has a compatible local quantization, or that a large one will fit. The local results record the actual LM Studio model ID rather than pretending to be the original Hugging Face weights.
