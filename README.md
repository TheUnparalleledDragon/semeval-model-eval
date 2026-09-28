# MMCultureQA Task 2 model comparison pipeline

This is the central guide for the SemEval 2027 Task 6 course project. It documents the task, the original dataset, both inference paths, the shared result contract, and the current experiment state. For a 16 GB Mac with models downloaded in LM Studio, use the [local runner guide](local/README.md) alongside the Mac section below. The Mac runner imports the root runner's dataset, image, prompt, and resume functions; changes to those shared functions can affect both backends.

This project runs open image and text models on the **original** [QCRI/MMCQA-SemEval27](https://huggingface.co/datasets/QCRI/MMCQA-SemEval27) data for SemEval 2027 Task 6, **Task 2: Textual Visual QA**. Change **one line**, `MODEL_KEY` in [`config.py`](config.py), to select a checkpoint. Each model receives two independent requests per image variant: the official-style question answering task and an image-only visual evidence description. Audio is never loaded.

The [task definition](https://mmcultureqa-semeval27.github.io/tasks/) asks for a short, open answer in the question's language, grounded in both the image and local cultural knowledge. Task 1 uses a **spoken** question; this repository implements Task 2's **written** question only. Tracks are ranked separately by language using BERTScore F1; BLEU and ROUGE are auxiliary. The [dataset card](https://huggingface.co/datasets/QCRI/MMCQA-SemEval27) currently lists English (`qa_mena_en`), Modern Standard Arabic (`qa_mena_msa`), Egyptian Arabic (`qa_mena_arz`), and Levantine Arabic (`qa_mena_ajp`), with **10,000 train and 1,000 dev rows per track**. More tracks are planned. Each row has `id` (image hash), image path, country, category, subcategory, question, and reference answer. The test release will omit reference answers and the cultural metadata. The `qa` request passes **only the image and question**; the `visual` request passes **only the image** with a fixed description prompt. Other fields are recorded for later analysis. The same image ID appears across language varieties, so comparison should be per track and split rather than treating all rows as independent images. Check the [participation page](https://mmcultureqa-semeval27.github.io/participate/) for current rules and submission format.

## Repository map and data flow

| Path | Role |
| --- | --- |
| [`config.py`](config.py) | Kaggle settings; change `MODEL_KEY` to select a Hugging Face checkpoint. |
| [`models.py`](models.py) | Checkpoint registry and model-specific multimodal loaders. |
| [`run.py`](run.py) | Kaggle runner **and** shared track discovery, image download/variants, prompt, output, and resume helpers. |
| [`requirements.txt`](requirements.txt) | Kaggle Python dependencies. |
| [`local/config.py`](local/config.py) | Mac settings; change `MODEL_ID` to the exact LM Studio ID. |
| [`local/run_local.py`](local/run_local.py) | Local runner; imports shared helpers from `run.py`. |
| [`local/progress.py`](local/progress.py) | Read-only live progress and rough time estimate for a Mac run. |
| [`local/lmstudio.py`](local/lmstudio.py) | LM Studio model inspection, HTTP inference, and image transport. |
| [`local/requirements.txt`](local/requirements.txt) | Mac Python dependencies, including CPU PyTorch for automatic BERTScore. |
| [`local/README.md`](local/README.md) | Detailed Mac setup and troubleshooting. |
| [`evaluate_run.py`](evaluate_run.py) | Automatic per-run BERTScore, coverage, visual format stats, and review sheet. |
| [`score_saved.py`](score_saved.py) | Score an existing run in a separate process without rerunning inference. |
| [`judge_predictions.py`](judge_predictions.py) | Optional OpenRouter judge for English QA and a combined aggregate summary; run separately. |
| [`compare.py`](compare.py) | Cross-run original-dev comparison for both tasks. |
| [`tests/`](tests/) | Small fixture/mock tests for the pipeline and local client. |
| [`sources.md`](sources.md) | Source links used to develop the project. |

Both runners query Hugging Face for a dataset revision, discover released `qa_*` text tracks, download the appropriate image ZIP once, and stream rows from the pinned revision. They make **separate stateless calls** with the same image: `qa` gets the image and dataset question plus a fixed short-answer instruction; `visual` gets the image and a fixed visual-evidence prompt, with **no dataset question**. The visual prompt asks for visible objects, clothing, symbols, readable text, location clues, and explicitly marked uncertainty in JSON. Neither call receives country, category, subcategory, or reference answer. A record is flushed after each call. After inference, BERTScore F1 is computed on `qa` original-dev outputs and visual descriptions receive format/coverage statistics and a human review sheet.

The visual prompt asks for visual evidence that could help a future two-model system. It is **not** part of the official SemEval Task 2 input or score. The QA call never receives the visual model's output, so its score measures a single-model answer without that extra context. There is no gold visual-description annotation in this dataset; JSON validity or number of details does not measure whether a description is correct. Rate groundedness using `visual_review.csv` before choosing a visual specialist.

## Checkpoints

| `MODEL_KEY` | Original Hugging Face checkpoint | Loader | Practical note |
| --- | --- | --- | --- |
| `internvl3_5_4b` | [OpenGVLab/InternVL3_5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B) | Transformers pipeline | Remote model code |
| `qwen3_vl_8b` | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Transformers pipeline | Recent Transformers required |
| `aya_vision_8b` | [CohereLabs/aya-vision-8b](https://huggingface.co/CohereLabs/aya-vision-8b) | Transformers pipeline | Gated: accept license and use an HF token |
| `culturalpangea_7b` | [neulab/CulturalPangea-7B](https://huggingface.co/neulab/CulturalPangea-7B) | LLaVA-NeXT | Extra install; see below |
| `minicpm_o_4_5` | [openbmb/MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5) | Native image chat | Audio/TTS modules disabled |
| `minicpm_v_4_5` | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) | Native image chat | Image model option for the “o/V” choice |
| `gemma_4_12b` | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | Native multimodal processor | Larger GPU footprint; response parser removes reasoning tags |
| `qwen3_8_27b` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Native multimodal processor | 27B; may exceed free Kaggle GPU memory even in 4-bit |

All loaders aim at the original full checkpoints. `LOAD_IN_4BIT=True` uses bitsandbytes NF4 at runtime to reduce memory; this is not a different published model. Keep this setting identical across runs. Inference is greedy (`do_sample=False`); QA has a 128-token cap and visual description has a separate 512-token cap. A model can still use a different internal image resolution or chat template; those are checkpoint-specific.

**Important limit:** the code paths were checked against the model cards, but these multi-GB checkpoints have **not all been executed on a Kaggle GPU**. A completed smoke run is required before claiming that a checkpoint works on your specific GPU. CulturalPangea has an older custom LLaVA-NeXT dependency and is the most likely to need environment adjustment. The 27B checkpoint may exceed both free Kaggle disk and GPU memory. Model-loading failures are written to `status.json`.

## From scratch on Kaggle

1. Create a Kaggle notebook, choose **GPU** in Notebook settings, and turn **Internet** on. A fresh session for each checkpoint avoids library and memory conflicts. Upload this directory as a Kaggle Dataset or clone your repository into `/kaggle/working`; then `cd` into this project directory.
2. For Aya Vision, accept the access terms on its model page while logged into Hugging Face. Add a Kaggle secret named `HF_TOKEN` containing an HF read token. In a Kaggle notebook cell, expose it with:

   ```python
   from kaggle_secrets import UserSecretsClient
   import os
   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
   ```

3. Install once in the fresh session:

   ```bash
   pip install -r requirements.txt
   ```

   Kaggle already supplies CUDA PyTorch. If the install changes PyTorch, restart the notebook kernel before running inference. Internet is needed to download images and model weights. The runner downloads **images only**; the dataset's audio archives are not needed.

4. In `config.py`, edit the first setting:

   ```python
   MODEL_KEY = "qwen3_vl_8b"
   ```

5. For a quick functional check, temporarily set `MAX_ROWS_PER_TRACK = 2`, then run:

   ```bash
   python run.py
   ```

   Inspect `status.json` and `predictions.jsonl` under `/kaggle/working/mmcqa_runs/<model_key>_<run_id>/`. If the predictions are meaningful, set `MAX_ROWS_PER_TRACK = None` and run again. A different limit creates a **different run directory** so a smoke run cannot silently contaminate the full run.

6. The checked-in selection run uses every released `qa_*` track on **original dev** images: **4,000 images × 2 independent tasks = 8,000 requests** per model. It can still exceed a free Kaggle session. Results flush after every request. Save/download the run directory and image cache to resume in another session. Keep the same settings for every model. For later analysis, adding train and the optional JPEG dev probe would make **96,000 requests** per model on the current release. If interrupted, rerun the same configuration and dataset/model revisions; successful task requests are skipped.

7. For each next checkpoint, change `MODEL_KEY` and repeat in a fresh notebook session. Keep `SPLITS`, `TRACKS`, `VARIANTS`, prompt, token cap, and 4-bit choice fixed for a fair comparison.

### CulturalPangea extra installation

Its [model card](https://huggingface.co/neulab/CulturalPangea-7B) requires [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT). In a fresh Kaggle session, after the normal requirements, install its package without the repository's training dependencies (which pin an old PyTorch version):

```bash
pip install 'git+https://github.com/LLaVA-VL/LLaVA-NeXT.git' --no-deps
pip install shortuuid ftfy open_clip_torch einops-exts timm
```

Then set `MODEL_KEY = "culturalpangea_7b"` and run the two-row smoke check. LLaVA-NeXT's code may need a Transformers version compatible with its checkpoint; its training dependency set and the newer Gemma/Qwen checkpoints are different, so run CulturalPangea in its own fresh session. If import or generation fails, keep the full traceback from `status.json` or `predictions.jsonl`. Do not substitute a text-only Qwen loader for this checkpoint: that would omit its vision encoder.

## From scratch on a 16 GB Mac with LM Studio

The local path uses an **image-capable model already downloaded and loaded in LM Studio**. It sends the same two independent requests to LM Studio's local HTTP server and writes the same JSONL prediction fields as the Kaggle runner. Python PyTorch is now installed **for automatic BERTScore after inference**; the VLM still runs in LM Studio. The local model ID and quantization are recorded separately from the original Hugging Face checkpoints; a local quantized model and its Hugging Face counterpart may give different answers.

1. In LM Studio, load a model with vision support and start its local server, normally at `http://127.0.0.1:1234`. Keep the server running throughout inference. Check LM Studio's memory estimate; a 27B model may not fit a 16 GB Mac.
2. From this repository root, create the Python environment and list the IDs the server reports:

   ```bash
   python3 -m venv .venv-local
   source .venv-local/bin/activate
   python -m pip install -r local/requirements.txt
   python local/run_local.py --list-models
   ```

3. Paste the **exact** vision model ID into `MODEL_ID` in [`local/config.py`](local/config.py), then run:

   ```bash
   python local/run_local.py
   ```

   The first run needs Internet to download the original Hugging Face data and image ZIP. Model inference stays on the Mac. Image/data cache: `local/data/`. Run output: `local/runs/`. Both can become large and are ignored by Git.

4. Inspect `status.json`, `evaluation.json`, and a few image-grounded answers in `predictions.jsonl`. Rerunning the same settings skips successful answers and retries failures. Change only `MODEL_ID` to test the next locally available model; keep splits, track selection, row limit, variants, prompt, and token caps consistent across runs.

**Current checked-in Mac settings:** `MODEL_ID = "google/gemma-4-e4b"`, `SPLITS = ("dev",)`, `VARIANTS = ("original",)`, and `MAX_ROWS_PER_TRACK = 2`. With four current tracks and two tasks, that is a **16-request smoke run**. Set `MAX_ROWS_PER_TRACK = None` for the full dev selection run: **4,000 images × 2 = 8,000 requests**. The Kaggle config currently selects `internvl3_5_4b`, dev/original, and no row limit.

The initial Gemma smoke attempt failed because LM Studio spent the 128-token allowance on `reasoning_content` and returned empty answer text with `finish_reason: "length"`. The local runner now checks the model's capabilities and uses LM Studio's [native chat endpoint](https://lmstudio.ai/docs/developer/rest/chat) with `reasoning: "off"` when supported. It records the endpoint and reasoning mode in the manifest; the native endpoint's seed is `null` because that API has no documented seed control. The old failed run folder is historical, and a changed reasoning setting gets a new run ID. If LM Studio does not expose capability metadata, disable thinking for that model in LM Studio and verify the returned answers. A connection failure or three consecutive failed answers stops the run after saving attempted records. See the [local guide](local/README.md) for further diagnosis.

## Configuration

[`config.py`](config.py) is intentionally small:

| Setting | Default | Meaning |
| --- | --- | --- |
| `MODEL_KEY` | `internvl3_5_4b` | Only line needed to switch models |
| `SPLITS` | `("dev",)` | Released labeled selection split; add train for later work |
| `TRACKS` | `"all"` | Discover all released text QA tracks, including future ones; or provide a tuple |
| `VARIANTS` | `("original",)` | Exact benchmark images; optionally add JPEG quality 85 probe |
| `ROBUSTNESS_SPLITS` | `("dev",)` | Run image variation only on dev |
| `MAX_ROWS_PER_TRACK` | `None` | No row limit; a positive integer is for a smoke run |
| `LOAD_IN_4BIT` | `True` | Quantize for Kaggle memory |
| `MAX_NEW_TOKENS` | `128` | Task QA answer limit |
| `VISUAL_MAX_NEW_TOKENS` | `512` | Image-only description limit |
| `BERTSCORE_BATCH_SIZE` | `8` | Scoring batch size; Mac defaults to `1` |
| `BERTSCORE_DEVICE` | `None` | Auto-select scoring device; Mac defaults to `cpu` |
| `BERTSCORE_MODEL` | `bert-base-multilingual-cased` | Shared multilingual scorer; keep identical across runs |

If enabled, `jpeg_85` recompresses the image in memory. It tests sensitivity to mild image degradation and is **never mixed into the main score**. It does not remove learned knowledge or prove that a model has never seen the image. Choose the model using original **dev** rows and keep any future hidden test data untouched. If you later add train, its scores should not decide the final model if you fine-tune on train.

The Mac equivalents live in [`local/config.py`](local/config.py), with `SERVER_URL` and `REQUEST_TIMEOUT_SECONDS` instead of Kaggle's 4-bit setting. Both runners use the same QA and visual prompts and separate token limits, but model-specific image processors, chat templates, quantizations, and runtime APIs can still differ. Keep that limitation in mind when comparing local and Kaggle results.

## Output contract and resume

Run directories are under `/kaggle/working/mmcqa_runs/` on Kaggle and `local/runs/` on Mac. A completed run contains:

```text
<output_dir>/<model_key>_<run_id>/
  manifest.json       # dataset/model identity and revisions, settings, backend details
  status.json         # preparing, running, evaluating, finished, or error state
  predictions.jsonl  # one JSON object per attempted row, variant, and task
  summary.json        # session and latest counts by task/track/split/variant
  qa_scores.jsonl     # original-dev QA BERTScore P/R/F1 and exact match per row
  evaluation.json     # per-track QA scores and visual coverage/format statistics
  visual_summary.json # visual coverage/format stats even if BERTScore download fails
  visual_review.csv   # image-only outputs and preserved human rating columns
```

Each JSONL row has `schema_version` (currently `2`), `run_id`, `dataset_revision`, `model_key`, `model_repo`, `model_revision`, `track`, `split`, `id`, `variant`, `task` (`qa` or `visual`), `question`, `reference`, `country`, `category`, `subcategory`, `image`, `prediction`, `status`, `error`, `latency_seconds`, and `completed_utc`. Dataset metadata is recorded for analysis but **never sent** in the visual request; its question is also excluded from the visual request. `status` is `ok` or `error`. A later successful retry supersedes an earlier failure for the same `(track, split, id, variant, task)` key. Image bytes are not embedded. If a run stops early, its attempted records remain saved; scoring files may not exist until inference completes.

Runs are named using a fingerprint of dataset and model revision plus all comparison settings. If the remote dataset or model changes, a new run directory is created. For a comparison, verify that the manifests identify the same dataset revision, prompt, row limit, and token cap. Preserve each complete run folder outside ephemeral Kaggle storage; `/kaggle/working` is not a permanent database.

## Compare the results

Put the completed run directories together, then run, for example:

```bash
python compare.py /kaggle/working/mmcqa_runs/* --output comparison.csv
```

For Mac runs, substitute `local/runs/*`, or pass explicit Mac and Kaggle run directories in the same command. A wildcard is appropriate only if every matched directory is a compatible **schema 2** run. `compare.py` requires the **same dataset revision, row limit, both prompts, and both generation caps**, and permits only one run for each unique `model_key`. Compare two-row smoke runs with two-row smoke runs and full runs with full runs. It scores dev only, regardless of whether train was also processed.

The CSV gives QA BERTScore F1 and exact match on the **intersection of successful original-dev IDs**, plus QA/visual coverage, latency, visual JSON validity, detail counts, and any completed human ratings. The coverage columns expose model failures. Read separate language-track results before selecting a model.

Scoring runs automatically after each completed inference run. To repeat scoring after a saved run, without loading the VLM again:

```bash
python score_saved.py score <run_directory>
# or: python score_saved.py score <run_directory>/predictions.jsonl
```

The organizers specify [BERTScore F1 for official ranking](https://mmcultureqa-semeval27.github.io/tasks/), but their current public dataset tree does not expose an evaluation script or exact BERTScore backbone/settings. This pipeline uses `bert-base-multilingual-cased` uniformly for all tracks and records that choice in `evaluation.json`. The earlier `xlm-roberta-large` default crashed on a 16 GB Mac during scoring. Keep `BERTSCORE_MODEL` the same in both config files for cross-platform comparisons; `compare.py` rejects runs with different metric settings. Treat these scores as **local estimates**, not guaranteed leaderboard scores. When the official scorer becomes available, apply it to saved original-dev QA outputs. The [participation page](https://mmcultureqa-semeval27.github.io/participate/) says CodaBench submission format will be posted for the evaluation phase; this JSONL is an internal research format.

Both requirements files include `bert-score`. On Mac, the local runner unloads its LM Studio model after inference, then scores on CPU with batch size one in a separate process to limit memory use. It downloads the multilingual scoring model on first use and may take time. `score_saved.py` can retry any saved run, even with LM Studio closed; it never redoes inference. If its scorer process fails, `status.json` records `evaluation_error` and `predictions.jsonl` stays untouched. Results appear in `evaluation.json` (track means and macro/micro F1), `qa_scores.jsonl` (per-row precision/recall/F1 and exact match), and `visual_summary.json` (format/coverage); visual quality requires human ratings in `visual_review.csv`. Exact match is a strict diagnostic and can undercount correct paraphrases. JSON validity and detail counts for visual responses measure **format and quantity**, not visual correctness. Inspect the image and fill `detail_accuracy_0_3`, `cultural_clue_accuracy_0_3`, and `unsupported_claims_0_3` in `visual_review.csv`: higher is better for the first two, while **lower is better** for unsupported claims. Rate the same image IDs for every model. `compare.py` averages human ratings only over IDs reviewed for all compared models; rerunning the evaluator retains ratings when a prediction is unchanged.

If inference ended before `summary.json` was written, `score_saved.py` still scores the saved successful original-dev QA rows and marks the status `scored_partial`. Such scores cover only the rows already saved; finish the run before using them for a model comparison. On macOS CPU scoring, the evaluator copies the downloaded model weights into ordinary RAM before inference because direct access to the memory-mapped weights caused a native `SIGBUS` crash on the tested Mac.

## Additional English evaluation with an LLM judge

BERTScore can reward a fluent answer containing the wrong landmark or cultural identity. Run the optional root-level evaluator on saved **Mac or Kaggle** predictions to add reference-based factual analysis. It is separate from `run.py`, `local/run_local.py`, and BERTScore scoring; inference runners never call OpenRouter.

The [OASIS paper, Section 3](https://arxiv.org/html/2510.06371v1#S3) reports a GPT-4.1 judge on a 1–10 rubric covering **helpfulness, relevance, accuracy, and faithfulness**. The [SemEval task page](https://mmcultureqa-semeval27.github.io/tasks/) lists LLM analysis as supplementary, outside official ranking. This script uses those four factors with an explicit project rubric and a cheaper judge, so its scores are **an adaptation, not a reproduction of the paper or an official SemEval score**.

### Run from scratch

1. Install the existing requirements and complete BERTScore evaluation first. `requests` is included in both requirements files. Keep `predictions.jsonl` beside `qa_scores.jsonl`, `manifest.json`, and `evaluation.json`. The evaluator can also read BERTScore fields embedded directly in prediction rows, but scorer settings may be unknown if `evaluation.json` is absent.
2. Create an OpenRouter API key. Install dotenv support with `python -m pip install 'python-dotenv>=1.0'` (also included in both requirements files). Copy `.env.example` to `.env` in the **project root**, and set `OPENROUTER_API_KEY=your-key` there. The script loads this file automatically even when launched from another directory. `.env` and `.env.*` secret files are ignored by Git. Alternatively export `OPENROUTER_API_KEY` in your shell; an exported value takes precedence over the default project `.env`. Use `--env-file /path/to/.env` to explicitly select a file and override a stale shell value. Keys are never printed or written to evaluation outputs. For Kaggle, enable Internet, add a Secret named `OPENROUTER_API_KEY`, and initialize the environment in a notebook cell:

   ```python
   import os
   from kaggle_secrets import UserSecretsClient
   os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
   ```

3. From the project root, validate the input without spending credits:

   ```bash
   python judge_predictions.py local/runs/<run_directory>/predictions.jsonl --dry-run
   ```

   Test the credential separately, without paid model calls:

   ```bash
   python judge_predictions.py --check-auth
   # Force the project file if a shell key is stale:
   python judge_predictions.py --check-auth --env-file .env
   ```

   The script reports the key source, not its value, and checks authentication before new judgments. HTTP 401 means an authentication rejection; HTTP 402 reports credit/spending-limit problems. If the key-status check succeeds but chat repeatedly returns `User not found`, the script retries up to three times and reports the inconsistent OpenRouter response. Retry later or contact OpenRouter support; do not hardcode credentials to bypass this. OpenRouter has [documented past infrastructure failures](https://openrouter.ai/blog/announcements/openrouter-outages-on-february-17-and-19-2026/) that produced this message even for valid keys.

4. Try two English records, inspect the ratings, then judge the complete English set:

   ```bash
   python judge_predictions.py local/runs/<run_directory>/predictions.jsonl --max-items 2
   python judge_predictions.py local/runs/<run_directory>/predictions.jsonl
   ```

   For Kaggle, substitute `/kaggle/working/mmcqa_runs/<run_directory>/predictions.jsonl`. GPU access and LM Studio are unnecessary. Only successful **original-dev English QA** rows with a gold reference are sent; Arabic, visual-description, train, and robustness-variant rows are excluded. English is detected by a track suffix `_en`, supporting future regional tracks. Every request contains the question and clearly labeled `ORIGINAL_REFERENCE_ANSWER` and `PREDICTED_MODEL_ANSWER`. Country/category metadata, BERTScore values, model identity, images, and earlier judgments are excluded from judge inputs.

The default judge is `google/gemini-2.5-flash-lite` in `judge_predictions.py`. Its [OpenRouter page](https://openrouter.ai/google/gemini-2.5-flash-lite) listed $0.10/M input tokens and $0.40/M output tokens when checked on September 28, 2026; prices and availability can change. Change the single `JUDGE_MODEL` line or use `--model <openrouter-model-id>`. The chosen model/provider must support strict [JSON structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs). Requests use temperature zero, an initial 600-token output cap, and providers supporting all requested parameters. Truncated-response retries can raise the cap to 1,200 then 2,400 tokens. Invalid JSON/schema responses get validation feedback before a retry; console messages and saved error details include the actual failure reason. Invalid/truncated responses, rate limits, connection errors, and server errors get up to three attempts. Other HTTP failures stop promptly. Retries can consume credits.

### Ratings and two output files

The script creates exactly two output files beside the source predictions:

| File | Contents |
| --- | --- |
| `llm_english_scores.jsonl` | English source records, matched BERTScores, four factor scores, verdict, short explanation/issues, computed final rating, judge identity, response usage, and resume fingerprints. |
| `final_summary.json` | Aggregate statistics only: BERTScore precision/recall/F1 per track, all-track micro F1 and macro F1, exact match, coverage, English means for each LLM factor/final rating, verdict counts/correct rate, BERTScore means by judge verdict, API usage/cost when returned, and comparison metadata. No individual answers or judgments. |

Each factor is **1–10, higher is better**. The judge assigns `correct`, `partially_correct`, `incorrect`, or `unjudgeable`. A wrong core entity or fact cannot earn a high accuracy/helpfulness rating just for having similar wording. The project's **final rating** is computed from the LLM factors: `0.50 × accuracy + 0.25 × faithfulness + 0.15 × relevance + 0.10 × helpfulness`, capped at **3 for incorrect** and **6 for partially correct**. This weighted formula and these caps are project choices, not a published organizer formula. Unjudgeable responses are counted but excluded from score averages and correct rate. Reference faithfulness is assessed from text; the judge cannot independently verify the image or guarantee that the gold reference is exhaustive. Review a sample of its explanations before using the scores to choose a model.

Score/verdict contradictions are normalized conservatively without raising any score or promoting a verdict: an incorrect verdict caps accuracy/helpfulness at 3, a partial verdict caps accuracy at 6, and a low accuracy downgrades an overly optimistic verdict. For example, `partially_correct` with accuracy 7 is recorded with accuracy 6. Adjusted judgments retain `raw_judge_scores` and `score_adjustments`; the summary counts `normalized_judgments`. Integral JSON numbers such as `7.0` are accepted as 7. Missing fields, out-of-range/non-integral ratings, and malformed JSON remain errors. The prompt, factor definitions, final weighting/caps, and already valid cached judgments remain compatible.

Successful judgments are saved immediately. Rerunning with the same judge/rubric skips unchanged successes and retries errors. The JSONL is an append-only log; the latest row for each `(track, split, id, variant, task)` is authoritative. An isolated invalid judge response is saved with `judge_error_details`, and evaluation proceeds to the next item. Three consecutive invalid responses, or an exhausted network/HTTP failure, stop the run and save `state: judge_error`. If evaluation reaches the end with failed judgments, it saves `state: finished_with_judge_errors`. Both cases exit with an error so incomplete coverage is visible; failed rows are excluded from score averages. Rerun the same command to resume. If you change the judge model or rubric, move both judge output files aside first; the script rejects mixing configurations. Source predictions, BERTScore files, and main-run status are untouched.

For later comparisons, check the summary's `run_comparison_settings`, BERTScore `metric_settings`, `judge_fingerprint`, and `comparison_id_hashes`. Require the same prompts, dataset revision, row limits, scorer, judge, and averaged item IDs. If any differ, aggregated means alone cannot produce a fair comparison; use per-record files to compare the shared IDs. BERTScore covers all available original-dev QA tracks; LLM ratings cover **English only**. The summary records missing BERTScores, pending judgments, errors, unjudgeable counts, and any smoke-run limit so reduced coverage stays visible. `api_usage_this_session` includes completed responses from retries when usage was returned; `api_usage_saved_successful_records` totals the stored successful judgments and excludes discarded retry responses. Do not sum those two counters together.

## Notes for interpreting a model choice

- Compare tracks separately. A single overall average can hide poor Arabic variety performance.
- Inspect errors and latency alongside semantic quality; a strong but unstable or too slow checkpoint may be unsuitable for a free GPU workflow.
- Category and country can support **after-the-fact** error analysis, but they are excluded from prompts because the future test set withholds them.
- The published data carries a non-commercial research license. Follow the [dataset and model license terms](https://huggingface.co/datasets/QCRI/MMCQA-SemEval27) and [registration steps](https://mmcultureqa-semeval27.github.io/participate/).
- The [OASIS dataset paper](https://arxiv.org/abs/2510.06371) describes the source of the task data. Cite it if you use the benchmark in course reports.

## Handoff for future agents

- Start with this README, the selected backend's config and runner, and `compare.py`. The root `run.py` also supplies shared functions to the Mac runner.
- Keep dataset revision pinning, both prompt texts, `(track, split, id, variant, task)` resume keys, and JSONL field meanings stable across backends. Increment `SCHEMA_VERSION` if a change makes old records incompatible.
- Keep `original` and `jpeg_85` separately labeled. Use original dev predictions for model selection and the eventual official-like score; use JPEG as a robustness probe only.
- Tests are in `tests/` and can be run with `python -m unittest discover -s tests -v`. They use fixtures and mocks and do **not** prove that all large checkpoints load on Kaggle or Mac.
- The listed Kaggle checkpoints were not all executed in this workspace. The first local Gemma two-row attempt failed before reasoning was disabled; the saved failed run is not a completed comparison. Verify a real smoke run and sample image-grounded answers before starting thousands of requests.
- Preserve actual backend, model ID/revision, quantization metadata, errors, and latency when reporting results. A local quantized model and an original Hugging Face checkpoint should not be described as identical merely because they share a model family name.
