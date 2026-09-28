"""Settings for the LM Studio runner on a Mac."""

# Run `python local/run_local.py --list-models`, then paste a vision model ID here.
# MODEL_ID = ""
MODEL_ID = "google/gemma-4-e4b"

SERVER_URL = "http://127.0.0.1:1234"
SPLITS = ("dev",)  # Use ("train", "dev") for every released row.
TRACKS = "all"
VARIANTS = ("original",)  # Add "jpeg_85" only for a separate robustness probe.
ROBUSTNESS_SPLITS = ("dev",)
MAX_ROWS_PER_TRACK = 100  # Use 2 for a quick check.
MAX_NEW_TOKENS = 128
VISUAL_MAX_NEW_TOKENS = 512
REQUEST_TIMEOUT_SECONDS = 600
SEED = 42
BERTSCORE_BATCH_SIZE = 1
BERTSCORE_DEVICE = "cpu"  # Keep scoring off the LM Studio GPU/Metal memory.
BERTSCORE_MODEL = "bert-base-multilingual-cased"  # Same default as Kaggle; fits a 16 GB Mac.
UNLOAD_MODEL_BEFORE_SCORING = True  # Releases LM Studio RAM after saved predictions.

# Paths are relative to the project root, so commands work from any directory.
DATA_DIR = "local/data"
OUTPUT_DIR = "local/runs"
