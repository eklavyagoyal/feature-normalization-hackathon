"""Central configuration — every magic number in one place."""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).resolve().parent.parent / "data"
TAXONOMY_PATH   = DATA_DIR / "taxonomy" / "taxonomy.parquet"
TRAIN_PROD_PATH = DATA_DIR / "train"    / "products.parquet"
TRAIN_FEAT_PATH = DATA_DIR / "train"    / "product_features.parquet"
VAL_PROD_PATH   = DATA_DIR / "val"      / "products.parquet"
VAL_FEAT_PATH   = DATA_DIR / "val"      / "product_features.parquet"
TEST_PROD_PATH  = DATA_DIR / "test"     / "products.parquet"
SUBMISSION_PATH = DATA_DIR / "test"     / "submission.parquet"
OUTPUT_DIR      = Path(__file__).resolve().parent.parent

# ── Layer thresholds ───────────────────────────────────────────────
SEMANTIC_THRESHOLD   = 0.72   # cosine sim cutoff for Layer 3 acceptance
LLM_ESCALATION_RATE  = 0.02   # max fraction of rows sent to LLM
LLM_BATCH_SIZE       = 30     # products per Claude call
LLM_MAX_CALLS        = 10_000 # hard budget cap

# ── Parallelism ────────────────────────────────────────────────────
NUM_WORKERS       = 8         # CPU workers for regex/trie
EMBEDDING_BATCH   = 8192      # texts per sentence-transformer encode call

# ── Sentence-Transformer ──────────────────────────────────────────
ST_MODEL_NAME = "all-MiniLM-L6-v2"
