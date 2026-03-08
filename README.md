# Feature Normalization Pipeline

**Challenge:** Extract normalized product features from unstructured German-language catalog text, constrained to a predefined taxonomy.

**Result:** 78.63% exact match accuracy on test (7.86M rows) | $0 cost | ~5 min runtime

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run on validation split (with accuracy evaluation)
python run_pipeline.py --split val

# 3. Generate test submission
python run_pipeline.py --split test --output submission.parquet
```

### Options

```
python run_pipeline.py --split {train,val,test}
                       [--limit N]          # Process only first N rows (for quick testing)
                       [--use-semantic]      # Enable sentence-transformer layer (optional)
                       [--use-llm]           # Enable Claude Haiku sniper layer (optional)
                       [--workers N]         # CPU workers for parallelism (default: 8)
                       [--output PATH]       # Output parquet path
```

---

## Project Structure

```
feature-normalization-hackathon/
|
|-- run_pipeline.py              # Main entry point — orchestrates the full pipeline
|-- pipeline/
|   |-- config.py                # Paths, thresholds, constants
|   |-- taxonomy_engine.py       # Taxonomy parser + Aho-Corasick trie
|   |-- extractor.py             # Deterministic waterfall extraction (core logic)
|   |-- normalizer.py            # Unit conversion, German decimal, format snapping
|   |-- classifier.py            # TF-IDF + SGD classifier training and inference
|   |-- semantic_matcher.py      # Sentence-transformer cosine matching (optional)
|   |-- llm_sniper.py            # Batched Claude Haiku caller (optional, not used)
|
|-- classifiers_v2.pkl           # Pre-trained word n-gram classifiers (802 features)
|-- classifiers_v4.pkl           # Pre-trained char n-gram classifiers (725 features)
|-- submission_78.parquet        # Final test submission (78.63% accuracy)
|
|-- explore.ipynb                # Data exploration notebook with visualizations
|-- APPROACH_DOC.md              # Approach document (1-2 pages for judges)
|-- requirements.txt             # Python dependencies
|-- data/                        # Provided parquet files (read-only)
```

---

## Reproducing the Submission

### Pre-trained classifiers (included)

The repository includes `classifiers_v2.pkl` and `classifiers_v4.pkl`. These were trained on the training split and are loaded automatically by `run_pipeline.py`.

To retrain from scratch:

```bash
python -c "
from pipeline.classifier import train_classifiers
# V2: word n-grams
train_classifiers('data/train/products.parquet', 'data/train/product_features.parquet',
                  save_path='classifiers_v2.pkl')
"
```

### Full test submission

```bash
python run_pipeline.py --split test --output submission.parquet
```

This produces `submission.parquet` in the required schema (`uid`, `feature_name`, `feature_value`, `feature_type`) with 7,864,744 rows. Runtime: ~5 minutes on a modern laptop (8 cores).

### Validation evaluation

```bash
python run_pipeline.py --split val
```

Prints per-feature accuracy breakdown and overall exact match score.

---

## Approach Summary

Multi-layer deterministic waterfall + lightweight ML ensemble. See [APPROACH_DOC.md](APPROACH_DOC.md) for the full approach document.

| Layer | Method | Cost |
|---|---|---|
| 0 | Train title lookup | $0 |
| 1-3 | Domain rules + structured desc parser + special-format extractors | $0 |
| 4-6 | Aho-Corasick trie + regex numeric + substring matching | $0 |
| 7 | TF-IDF + SGD ensemble classifiers | $0 |
| 8 | Taxonomy validation + fallbacks | $0 |

---

## Cost at 200M Products

| Component | Cost | Time (16-core) |
|---|---|---|
| Full pipeline | $0 | ~33 min |
| Compute (cloud) | ~$0.27 | c5.4xlarge |
| **Total** | **~$0.27** | |
