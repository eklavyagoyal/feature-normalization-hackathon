#!/usr/bin/env python3
"""
run_pipeline.py — Main orchestrator for the Waterfall Pipeline.

Usage:
  # Run on val split (with evaluation):
  python run_pipeline.py --split val

  # Run on test split (generates submission.parquet):
  python run_pipeline.py --split test

  # Run with LLM layer enabled:
  python run_pipeline.py --split test --use-llm

  # Control parallelism:
  python run_pipeline.py --split test --workers 16
"""
import argparse
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline.config import (
    DATA_DIR,
    EMBEDDING_BATCH,
    NUM_WORKERS,
    OUTPUT_DIR,
    SEMANTIC_THRESHOLD,
    SUBMISSION_PATH,
    TAXONOMY_PATH,
    TEST_PROD_PATH,
    TRAIN_FEAT_PATH,
    TRAIN_PROD_PATH,
    VAL_FEAT_PATH,
    VAL_PROD_PATH,
)
from pipeline.taxonomy_engine import TaxonomyEngine
from pipeline.extractor import Extractor
from pipeline.normalizer import snap_categorical, snap_numeric


def load_data(split: str):
    """Load products and (optionally) ground truth for a split."""
    if split == "val":
        products = pd.read_parquet(VAL_PROD_PATH)
        features = pd.read_parquet(VAL_FEAT_PATH)
    elif split == "test":
        products = pd.read_parquet(TEST_PROD_PATH)
        features = None
    elif split == "train":
        products = pd.read_parquet(TRAIN_PROD_PATH)
        features = pd.read_parquet(TRAIN_FEAT_PATH)
    else:
        raise ValueError(f"Unknown split: {split}")
    return products, features


def build_train_lookup() -> dict:
    """Build (feature_name, title) → most_common_value lookup from train."""
    print("[Train Lookup] Building title→value index...")
    t0 = time.time()
    train_prod = pd.read_parquet(TRAIN_PROD_PATH)
    train_feat = pd.read_parquet(TRAIN_FEAT_PATH)

    merged = train_feat.merge(train_prod[["uid", "title"]], on="uid")
    lookup = (
        merged.groupby(["feature_name", "title"])["feature_value"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    print(f"[Train Lookup] {len(lookup):,} entries built in {time.time()-t0:.1f}s")
    return lookup


def build_train_feature_mode() -> dict:
    """Global most-frequent value per feature_name — last-resort fallback."""
    train_feat = pd.read_parquet(TRAIN_FEAT_PATH)
    return (
        train_feat.groupby("feature_name")["feature_value"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )


def run_deterministic_layers(
    working_df: pd.DataFrame,
    tax: TaxonomyEngine,
    extractor: Extractor,
    train_lookup: dict,
    classifiers: dict = None,
    classifiers_v4: dict = None,
) -> pd.DataFrame:
    """Run full waterfall: Train lookup → Deterministic → Ensemble Classifier."""
    print("[Pipeline] Running deterministic extraction...")
    t0 = time.time()

    predictions = []
    confidences = []

    for idx, row in tqdm(working_df.iterrows(), total=len(working_df), desc="Extracting"):
        category = row["category"]
        fn = row["feature_name"]
        ft = row["feature_type"]
        title = str(row["title"])
        desc = str(row.get("description", "") or "")
        allowed = tax.allowed_values(category, fn)

        pred = None
        conf = 0.0

        # ── Layer 0: Train title lookup ───────────────────────
        title_key = (fn, title)
        if title_key in train_lookup:
            candidate = train_lookup[title_key]
            if not allowed or candidate in allowed:
                pred = candidate
                conf = 0.95

        # ── Layer 1: Deterministic extraction ─────────────────
        if pred is None:
            pred, conf = extractor.extract(category, fn, ft, title, desc)

        predictions.append(pred)
        confidences.append(conf)

    working_df["predicted_value"] = predictions
    working_df["confidence"] = confidences

    elapsed = time.time() - t0
    coverage = (working_df["predicted_value"].notna()).mean()
    print(f"[Deterministic] Done in {elapsed:.1f}s. Coverage: {coverage*100:.1f}%")

    # ── Layer 2: Ensemble classifier ──────────────────────────
    if classifiers is not None and len(classifiers) > 0:
        working_df = _run_classifier_layer(working_df, tax, classifiers, classifiers_v4)

    return working_df


def _run_classifier_layer(
    working_df: pd.DataFrame,
    tax: TaxonomyEngine,
    classifiers: dict,
    classifiers_v4: dict = None,
) -> pd.DataFrame:
    """Run V2+V4 ensemble classifiers. Strict ensemble: only override det when both agree."""
    import numpy as np
    from pipeline.classifier import classify

    needs_clf = (working_df["predicted_value"].isna()) | (
        (working_df["confidence"] < 0.90) & (working_df["feature_type"] == "categorical")
    )
    clf_rows = working_df[needs_clf].copy()

    if len(clf_rows) == 0:
        return working_df

    print(f"[Ensemble] Processing {len(clf_rows):,} rows...")
    t0 = time.time()
    updated = 0

    for fn, group in clf_rows.groupby("feature_name"):
        has_v2 = fn in classifiers
        has_v4 = classifiers_v4 is not None and fn in classifiers_v4

        if not has_v2 and not has_v4:
            continue

        # Prepare texts
        texts_plain = (
            group["title"].astype(str) + " " +
            group["description"].fillna("").astype(str).str[:500]
        ).tolist()

        # Get V2 predictions (word n-grams)
        v2_preds = [None] * len(group)
        v2_confs = [0.0] * len(group)
        if has_v2:
            tfidf2, clf2 = classifiers[fn]
            has_pipe = any("|" in w for w in list(tfidf2.vocabulary_.keys())[:50])
            if has_pipe:
                texts_v2 = (
                    group["category"].astype(str).str.replace("_", " ") + " | " +
                    group["title"].astype(str) + " | " +
                    group["description"].fillna("").astype(str).str[:500]
                ).tolist()
            else:
                texts_v2 = texts_plain
            X2 = tfidf2.transform(texts_v2)
            try:
                probs2 = clf2.predict_proba(X2)
                classes2 = clf2.classes_
                for i, (idx, row) in enumerate(group.iterrows()):
                    allowed = tax.allowed_values(row["category"], fn)
                    bp, bv = 0.0, None
                    if allowed:
                        for j, cls in enumerate(classes2):
                            if cls in allowed and probs2[i][j] > bp:
                                bp = float(probs2[i][j])
                                bv = cls
                    else:
                        bj = np.argmax(probs2[i])
                        bv = classes2[bj]
                        bp = float(probs2[i][bj])
                    v2_preds[i] = bv
                    v2_confs[i] = bp
            except:
                pass

        # Get V4 predictions (char n-grams)
        v4_preds = [None] * len(group)
        v4_confs = [0.0] * len(group)
        if has_v4:
            tfidf4, clf4 = classifiers_v4[fn]
            X4 = tfidf4.transform(texts_plain)
            try:
                probs4 = clf4.predict_proba(X4)
                classes4 = clf4.classes_
                for i, (idx, row) in enumerate(group.iterrows()):
                    allowed = tax.allowed_values(row["category"], fn)
                    bp, bv = 0.0, None
                    if allowed:
                        for j, cls in enumerate(classes4):
                            if cls in allowed and probs4[i][j] > bp:
                                bp = float(probs4[i][j])
                                bv = cls
                    else:
                        bj = np.argmax(probs4[i])
                        bv = classes4[bj]
                        bp = float(probs4[i][bj])
                    v4_preds[i] = bv
                    v4_confs[i] = bp
            except:
                pass

        # Combine: strict ensemble (both must agree for override)
        for i, (idx, row) in enumerate(group.iterrows()):
            det_pred = working_df.at[idx, "predicted_value"]
            det_conf = working_df.at[idx, "confidence"]
            ft = row["feature_type"]

            v2p, v2c = v2_preds[i], v2_confs[i]
            v4p, v4c = v4_preds[i], v4_confs[i]

            # Ensemble: pick best candidate
            if v2p and v4p and v2p == v4p:
                ens_pred = v2p
                ens_conf = (v2c + v4c) / 2 + 0.1  # agreement bonus
            elif v2c > v4c:
                ens_pred = v2p
                ens_conf = v2c
            else:
                ens_pred = v4p
                ens_conf = v4c

            if ens_pred is None:
                continue

            both_agree = (v2p and v4p and v2p == v4p)

            if det_pred is None:
                # Unresolved: use ensemble
                working_df.at[idx, "predicted_value"] = ens_pred
                working_df.at[idx, "confidence"] = ens_conf * 0.7
                updated += 1
            elif ft == "numeric":
                pass  # Never override numeric
            elif both_agree and det_conf < 0.90 and ens_conf > 0.5:
                # Override only when BOTH classifiers agree (strict ensemble)
                working_df.at[idx, "predicted_value"] = ens_pred
                working_df.at[idx, "confidence"] = ens_conf * 0.7
                updated += 1

    elapsed = time.time() - t0
    coverage = (working_df["predicted_value"].notna()).mean()
    print(f"[Ensemble] Updated {updated:,} rows in {elapsed:.1f}s. Coverage: {coverage*100:.1f}%")
    return working_df


def run_semantic_layer(
    working_df: pd.DataFrame,
    tax: TaxonomyEngine,
    threshold: float = SEMANTIC_THRESHOLD,
) -> pd.DataFrame:
    """Run Layer 3: Semantic similarity for unresolved rows."""
    unresolved = working_df[working_df["predicted_value"].isna()].copy()
    if len(unresolved) == 0:
        print("[Semantic] No unresolved rows. Skipping.")
        return working_df

    print(f"[Semantic] Processing {len(unresolved):,} unresolved rows...")
    from pipeline.semantic_matcher import SemanticMatcher

    matcher = SemanticMatcher(tax)

    # Batch process
    texts = unresolved["title"].tolist()
    keys = list(zip(unresolved["category"], unresolved["feature_name"]))

    results = []
    batch_size = EMBEDDING_BATCH
    for i in tqdm(range(0, len(texts), batch_size), desc="Semantic matching"):
        batch_texts = texts[i : i + batch_size]
        batch_keys = keys[i : i + batch_size]
        batch_results = matcher.match_batch(batch_texts, batch_keys, threshold)
        results.extend(batch_results)

    # Update working_df
    for (idx, _row), (val, conf) in zip(unresolved.iterrows(), results):
        if val is not None:
            working_df.at[idx, "predicted_value"] = val
            working_df.at[idx, "confidence"] = conf

    new_coverage = (working_df["predicted_value"].notna()).mean()
    print(f"[Semantic] Coverage after semantic layer: {new_coverage*100:.1f}%")
    return working_df


def build_taxonomy_value_freq(tax: TaxonomyEngine) -> dict:
    """Build global frequency of each value per feature_name across all categories.

    Returns: {feature_name: {value: count_of_categories_listing_it}}
    """
    from collections import Counter, defaultdict
    freq = defaultdict(Counter)
    for (cat, fn), vals in tax.all_allowed.items():
        for v in vals:
            freq[fn][v] += 1
    return dict(freq)


def apply_fallbacks(
    working_df: pd.DataFrame,
    tax: TaxonomyEngine,
    train_feature_mode: dict,
) -> pd.DataFrame:
    """Fill remaining nulls with fallback strategies.

    Hierarchy:
      1. Train global mode (if value is in allowed set for this category)
      2. Taxonomy frequency mode (most common value across categories)
      3. Any allowed value (last resort)
    """
    still_null = working_df["predicted_value"].isna()
    n_null = still_null.sum()
    if n_null == 0:
        return working_df

    print(f"[Fallback] Filling {n_null:,} remaining null predictions...")

    tax_freq = build_taxonomy_value_freq(tax)

    for idx in working_df[still_null].index:
        row = working_df.loc[idx]
        fn = row["feature_name"]
        category = row["category"]
        allowed = tax.allowed_values(category, fn)

        # Strategy 1: Train global mode if it's valid for this category
        if fn in train_feature_mode:
            candidate = train_feature_mode[fn]
            if not allowed or candidate in allowed:
                working_df.at[idx, "predicted_value"] = candidate
                working_df.at[idx, "confidence"] = 0.20
                continue

        # Strategy 2: Taxonomy frequency mode — pick the allowed value that
        # appears in the most categories (strongest prior)
        if allowed and fn in tax_freq:
            freq = tax_freq[fn]
            best = max(allowed, key=lambda v: freq.get(v, 0))
            working_df.at[idx, "predicted_value"] = best
            working_df.at[idx, "confidence"] = 0.15
            continue

        # Strategy 3: Any allowed value
        if allowed:
            working_df.at[idx, "predicted_value"] = next(iter(allowed))
            working_df.at[idx, "confidence"] = 0.10

    final_null = working_df["predicted_value"].isna().sum()
    print(f"[Fallback] {final_null} rows still null after fallbacks")
    return working_df


def validate_predictions(working_df: pd.DataFrame, tax: TaxonomyEngine) -> pd.DataFrame:
    """Final validation gate: snap all predictions to valid taxonomy values."""
    print("[Validator] Validating all predictions against taxonomy...")
    corrected = 0

    for idx, row in working_df.iterrows():
        pred = row["predicted_value"]
        if pred is None:
            continue

        category = row["category"]
        fn = row["feature_name"]
        ft = row["feature_type"]
        allowed = tax.allowed_values(category, fn)

        if not allowed:
            continue

        if pred in allowed:
            continue

        # Try snapping
        if ft == "categorical":
            snapped = snap_categorical(pred, allowed, cutoff=0.75)
            if snapped:
                working_df.at[idx, "predicted_value"] = snapped
                corrected += 1
        else:
            snapped = snap_numeric(pred, allowed)
            if snapped != pred:
                working_df.at[idx, "predicted_value"] = snapped
                corrected += 1

    print(f"[Validator] Corrected {corrected:,} predictions via snapping")
    return working_df


def evaluate(working_df: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Evaluate predictions against ground truth. Returns accuracy."""
    merged = working_df[["uid", "feature_name", "predicted_value"]].merge(
        ground_truth[["uid", "feature_name", "feature_value"]],
        on=["uid", "feature_name"],
        how="inner",
    )
    if len(merged) == 0:
        print("[Evaluate] No matching rows found!")
        return 0.0

    exact_match = (merged["predicted_value"] == merged["feature_value"]).mean()
    print(f"\n{'='*60}")
    print(f"  EXACT MATCH ACCURACY: {exact_match:.4f} ({exact_match*100:.2f}%)")
    print(f"  Evaluated on {len(merged):,} rows")
    print(f"{'='*60}\n")

    # Per-feature-type breakdown
    merged_with_type = merged.merge(
        working_df[["uid", "feature_name", "feature_type"]].drop_duplicates(),
        on=["uid", "feature_name"],
    )
    for ft in ["categorical", "numeric"]:
        sub = merged_with_type[merged_with_type["feature_type"] == ft]
        if len(sub) > 0:
            acc = (sub["predicted_value"] == sub["feature_value"]).mean()
            print(f"  {ft:12s}: {acc:.4f} ({len(sub):,} rows)")

    # Per-feature-name top errors
    merged["is_correct"] = merged["predicted_value"] == merged["feature_value"]
    feat_acc = merged.groupby("feature_name")["is_correct"].agg(["mean", "count"])
    feat_acc = feat_acc.sort_values("mean")
    print(f"\n  Worst 15 features:")
    for fn, row in feat_acc.head(15).iterrows():
        print(f"    {fn:30s}: {row['mean']:.3f} ({int(row['count']):>6d} rows)")

    print(f"\n  Best 10 features:")
    for fn, row in feat_acc.tail(10).iterrows():
        print(f"    {fn:30s}: {row['mean']:.3f} ({int(row['count']):>6d} rows)")

    return exact_match


def main():
    parser = argparse.ArgumentParser(description="Feature Normalization Pipeline")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM sniper layer")
    parser.add_argument("--use-semantic", action="store_true", help="Enable semantic layer")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Feature Normalization Pipeline — Split: {args.split}")
    print(f"{'='*60}\n")

    # ── Step 1: Load taxonomy and build engines ───────────────
    t_start = time.time()
    tax = TaxonomyEngine(str(TAXONOMY_PATH))
    print(f"[Taxonomy] {tax.n_keys:,} feature slots across {tax.n_categories:,} categories")

    extractor = Extractor(tax)
    train_lookup = build_train_lookup()
    train_feature_mode = build_train_feature_mode()

    # ── Step 1b: Load ensemble classifiers (V2 word + V4 char) ──
    from pipeline.classifier import CLASSIFIERS_PATH
    import pickle
    v2_path = CLASSIFIERS_PATH.parent / "classifiers_v2.pkl"
    v4_path = CLASSIFIERS_PATH.parent / "classifiers_v4.pkl"

    classifiers = {}
    classifiers_v4 = {}

    if v2_path.exists():
        with open(v2_path, "rb") as f:
            classifiers = pickle.load(f)
        print(f"[Classifier] Loaded {len(classifiers)} V2 classifiers")

    if v4_path.exists():
        with open(v4_path, "rb") as f:
            classifiers_v4 = pickle.load(f)
        print(f"[Classifier] Loaded {len(classifiers_v4)} V4 char-ngram classifiers")

    # ── Step 2: Load products and build working dataframe ─────
    products, ground_truth = load_data(args.split)

    if args.split == "test":
        # Use the submission template which already has the right (uid, feature_name, feature_type) rows
        # CRITICAL: preserve the original index — the evaluator may depend on it
        submission_template = pd.read_parquet(SUBMISSION_PATH)
        original_index = submission_template.index.copy()
        working_df = submission_template.merge(
            products[["uid", "category", "title", "description"]], on="uid"
        )
        # Store original template for output reconstruction
        working_df._original_template = submission_template
        working_df._original_index = original_index
    else:
        # For train/val: build from ground truth
        working_df = ground_truth.merge(
            products[["uid", "category", "title", "description"]], on="uid"
        )

    if args.limit:
        working_df = working_df.head(args.limit)

    print(f"[Data] {len(working_df):,} rows to process")

    # ── Step 3: Run deterministic layers + ensemble classifiers ─
    working_df = run_deterministic_layers(working_df, tax, extractor, train_lookup, classifiers, classifiers_v4)

    # ── Step 4: Run semantic layer (if enabled) ───────────────
    if args.use_semantic:
        working_df = run_semantic_layer(working_df, tax)

    # ── Step 5: Apply fallbacks ───────────────────────────────
    working_df = apply_fallbacks(working_df, tax, train_feature_mode)

    # ── Step 6: Validate all predictions ──────────────────────
    working_df = validate_predictions(working_df, tax)

    # ── Step 7: Evaluate (if ground truth available) ──────────
    if ground_truth is not None:
        accuracy = evaluate(working_df, ground_truth)

    # ── Step 8: Write output ──────────────────────────────────
    output_path = args.output or str(OUTPUT_DIR / "submission.parquet")

    if args.split == "test":
        # CRITICAL: Reconstruct output with EXACT same structure as template
        # The evaluator may match by index position, row order, or join on (uid, feature_name)
        submission_template = pd.read_parquet(SUBMISSION_PATH)

        # Build a lookup from our predictions
        pred_lookup = {}
        for _, row in working_df.iterrows():
            # Use (uid, feature_name) as key; handle duplicates by keeping first
            key = (row["uid"], row["feature_name"])
            if key not in pred_lookup:
                pred_lookup[key] = row.get("predicted_value", "")

        # Fill the template in-place, preserving exact row order and index
        values = []
        for _, row in submission_template.iterrows():
            key = (row["uid"], row["feature_name"])
            val = pred_lookup.get(key, "")
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = ""
            values.append(val)

        submission_template["feature_value"] = values
        submission_template.to_parquet(output_path)
        print(f"\n[Output] Written to {output_path} (preserved template structure)")
        print(f"[Output] {len(submission_template):,} rows, {sum(1 for v in values if v == ''):,} empty")
    else:
        output_df = working_df[["uid", "feature_name", "predicted_value", "feature_type"]].copy()
        output_df = output_df.rename(columns={"predicted_value": "feature_value"})
        output_df["feature_value"] = output_df["feature_value"].fillna("")
        output_df.to_parquet(output_path, index=False)
        print(f"\n[Output] Written to {output_path}")
        print(f"[Output] {len(output_df):,} rows, {output_df['feature_value'].eq('').sum():,} empty")

    print(f"\n[Total time] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
