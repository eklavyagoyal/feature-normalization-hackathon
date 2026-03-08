"""
TF-IDF + SGD classifiers per feature_name.

Trained on train split, used as fallback when deterministic extraction
fails or has low confidence.
"""
import pickle
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


CLASSIFIERS_PATH = Path(__file__).resolve().parent.parent / "classifiers_v2.pkl"


def train_classifiers(
    train_products_path: str,
    train_features_path: str,
    save_path: str = str(CLASSIFIERS_PATH),
    min_samples: int = 30,
    max_classes: int = 1000,
) -> Dict:
    """Train one TF-IDF + SGD classifier per feature_name."""
    print("[Classifier] Loading training data...")
    train_prod = pd.read_parquet(train_products_path)
    train_feat = pd.read_parquet(train_features_path)
    merged = train_feat.merge(train_prod[["uid", "title", "description"]], on="uid")

    classifiers = {}
    feature_groups = merged.groupby("feature_name")

    for fn, group in feature_groups:
        n = len(group)
        n_classes = group["feature_value"].nunique()

        if n < min_samples or n_classes < 2 or n_classes > max_classes:
            continue

        texts = (
            group["title"] + " " + group["description"].fillna("").str[:200]
        ).tolist()
        labels = group["feature_value"].values

        try:
            tfidf = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2), sublinear_tf=True
            )
            X = tfidf.fit_transform(texts)

            clf = SGDClassifier(
                loss="modified_huber",
                max_iter=100,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
            clf.fit(X, labels)

            classifiers[fn] = (tfidf, clf)
        except Exception:
            continue

    print(f"[Classifier] Trained {len(classifiers)} classifiers")

    with open(save_path, "wb") as f:
        pickle.dump(classifiers, f, protocol=4)
    print(f"[Classifier] Saved to {save_path}")

    return classifiers


def load_classifiers(path: str = str(CLASSIFIERS_PATH)) -> Dict:
    """Load pre-trained classifiers from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def classify(
    classifiers: Dict,
    feature_name: str,
    title: str,
    description: str,
    allowed_values: Set[str],
) -> Tuple[Optional[str], float]:
    """Predict feature_value using the trained classifier.

    Returns (predicted_value, confidence) constrained to allowed_values.
    """
    if feature_name not in classifiers:
        return None, 0.0

    tfidf, clf = classifiers[feature_name]
    text = title + " " + (description or "")[:200]
    X = tfidf.transform([text])

    try:
        probs = clf.predict_proba(X)[0]
        classes = clf.classes_

        if allowed_values:
            best_prob = 0.0
            best_val = None
            for i, cls in enumerate(classes):
                if cls in allowed_values and probs[i] > best_prob:
                    best_prob = probs[i]
                    best_val = cls
            return best_val, best_prob
        else:
            best_i = np.argmax(probs)
            return classes[best_i], float(probs[best_i])
    except Exception:
        pred = clf.predict(X)[0]
        if not allowed_values or pred in allowed_values:
            return pred, 0.5
        return None, 0.0
