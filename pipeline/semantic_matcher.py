"""
SemanticMatcher — Layer 3: Sentence-Transformer cosine similarity.

Pre-embeds all unique taxonomy values once, then for each unresolved
product text, finds the closest allowed value via cosine similarity.

Optimizations:
  - Embeddings computed once and cached as numpy array
  - Per-(category, feature_name) index into the embedding matrix
  - Batch encodes product texts in chunks for throughput
  - Normalized embeddings → dot product = cosine similarity
"""
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from .taxonomy_engine import TaxonomyEngine

Key = Tuple[str, str]


class SemanticMatcher:
    """Embed taxonomy values, match product text by cosine similarity."""

    def __init__(self, tax: TaxonomyEngine, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.tax = tax
        self.model = SentenceTransformer(model_name)

        # Collect all unique values across the entire taxonomy
        all_values_set: Set[str] = set()
        for key, vals in tax.all_allowed.items():
            all_values_set.update(vals)

        self.value_list: List[str] = sorted(all_values_set)
        self._value_to_idx: Dict[str, int] = {
            v: i for i, v in enumerate(self.value_list)
        }

        # Encode all values (one-time cost: ~16K values → ~2 seconds)
        print(f"[SemanticMatcher] Encoding {len(self.value_list)} taxonomy values...")
        self.value_embeddings: np.ndarray = self.model.encode(
            self.value_list,
            batch_size=512,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # Build per-key index: which rows in value_embeddings are valid
        self._key_indices: Dict[Key, np.ndarray] = {}
        for key, vals in tax.all_allowed.items():
            indices = [self._value_to_idx[v] for v in vals if v in self._value_to_idx]
            if indices:
                self._key_indices[key] = np.array(indices, dtype=np.int32)

    def match_single(
        self,
        text: str,
        category: str,
        feature_name: str,
        threshold: float = 0.72,
    ) -> Tuple[Optional[str], float]:
        """Match a single product text against allowed values for (category, feature_name).

        Returns (best_value, similarity_score) or (None, 0.0).
        """
        key = (category, feature_name)
        indices = self._key_indices.get(key)
        if indices is None or len(indices) == 0:
            return None, 0.0

        text_emb = self.model.encode([text], normalize_embeddings=True)
        candidate_embs = self.value_embeddings[indices]

        # Dot product of normalized vectors = cosine similarity
        sims = candidate_embs @ text_emb[0]
        best_idx = np.argmax(sims)
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            return self.value_list[indices[best_idx]], best_sim
        return None, best_sim

    def match_batch(
        self,
        texts: List[str],
        keys: List[Key],
        threshold: float = 0.72,
    ) -> List[Tuple[Optional[str], float]]:
        """Batch match: encode all texts at once, then match per-key.

        Args:
            texts: Product texts (title or title+desc)
            keys: Corresponding (category, feature_name) per text
            threshold: Minimum cosine similarity to accept

        Returns: List of (value, confidence) tuples
        """
        if not texts:
            return []

        # Batch encode all product texts
        text_embs = self.model.encode(
            texts,
            batch_size=512,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        results = []
        for i, (text_emb, key) in enumerate(zip(text_embs, keys)):
            indices = self._key_indices.get(key)
            if indices is None or len(indices) == 0:
                results.append((None, 0.0))
                continue

            candidate_embs = self.value_embeddings[indices]
            sims = candidate_embs @ text_emb
            best_idx = np.argmax(sims)
            best_sim = float(sims[best_idx])

            if best_sim >= threshold:
                results.append((self.value_list[indices[best_idx]], best_sim))
            else:
                results.append((None, best_sim))

        return results
