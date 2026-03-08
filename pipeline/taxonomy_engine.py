"""
TaxonomyEngine — Parses taxonomy.parquet once and builds every lookup
structure the downstream layers need.

Artifacts produced:
  .cat_lookup   : {(category, feature_name): [sorted allowed values]}
  .cat_sets     : {(category, feature_name): {lower_val: canonical_val}}
  .num_units    : {(category, feature_name): dominant_unit}
  .num_allowed  : {(category, feature_name): set(allowed_values)}
  .feature_type : {(category, feature_name): "categorical"|"numeric"}
  .trie         : Aho-Corasick automaton over ALL unique categorical values
  .value_to_keys: {lower_val: [(category, feature_name), ...]}
"""
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

Key = Tuple[str, str]  # (category, feature_name)


def _parse_agg_values(agg_str: str) -> List[str]:
    """Extract values from taxonomy format: {'[val1]','[val2]',...}"""
    if pd.isna(agg_str) or not agg_str:
        return []
    return [m.group(1) for m in re.finditer(r"\[([^\]]*)\]", agg_str)]


def _dominant_unit(values: List[str]) -> Optional[str]:
    """Detect the most common unit from a list of numeric values."""
    unit_counts: Dict[str, int] = defaultdict(int)
    for v in values:
        m = re.match(r"^[\d.,\s\-+/ca.]+\s+(.+)$", v.strip())
        if m:
            unit_counts[m.group(1)] += 1
    if not unit_counts:
        return None
    return max(unit_counts, key=unit_counts.get)


class TaxonomyEngine:
    """One-time parse of taxonomy.parquet → every lookup we need."""

    def __init__(self, taxonomy_path: str):
        df = pd.read_parquet(taxonomy_path)

        # Core lookups
        self.feature_type: Dict[Key, str] = {}
        self.cat_lookup: Dict[Key, List[str]] = {}
        self.cat_sets: Dict[Key, Dict[str, str]] = {}      # lower→canonical
        self.num_units: Dict[Key, Optional[str]] = {}
        self.num_allowed: Dict[Key, Set[str]] = {}
        self.all_allowed: Dict[Key, Set[str]] = {}

        # Reverse index: lowercase value → list of keys it belongs to
        self.value_to_keys: Dict[str, List[Key]] = defaultdict(list)

        # All unique categorical values (for Aho-Corasick / FlashText)
        self._all_cat_values: Set[str] = set()

        for _, row in df.iterrows():
            key: Key = (row["category"], row["feature_name"])
            ft = row["feature_type"]
            vals = _parse_agg_values(row["aggregated_feature_values"])

            self.feature_type[key] = ft
            self.all_allowed[key] = set(vals)

            if ft == "categorical":
                self.cat_lookup[key] = sorted(vals, key=len, reverse=True)
                self.cat_sets[key] = {v.lower(): v for v in vals}
                self._all_cat_values.update(vals)
            else:
                self.num_allowed[key] = set(vals)
                self.num_units[key] = _dominant_unit(vals)

            for v in vals:
                self.value_to_keys[v.lower()].append(key)

        # Build Aho-Corasick automaton for categorical values
        self.trie = self._build_trie()

        # Stats
        self.n_keys = len(self.feature_type)
        self.n_categories = df["category"].nunique()
        self.n_cat_values = len(self._all_cat_values)

    def _build_trie(self):
        """Build Aho-Corasick automaton for fast multi-pattern matching."""
        try:
            import ahocorasick
            A = ahocorasick.Automaton()
            for val in self._all_cat_values:
                # Store lowercase for matching, original for output
                A.add_word(val.lower(), val)
            A.make_automaton()
            return A
        except ImportError:
            # Fallback: no Aho-Corasick, will use sorted-substring
            return None

    def allowed_values(self, category: str, feature_name: str) -> Set[str]:
        return self.all_allowed.get((category, feature_name), set())

    def get_type(self, category: str, feature_name: str) -> Optional[str]:
        return self.feature_type.get((category, feature_name))

    def get_unit(self, category: str, feature_name: str) -> Optional[str]:
        return self.num_units.get((category, feature_name))

    def trie_search(self, text: str) -> List[Tuple[int, str]]:
        """Return all Aho-Corasick matches as (end_pos, original_value)."""
        if self.trie is None:
            return []
        results = []
        for end_idx, val in self.trie.iter(text.lower()):
            results.append((end_idx, val))
        return results
