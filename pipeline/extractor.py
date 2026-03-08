"""
Extractor — The Waterfall Pipeline.

Each extraction method returns (value, confidence) or (None, 0.0).
The waterfall stops at the first confident match.

Layer order:
  1. Structured Description Parser   ("Technische Daten" key:value pairs — broadened)
  2. Special-format extractors        (thread size, scale, drive size)
  3. Aho-Corasick / Trie Matching    (categoricals with specificity-aware ranking)
  4. Regex Numeric Extraction         (with allowed-value preference + position heuristics)
  5. Longest Substring Match          (fallback over title+desc)
  6. Single-value feature             (if taxonomy has exactly one allowed value)
  7. → unresolved (passed to semantic / LLM layers)
"""
import re
from typing import Dict, List, Optional, Set, Tuple

from .normalizer import (
    extract_drive_size,
    extract_scale,
    extract_thread_size,
    normalize_german_decimal,
    snap_categorical,
    snap_numeric,
)
from .taxonomy_engine import TaxonomyEngine

Key = Tuple[str, str]

# ── Dimension keyword mapping for positional extraction ──────────
# "HxBxT 1950x900x480mm" → {H: 1950, B: 900, T: 480}
# "BxTxH 1200x400x748mm" → {B: 1200, T: 400, H: 748}
_DIM_FEATURE_MAP = {
    "Höhe": "H", "Gesamthöhe": "H",
    "Breite": "B", "Gesamtbreite": "B",
    "Tiefe": "T", "Gesamttiefe": "T",
    "Länge": "L",
}

_DIM_ORDER_PATTERNS = [
    # Each tuple: (regex, ordered dimension letters)
    (re.compile(r"HxBxT\s*[:=]?\s*(\d+)\s*[xX×]\s*(\d+)\s*[xX×]\s*(\d+)\s*mm", re.I), ["H", "B", "T"]),
    (re.compile(r"BxTxH\s*[:=]?\s*(\d+)\s*[xX×]\s*(\d+)\s*[xX×]\s*(\d+)\s*mm", re.I), ["B", "T", "H"]),
    (re.compile(r"BxHxT\s*[:=]?\s*(\d+)\s*[xX×]\s*(\d+)\s*[xX×]\s*(\d+)\s*mm", re.I), ["B", "H", "T"]),
    (re.compile(r"LxBxH\s*[:=]?\s*(\d+)\s*[xX×]\s*(\d+)\s*[xX×]\s*(\d+)\s*mm", re.I), ["L", "B", "H"]),
    (re.compile(r"LxB\s*[:=]?\s*(\d+)\s*[xX×]\s*(\d+)\s*mm", re.I), ["L", "B"]),
    (re.compile(r"\(L\s*[xX×]\s*B\)\s*(\d+)\s*mm\s*[xX×]\s*(\d+)\s*mm", re.I), ["L", "B"]),
    (re.compile(r"\(B\s*[xX×]\s*H\)\s*(\d+)\s*mm\s*[xX×]\s*(\d+)\s*mm", re.I), ["B", "H"]),
    # Unlabeled 3-dim with "mm" at end: treat as BxTxH or HxBxT based on position
    (re.compile(r"(\d{3,4})\s*[xX×]\s*(\d{2,4})\s*[xX×]\s*(\d{2,4})\s*mm\b"), ["H", "B", "T"]),
]

# Feature names that are related to screws: "Mx16" patterns
_SCREW_FEATURES = {"Länge", "Durchmesser", "Gewinde-Ø", "Gewindegröße", "Gewindelänge"}


class Extractor:
    """Deterministic extraction engine — Layers 1-2 of the waterfall."""

    def __init__(self, tax: TaxonomyEngine):
        self.tax = tax

        # Pre-compile regex patterns for numeric features with known units
        self._num_patterns: Dict[Key, re.Pattern] = {}
        for key, unit in tax.num_units.items():
            if unit:
                esc = re.escape(unit)
                self._num_patterns[key] = re.compile(
                    rf"(\d+[.,]?\d*)\s*{esc}\b", re.IGNORECASE
                )

        # Feature-name → special extractor dispatch
        self._special_extractors = {
            "Gewinde-Ø": extract_thread_size,
            "Gewindegröße": extract_thread_size,
            "Maßstab": extract_scale,
        }

        # Drive-size feature names
        self._drive_features = {"Größe", "Antriebsgröße", "Antriebsprofil"}

        # Build feature-name keyword aliases for structured desc parsing
        # Includes both Ø↔Durchmesser variants AND domain-specific aliases
        _DOMAIN_ALIASES = {
            "Felgenmaterial": ["Radkörper"],
            "Ladeflächenlänge": ["Ladeflächen-Länge"],
            "Ladeflächenbreite": ["Ladeflächen-Breite"],
            "Für Modell": ["Geeignet für", "Passend für"],
            "Spannutenlänge": ["Spirallänge"],
            "Gewindeausführung": ["Gewindeform"],
            "Körnung": ["Korngröße"],
            "SW von": ["Schlüsselweite"],
            "Dornmaterial": ["Material (Dorn)"],
            "Laufbelag": ["Reifen", "Bandage"],
        }
        self._fn_aliases: Dict[str, List[str]] = {}
        for key in tax.feature_type:
            fn = key[1]
            aliases = [fn]
            if "Ø" in fn:
                aliases.append(fn.replace("Ø", "Durchmesser"))
            if "Durchmesser" in fn:
                aliases.append(fn.replace("Durchmesser", "Ø"))
            if fn in _DOMAIN_ALIASES:
                aliases.extend(_DOMAIN_ALIASES[fn])
            self._fn_aliases[fn] = aliases

    # ── PUBLIC API ────────────────────────────────────────────────

    def extract(
        self,
        category: str,
        feature_name: str,
        feature_type: str,
        title: str,
        description: str,
    ) -> Tuple[Optional[str], float]:
        """Run the full deterministic waterfall. Returns (value, confidence)."""

        key = (category, feature_name)
        allowed = self.tax.all_allowed.get(key, set())
        text = f"{title} {description}"

        # ── Layer 0: Domain-specific rules (highest precision) ──
        val = self._domain_rules(category, feature_name, feature_type, title, text, allowed)
        if val is not None:
            return val, 0.93

        # ── Layer 1: Structured description parsing ───────────
        val = self._parse_structured_desc(feature_name, feature_type, description, allowed)
        if val is not None:
            return val, 0.92

        # ── Layer 2: Special-format extractors ────────────────
        if feature_name in self._special_extractors:
            val = self._special_extractors[feature_name](text)
            if val is not None and val in allowed:
                return val, 0.95

        if feature_name in self._drive_features and feature_type == "numeric":
            val = extract_drive_size(text)
            if val is not None and val in allowed:
                return val, 0.90

        # ── Layer 3: Dimension extraction (HxBxT patterns) ───
        if feature_name in _DIM_FEATURE_MAP and feature_type == "numeric":
            val = self._extract_dimension(text, feature_name, allowed)
            if val is not None:
                return val, 0.90

        # ── Layer 4: Aho-Corasick categorical matching ────────
        if feature_type == "categorical":
            val = self._trie_match(text, key, allowed)
            if val is not None:
                return val, 0.88

        # ── Layer 5: Regex numeric extraction (improved) ──────
        if feature_type == "numeric":
            val = self._regex_numeric_improved(text, title, key, allowed, feature_name)
            if val is not None:
                return val, 0.85

        # ── Layer 6: Brute-force longest substring match ──────
        val = self._longest_substring(text, allowed)
        if val is not None:
            return val, 0.75

        # ── Layer 7: Single-value features (free accuracy) ────
        if len(allowed) == 1:
            return next(iter(allowed)), 0.50

        return None, 0.0

    # ── DOMAIN-SPECIFIC RULES ─────────────────────────────────────

    def _domain_rules(
        self, category: str, feature_name: str, feature_type: str,
        title: str, text: str, allowed: Set[str],
    ) -> Optional[str]:
        """High-precision rules for specific feature patterns discovered via error analysis."""

        # ── Fächeranzahl: "4x5 Fächer" → 4*5=20 Fächer ──────
        if feature_name == "Fächeranzahl":
            m = re.search(r"(\d+)\s*[xX×]\s*(\d+)\s*Fächer", text)
            if m:
                total = int(m.group(1)) * int(m.group(2))
                candidate = f"{total} Fächer"
                if candidate in allowed:
                    return candidate

        # ── Einzelzeichen: extract letter from title ──────────
        if feature_name == "Einzelzeichen":
            # Pattern: Buchstabe "z" or Buchstabe Z
            m = re.search(r'Buchstabe\s*["\']?\s*([A-Za-z0-9])\s*["\']?', text, re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                # Try formats: "Z •", "Z", etc.
                for fmt in [f"{letter} •", f"{letter} ·", letter]:
                    if fmt in allowed:
                        return fmt

        # ── Wärmequelle: "Luft/Wasser" → Luft, "Sole-Wasser" → Sole
        if feature_name == "Wärmequelle":
            text_l = text.lower()
            # Order matters: check most specific first
            if "sole" in text_l and "Sole" in allowed:
                return "Sole"
            if "luft" in text_l and "Luft" in allowed:
                return "Luft"
            if "wasser" in text_l and "Wasser" in allowed:
                return "Wasser"

        # ── Antrieb: map German screw drive terms ─────────────
        if feature_name == "Antrieb" and feature_type == "categorical":
            text_l = text.lower()
            # Order: most specific patterns first
            drive_map = [
                (r"t-profil|t-star|torx|innensechsrund", "Torx"),
                (r"kreuzschlitz\s*phillips|phillips", "Kreuzschlitz (Phillips)"),
                (r"kreuzschlitz\s*pozidriv|pozidriv", "Kreuzschlitz (Pozidriv)"),
                (r"innensechskant|inbus", "Innensechskant"),
                (r"sechskant|außensechskant", "Sechskant"),
                (r"schlitz", "Schlitz"),
            ]
            for pattern, value in drive_map:
                if re.search(pattern, text_l):
                    if value in allowed:
                        return value
                    # Try finding a close match in allowed
                    for av in allowed:
                        if value.lower() in av.lower():
                            return av

        # ── Verpackungseinheit: extract "NNN St." from end of title ──
        if feature_name == "Verpackungseinheit":
            # Common pattern: "... 1000 St." at end of title
            m = re.search(r"(\d+)\s*St\.?\s*$", title)
            if m:
                num = m.group(1)
                for fmt in [f"{num} Stück", f"{num} St."]:
                    if fmt in allowed:
                        return fmt
            # Also try in description
            m = re.search(r"Verpackungseinheit[:\s]+(\d+)\s*(?:Stück|St\.?)", text, re.I)
            if m:
                num = m.group(1)
                for fmt in [f"{num} Stück", f"{num} St."]:
                    if fmt in allowed:
                        return fmt

        # ── Material: prefer compound matches like "Edelstahl (A2)" over "Edelstahl"
        if feature_name == "Material" and feature_type == "categorical":
            text_l = text.lower()
            # Try each allowed value; score by how many of its tokens are in text
            best_val = None
            best_score = 0
            for av in allowed:
                av_l = av.lower()
                if av_l in text_l:
                    # Exact substring — score by length
                    score = len(av) * 10 + 1
                    if score > best_score:
                        best_score = score
                        best_val = av
                else:
                    # Check component matching: "Edelstahl (A2)" — is "Edelstahl" AND "A2" in text?
                    m_compound = re.match(r"^(.+?)\s*\((.+?)\)$", av)
                    if m_compound:
                        base = m_compound.group(1).strip().lower()
                        grade = m_compound.group(2).strip().lower()
                        if base in text_l and grade in text_l:
                            score = len(av) * 10 + 2  # bonus for compound match
                            if score > best_score:
                                best_score = score
                                best_val = av
                    # Check "Stahl 4.8" — is "Stahl" AND "4.8" in text?
                    m_grade = re.match(r"^(.+?)\s+([\d.]+)$", av)
                    if m_grade:
                        base = m_grade.group(1).strip().lower()
                        grade = m_grade.group(2)
                        if base in text_l and grade in text_l:
                            score = len(av) * 10 + 2
                            if score > best_score:
                                best_score = score
                                best_val = av
            if best_val is not None:
                return best_val

        # ── Innen-Ø for gaskets/flanges: first number in "NNN x NNN x N mm" ──
        if feature_name == "Innen-Ø" and "flansch" in text.lower():
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*[xX×]\s*\d+(?:[.,]\d+)?\s*[xX×]?\s*\d*(?:[.,]\d+)?\s*mm", text)
            if m:
                num = normalize_german_decimal(m.group(1))
                candidate = f"{num} mm"
                if candidate in allowed:
                    return candidate

        # ── Korpusfarbe: extract "Korpus RAL7035" → "RAL 7035 Lichtgrau" ──
        if feature_name == "Korpusfarbe" and feature_type == "categorical":
            m = re.search(r"Korpus\s*RAL\s*(\d{4})", text, re.IGNORECASE)
            if m:
                ral_code = m.group(1)
                for av in allowed:
                    if ral_code in av.replace(" ", ""):
                        return av

        # ── Frontfarbe: extract "Front RAL7021" → "RAL 7021 Schwarzgrau" ──
        if feature_name == "Frontfarbe" and feature_type == "categorical":
            m = re.search(r"Front\s*RAL\s*(\d{4})", text, re.IGNORECASE)
            if m:
                ral_code = m.group(1)
                for av in allowed:
                    if ral_code in av.replace(" ", ""):
                        return av

        # ── Bodenausführung: "Sockel" in title → "mit Sockel" ──
        if feature_name == "Bodenausführung" and feature_type == "categorical":
            text_l = text.lower()
            if "sockel" in text_l and "mit Sockel" in allowed:
                return "mit Sockel"
            if "füße" in text_l or "fuß" in text_l or "füssen" in text_l:
                if "mit Füßen" in allowed:
                    return "mit Füßen"

        # ── Säulentyp: HPLC columns — classify by inner diameter ──
        if feature_name == "Säulentyp" and feature_type == "categorical":
            # EC columns: "EC 250/4" = 250mm length / 4mm ID → Analytisch
            # ID ≤ 4.6mm = Analytisch, > 4.6mm = Semipräparativ/Präparativ
            m = re.search(r"EC\s+\d+/(\d+(?:\.\d+)?)", text)
            if m:
                inner_d = float(m.group(1))
                if inner_d <= 4.6 and "Analytisch" in allowed:
                    return "Analytisch"
                elif inner_d > 4.6 and inner_d <= 10 and "Semipräparativ" in allowed:
                    return "Semipräparativ"
            # Guard/Vorsäule columns
            if re.search(r"Vorsäule|Guard", text, re.IGNORECASE):
                if "Vorsäule" in allowed:
                    return "Vorsäule"

        # ── DxL pattern: "diameter x length" for screws, dübel, fasteners ──
        # Applies to MULTIPLE features: Länge=2nd number, Durchmesser=1st number
        # Patterns: "3.5x35mm", "Ø10x140mm", "M6X40", "6 x 30"
        _DXL_FEATURES = {
            "Länge": 1, "Gesamtlänge": 1, "Gewindelänge": 1,  # 2nd number = length
            "Durchmesser": 0, "Schaft-Ø": 0,  # 1st number = diameter
        }
        if feature_name in _DXL_FEATURES and feature_type == "numeric":
            text_norm = normalize_german_decimal(text)
            pos = _DXL_FEATURES[feature_name]  # 0=first, 1=second
            # Look for NxN pattern (with optional Ø or M prefix)
            m = re.search(r"(?:Ø|ø)?\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)", text_norm)
            if m:
                val_str = m.group(pos + 1)
                unit = self.tax.num_units.get((category, feature_name))
                if not unit:
                    unit = "mm"
                candidate = f"{val_str} {unit}"
                if candidate in allowed:
                    return candidate
                # Try int version
                try:
                    fval = float(val_str)
                    if fval == int(fval):
                        candidate_int = f"{int(fval)} {unit}"
                        if candidate_int in allowed:
                            return candidate_int
                except: pass

        # ── Breite for labels/etiketten: "WxH" format ──
        # "50x25mm" → Breite=50 (first), Höhe=25 (second)
        # "19x64mm" → Breite=19 (first, shorter dim), Höhe/Länge=64
        if feature_name == "Breite" and feature_type == "numeric":
            text_norm = normalize_german_decimal(text)
            # For etiketten/labels, BxH or WxH format
            if re.search(r"etikett|label|aufkleber|schild", text, re.IGNORECASE):
                m = re.search(r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*mm", text_norm)
                if m:
                    # First number is typically the width for labels
                    w_str = m.group(1)
                    unit = self.tax.num_units.get((category, feature_name), "mm")
                    candidate = f"{w_str} {unit or 'mm'}"
                    if candidate in allowed:
                        return candidate

        # ── Spannbereich/Schneidbereich: extract range patterns ──
        if feature_name in ("Spannbereich von", "Spannbereich bis", "Schneidbereich min.", "Schneidbereich max."):
            text_norm = normalize_german_decimal(text)
            # Look for range pattern: "22-28mm" or "22 - 28 mm"
            m = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)\b", text_norm)
            if m:
                v1, v2, unit = m.group(1), m.group(2), m.group(3)
                if "von" in feature_name or "min" in feature_name.lower():
                    candidate = f"{v1} {unit}"
                else:
                    candidate = f"{v2} {unit}"
                if candidate in allowed:
                    return candidate

        return None

    # ── PRIVATE METHODS ───────────────────────────────────────────

    def _parse_structured_desc(
        self, feature_name: str, feature_type: str, description: str, allowed: Set[str]
    ) -> Optional[str]:
        """Parse key:value sections in description.

        Strategy: extract ALL key:value pairs from description first,
        then match any key against feature name aliases.
        """
        if not description:
            return None

        # Extract ALL key:value pairs from description at once
        # Separator-bounded: "key: value" between · or • or <br> or newline
        all_pairs = re.findall(
            r"([^·•<>\n]{2,60}?)\s*:\s*([^·•<>\n]{1,100})", description
        )
        if not all_pairs:
            return None

        # Build set of search terms for this feature name
        aliases = self._fn_aliases.get(feature_name, [feature_name])
        search_terms = set()
        for alias in aliases:
            search_terms.add(alias.lower())
            # Also add without hyphens/special chars
            cleaned = alias.lower().replace("-", " ").replace("ø", "durchmesser")
            search_terms.add(cleaned)

        # Search all pairs for a key match
        for k, v in all_pairs:
            k_clean = k.strip().lstrip("- ").strip()
            k_lower = k_clean.lower()

            # Check if any alias matches this key
            matched = False
            for term in search_terms:
                if term in k_lower or k_lower in term:
                    if len(term) >= 3 and len(k_lower) >= 3:  # avoid trivial matches
                        matched = True
                        break

            if not matched:
                continue

            # Found a matching key — try to extract the value
            raw_val = v.strip().rstrip(".")
            raw_val = normalize_german_decimal(raw_val)

            # Direct match against allowed values
            if raw_val in allowed:
                return raw_val

            # For numeric: try extracting number+unit
            if feature_type == "numeric":
                cleaned = self._clean_numeric_from_desc(raw_val, allowed)
                if cleaned:
                    return cleaned

            # For categorical: try snap
            snapped = snap_categorical(raw_val, allowed, cutoff=0.85)
            if snapped:
                return snapped

            # Try: allowed value contained in desc value
            for av in sorted(allowed, key=len, reverse=True):
                if av in raw_val or av.lower() in raw_val.lower():
                    return av

        return None

    def _clean_numeric_from_desc(self, raw_val: str, allowed: Set[str]) -> Optional[str]:
        """Extract a clean numeric value from a description snippet."""
        raw_val = normalize_german_decimal(raw_val.strip())
        # Try direct
        if raw_val in allowed:
            return raw_val
        # Try "number unit" extraction
        m = re.match(r"^([\d.]+\s+\S+)", raw_val)
        if m and m.group(1) in allowed:
            return m.group(1)
        # Try just the first number+unit token
        m = re.match(r"^([\d.]+)\s*(\S+)", raw_val)
        if m:
            candidate = f"{m.group(1)} {m.group(2)}"
            if candidate in allowed:
                return candidate
        return None

    def _extract_dimension(
        self, text: str, feature_name: str, allowed: Set[str]
    ) -> Optional[str]:
        """Extract from dimension patterns like HxBxT 1950x900x480mm."""
        target_letter = _DIM_FEATURE_MAP.get(feature_name)
        if not target_letter:
            return None

        text_norm = normalize_german_decimal(text)
        for pattern, dim_letters in _DIM_ORDER_PATTERNS:
            m = pattern.search(text_norm)
            if m:
                groups = m.groups()
                for i, letter in enumerate(dim_letters):
                    if letter == target_letter and i < len(groups):
                        candidate = f"{groups[i]} mm"
                        if candidate in allowed:
                            return candidate
                        # If not in allowed, still return (numeric can be novel)
                        if allowed:
                            # Check if the unit matches what's in allowed
                            sample_unit = None
                            for av in allowed:
                                if "mm" in av:
                                    sample_unit = "mm"
                                    break
                            if sample_unit == "mm":
                                return candidate
        return None

    def _trie_match(
        self, text: str, key: Key, allowed: Set[str]
    ) -> Optional[str]:
        """Categorical matching with specificity-aware ranking.

        Fix: prefer "Edelstahl (A2)" over "Edelstahl" when both match,
        by checking if the more-specific value's extra terms also appear.
        """
        text_lower = text.lower()
        allowed_lower = {v.lower(): v for v in allowed}

        if self.tax.trie is not None:
            hits = self.tax.trie_search(text)
            # Collect all matches that are in our allowed set
            matches = []
            for _end_pos, original_val in hits:
                if original_val.lower() in allowed_lower:
                    matches.append(allowed_lower[original_val.lower()])
        else:
            matches = [v for v in allowed if v.lower() in text_lower]

        if not matches:
            return None

        if len(matches) == 1:
            return matches[0]

        # Specificity ranking: prefer the value where ALL tokens are in the text
        best = None
        best_score = -1
        for v in matches:
            # Score = number of tokens in the value that appear in text
            tokens = v.lower().split()
            token_hits = sum(1 for t in tokens if t in text_lower)
            # Bonus for longer values (more specific)
            score = token_hits * 1000 + len(v)
            if score > best_score:
                best_score = score
                best = v

        return best

    def _regex_numeric_improved(
        self, text: str, title: str, key: Key, allowed: Set[str],
        feature_name: str,
    ) -> Optional[str]:
        """Improved numeric extraction with allowed-value preference.

        Key improvements over naive regex:
        1. If a candidate matches an allowed value, return it immediately
        2. If multiple candidates exist, prefer those in allowed set
        3. Use title-only matching to reduce noise from description
        4. For screw patterns (MxL), use positional heuristics
        """
        unit = self.tax.num_units.get(key)
        if not unit:
            return None

        pattern = self._num_patterns.get(key)
        if pattern is None:
            return None

        # Strategy 1: Search title first (higher signal density)
        title_norm = normalize_german_decimal(title)
        title_matches = pattern.findall(title_norm)

        # Strategy 2: Search full text
        text_norm = normalize_german_decimal(text)
        all_matches = pattern.findall(text_norm)

        if not all_matches:
            return None

        # Build candidate list
        all_candidates = []
        for num_str in all_matches:
            num_str = num_str.replace(",", ".")
            candidate = f"{num_str} {unit}"
            in_title = num_str in [m.replace(",", ".") for m in title_matches]
            in_allowed = candidate in allowed
            all_candidates.append({
                "value": candidate,
                "num_str": num_str,
                "in_title": in_title,
                "in_allowed": in_allowed,
            })

        # Priority 1: In allowed set AND in title → highest confidence
        for c in all_candidates:
            if c["in_allowed"] and c["in_title"]:
                return c["value"]

        # Priority 2: In allowed set (from anywhere in text)
        for c in all_candidates:
            if c["in_allowed"]:
                return c["value"]

        # Priority 3: In title (even if not in allowed — numeric can be novel)
        title_candidates = [c for c in all_candidates if c["in_title"]]
        if title_candidates:
            # For screw patterns like "M2 16 mm": the LAST mm-value in title is usually Länge
            # The first is usually Durchmesser
            if feature_name in ("Länge", "Gesamtlänge", "Klingenlänge", "Schneidenlänge", "Gewindelänge"):
                return title_candidates[-1]["value"]
            elif feature_name in ("Durchmesser", "Schneiden-Ø", "Bohrungs-Ø", "Schaft-Ø"):
                return title_candidates[0]["value"]
            # Default: return first title match
            return title_candidates[0]["value"]

        # Priority 4: First match from full text
        return all_candidates[0]["value"]

    def _longest_substring(
        self, text: str, allowed: Set[str]
    ) -> Optional[str]:
        """Brute-force: find the longest allowed value that appears in text."""
        text_lower = text.lower()
        sorted_vals = sorted(allowed, key=len, reverse=True)
        for v in sorted_vals:
            if v.lower() in text_lower:
                return v
        return None
