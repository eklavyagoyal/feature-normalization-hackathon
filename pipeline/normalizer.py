"""
Normalizer — Unit conversion, format standardization, and value snapping.

Handles:
  - "500 ml" → "0.5 l"   (unit conversion within families)
  - "0,5"   → "0.5"      (German decimal comma)
  - "M2"    → "M 2"      (spacing normalization)
  - "1:10"  → ":10"      (scale format)
  - Snapping predicted values to nearest allowed taxonomy value
"""
import difflib
import re
from typing import Dict, Optional, Set

# ── Unit conversion families ──────────────────────────────────────
UNIT_FAMILIES: Dict[str, Dict[str, float]] = {
    # Length
    "mm":  {"mm": 1, "cm": 0.1, "m": 0.001},
    "cm":  {"mm": 10, "cm": 1, "m": 0.01},
    "m":   {"mm": 1000, "cm": 100, "m": 1},
    # Volume
    "ml":  {"ml": 1, "cl": 0.1, "l": 0.001},
    "cl":  {"ml": 10, "cl": 1, "l": 0.01},
    "l":   {"ml": 1000, "cl": 100, "l": 1},
    # Weight
    "g":   {"g": 1, "kg": 0.001},
    "kg":  {"g": 1000, "kg": 1},
    # Power
    "W":   {"W": 1, "kW": 0.001},
    "kW":  {"W": 1000, "kW": 1},
    # Electrical
    "mAh": {"mAh": 1, "Ah": 0.001},
    "Ah":  {"mAh": 1000, "Ah": 1},
    "mV":  {"mV": 1, "V": 0.001},
    "V":   {"mV": 1000, "V": 1},
}


def _format_number(n: float) -> str:
    """Format a number: no trailing zeros, but preserve meaningful decimals."""
    if n == int(n) and abs(n) < 1e12:
        return str(int(n))
    # Up to 4 decimal places, strip trailing zeros
    s = f"{n:.4f}".rstrip("0").rstrip(".")
    return s


def convert_unit(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    """Convert between units in the same family."""
    if from_unit == to_unit:
        return value
    family = UNIT_FAMILIES.get(from_unit)
    if family and to_unit in family:
        return value * family[to_unit]
    return None


def normalize_german_decimal(text: str) -> str:
    """Convert German comma decimal to dot: '0,5' → '0.5'."""
    return re.sub(r"(\d),(\d)", r"\1.\2", text)


def parse_numeric_from_text(text: str, target_unit: str) -> Optional[str]:
    """Extract a numeric value with matching unit from text.

    Returns the normalized string (e.g. '125 mm') or None.
    """
    text = normalize_german_decimal(text)
    esc_unit = re.escape(target_unit)

    # Direct match: "number unit"
    pattern = rf"(\d+\.?\d*)\s*{esc_unit}\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        # Return all candidates; caller picks the right one
        return [f"{m} {target_unit}" for m in matches]

    # Try converting from related units
    for src_unit, family in UNIT_FAMILIES.items():
        if target_unit in family and src_unit != target_unit:
            esc_src = re.escape(src_unit)
            src_pattern = rf"(\d+\.?\d*)\s*{esc_src}\b"
            src_matches = re.findall(src_pattern, text, re.IGNORECASE)
            for m in src_matches:
                converted = convert_unit(float(m), src_unit, target_unit)
                if converted is not None:
                    return [f"{_format_number(converted)} {target_unit}"]

    return None


def snap_categorical(predicted: str, allowed: Set[str], cutoff: float = 0.8) -> Optional[str]:
    """Snap a predicted categorical value to the nearest allowed value.

    1. Exact match
    2. Case-insensitive match
    3. Fuzzy match (difflib) with cutoff
    """
    if predicted in allowed:
        return predicted

    # Case-insensitive
    lower_map = {v.lower(): v for v in allowed}
    if predicted.lower() in lower_map:
        return lower_map[predicted.lower()]

    # Fuzzy
    close = difflib.get_close_matches(predicted, list(allowed), n=1, cutoff=cutoff)
    return close[0] if close else None


def snap_numeric(predicted: str, allowed: Set[str]) -> Optional[str]:
    """Snap a predicted numeric value to the allowed set if possible.

    Numeric values CAN be novel (not in taxonomy examples), so we're
    more lenient here — we just validate format.
    """
    if predicted in allowed:
        return predicted

    # Try case/space normalization
    normalized = re.sub(r"\s+", " ", predicted.strip())
    if normalized in allowed:
        return normalized

    # Not in allowed set, but might still be valid if format is correct
    return predicted


# ── Special-format extractors ──────────────────────────────────────

def extract_thread_size(text: str) -> Optional[str]:
    """Extract thread sizes: M2, M12 → 'M 2', 'M 12'."""
    text = normalize_german_decimal(text)
    m = re.search(r"\bM\s*(\d+(?:\.\d+)?)\b", text)
    if m:
        return f"M {m.group(1)}"
    return None


def extract_scale(text: str) -> Optional[str]:
    """Extract scale: 1:10, Maßstab 1/10 → ':10'."""
    m = re.search(r"1\s*[:/]\s*(\d+)", text)
    if m:
        return f":{m.group(1)}"
    return None


def extract_drive_size(text: str) -> Optional[str]:
    """Extract drive sizes: PZ1, PH2, T25 → 'PZ 1', 'PH 2', 'T 25'."""
    m = re.search(r"\b(PZ|PH|TX|T)\s*(\d+)\b", text, re.IGNORECASE)
    if m:
        prefix = m.group(1).upper()
        if prefix == "TX":
            prefix = "T"
        return f"{prefix} {m.group(2)}"
    return None


def extract_with_prefix(text: str, prefix: str, unit: str) -> Optional[str]:
    """Extract values with prefix: 'ca. 700 g' pattern."""
    esc = re.escape(prefix)
    esc_unit = re.escape(unit)
    m = re.search(rf"{esc}\s*(\d+[.,]?\d*)\s*{esc_unit}", text)
    if m:
        num = normalize_german_decimal(m.group(1))
        return f"{prefix}{num} {unit}"
    return None
