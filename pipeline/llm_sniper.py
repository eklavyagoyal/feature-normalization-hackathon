"""
LLMSniper — Layer 5: Claude Haiku "Quality Guard".

Only invoked for the bottom 1-3% of rows where all deterministic and
semantic layers failed. Batches products by category to maximize
tokens-per-dollar efficiency.

Cost control:
  - Hard cap on total API calls (LLM_MAX_CALLS from config)
  - Batch 30 products per call
  - Title-only (no descriptions) to minimize input tokens
  - Temperature=0 for deterministic output
  - Strict JSON output format
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

Key = Tuple[str, str]

# ═══════════════════════════════════════════════════════════════════
# THE GOLDEN PROMPT — Hyper-concise, zero-waste, taxonomy-locked
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You extract product attributes from German product catalog titles.

RULES:
1. Output ONLY valid JSON — an array of objects, one per product.
2. Each object has keys matching the requested feature names.
3. For CATEGORICAL features: use ONLY values from the provided allowed list. If unsure, use null.
4. For NUMERIC features: output as "number unit" (e.g. "125 mm", "0.5 l"). Match the unit shown in examples.
5. If a feature cannot be determined from the title, use null.
6. Never invent values. Never explain. JSON only."""


def build_extraction_prompt(
    category: str,
    features: List[Dict[str, Any]],
    products: List[Dict[str, str]],
) -> str:
    """Build the user prompt for a batch of products in one category.

    Args:
        category: Product category name
        features: List of {name, type, allowed_values} dicts
        products: List of {idx, title} dicts

    Returns: User message string
    """
    # Feature definitions
    feat_lines = []
    for f in features:
        if f["type"] == "categorical":
            vals = f["allowed_values"][:50]  # cap to control prompt size
            feat_lines.append(f'  "{f["name"]}" (categorical): {json.dumps(vals, ensure_ascii=False)}')
        else:
            # Show a few examples for unit/format reference
            examples = list(f["allowed_values"])[:5]
            feat_lines.append(f'  "{f["name"]}" (numeric, e.g. {", ".join(examples)})')

    feat_block = "\n".join(feat_lines)

    # Product list
    prod_lines = [f'  {p["idx"]}. {p["title"]}' for p in products]
    prod_block = "\n".join(prod_lines)

    return f"""Category: {category}
Features to extract:
{feat_block}

Products:
{prod_block}

Return JSON array. Object keys = feature names. Example format:
[{{"feature1": "value1", "feature2": "value2"}}, ...]"""


def parse_llm_response(response_text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse LLM JSON response, handling common formatting issues."""
    text = response_text.strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array from markdown code block
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return None


class LLMSniper:
    """Batched Claude Haiku caller for ambiguous products."""

    def __init__(self, tax_engine, batch_size: int = 30, max_calls: int = 10_000):
        self.tax = tax_engine
        self.batch_size = batch_size
        self.max_calls = max_calls
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def process_batch_async(
        self,
        client,
        category: str,
        feature_defs: List[Dict[str, Any]],
        products: List[Dict[str, str]],
        model: str = "claude-haiku-4-5-20251001",
    ) -> List[Dict[str, Any]]:
        """Send one batch to Claude and parse results.

        Args:
            client: anthropic.AsyncAnthropic instance
            category: Product category
            feature_defs: [{name, type, allowed_values}, ...]
            products: [{idx, uid, title}, ...]
            model: Model to use

        Returns: List of {feature_name: value} dicts, one per product
        """
        if self.total_calls >= self.max_calls:
            return [{}] * len(products)

        prompt = build_extraction_prompt(category, feature_defs, products)

        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        self.total_calls += 1
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        parsed = parse_llm_response(response.content[0].text)
        if parsed and len(parsed) == len(products):
            return parsed

        # Fallback: return empty dicts
        return [{}] * len(products)

    def process_batch_sync(
        self,
        client,
        category: str,
        feature_defs: List[Dict[str, Any]],
        products: List[Dict[str, str]],
        model: str = "claude-haiku-4-5-20251001",
    ) -> List[Dict[str, Any]]:
        """Synchronous version for simpler execution."""
        if self.total_calls >= self.max_calls:
            return [{}] * len(products)

        prompt = build_extraction_prompt(category, feature_defs, products)

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        self.total_calls += 1
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        parsed = parse_llm_response(response.content[0].text)
        if parsed and len(parsed) == len(products):
            return parsed

        return [{}] * len(products)

    def cost_report(self) -> Dict[str, Any]:
        """Return cost stats for reproducibility documentation."""
        input_cost = self.total_input_tokens * 0.80 / 1_000_000
        output_cost = self.total_output_tokens * 4.00 / 1_000_000
        return {
            "total_api_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(input_cost + output_cost, 4),
        }
