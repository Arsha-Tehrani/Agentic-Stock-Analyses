"""
json_repair.py - Shared JSON truncation recovery utilities.

Used by both ToneAnalystNode and RegimeAnalystNode to recover from
LLM responses that are truncated mid-string or missing closing brackets.

Strategy (3-layer defense):
  1. Try raw json.loads()
  2. Attempt structural repair (close brackets/quotes)
  3. Last-ditch regex extraction of individual fields
"""

import json
import re


def repair_truncated_json(text: str) -> str:
    """
    Attempt to repair truncated/malformed JSON by:
    1. Appending missing closing brackets/braces and unterminated strings
    2. Inserting missing commas between adjacent fields
    3. Removing stray trailing commas
    """
    text = text.strip()

    # Remove trailing commas (common in truncated JSON)
    while text.endswith(","):
        text = text[:-1].rstrip()

    # --- Fix missing commas between adjacent fields ---
    # Pattern: a field value followed directly by a field key without a comma
    # e.g. "emotional_score": 0.30"factual_score": 0.50
    # or   "emotional_score": 0.30\n"factual_score": 0.50
    # Match: end-of-value (quote or digit/]}) followed by optional whitespace then a quote
    import re
    text = re.sub(r'(["\d}\]](?:\s*\n\s*))(")', r'\1,\2', text)

    # Count open vs close brackets/braces
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    # If we are inside a string (odd number of unescaped quotes on
    # the last line), try appending a closing quote first.
    quote_count = text.count('"') - text.count('\\"')
    if quote_count % 2 != 0:
        text += '"'

    # Append missing closing brackets/braces (inside-out order)
    text += "]" * open_brackets
    text += "}" * open_braces

    return text


def parse_json_with_repair(text: str) -> dict:
    """Parse JSON string, attempting automatic repair on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    🔧 JSON parse failed ({e}), attempting truncation repair...")
        repaired = repair_truncated_json(text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Last-ditch: extract whatever valid JSON keys we can via regex
            print("    ⚠️  JSON repair failed, extracting partial fields")
            result: dict = {}
            for key in ["score", "reasoning", "emotional_score", "factual_score",
                        "tonality_label", "Macro_Analysis", "Rotation_Analysis",
                        "Emotional_Arbitrage_Analysis",
                        "macro_score", "rotation_score", "emotional_arbitrage_score"]:
                pattern = rf'"{key}"\s*:\s*("[^"]*"|[-+]?\d*\.?\d+)'
                match = re.search(pattern, text)
                if match:
                    val = match.group(1)
                    if val.startswith('"'):
                        result[key] = val.strip('"')
                    else:
                        try:
                            result[key] = float(val) if "." in val else int(val)
                        except ValueError:
                            continue
            return result