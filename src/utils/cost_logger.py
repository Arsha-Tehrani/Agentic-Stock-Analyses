"""
cost_logger.py – Lightweight Gemini token-usage tracking for the pipeline.

Reads usage_metadata off each generate_content() response and accumulates
per-node token totals for the current process, so the actual cost driver
behind a run can be inspected directly instead of estimated from a monthly
bill. Call log_gemini_usage() right after each Gemini call, then
print_run_summary() once at the end of the run.
"""

from collections import defaultdict

_totals = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "output_tokens": 0})


def log_gemini_usage(node: str, model: str, response) -> None:
    """Record token usage from a Gemini generate_content() response."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
    output_tokens = getattr(usage, "candidates_token_count", None) or 0
    bucket = _totals[(node, model)]
    bucket["calls"] += 1
    bucket["prompt_tokens"] += prompt_tokens
    bucket["output_tokens"] += output_tokens
    print(f"    💰 [{node}] {model} — in={prompt_tokens} out={output_tokens} tokens")


def print_run_summary() -> None:
    """Print a per-node token usage breakdown for everything logged this run."""
    if not _totals:
        return
    print("\n" + "=" * 60)
    print("💰 Gemini Token Usage Summary (this run)")
    print("=" * 60)
    grand_calls = grand_prompt = grand_output = 0
    for (node, model), bucket in sorted(_totals.items()):
        total = bucket["prompt_tokens"] + bucket["output_tokens"]
        print(
            f"  {node:22s} {model:28s} calls={bucket['calls']:3d}  "
            f"in={bucket['prompt_tokens']:7d}  out={bucket['output_tokens']:7d}  total={total:7d}"
        )
        grand_calls += bucket["calls"]
        grand_prompt += bucket["prompt_tokens"]
        grand_output += bucket["output_tokens"]
    print("-" * 60)
    print(
        f"  {'TOTAL':51s} calls={grand_calls:3d}  "
        f"in={grand_prompt:7d}  out={grand_output:7d}  total={grand_prompt + grand_output:7d}"
    )
    print("=" * 60)
