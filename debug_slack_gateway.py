"""
debug_slack_gateway.py — Lightweight test runner for the Slack Ingestion Gateway.

Tests Gemini classification of sample messages and verifies DB persistence
without needing a live Slack connection.

Usage:
    python3 debug_slack_gateway.py

Token cost: ~5 small Gemini calls. Fast iteration on signal classification.
"""

import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime
from enum import Enum
from typing import Optional

from google import genai
from pydantic import BaseModel, Field

from src.config import (
    GEMINI_API_KEY,
    SLACK_GATEWAY_MODEL,
    SLACK_SIGNAL_TEMPERATURE,
    SLACK_SIGNAL_MAX_TOKENS,
)

# ── Debug config ──────────────────────────────────────────────────────────
TEST_MESSAGES = [
    # TRADE signals
    "Sold 3 shares of Tesla, I want less exposure to the spacex family and Elon. also how can I short SPCX",
    "DILUTE AAPL, trim the position by 1%",
    "ADD NVDA to the portfolio, good entry point",
    
    # THESIS signals
    "I think PLTR is undervalued because their government contracts are expanding rapidly and the AIP platform is gaining enterprise traction. Long term hold.",
    "TSLA's energy storage business could be a massive catalyst. The Megapack deployments are accelerating. Short term momentum play.",
    "QBTS has a quantum moat with their annealing technology. Long term structural thesis as quantum computing adoption grows.",
    
    # NOISE
    "Good morning! How's the market today?",
    "What's the weather like?",
]
WRITE_TO_DB = True          # Set True to persist THESIS records to the database
# ───────────────────────────────────────────────────────────────────────────

DIVIDER = "=" * 60


class SignalType(str, Enum):
    TRADE = "TRADE"
    THESIS = "THESIS"
    NOISE = "NOISE"


class OutboundSlackSignalSchema(BaseModel):
    signal_type: SignalType = Field(description="TRADE, THESIS, or NOISE")
    ticker: str = Field(default="", description="Uppercase clean ticker symbol")
    action: Optional[str] = Field(default=None, description="EXPAND, DILUTE, or ADD")
    thesis_text: Optional[str] = Field(default=None, description="Structural reasoning")
    time_horizon: str = Field(default="LONG_TERM_HOLD", description="SHORT_TERM_MOMENTUM or LONG_TERM_HOLD")


async def classify_message(client: genai.Client, raw_text: str) -> OutboundSlackSignalSchema:
    """Send a single message to Gemini and return the classified signal."""
    prompt = (
        "You are a financial signal classifier for a Slack ingestion gateway.\n\n"
        "Classify the message as TRADE, THESIS, or NOISE.\n"
        "- TRADE: explicit trade command (buy/sell/expand/dilute/add) with ticker\n"
        "- THESIS: research argument about a company with structural reasoning\n"
        "- NOISE: everything else\n\n"
        "Extract: ticker (uppercase), action (EXPAND/DILUTE/ADD for TRADE), "
        "thesis_text (for THESIS), time_horizon (SHORT_TERM_MOMENTUM or LONG_TERM_HOLD).\n\n"
        f"Message: {raw_text}"
    )

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=SLACK_GATEWAY_MODEL,
            contents=prompt,
            config={
                "temperature": SLACK_SIGNAL_TEMPERATURE,
                "max_output_tokens": SLACK_SIGNAL_MAX_TOKENS,
                "response_mime_type": "application/json",
                "response_schema": OutboundSlackSignalSchema,
            },
        )

        if response.text is None:
            return OutboundSlackSignalSchema(signal_type=SignalType.NOISE)

        if hasattr(response, "parsed") and response.parsed is not None:
            signal = response.parsed
        else:
            signal = OutboundSlackSignalSchema.model_validate_json(response.text.strip())

        if signal.ticker:
            signal.ticker = signal.ticker.upper().strip()
        return signal

    except Exception as e:
        print(f"    ⚠️  Classification error: {e}")
        return OutboundSlackSignalSchema(signal_type=SignalType.NOISE)


async def main():
    print(DIVIDER)
    print("🐞 DEBUG SLACK GATEWAY — Test Gemini classification")
    print(f"   Model: {SLACK_GATEWAY_MODEL}")
    print(f"   Temperature: {SLACK_SIGNAL_TEMPERATURE}")
    print(f"   Max tokens: {SLACK_SIGNAL_MAX_TOKENS}")
    print(f"   Test messages: {len(TEST_MESSAGES)}")
    print(f"   Write to DB: {WRITE_TO_DB}")
    print(DIVIDER)

    # ── Initialize Gemini client ──
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set. Aborting.")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    t0 = time.monotonic()

    # ── DB path ──
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "news.db")

    results = []
    for i, msg in enumerate(TEST_MESSAGES, 1):
        print(f"\n  [{i}/{len(TEST_MESSAGES)}] Classifying: {msg[:80]}...")
        t_start = time.monotonic()
        signal = await classify_message(client, msg)
        elapsed = time.monotonic() - t_start

        # Pretty print result
        icon = {"TRADE": "💰", "THESIS": "📝", "NOISE": "🔇"}.get(signal.signal_type.value, "❓")
        print(f"    {icon} Signal: {signal.signal_type.value:<8} | Ticker: {signal.ticker or '—':<6} | "
              f"Action: {signal.action or '—':<8} | Horizon: {signal.time_horizon:<20}")
        if signal.thesis_text:
            print(f"    Thesis: {signal.thesis_text[:120]}...")
        print(f"    ⏱️  {elapsed:.2f}s")

        results.append({
            "message": msg,
            "signal": signal,
            "elapsed": elapsed,
        })

        # ── Persist THESIS signals to DB if enabled ──
        if WRITE_TO_DB and signal.signal_type == SignalType.THESIS and signal.ticker:
            try:
                conn = sqlite3.connect(db_path)
                conn.execute(
                    "INSERT INTO pending_user_theses "
                    "(ticker, core_argument, time_horizon, raw_message, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        signal.ticker.upper(),
                        signal.thesis_text or msg,
                        signal.time_horizon or "LONG_TERM_HOLD",
                        msg,
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
                conn.close()
                print(f"    💾 Persisted to pending_user_theses table")
            except Exception as e:
                print(f"    ⚠️  DB write error: {e}")

        # ── Persist TRADE signals to DB if enabled ──
        if WRITE_TO_DB and signal.signal_type == SignalType.TRADE and signal.ticker and signal.action:
            try:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT portfolio_allocations, version FROM portfolio_state WHERE id = 1"
                ).fetchone()
                if row:
                    allocs = json.loads(row["portfolio_allocations"])
                    version = row["version"]
                    ticker = signal.ticker.upper()
                    action = signal.action

                    # Find and modify the position
                    for sector_data in allocs.get("sectors", {}).values():
                        for holding in sector_data.get("holdings", []):
                            if holding["ticker"].upper() == ticker:
                                if action == "EXPAND":
                                    holding["concentration_percent"] = round(
                                        holding["concentration_percent"] + 1.0, 2
                                    )
                                    sector_data["weight_percent"] = round(
                                        sector_data["weight_percent"] + 1.0, 2
                                    )
                                elif action == "DILUTE":
                                    new_pct = max(0.0, holding["concentration_percent"] - 1.0)
                                    weight_reduction = holding["concentration_percent"] - new_pct
                                    holding["concentration_percent"] = round(new_pct, 2)
                                    sector_data["weight_percent"] = round(
                                        sector_data["weight_percent"] - weight_reduction, 2
                                    )
                                break

                    conn.execute(
                        "UPDATE portfolio_state SET "
                        "portfolio_allocations = ?, version = version + 1, "
                        "updated_by = 'slack_debug' WHERE id = 1",
                        (json.dumps(allocs),),
                    )
                    conn.commit()
                    print(f"    💾 Portfolio updated: {action} {ticker} (v{version + 1})")
                conn.close()
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    print(f"    ⏳ DB locked, retrying after 1s...")
                    await asyncio.sleep(1)
                    # Retry once
                    try:
                        conn = sqlite3.connect(db_path)
                        conn.row_factory = sqlite3.Row
                        row = conn.execute(
                            "SELECT portfolio_allocations, version FROM portfolio_state WHERE id = 1"
                        ).fetchone()
                        if row:
                            allocs = json.loads(row["portfolio_allocations"])
                            version = row["version"]
                            ticker = signal.ticker.upper()
                            action = signal.action
                            for sector_data in allocs.get("sectors", {}).values():
                                for holding in sector_data.get("holdings", []):
                                    if holding["ticker"].upper() == ticker:
                                        if action == "EXPAND":
                                            holding["concentration_percent"] = round(
                                                holding["concentration_percent"] + 1.0, 2
                                            )
                                            sector_data["weight_percent"] = round(
                                                sector_data["weight_percent"] + 1.0, 2
                                            )
                                        elif action == "DILUTE":
                                            new_pct = max(0.0, holding["concentration_percent"] - 1.0)
                                            weight_reduction = holding["concentration_percent"] - new_pct
                                            holding["concentration_percent"] = round(new_pct, 2)
                                            sector_data["weight_percent"] = round(
                                                sector_data["weight_percent"] - weight_reduction, 2
                                            )
                                        break
                            conn.execute(
                                "UPDATE portfolio_state SET "
                                "portfolio_allocations = ?, version = version + 1, "
                                "updated_by = 'slack_debug' WHERE id = 1",
                                (json.dumps(allocs),),
                            )
                            conn.commit()
                            print(f"    💾 Portfolio updated (retry): {action} {ticker} (v{version + 1})")
                        conn.close()
                    except Exception as e2:
                        print(f"    ⚠️  Trade DB retry error: {e2}")
                else:
                    print(f"    ⚠️  Trade DB error: {e}")
            except Exception as e:
                print(f"    ⚠️  Trade DB error: {e}")

    # ── Summary ──
    elapsed_total = time.monotonic() - t0
    trade_count = sum(1 for r in results if r["signal"].signal_type == SignalType.TRADE)
    thesis_count = sum(1 for r in results if r["signal"].signal_type == SignalType.THESIS)
    noise_count = sum(1 for r in results if r["signal"].signal_type == SignalType.NOISE)

    print(f"\n{DIVIDER}")
    print("📋 CLASSIFICATION RESULTS")
    print(DIVIDER)
    print(f"  Total messages:     {len(results)}")
    print(f"  💰 TRADE signals:   {trade_count}")
    print(f"  📝 THESIS signals:  {thesis_count}")
    print(f"  🔇 NOISE signals:   {noise_count}")
    print(f"  ⏱️  Total time:      {elapsed_total:.2f}s")

    # Show accuracy expectations
    print(f"\n  Expected classification (for reference):")
    expected = ["TRADE", "TRADE", "TRADE", "THESIS", "THESIS", "THESIS", "NOISE", "NOISE"]
    for i, (r, exp) in enumerate(zip(results, expected)):
        status = "✅" if r["signal"].signal_type.value == exp else "❌"
        print(f"    {status} [{i+1}] Got '{r['signal'].signal_type.value}' expected '{exp}'")

    # ── Verify DB state if write was enabled ──
    if WRITE_TO_DB:
        try:
            conn = sqlite3.connect(db_path)
            pending = conn.execute("SELECT COUNT(*) FROM pending_user_theses").fetchone()[0]
            print(f"\n  📊 Database: {pending} thesis record(s) in pending_user_theses")
            conn.close()
        except Exception as e:
            print(f"\n  ⚠️  Could not query pending_user_theses table: {e}")

    print(f"\n{DIVIDER}")
    print("🐞 Debug complete. Check classification accuracy above.")
    print(DIVIDER)


if __name__ == "__main__":
    asyncio.run(main())