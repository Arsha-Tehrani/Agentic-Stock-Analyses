"""
slack_gateway.py – Slack Ingestion Gateway

Provides a production-grade asynchronous Slack bot using Socket Mode
(slack_bolt.app.async_app.AsyncApp) that:

1. Listens to incoming direct messages or channel mentions.
2. Sends the unformatted raw message string to gemini-2.5-flash using
   native structured output mode (response_mime_type="application/json").
3. Routes classified signals:
   - "TRADE"  → Directly mutates the live portfolio_allocations in SQLite
   - "THESIS" → Stores structured record in pending_user_theses staging table
   - "NOISE"  → Logs and discards
4. Replies to Slack with a confirmation / ephemeral message.

Usage:
    # Set environment variables
    export SLACK_BOT_TOKEN="xoxb-..."
    export SLACK_APP_TOKEN="your-app-token"
    export GEMINI_API_KEY="..."

    python -c "from src.utils.slack_gateway import SlackIngestionGateway; \
    import asyncio; asyncio.run(SlackIngestionGateway().start())"

Or run directly:
    python -m src.utils.slack_gateway
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from enum import Enum
from typing import Optional

from google import genai
from pydantic import BaseModel, Field

from src.config import (
    GEMINI_API_KEY,
    SLACK_BOT_TOKEN,
    SLACK_APP_TOKEN,
    SLACK_CHANNEL_ID,
    SLACK_GATEWAY_MODEL,
    SLACK_SIGNAL_TEMPERATURE,
    SLACK_SIGNAL_MAX_TOKENS,
)

# Resolve the shared database path relative to the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_THESES_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "news.db")

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Schemas
# =============================================================================


class SignalType(str, Enum):
    """Classification of an incoming Slack signal."""
    TRADE = "TRADE"
    THESIS = "THESIS"
    NOISE = "NOISE"


class OutboundSlackSignalSchema(BaseModel):
    """
    Target schema for Gemini structured-output classification of Slack messages.

    The LLM classifies each incoming raw message and extracts structured fields
    that the gateway can act upon.
    """
    signal_type: SignalType = Field(
        description="Classification: TRADE (execution signal), THESIS (research thesis), or NOISE (discard)"
    )
    ticker: str = Field(
        default="",
        description="Stock ticker symbol, uppercase clean string (e.g. 'TSLA', 'AAPL')"
    )
    action: Optional[str] = Field(
        default=None,
        description="Optional trade action: EXPAND, DILUTE, or ADD"
    )
    thesis_text: Optional[str] = Field(
        default=None,
        description="Structural reasoning captured from the user's research thesis"
    )
    time_horizon: str = Field(
        default="LONG_TERM_HOLD",
        description="Investment time horizon: SHORT_TERM_MOMENTUM or LONG_TERM_HOLD"
    )


# =============================================================================
# Slack Ingestion Gateway
# =============================================================================


class SlackIngestionGateway:
    """
    Asynchronous Slack ingestion gateway that classifies messages via Gemini
    and persists structured signals into the pipeline's SQLite database.
    """

    def __init__(self):
        # ── Gemini client ──
        self._client: Optional[genai.Client] = None
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)

        # ── Slack Bolt AsyncApp (lazy init in start()) ──
        self._app = None

    async def start(self):
        """
        Start the Slack Socket Mode listener.

        Requires SLACK_BOT_TOKEN and SLACK_APP_TOKEN to be set in environment.
        """
        if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
            logger.error(
                "SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set. "
                "Skipping Slack ingestion gateway."
            )
            return

        try:
            from slack_bolt.app.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        except ImportError:
            logger.error(
                "slack-bolt library not installed. Run: pip install slack-bolt"
            )
            return

        self._app = AsyncApp(token=SLACK_BOT_TOKEN)

        # Register the message handler
        @self._app.event("message")
        async def handle_message(event, say):
            await self._handle_message(event, say)

        logger.info("🚀 Slack Ingestion Gateway starting (Socket Mode)...")
        handler = AsyncSocketModeHandler(self._app, SLACK_APP_TOKEN)
        await handler.start_async()

    async def _handle_message(self, event: dict, say):
        """
        Process an incoming Slack message event.

        Extracts the raw text, classifies it via Gemini, and routes to the
        appropriate persistence handler.
        """
        try:
            # Extract the raw message text
            raw_text = (event.get("text") or "").strip()
            if not raw_text:
                return

            channel_type = event.get("channel_type", "")
            # Only handle direct messages or channel mentions
            if channel_type not in ("im", "channel", "group", "mpim"):
                return

            # If this is a channel message, check if the bot is mentioned
            if channel_type in ("channel", "group", "mpim"):
                # Get the bot user ID from the event's authorizations or use the app
                bot_user_id = event.get("user", "")
                mention = f"<@{bot_user_id}>"
                if mention not in raw_text:
                    return

            user_id = event.get("user", "")
            logger.info(f"📩 Received message from user {user_id}: {raw_text[:100]}...")

            # ── Classify the signal via Gemini ──
            signal = await self._classify_signal(raw_text)

            # ── Route based on signal type ──
            if signal.signal_type == SignalType.TRADE:
                await self._handle_trade_signal(signal, raw_text, say)
            elif signal.signal_type == SignalType.THESIS:
                await self._handle_thesis_signal(signal, raw_text, say)
            else:
                # NOISE
                logger.info(f"🔇 Signal classified as NOISE — discarding: {raw_text[:100]}...")
                await say(f"Thanks <@{user_id}>! I couldn't identify a trade signal or research thesis in your message. "
                          "Try being more specific about a ticker and your reasoning.")

        except Exception as e:
            logger.error(f"⚠️  Slack message handler error: {e}", exc_info=True)
            try:
                await say("Sorry, I encountered an error processing your message. Please try again.")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Gemini Classification
    # ------------------------------------------------------------------

    async def _classify_signal(self, raw_text: str) -> OutboundSlackSignalSchema:
        """
        Send the raw message to gemini-2.5-flash with native structured output.
        Falls back to NOISE if the LLM is unavailable or parsing fails.
        """
        if not self._client:
            logger.warning("⚠️  No Gemini client available — classifying all signals as NOISE.")
            return OutboundSlackSignalSchema(
                signal_type=SignalType.NOISE,
                ticker="",
                action=None,
                thesis_text=None,
                time_horizon="LONG_TERM_HOLD",
            )

        prompt = (
            "You are a financial signal classifier for a Slack ingestion gateway. "
            "Your job is to read the incoming raw message and classify it into one of three categories:\n\n"
            "1. TRADE — The user is giving an explicit trade execution command. "
            "Look for phrases like 'buy', 'sell', 'add to', 'cut', 'increase position', "
            "'reduce', 'expand', 'dilute', along with a ticker symbol.\n\n"
            "2. THESIS — The user is submitting a research thesis or investment idea. "
            "Look for structural reasoning, arguments about a company's prospects, "
            "mentions of time horizon, and ticker references.\n\n"
            "3. NOISE — Everything else: casual conversation, greetings, questions "
            "about the system, non-investment content.\n\n"
            "Extract the following fields from the message:\n"
            "- ticker: Uppercase clean ticker symbol (e.g., 'TSLA', 'AAPL'). Empty string if none found.\n"
            "- action: For TRADE signals, one of EXPAND, DILUTE, or ADD. Null for THESIS/NOISE.\n"
            "- thesis_text: For THESIS signals, the user's structural reasoning. Null for TRADE/NOISE.\n"
            "- time_horizon: SHORT_TERM_MOMENTUM or LONG_TERM_HOLD. Default LONG_TERM_HOLD.\n\n"
            "Raw message to classify:\n"
            f"{raw_text}"
        )

        try:
            # Use synchronous call in thread pool since the genai client is sync
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=SLACK_GATEWAY_MODEL,
                contents=prompt,
                config={
                    "temperature": SLACK_SIGNAL_TEMPERATURE,
                    "max_output_tokens": SLACK_SIGNAL_MAX_TOKENS,
                    "response_mime_type": "application/json",
                    "response_schema": OutboundSlackSignalSchema,
                },
            )

            # Handle safety-filtered responses
            if response.text is None:
                logger.warning("⚠️  Gemini returned safety-filtered response — classifying as NOISE.")
                return OutboundSlackSignalSchema(
                    signal_type=SignalType.NOISE,
                    ticker="",
                    action=None,
                    thesis_text=None,
                    time_horizon="LONG_TERM_HOLD",
                )

            # Native JSON mode: response.parsed contains the typed model
            if hasattr(response, "parsed") and response.parsed is not None:
                signal = response.parsed
            else:
                # Fallback: parse from text
                signal = OutboundSlackSignalSchema.model_validate_json(response.text.strip())

            # Post-process: ensure ticker is uppercase and trimmed
            if signal.ticker:
                signal.ticker = signal.ticker.upper().strip()

            logger.info(
                f"📊 Classified signal: type={signal.signal_type.value}, "
                f"ticker={signal.ticker}, action={signal.action}, "
                f"horizon={signal.time_horizon}"
            )
            return signal

        except Exception as e:
            logger.error(f"⚠️  Gemini classification failed: {e}")
            return OutboundSlackSignalSchema(
                signal_type=SignalType.NOISE,
                ticker="",
                action=None,
                thesis_text=None,
                time_horizon="LONG_TERM_HOLD",
            )

    # ------------------------------------------------------------------
    # TRADE Signal Handler
    # ------------------------------------------------------------------

    async def _handle_trade_signal(
        self,
        signal: OutboundSlackSignalSchema,
        raw_text: str,
        say,
    ):
        """
        Directly modify the underlying SQLite portfolio_allocations to ensure
        live metrics match execution reality immediately.

        Loads the current portfolio state, locates the ticker, and adjusts
        its concentration_percent by a default increment (e.g., 1% for EXPAND,
        1% for DILUTE, or inserts a new position at 1% for ADD).
        """
        ticker = signal.ticker
        action = signal.action

        if not ticker or not action:
            await say(f"⚠️ Trade signal received but missing ticker or action. "
                      f"Got ticker='{ticker}', action='{action}'.")
            return

        if action not in ("EXPAND", "DILUTE", "ADD"):
            await say(f"⚠️ Invalid trade action '{action}'. Must be EXPAND, DILUTE, or ADD.")
            return

        try:
            # Load current portfolio state
            conn = sqlite3.connect(_THESES_DB_PATH)
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    "SELECT portfolio_allocations, version FROM portfolio_state WHERE id = 1"
                ).fetchone()
            finally:
                conn.close()

            if not row:
                await say("⚠️ No portfolio state found in database. Cannot execute trade.")
                return

            portfolio_allocations = json.loads(row["portfolio_allocations"])
            current_version = row["version"]

            # Determine the sector for this ticker and modify it
            ticker_found = False
            for sector_name, sector_data in portfolio_allocations.get("sectors", {}).items():
                for holding in sector_data.get("holdings", []):
                    if holding["ticker"].upper() == ticker:
                        ticker_found = True
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
                if ticker_found:
                    break

            if not ticker_found and action == "ADD":
                # Add as a new position in a default sector or cash allocation
                # Place in a generic sector based on common mappings
                # For simplicity, add to the first sector with room or create an entry
                for sector_name, sector_data in portfolio_allocations.get("sectors", {}).items():
                    existing_pct = sum(h["concentration_percent"] for h in sector_data.get("holdings", []))
                    room = sector_data["weight_percent"] - existing_pct
                    if room >= 1.0:
                        sector_data["holdings"].append({
                            "ticker": ticker,
                            "concentration_percent": 1.0,
                        })
                        ticker_found = True
                        break

                if not ticker_found:
                    await say(f"⚠️ No sector with sufficient room to ADD position {ticker}.")
                    return
            elif not ticker_found:
                await say(f"⚠️ Ticker {ticker} not found in portfolio and action is {action} (not ADD).")
                return

            # Save the updated portfolio state
            from src.db.DatabaseSink import DatabaseSink
            db_sink = DatabaseSink(db_path=_THESES_DB_PATH)
            db_sink.save_portfolio_state(
                state_dict={
                    "timestamp": datetime.now().strftime("%Y-%m-%d"),
                    "macro_baseline": {},  # Preserved by save_portfolio_state
                    "portfolio_allocations": portfolio_allocations,
                },
                updated_by="slack_gateway",
                reason=f"TRADE signal: {action} {ticker} via Slack",
            )

            await say(
                f"✅ Trade executed: **{action} {ticker}**. "
                f"Portfolio state updated (v{current_version + 1})."
            )
            logger.info(f"✅ TRADE executed: {action} {ticker} (v{current_version + 1})")

        except Exception as e:
            logger.error(f"⚠️  Trade execution error: {e}", exc_info=True)
            await say(f"⚠️ Error executing trade: {e}")

    # ------------------------------------------------------------------
    # THESIS Signal Handler
    # ------------------------------------------------------------------

    async def _handle_thesis_signal(
        self,
        signal: OutboundSlackSignalSchema,
        raw_text: str,
        say,
    ):
        """
        Store the structured thesis record inside the pending_user_theses
        staging table for the next pipeline run to pick up.
        """
        ticker = signal.ticker
        thesis_text = signal.thesis_text or raw_text
        time_horizon = signal.time_horizon or "LONG_TERM_HOLD"
        timestamp = datetime.now().isoformat()

        if not ticker:
            await say(
                "⚠️ Thesis signal received but no ticker could be identified. "
                "Please mention a specific stock ticker (e.g., 'TSLA')."
            )
            return

        # Validate time_horizon
        if time_horizon not in ("SHORT_TERM_MOMENTUM", "LONG_TERM_HOLD"):
            time_horizon = "LONG_TERM_HOLD"

        try:
            conn = sqlite3.connect(_THESES_DB_PATH)
            try:
                conn.execute(
                    "INSERT INTO pending_user_theses "
                    "(ticker, core_argument, time_horizon, raw_message, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ticker.upper(), thesis_text, time_horizon, raw_text, timestamp),
                )
                conn.commit()
            finally:
                conn.close()

            await say(
                f"📝 Thesis recorded for **{ticker.upper()}** ({time_horizon}). "
                f"It will be processed in the next pipeline run. "
                f"Summary: {thesis_text[:200]}{'...' if len(thesis_text) > 200 else ''}"
            )
            logger.info(f"📝 THESIS stored: {ticker.upper()} ({time_horizon})")

        except Exception as e:
            logger.error(f"⚠️  Thesis persistence error: {e}", exc_info=True)
            await say(f"⚠️ Error saving your thesis: {e}")

    # ------------------------------------------------------------------
    # Utility: display help message
    # ------------------------------------------------------------------

    @staticmethod
    def get_usage_guide() -> str:
        """Return a markdown-formatted usage guide for Slack users."""
        return (
            "📈 *Slack Ingestion Gateway — Usage Guide*\n\n"
            "Send me a direct message or mention me in a channel with:\n\n"
            "• *Trade signals:* `EXPAND TSLA` or `DILUTE AAPL` or `ADD NVDA`\n"
            "• *Research theses:* `I think TSLA is undervalued because... [reasons]`\n\n"
            "I'll classify your message, extract the relevant data, and route it "
            "to the portfolio pipeline."
        )


# =============================================================================
# Entry point for direct execution
# =============================================================================

async def main():
    """Run the Slack gateway."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    gateway = SlackIngestionGateway()
    await gateway.start()


if __name__ == "__main__":
    asyncio.run(main())