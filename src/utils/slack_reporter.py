"""
slack_reporter.py — Posts pipeline results to a Slack output channel.

After the daily pipeline run completes, this reporter takes the final
PortfolioRecommendation and posts a clean, formatted summary to a
separate Slack channel (e.g. #portfolio-updates) so the user can see
results without digging through log files.

Uses the same SLACK_BOT_TOKEN as the ingestion gateway but posts to
a different channel (SLACK_OUTPUT_CHANNEL_ID).
"""

import logging
from typing import Optional

from src.config import (
    SLACK_BOT_TOKEN,
    SLACK_OUTPUT_CHANNEL_ID,
)
from src.state import PortfolioRecommendation

logger = logging.getLogger(__name__)


class SlackOutputReporter:
    """
    Posts formatted pipeline results to a Slack output channel.

    Usage:
        reporter = SlackOutputReporter()
        reporter.post_recommendation(recommendation)
    
    If SLACK_BOT_TOKEN or SLACK_OUTPUT_CHANNEL_ID is not configured,
    prints the recommendation to stdout instead (graceful degradation).
    """

    def __init__(self):
        self._client: Optional[object] = None
        self._initialized = False

        if SLACK_BOT_TOKEN and SLACK_OUTPUT_CHANNEL_ID:
            try:
                from slack_sdk import WebClient
                self._client = WebClient(token=SLACK_BOT_TOKEN)
                self._initialized = True
            except ImportError:
                logger.warning("slack_sdk not installed — falling back to stdout output.")
            except Exception as e:
                logger.warning(f"Failed to initialize Slack WebClient: {e}")

    def post_recommendation(self, recommendation: Optional[PortfolioRecommendation]) -> bool:
        """
        Post the portfolio recommendation to the output Slack channel.

        Args:
            recommendation: The PortfolioRecommendation from GraphState.

        Returns:
            True if posted successfully, False if skipped or failed.
        """
        if recommendation is None:
            self._fallback_print("No recommendation generated — pipeline produced no output.")
            return False

        if not recommendation.Proposed_Actions:
            self._fallback_print(
                "No trades proposed. "
                f"Regime significance: {recommendation.regime_significance_score}/100. "
                "Portfolio manager assessed no actionable opportunities."
            )
            return False

        # Build the Slack message blocks
        blocks = self._build_message_blocks(recommendation)

        if self._initialized and self._client is not None:
            try:
                response = self._client.chat_postMessage(
                    channel=SLACK_OUTPUT_CHANNEL_ID,
                    text=f"📊 Pipeline Results — {recommendation.generated_at[:10]}",
                    blocks=blocks,
                    mrkdwn=True,
                )
                logger.info(f"✅ Posted recommendation to Slack channel {SLACK_OUTPUT_CHANNEL_ID}")
                return response["ok"]
            except Exception as e:
                logger.error(f"⚠️  Failed to post to Slack: {e}")
                self._fallback_print(recommendation)
                return False
        else:
            self._fallback_print(recommendation)
            return False

    def _build_message_blocks(self, rec: PortfolioRecommendation) -> list:
        """Build Slack Block Kit blocks for a formatted recommendation message."""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📊 Daily Pipeline Report — {rec.generated_at[:10]}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Regime Significance:*\n{rec.regime_significance_score}/100"},
                    {"type": "mrkdwn", "text": f"*Research Queries:*\n{len(rec.queries_used)}"},
                ],
            },
            {"type": "divider"},
        ]

        # Add action items
        if rec.Proposed_Actions:
            action_text = "*Proposed Actions:*\n"
            for i, action in enumerate(rec.Proposed_Actions, 1):
                icon = {"EXPAND": "📈", "DILUTE": "📉", "ADD": "➕"}.get(action.action, "•")
                action_text += (
                    f"\n{icon} *{action.ticker}* — {action.action} ({action.time_horizon})\n"
                    f"  _{action.reasoning[:200]}_"
                )
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": action_text},
            })

        blocks.append({"type": "divider"})

        # Add narrative summaries (collapsed into a single block)
        narrative = ""
        if rec.Portfolio_Impact_Assessment:
            narrative += f"*Portfolio Impact:*\n{rec.Portfolio_Impact_Assessment[:300]}\n\n"
        if rec.Abstract_Proxy_Discoveries:
            narrative += f"*Proxy Discoveries:*\n{rec.Abstract_Proxy_Discoveries[:200]}\n\n"
        if rec.Momentum_vs_Valuation_Analysis:
            narrative += f"*Momentum vs Valuation:*\n{rec.Momentum_vs_Valuation_Analysis[:200]}"

        if narrative:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": narrative},
            })

        return blocks

    def _fallback_print(self, rec_or_msg) -> None:
        """Print to stdout when Slack is unavailable."""
        if isinstance(rec_or_msg, str):
            print(f"\n[Slack Reporter] {rec_or_msg}")
            return

        rec = rec_or_msg
        print("\n" + "=" * 60)
        print("📊 PIPELINE RESULTS (Slack unavailable — printed to stdout)")
        print("=" * 60)
        print(f"  Regime significance: {rec.regime_significance_score}/100")
        print(f"  Proxies discovered:  {len(rec.queries_used)} queries executed")
        print(f"  Proposed actions:    {len(rec.Proposed_Actions)}")
        print("-" * 60)
        if rec.Proposed_Actions:
            for i, a in enumerate(rec.Proposed_Actions, 1):
                print(f"  [{i}] {a.ticker} {a.action} {a.time_horizon}")
                print(f"       {a.reasoning[:150]}")
        print("-" * 60)
        print(f"  Impact: {rec.Portfolio_Impact_Assessment[:200]}")
        print("=" * 60)