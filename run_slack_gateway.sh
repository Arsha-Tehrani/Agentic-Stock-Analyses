#!/bin/bash
# run_slack_gateway.sh — Start the Slack Ingestion Gateway
# Sources environment variables from data/slack_gateway.env and starts the gateway.
#
# Usage:
#   ./run_slack_gateway.sh                        # Run in foreground
#   nohup ./run_slack_gateway.sh &                # Run in background
#
# Required env vars (set in slack_gateway.env):
#   SLACK_BOT_TOKEN, SLACK_APP_TOKEN, GEMINI_API_KEY

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/data/slack_gateway.env"

# Source the env file if it exists
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# Validate required env vars
if [ -z "$SLACK_BOT_TOKEN" ] || [ "$SLACK_BOT_TOKEN" = "your-slack-bot-token" ]; then
    echo "❌ SLACK_BOT_TOKEN not set or still has placeholder value."
    echo "   Edit $ENV_FILE with your real tokens."
    exit 1
fi

if [ -z "$SLACK_APP_TOKEN" ] || [ "$SLACK_APP_TOKEN" = "your-slack-app-token" ]; then
    echo "❌ SLACK_APP_TOKEN not set or still has placeholder value."
    echo "   Edit $ENV_FILE with your real tokens."
    exit 1
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set."
    echo "   Edit $ENV_FILE with your real API key."
    exit 1
fi

echo "🚀 Starting Slack Ingestion Gateway..."
echo "   Bot token:    ${SLACK_BOT_TOKEN:0:10}...${SLACK_BOT_TOKEN: -4}"
echo "   App token:    ${SLACK_APP_TOKEN:0:10}...${SLACK_APP_TOKEN: -4}"
echo "   Gemini model: $(python3 -c "from src.config import SLACK_GATEWAY_MODEL; print(SLACK_GATEWAY_MODEL)")"
echo "   DB path:      $SCRIPT_DIR/data/news.db"
echo ""
echo "   Listening for Slack messages... (Ctrl+C to stop)"

# Activate venv and run
cd "$SCRIPT_DIR" || exit 1
source .venv/bin/activate
python -m src.utils.slack_gateway