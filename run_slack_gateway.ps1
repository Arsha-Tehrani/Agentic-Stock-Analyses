# run_slack_gateway.ps1 — Start the Slack Ingestion Gateway (Windows)
# Loads environment variables from data\slack_gateway.env and starts the gateway.
#
# Usage:
#   .\run_slack_gateway.ps1
#
# Required env vars (set in slack_gateway.env):
#   SLACK_BOT_TOKEN, SLACK_APP_TOKEN, GEMINI_API_KEY

$ScriptDir = $PSScriptRoot
$EnvFile = Join-Path $ScriptDir "data\slack_gateway.env"

# Load the env file if it exists (KEY="VALUE" or KEY=VALUE lines, '#' comments ignored)
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $key, $value = $line -split "=", 2
            $key = $key.Trim()
            $value = $value.Trim().Trim('"')
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Validate required env vars
if ([string]::IsNullOrEmpty($env:SLACK_BOT_TOKEN) -or $env:SLACK_BOT_TOKEN -eq "your-slack-bot-token") {
    Write-Host "❌ SLACK_BOT_TOKEN not set or still has placeholder value."
    Write-Host "   Edit $EnvFile with your real tokens."
    exit 1
}

if ([string]::IsNullOrEmpty($env:SLACK_APP_TOKEN) -or $env:SLACK_APP_TOKEN -eq "your-slack-app-token") {
    Write-Host "❌ SLACK_APP_TOKEN not set or still has placeholder value."
    Write-Host "   Edit $EnvFile with your real tokens."
    exit 1
}

if ([string]::IsNullOrEmpty($env:GEMINI_API_KEY)) {
    Write-Host "❌ GEMINI_API_KEY not set."
    Write-Host "   Edit $EnvFile with your real API key."
    exit 1
}

Set-Location $ScriptDir

$botTokenMasked = $env:SLACK_BOT_TOKEN.Substring(0, [Math]::Min(10, $env:SLACK_BOT_TOKEN.Length)) + "..." + $env:SLACK_BOT_TOKEN.Substring([Math]::Max(0, $env:SLACK_BOT_TOKEN.Length - 4))
$appTokenMasked = $env:SLACK_APP_TOKEN.Substring(0, [Math]::Min(10, $env:SLACK_APP_TOKEN.Length)) + "..." + $env:SLACK_APP_TOKEN.Substring([Math]::Max(0, $env:SLACK_APP_TOKEN.Length - 4))
$gatewayModel = & "$ScriptDir\.venv\Scripts\python.exe" -c "from src.config import SLACK_GATEWAY_MODEL; print(SLACK_GATEWAY_MODEL)"

Write-Host "🚀 Starting Slack Ingestion Gateway..."
Write-Host "   Bot token:    $botTokenMasked"
Write-Host "   App token:    $appTokenMasked"
Write-Host "   Gemini model: $gatewayModel"
Write-Host "   DB path:      $ScriptDir\data\news.db"
Write-Host ""
Write-Host "   Listening for Slack messages... (Ctrl+C to stop)"

& "$ScriptDir\.venv\Scripts\python.exe" -m src.utils.slack_gateway
