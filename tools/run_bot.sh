#!/usr/bin/env bash
# Start the Valkeyrie Slack bot with everything it needs in one place.
#
# Run this INSIDE the tmux session that holds the Slack tokens (SLACK_BOT_TOKEN, SLACK_APP_TOKEN):
#   tmux attach -t valkeyrie
#   ./tools/run_bot.sh
# A restart that omitted AWS_PROFILE once left the bot connected to Slack and unable to answer;
# the bot now refuses to start in that state, and this script makes the state impossible.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SLACK_BOT_TOKEN:?export SLACK_BOT_TOKEN first}"
: "${SLACK_APP_TOKEN:?export SLACK_APP_TOKEN first}"
export AWS_PROFILE="${AWS_PROFILE:-valkeyrie-personal}"
export SLACK_TEAM_ID="${SLACK_TEAM_ID:-THDV9665P}"
git pull -q origin main
exec uv run --with boto3==1.40.21 --with slack-bolt==1.21.2 python tools/slack_bot.py
