#!/usr/bin/env bash
# Start the Slack bot in a tmux session named "valkeyrie" if none is running. Idempotent, so it is
# safe from cron's @reboot and from a shell. The host reboot of 2026-10-03 21:33 UTC took the
# session and the bot with it, and the bot was down 34 hours before anyone noticed.
set -euo pipefail
cd "$(dirname "$0")/.."
if tmux has-session -t valkeyrie 2>/dev/null; then
  echo "valkeyrie session already running"
  exit 0
fi
: "${VALKEYRIE_ENV:=$HOME/.config/valkeyrie/env}"
[ -r "$VALKEYRIE_ENV" ] || { echo "no env file at $VALKEYRIE_ENV" >&2; exit 1; }
tmux new-session -d -s valkeyrie
tmux send-keys -t valkeyrie "cd $(pwd) && set -a && source $VALKEYRIE_ENV && set +a && ./tools/run_bot.sh" Enter
echo "valkeyrie session started"
