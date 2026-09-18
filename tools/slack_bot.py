"""Slack bot that answers Valkey questions from the deployed Valkeyrie assistant.

Runs in Socket Mode, which opens an outbound WebSocket to Slack rather than
receiving webhooks. That matters here: this AWS account strips public Lambda
permissions, so an Events API endpoint would be unreachable. Socket Mode needs no
inbound endpoint at all.

Run it:

    export SLACK_BOT_TOKEN=xoxb-...      # Bot User OAuth Token
    export SLACK_APP_TOKEN=xapp-...      # App-Level Token, connections:write
    AWS_PROFILE=valkeyrie-personal uv run \
      --with boto3==1.40.21 --with slack-bolt==1.21.2 python tools/slack_bot.py

Slack app needs: Socket Mode enabled, bot scopes `app_mentions:read` and
`chat:write`, and the `app_mention` event subscribed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

import boto3
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

FUNCTION = "valkeyrie-development-application"
# Version 20 is Fable, the model that passed qualification, with a 120s timeout. It carries
# the live issue-search fix and the refusals that name a next step, including the model's
# own abstention reason. Version 8 is Opus, which failed the claim-to-evidence gate at 0.743.
QUALIFIER = "20"
KNOWLEDGE_BASE_ID = "ONVASJDDNX"
MAX_QUESTION_BYTES = 2048
# Questions about present project state must route live; everything else answers
# from the pinned corpus. Same list the HTTP adapter and tools/ask.py use.
LIVE_HINTS = ("current", "currently", "latest", "right now", "upcoming", "recent", "status of")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("valkeyrie-slack")

lambda_client = boto3.client("lambda", region_name="us-east-1")


def answer_mention(event: dict[str, Any], say: Any) -> None:
    """Answer one mention in a thread, or explain why it could not be answered."""
    question = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", "")).strip()
    # Keep the conversation in a thread so a busy channel stays readable.
    thread = event.get("thread_ts") or event["ts"]

    if not question:
        say(text="Ask me a Valkey question, for example: what is the TSC?", thread_ts=thread)
        return
    if len(question.encode("utf-8")) > MAX_QUESTION_BYTES:
        say(text=f"That question is over the {MAX_QUESTION_BYTES}-byte limit.", thread_ts=thread)
        return

    try:
        result = _ask(question, event)
    except Exception:
        # Log the detail, tell the channel only that it failed.
        log.exception("answer failed")
        say(text="Something went wrong answering that. The failure is logged.", thread_ts=thread)
        return

    say(text=_format(result), thread_ts=thread, unfurl_links=False)


def _ask(question: str, event: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    live = any(hint in question.casefold() for hint in LIVE_HINTS)
    payload = {
        "action": "answer",
        # Derived from the Slack event identity, never from message text. Slack
        # redelivers an event when an ack is missed, and request_id is the
        # idempotency and lease key, so a stable id makes a redelivery reuse the
        # original answer instead of paying for a second one.
        "request_id": f"req_slack-{_event_key(event)}",
        "question": question,
        "version_requirement": "current_state" if live else "none",
        "requested_version": None,
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
        "owner": "slack-bot",
        "lease_duration_seconds": 300,
        "now": now,
        "completed_at": now,
    }
    response = lambda_client.invoke(
        FunctionName=FUNCTION,
        Qualifier=QUALIFIER,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    body = json.loads(response["Payload"].read())
    if "FunctionError" in response:
        raise RuntimeError(f"lambda reported an error: {body}")
    return body if isinstance(body, dict) else {}


def _event_key(event: dict[str, Any]) -> str:
    identity = f"{event.get('team')}/{event.get('channel')}/{event['ts']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _format(result: dict[str, Any]) -> str:
    """Render a response by its outcome. The runtime explains itself in `message`."""
    outcome = result.get("outcome", "unknown")
    claims = result.get("claims") or []
    citations = result.get("citations") or []
    message = result.get("message")

    if claims:
        body = "\n".join(f"• {c['text']}" for c in claims if c.get("text"))
        if citations:
            sources = "\n".join(
                f"  <{c.split(': ', 1)[-1]}|{c.split(': ', 1)[0]}>" for c in citations
            )
            body += f"\n\n*Sources*\n{sources}"
        return body

    # No claims is not one situation. A clarification is the assistant asking something
    # back, a partial means evidence was reachable but incomplete, and an abstention is a
    # deliberate refusal. Collapsing all three into a refusal loses the actual reply.
    if outcome == "clarification":
        return message or "What would you like to know about the Valkey project?"
    if outcome == "partial":
        detail = message or "Some evidence could not be retrieved."
        return f"{detail}\nI won't guess at the rest. Try asking without the live-status wording."
    if message:
        return message
    return f"I don't have grounded evidence for that ({outcome})."


def main() -> None:
    """Build the Slack app at startup so the answer path stays importable and testable."""
    # request_verification_enabled=False because that middleware is for HTTP mode: it
    # verifies a signing secret on inbound POSTs. Socket Mode has no inbound endpoint and
    # events arrive over a pre-authenticated WebSocket, so Bolt would otherwise demand a
    # signing secret that serves no purpose here. Token verification stays on, so a bad
    # bot token fails at startup rather than on the first mention.
    app = App(token=os.environ["SLACK_BOT_TOKEN"], request_verification_enabled=False)
    app.event("app_mention")(answer_mention)
    log.info("connecting to Slack in Socket Mode, serving %s:%s", FUNCTION, QUALIFIER)
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
