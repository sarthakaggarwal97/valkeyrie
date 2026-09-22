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

Slack app needs: Socket Mode enabled, bot scopes `app_mentions:read`, `chat:write`, and
`channels:history` (plus `groups:history` for private channels) so a follow-up in a thread can be
read against the turns before it, and the `app_mention` event subscribed. Without the history
scopes the bot still answers; each question is simply read on its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from datetime import UTC, datetime
from typing import Any

import boto3
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

FUNCTION = "valkeyrie-development-application"
# Version 38 is Fable, the model that passed qualification (requalified for the prompt that
# treats a completed empty search as evidence), with a 120s timeout. It carries whole board
# reads totalled by status, release membership on merged pull requests, cross-repository
# searches, a router-composed search with a date window, releases read by tag, one retry of a
# corpus abstention as a persisted plan revision, and a named repository always searched. Version
# 8 is Opus, which failed the claim-to-evidence gate at 0.743.
QUALIFIER = "38"
KNOWLEDGE_BASE_ID = "ONVASJDDNX"
MAX_QUESTION_BYTES = 2048
# Questions about present project state must route live; everything else answers
# from the pinned corpus. Same list the HTTP adapter and tools/ask.py use.
LIVE_HINTS = ("current", "currently", "latest", "right now", "upcoming", "recent", "status of")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("valkeyrie-slack")

lambda_client = boto3.client("lambda", region_name="us-east-1")


def answer_mention(event: dict[str, Any], say: Any, client: Any) -> None:
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

    # Slack redelivers an event whose ack it did not see. The runtime already replays the same
    # answer for the same event, so inference is not repeated; the REPLY was, and a thread got
    # the same answer twice. Each event is answered once per process; a redelivery that arrives
    # after a restart still replays the stored result, which is the right answer to send.
    key = _event_key(event)
    with _answered_lock:
        if key in _answered:
            log.info("duplicate delivery of %s ignored", key)
            return
        _answered[key] = True
        while len(_answered) > MAX_ANSWERED_EVENTS:
            _answered.pop(next(iter(_answered)))

    conversation = _thread_history(event, client)
    try:
        result = _ask(question, event, conversation)
    except Exception:
        # Log the detail, tell the channel only that it failed.
        log.exception("answer failed")
        say(text="Something went wrong answering that. The failure is logged.", thread_ts=thread)
        return

    say(text=_format(result), thread_ts=thread, unfurl_links=False)


def _ask(
    question: str, event: dict[str, Any], conversation: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    live = any(hint in question.casefold() for hint in LIVE_HINTS)
    payload: dict[str, Any] = {
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
    if conversation:
        # Prior turns of this thread, so "and what about failover?" is read against what came
        # before it. The runtime resolves the follow-up into a standalone question and answers
        # that from evidence; the history decides what was asked, never what may be claimed.
        payload["conversation"] = conversation
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


# Events answered by this process, oldest first. Bounded so a long-lived bot does not grow.
_answered: dict[str, bool] = {}
_answered_lock = threading.Lock()
MAX_ANSWERED_EVENTS = 4096


def _event_key(event: dict[str, Any]) -> str:
    identity = f"{event.get('team')}/{event.get('channel')}/{event['ts']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


# Conversation memory. The thread itself is the store: Slack already holds every turn, so the
# bot reads the ones before this mention rather than keeping a copy anywhere. Bounds mirror the
# runtime's, which validates them again.
MAX_HISTORY_TURNS = 6
MAX_HISTORY_TURN_BYTES = 2000
_bot_user_id: str | None = None
_history_unavailable_logged = False


def _thread_history(event: dict[str, Any], client: Any) -> list[dict[str, str]]:
    """Return the turns before this mention in its thread, oldest first, or [] if unavailable."""
    global _bot_user_id, _history_unavailable_logged
    thread = event.get("thread_ts")
    if not thread or thread == event.get("ts"):
        return []  # A thread root has nothing before it.
    try:
        if _bot_user_id is None:
            _bot_user_id = client.auth_test()["user_id"]
        replies = client.conversations_replies(
            channel=event["channel"], ts=thread, limit=MAX_HISTORY_TURNS * 2 + 2
        )
    except Exception as error:  # noqa: BLE001 - history is optional; the answer is not.
        if not _history_unavailable_logged:
            log.warning("thread history unavailable, answering without it: %s", error)
            _history_unavailable_logged = True
        return []
    return _turns_before(replies.get("messages", []), event.get("ts", ""), _bot_user_id or "")


def _turns_before(
    messages: list[dict[str, Any]], current_ts: str, bot_user_id: str
) -> list[dict[str, str]]:
    """Map thread messages to bounded user/assistant turns, excluding the current mention."""
    turns: list[dict[str, str]] = []
    for message in messages:
        if message.get("ts") == current_ts:
            break
        text = re.sub(r"<@[A-Z0-9]+>", "", message.get("text") or "").strip()
        if not text:
            continue
        # Only THIS bot's messages are assistant turns. Another bot in the thread is neither the
        # asker nor us; treating every bot_id as ours would let it speak as the assistant and
        # steer how the next follow-up is read.
        if message.get("user") == bot_user_id:
            role = "assistant"
        elif message.get("bot_id"):
            continue
        else:
            role = "user"
        if role == "assistant":
            # Keep the claims, drop the Sources block: links are not conversation.
            text = text.split("\n\n*Sources*", 1)[0].strip()
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_HISTORY_TURN_BYTES:
            text = encoded[:MAX_HISTORY_TURN_BYTES].decode("utf-8", "ignore")
        turns.append({"role": role, "text": text})
    return turns[-MAX_HISTORY_TURNS:]


def _format(result: dict[str, Any]) -> str:
    """Render a response by its outcome. The runtime explains itself in `message`."""
    outcome = result.get("outcome", "unknown")
    claims = result.get("claims") or []
    citations = result.get("citations") or []
    message = result.get("message")

    if claims:
        body = "\n".join(f"• {_plain(c['text'])}" for c in claims if c.get("text"))
        if citations:
            # Citations are application-authored from validated GitHub URLs, so the link markup
            # is built here; the label is still escaped since it carries a path.
            sources = "\n".join(
                f"  <{c.split(': ', 1)[-1]}|{_plain(c.split(': ', 1)[0])}>" for c in citations
            )
            body += f"\n\n*Sources*\n{sources}"
        return body

    # No claims is not one situation. A clarification is the assistant asking something
    # back, a partial means evidence was reachable but incomplete, and an abstention is a
    # deliberate refusal. Collapsing all three into a refusal loses the actual reply.
    if outcome == "clarification":
        return (
            _plain(message) if message else "What would you like to know about the Valkey project?"
        )
    if outcome == "partial":
        detail = _plain(message) if message else "Some evidence could not be retrieved."
        return f"{detail}\nI won't guess at the rest. Try asking without the live-status wording."
    if message:
        return _plain(message)
    return f"I don't have grounded evidence for that ({outcome})."


def _plain(text: str) -> str:
    """Escape Slack's three control characters in model-authored text.

    Claims and messages are grounded in GitHub content anyone can edit. In mrkdwn, `<!channel>`
    pages the channel and `<@U…>` mentions a person; escaped, they are just the text they were.
    The bot's own link markup is added after escaping, so it is never affected.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
