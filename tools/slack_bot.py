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
from pathlib import Path
from typing import Any

import boto3
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

FUNCTION = "valkeyrie-development-application"
# Version 63 is Fable, the model that passed qualification (requalified for the prompt that
# treats a completed empty search as evidence), with a 120s timeout. It carries whole board
# reads totalled by status, release membership on merged pull requests, cross-repository
# searches, a router-composed search with a date window, releases read by tag, one retry of a
# corpus abstention as a persisted plan revision, a named repository always searched, the
# discussion on an issue or pull request, files at a ref, security advisories, per-repository
# retrieval quotas, and credential screening. Version 8 is Opus, which failed the
# claim-to-evidence gate at 0.743.
QUALIFIER = "63"
KNOWLEDGE_BASE_ID = "ONVASJDDNX"
# Matches the runtime's own bound. It was a quarter of that, which refused every pasted diagnostic:
# INFO is four to nine kilobytes, and "here is my INFO output, why is used_memory climbing?" is the
# question this assistant most wants to be asked. A paste larger than this is trimmed by the asker,
# who knows which section matters, rather than silently truncated here.
MAX_QUESTION_BYTES = 8 * 1024
# Questions about present project state must route live; everything else answers
# from the pinned corpus. Same list the HTTP adapter and tools/ask.py use.
LIVE_HINTS = ("current", "currently", "latest", "right now", "upcoming", "recent", "status of")

# What a reaction on one of the bot's own answers means. Deliberately small: these are the marks
# people already use, and anything else is left unread rather than guessed at.
_FEEDBACK = {
    "+1": "helpful",
    "thumbsup": "helpful",
    "white_check_mark": "helpful",
    "tada": "helpful",
    "heavy_check_mark": "helpful",
    "-1": "unhelpful",
    "thumbsdown": "unhelpful",
    "x": "unhelpful",
    "confused": "unhelpful",
}
FEEDBACK_PATH = Path(os.environ.get("VALKEYRIE_FEEDBACK_PATH", "/tmp/valkeyrie-feedback.jsonl"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("valkeyrie-slack")

lambda_client = boto3.client("lambda", region_name="us-east-1")


# A question takes twenty to forty seconds to answer, and for all that time the thread looked dead.
# :eyes: on the asker's message says it was seen, and it comes OFF when the reply lands. Nothing
# replaces it: the reply itself is the outcome, and a tick or a cross added underneath read as the
# bot grading its own work.
_WORKING = "eyes"
# Reactions need the reactions:write scope. Without it every call fails the same way, so the miss
# is logged ONCE and the bot carries on: an answer that arrives without a tick mark is still an
# answer, and a bot that crashes over decoration is not.
_reaction_scope_missing = False


def _unreact(client: Any, event: dict[str, Any], name: str) -> None:
    """Take a mark back off, best effort. The mark may never have landed."""
    if _reaction_scope_missing:
        return
    channel, timestamp = event.get("channel"), event.get("ts")
    if not channel or not timestamp:
        return
    try:
        client.reactions_remove(channel=channel, timestamp=timestamp, name=name)
    except Exception as error:  # noqa: BLE001 - removal failing is not a failure
        log.debug("could not remove :%s: (%s)", name, error)


def _react(client: Any, event: dict[str, Any], name: str) -> None:
    """Mark the asker's message, best effort. Never let decoration break an answer."""
    global _reaction_scope_missing
    if _reaction_scope_missing:
        return
    channel, timestamp = event.get("channel"), event.get("ts")
    if not channel or not timestamp:
        return
    try:
        client.reactions_add(channel=channel, timestamp=timestamp, name=name)
    except Exception as error:  # noqa: BLE001 - see _reaction_scope_missing
        text = str(error)
        if "missing_scope" in text or "not_allowed_token_type" in text:
            _reaction_scope_missing = True
            log.warning(
                "reactions are disabled: the Slack app needs the reactions:write scope (%s)", error
            )
            return
        # already_reacted is the common one and means the mark is already there.
        log.debug("could not add :%s: (%s)", name, error)


def record_feedback(event: dict[str, Any], client: Any) -> None:
    """Record a person's reaction to one of this bot's answers.

    The battery measures answers against expectations I wrote. This measures them against the
    people asking, which is the only source that can tell us an answer was correct but useless.
    Written to a local file rather than a service: it is a signal to read, not state to depend on.
    """
    if event.get("item_user") != _bot_user_id:
        return
    item = event.get("item") or {}
    if item.get("type") != "message":
        return
    verdict = _FEEDBACK.get(event.get("reaction", ""))
    if verdict is None:
        return
    row = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "verdict": verdict,
        "reaction": event.get("reaction"),
        "channel": item.get("channel"),
        "answer_ts": item.get("ts"),
        "by": event.get("user"),
        "qualifier": QUALIFIER,
    }
    try:
        with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError as error:  # noqa: BLE001 - feedback is a nicety, never a failure
        log.debug("could not record feedback (%s)", error)
        return
    log.info("feedback %s on %s by %s", verdict, item.get("ts"), event.get("user"))


def answer_mention(event: dict[str, Any], say: Any, client: Any) -> None:
    """Answer one mention in a thread, or explain why it could not be answered."""
    # Who may trigger an inference. Every mention delivered to this installation used to run one,
    # including one posted by another app, which is a loop waiting to happen, and one from a
    # workspace this bot was never meant to serve. EXPECTED_TEAM unset keeps the old behaviour so
    # a local run needs no configuration.
    if EXPECTED_TEAM and event.get("team") not in {EXPECTED_TEAM, None}:
        log.warning("ignoring mention from unexpected team %s", event.get("team"))
        return
    if event.get("bot_id") or event.get("subtype"):
        # A bot's own mention, or an edit/join/share event that is not a person asking.
        return

    question = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", "")).strip()
    # Keep the conversation in a thread so a busy channel stays readable.
    thread = event.get("thread_ts") or event["ts"]

    if not question:
        say(text="Ask me a Valkey question, for example: what is the TSC?", thread_ts=thread)
        return
    if len(question.encode("utf-8")) > MAX_QUESTION_BYTES:
        say(
            text=(
                f"That is over my {MAX_QUESTION_BYTES // 1024} KB limit for one message. "
                "Paste the part that matters, for example the Memory or Clients section of INFO, "
                "or the few SLOWLOG entries you are asking about."
            ),
            thread_ts=thread,
        )
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

    _react(client, event, _WORKING)
    conversation = _thread_history(event, client)
    try:
        result = _ask(question, event, conversation)
    except Exception:
        # Log the detail, tell the channel only that it failed.
        log.exception("answer failed")
        _unreact(client, event, _WORKING)
        say(text="Something went wrong answering that. The failure is logged.", thread_ts=thread)
        return

    say(text=_format(result), thread_ts=thread, unfurl_links=False)
    _unreact(client, event, _WORKING)


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
# The workspace this bot serves. Set SLACK_TEAM_ID to enforce it; unset serves any team the app
# is installed in, which is the behaviour a local run expects.
EXPECTED_TEAM = os.environ.get("SLACK_TEAM_ID", "")


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
    return _turns_before(
        replies.get("messages", []),
        event.get("ts", ""),
        _bot_user_id or "",
        asker=event.get("user") or "",
    )


def _turns_before(
    messages: list[dict[str, Any]], current_ts: str, bot_user_id: str, *, asker: str = ""
) -> list[dict[str, str]]:
    """Map thread messages to bounded user/assistant turns, excluding the current mention.

    The asker's own turns, this bot's replies, and the message that OPENED the thread are kept.
    Another person's mid-thread message is not context for this person's follow-up, and taking it
    as one let a third party supply the subject that "is it released?" resolves against. The
    opening message is different: it is what the thread is about, and dropping it broke a real
    conversation where one person asked "how does the project handle content?", the bot asked
    which kind, and a SECOND person answered "content for social media and blogs" to a bot that
    could no longer see the question.
    """
    turns: list[dict[str, str]] = []
    root_ts = messages[0].get("ts") if messages else None
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
        elif asker and message.get("user") != asker and message.get("ts") != root_ts:
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
            # Citations are application-authored from validated GitHub URLs, so the link markup is
            # built here; the label is still escaped since it carries a path.
            sources = "\n".join(f"  {_source_link(c)}" for c in citations)
            body += f"\n\n*Sources*\n{sources}"
        if message:
            # An answer may carry one limitation: the part of the question the evidence did not
            # support. It is the difference between a useful partial answer and a silent gap.
            body = f"{body}\n\n_{_plain(message)}_"
        # Where to go next. An answer with a stated limitation is the case where someone most needs
        # somewhere else to go, so that one names the human channels; every other answer just says
        # that the thread is open, which is the cheapest way to get a better second answer.
        return f"{body}\n\n_{_MORE_HELP if message else _FOLLOW_UP}_"

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


# Closing pointers. Short by design: a paragraph of boilerplate under every answer trains people to
# stop reading the part that matters.
_FOLLOW_UP = "Ask a follow-up in this thread if you need more detail."
_MORE_HELP = (
    "For more than I can ground: ask in the Valkey Slack help channels, "
    "open a GitHub Discussion in valkey-io, or read the topic pages on valkey.io."
)


def _names(url: str) -> str:
    """What a GitHub URL is ABOUT, for a citation label: a tag, a number, or a file path."""
    parts = [part for part in url.split("/") if part]
    for marker, render in (
        ("tag", lambda rest: rest[0] if rest else ""),
        ("pull", lambda rest: f"#{rest[0]}" if rest else ""),
        ("issues", lambda rest: f"#{rest[0]}" if rest else ""),
        # A file URL carries the commit between blob and the path, which the link already pins.
        ("blob", lambda rest: "/".join(rest[1:])),
        ("tree", lambda rest: "/".join(rest[1:])),
    ):
        if marker in parts:
            return render(parts[parts.index(marker) + 1 :])
    return ""


def _source_link(citation: str) -> str:
    """One Slack link per citation, labelled by what it is rather than by its commit.

    A citation arrives as "label: url". The commit belongs IN the link, which already pins it, not
    in the label, where forty hex characters push the filename off the line. A live observation is
    labelled by what was read, taken from the tail of its own URL, because "live GitHub release
    observed 2026-09-24T15:25:25Z" does not say WHICH release.
    """
    label, separator, url = citation.rpartition(": ")
    # rpartition puts the WHOLE string in the last field when the separator is absent, so a
    # citation without one would otherwise become a link whose label is empty.
    if not separator or not url.startswith("https://"):
        return _plain(citation)
    if label.startswith("live GitHub"):
        kind = label.removeprefix("live GitHub").split(" observed ")[0].strip().replace("_", " ")
        label = f"{kind} ({_names(url)})" if _names(url) else kind
    else:
        # repo/path@commit -> repo/path, since the link carries the commit already.
        label = label.split("@", 1)[0]
    return f"<{url}|{_plain(label)}>"


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
    # A reaction on one of the bot's own answers is the only quality signal that comes from the
    # people asking rather than from expectations someone wrote down. Needs the reactions:read
    # scope and the reaction_added event subscription; without them no event arrives and nothing
    # here runs, which is why it is registered unconditionally and never asserted.
    app.event("reaction_added")(record_feedback)
    # Resolved once here so the feedback handler can tell this bot's messages from anyone else's.
    global _bot_user_id
    try:
        _bot_user_id = app.client.auth_test()["user_id"]
    except Exception as error:  # noqa: BLE001 - the answer path does not need this
        log.warning("could not resolve the bot user id, feedback will be ignored (%s)", error)
    log.info("connecting to Slack in Socket Mode, serving %s:%s", FUNCTION, QUALIFIER)
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
