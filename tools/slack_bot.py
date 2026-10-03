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
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from slack_bolt import App, Assistant
from slack_bolt.adapter.socket_mode import SocketModeHandler

FUNCTION = "valkeyrie-development-application"
# Version 94 is Fable, the model that passed qualification (requalified for the prompt that
# treats a completed empty search as evidence), with a 120s timeout. It carries whole board
# reads totalled by status, release membership on merged pull requests, cross-repository
# searches, a router-composed search with a date window, releases read by tag, one retry of a
# corpus abstention as a persisted plan revision, a named repository always searched, the
# discussion on an issue or pull request, files at a ref, security advisories, per-repository
# retrieval quotas, and credential screening. Version 8 is Opus, which failed the
# claim-to-evidence gate at 0.743.
QUALIFIER = "94"
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

# The command module lives beside this file; the bot is launched as a script, not a package, so
# the directory has to be on the path before the import can be spelled plainly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import actions  # noqa: E402

# The function may run for its full timeout when an answer chains several lookups. boto3's default
# client gives up reading after 60 seconds and RETRIES the invoke with the same payload; the retry
# reaches the runtime while the first invocation still holds the request lease, and the user gets
# "request lease is still active" instead of the answer that lands seconds later. Reproduced on the
# first sixty-plus-second answer. Read past the function's timeout, and never retry: the runtime's
# own replay is the retry, and the question is not idempotent in time.
lambda_client = boto3.client(
    "lambda",
    region_name="us-east-1",
    config=Config(read_timeout=330, connect_timeout=20, retries={"max_attempts": 0}),
)


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
        log.debug("could not remove :%s: (%s)", name, _brief(error))


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
                "reactions are disabled: the Slack app needs the reactions:write scope (%s)",
                _brief(error),
            )
            return
        # already_reacted is the common one and means the mark is already there.
        log.debug("could not add :%s: (%s)", name, _brief(error))


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


def _brief(error: BaseException) -> str:
    """An exception as type and Slack error code only: str(SlackApiError) carries the whole
    response, which can include tokens or user text."""
    code = getattr(getattr(error, "response", None), "get", lambda k, d=None: d)("error")
    return f"{type(error).__name__}" + (f": {code}" if isinstance(code, str) else "")


def _claim(key: str) -> bool:
    """Take the delivery fence for an event; False when this process already holds it."""
    with _answered_lock:
        if key in _answered:
            return False
        _answered[key] = True
        while len(_answered) > MAX_ANSWERED_EVENTS:
            _answered.pop(next(iter(_answered)))
        return True


def _release(key: str) -> None:
    with _answered_lock:
        _answered.pop(key, None)


_MENTION = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]*)?>")
# The credential shapes the runtime redacts from the question (application_runtime._CREDENTIAL),
# applied here to history turns before they leave the process.
_CREDENTIAL_SHAPES = re.compile(
    r"gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    r"|AKIA[A-Z0-9]{16}"
)


def _without_mentions(text: str, client: Any) -> str:
    """The question with this bot's mention removed and other people's mentions kept as words.

    Removing every mention outright turned "compare GET<@bot>SET" into "compare GETSET" and
    dropped the person in "ask <@someone> about ownership"; a space where the bot was and
    "@someone" for anyone else keeps the sentence the asker wrote.
    """
    global _bot_user_id
    bot_id = _bot_user_id
    if bot_id is None:
        try:
            bot_id = _bot_user_id = client.auth_test()["user_id"]
        except Exception:  # noqa: BLE001 - the mention is still answerable without the id
            # Not cached: a transient failure must not turn every later bot mention into
            # "@someone" for the life of the process. With no id, every mention is removed.
            bot_id = None

    def replace(match: re.Match[str]) -> str:
        return " " if bot_id is None or match.group(1) == bot_id else " @someone "

    return re.sub(r"\s{2,}", " ", _MENTION.sub(replace, text)).strip()


def _bounded_reply(text: str) -> str:
    """Every reply under Slack's 40,000-character limit, whatever path produced it."""
    if len(text) <= MAX_REPLY_CHARS:
        return text
    limit = MAX_REPLY_CHARS - 200
    # Cut on a line boundary when one is near, never inside <url|label> markup or a backtick run.
    cut_at = text.rfind("\n", limit - 2000, limit)
    if cut_at < 0:
        cut_at = limit
    cut = text[:cut_at]
    open_link = cut.rfind("<")
    if open_link > cut.rfind(">"):
        cut = cut[:open_link]
    cut = cut.rstrip("`")
    if cut.count("```") % 2:
        cut += "\n```"
    return cut.rstrip() + "\n\n_The rest of this reply did not fit in one Slack message._"


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

    question = _without_mentions(event.get("text", ""), client)
    # Keep the conversation in a thread so a busy channel stays readable.
    thread = event.get("thread_ts") or event["ts"]

    # Slack redelivers an event whose ack it did not see. Every user-facing reply, including the
    # prompt for an empty mention, sits behind this fence; the fence is RELEASED again if the
    # reply never reaches Slack, so a transient post failure is retried on redelivery rather
    # than lost for the life of the process.
    key = _event_key(event)
    if not _claim(key):
        log.info("duplicate delivery of %s ignored", key)
        return

    def deliver(**kwargs: Any) -> None:
        try:
            say(**kwargs)
        except Exception:
            _release(key)
            raise

    if not question:
        deliver(text="Ask me a Valkey question, for example: what is the TSC?", thread_ts=thread)
        return
    if len(question.encode("utf-8")) > MAX_QUESTION_BYTES:
        deliver(
            text=(
                f"That is over my {MAX_QUESTION_BYTES // 1024} KB limit for one message. "
                "Paste the part that matters, for example the Memory or Clients section of INFO, "
                "or the few SLOWLOG entries you are asking about."
            ),
            thread_ts=thread,
        )
        return

    # A command, not a question. Checked AFTER dedup so a redelivered command cannot dispatch
    # twice, and before any retrieval, because a command needs no evidence. Only a message whose
    # second word names a catalogued action is a command; "run down the list of eviction
    # policies" is a question and falls through.
    command_text = actions.match_command(question)
    if command_text is not None:
        try:
            parsed = actions.parse_command(command_text, str(event.get("user", "")))
        except actions.CommandError as refusal:
            deliver(text=_plain(str(refusal)), thread_ts=thread)
            return
        except Exception:
            log.exception("command handling failed")
            deliver(
                text="That command could not be processed. The failure is logged.", thread_ts=thread
            )
            return
        if isinstance(parsed, str):
            deliver(text=_plain(parsed), thread_ts=thread)
            return
        _react(client, event, _WORKING)
        try:
            reply = actions.execute(parsed)
        except actions.CommandError as refusal:
            # A refusal (the cooldown) is an answer, not a failure.
            deliver(text=_plain(str(refusal)), thread_ts=thread)
            return
        except Exception:
            log.exception("command execution failed")
            deliver(text="That command failed to run. The failure is logged.", thread_ts=thread)
            return
        finally:
            # The eyes come off on EVERY exit, including a failed dispatch or a failed post;
            # a reaction left behind reads as the bot still working on it.
            _unreact(client, event, _WORKING)
        # The dispatch may have happened: a failed confirmation must NOT release the fence, or a
        # redelivery dispatches the workflow a second time.
        say(text=reply, thread_ts=thread, unfurl_links=False)
        return

    _react(client, event, _WORKING)
    failed = False
    try:
        conversation = _thread_history(event, client)
        started = time.monotonic()
        result = _ask(question, event, conversation)
        text = _format(result, seconds=time.monotonic() - started)
    except Exception:
        # Log the detail, tell the channel only that it failed. The fence is released so a
        # redelivery (or the asker's retry) is answered: the runtime replays a stored result and
        # re-runs an unfinished one, so neither costs a second inference.
        log.exception("answer failed")
        failed = True
    finally:
        # The eyes come off BEFORE the fence is released below: a retry that started in between
        # added its own eyes, which this handler then removed.
        _unreact(client, event, _WORKING)
    if failed:
        _release(key)
        say(text="Something went wrong answering that. The failure is logged.", thread_ts=thread)
        return
    deliver(text=_bounded_reply(text), thread_ts=thread, unfurl_links=False)


assistant = Assistant()

SUGGESTED_PROMPTS = [
    {"title": "What happened lately", "message": "What happened in Valkey in the last two weeks?"},
    {"title": "Review queue", "message": "Which pull requests need review?"},
    {
        "title": "CI on unstable",
        "message": "Why is CI failing on unstable? Are there fixes open already?",
    },
    {
        "title": "A config option",
        "message": "What does maxmemory-policy do and what is its default?",
    },
]


@assistant.thread_started
def greet_assistant_thread(say: Any, set_suggested_prompts: Any) -> None:
    """The first thing a person sees in the assistant panel: what to ask, with examples."""
    say(
        "I answer questions about Valkey from its repositories and live GitHub state, with "
        "sources. Ask about commands, configuration, governance, releases, CI, or what is "
        "happening in the project."
    )
    set_suggested_prompts(prompts=SUGGESTED_PROMPTS)


@assistant.user_message
def answer_assistant_message(
    payload: dict[str, Any], say: Any, set_status: Any, set_title: Any, client: Any
) -> None:
    """A message typed in the assistant panel or DM: the same answer path as a mention.

    The status line is the progress signal a mention never had (the reactions scopes were never
    granted); Slack clears it when the reply posts. The request id derives from the event the
    same way, so a redelivery replays rather than re-infers.
    """
    if EXPECTED_TEAM and payload.get("team") not in {EXPECTED_TEAM, None}:
        log.warning("ignoring assistant message from unexpected team %s", payload.get("team"))
        return
    if payload.get("bot_id") or payload.get("subtype"):
        return
    question = (payload.get("text") or "").strip()
    if not question:
        say("Ask me a Valkey question, for example: what is the TSC?")
        return
    if len(question.encode("utf-8")) > MAX_QUESTION_BYTES:
        say(f"That is over my {MAX_QUESTION_BYTES // 1024} KB limit for one message.")
        return
    key = _event_key(payload)
    if not _claim(key):
        return

    def deliver(*args: Any, **kwargs: Any) -> None:
        try:
            say(*args, **kwargs)
        except Exception:
            _release(key)
            raise

    try:
        set_status("reading Valkey sources...")
    except Exception as error:  # noqa: BLE001 - the status line is decoration
        log.debug("could not set assistant status (%s)", _brief(error))
    try:
        conversation = _thread_history(payload, client)
        started = time.monotonic()
        result = _ask(question, payload, conversation)
        text = _format(result, seconds=time.monotonic() - started)
    except Exception:
        log.exception("assistant answer failed")
        _release(key)
        say("Something went wrong answering that. The failure is logged.")
        return
    if not payload.get("thread_ts") or payload.get("thread_ts") == payload.get("ts"):
        # The panel names each conversation; the question, cut short, is the natural title.
        try:
            set_title(question[:60])
        except Exception as error:  # noqa: BLE001 - a title is decoration
            log.debug("could not set assistant thread title (%s)", _brief(error))
    deliver(text=_bounded_reply(text), unfurl_links=False)


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
    raw = response["Payload"].read()
    if "FunctionError" in response:
        # The error body can carry anything the function raised. Its type is enough to act on.
        kind = "unknown"
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("errorType"), str):
                kind = parsed["errorType"]
        except ValueError:
            pass
        raise RuntimeError(f"lambda reported an error of type {kind}")
    body = json.loads(raw)
    if not isinstance(body, dict) or not isinstance(body.get("outcome"), str):
        # A list or a bare value is a broken contract, not an abstention; it used to render as
        # "I don't have grounded evidence for that (unknown)".
        raise RuntimeError("lambda response is not a result object")
    return body


# Events answered by this process, oldest first. Bounded so a long-lived bot does not grow.
_answered: dict[str, bool] = {}
_answered_lock = threading.Lock()
# Slack's hard limit is 40,000 characters per message; the reply body keeps well under it so the
# sources and closers always fit after the claims.
MAX_REPLY_CHARS = 30_000
MAX_ANSWERED_EVENTS = 4096
MAX_HISTORY_PAGES = 5
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
        messages: list[dict[str, Any]] = []
        cursor: str | None = None
        current = event.get("ts", "")
        # Slack pages a long thread. The first page is the OLDEST messages, so a 20-message
        # thread answered a follow-up against messages 9 to 14 instead of the ones just before
        # it. Pages are followed until the current mention is on one, within a small bound.
        for _ in range(MAX_HISTORY_PAGES):
            kwargs: dict[str, Any] = {"channel": event["channel"], "ts": thread, "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            replies = client.conversations_replies(**kwargs)
            messages.extend(replies.get("messages", []))
            if any(m.get("ts") == current for m in messages):
                break
            cursor = (replies.get("response_metadata") or {}).get("next_cursor") or None
            if not cursor:
                break
        else:
            return []  # The mention was never found: stale history is worse than none.
    except Exception as error:  # noqa: BLE001 - history is optional; the answer is not.
        if not _history_unavailable_logged:
            log.warning("thread history unavailable, answering without it: %s", _brief(error))
            _history_unavailable_logged = True
        return []
    if not any(m.get("ts") == current for m in messages):
        return []
    return _turns_before(
        messages,
        current,
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
        elif message.get("bot_id") or not message.get("user"):
            # Another bot, or a system message with no human author, is not the asker's words
            # even when it opened the thread.
            continue
        elif asker and message.get("user") != asker and message.get("ts") != root_ts:
            continue
        else:
            role = "user"
        if role == "assistant":
            # Keep the claims, drop the Sources block: links are not conversation.
            text = text.split("\n\n_Sources", 1)[0].split("\n\n*Sources*", 1)[0].strip()
        else:
            # A token pasted earlier in the thread must not travel to the Lambda and the router
            # model inside the history; the same shapes the runtime redacts are redacted here.
            text = _CREDENTIAL_SHAPES.sub("[redacted credential]", text)
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_HISTORY_TURN_BYTES:
            text = encoded[:MAX_HISTORY_TURN_BYTES].decode("utf-8", "ignore")
        turns.append({"role": role, "text": text})
    return turns[-MAX_HISTORY_TURNS:]


def _format(result: dict[str, Any], *, seconds: float | None = None) -> str:
    """Render a response by its outcome. The runtime explains itself in `message`."""
    outcome = result.get("outcome", "unknown")
    claims = result.get("claims") or []
    citations = result.get("citations") or []
    message = result.get("message")

    if claims:
        # Slack refuses a message over 40,000 characters and would truncate a fence mid-block if
        # it did not. The answer is assembled whole claim by whole claim under a cap that leaves
        # room for the sources and closers, and the count of claims left out is said.
        # The sources and closers are rendered FIRST and their size reserved, so the whole
        # message, not only the claims, stays under the cap.
        sources = ""
        if citations:
            # Citations are application-authored from validated GitHub URLs, so the link markup is
            # built here; the label is still escaped since it carries a path.
            # Two windows of the same file at the same commit render identically; one line.
            links = list(dict.fromkeys(_source_link(c) for c in citations))
            labels = sum(len(_label_of(link)) for link in links)
            if len(links) <= 3 and labels <= 120:
                # A few short sources fit on one line; a heading over one link was three rows
                # of chrome under a one-sentence answer.
                sources = "\n\n_Sources:_ " + " \u00b7 ".join(links)
            else:
                sources = "\n\n_Sources_\n" + "\n".join(f"  {link}" for link in links)
        # The answer ends with its stated limitation, when it has one, and nothing else: the
        # "Checked 3 sources in 25 s" line and the pointer to the human channels were chrome
        # that the fifth reply taught people to skip. An abstention still gets the pointer, in
        # _format, because it is the one reply with nothing else to offer.
        tail = f"\n\n_{_spoken(_plain(message))}_" if message else ""
        budget = MAX_REPLY_CHARS - len(sources) - len(tail) - 80
        rendered: list[str] = []
        used = 0
        repository = _sole_repository(citations)
        texts = [c for c in claims if c.get("text")]
        for position, claim in enumerate(texts):
            body = _spoken(_linked(_plain(claim["text"]), repository))
            if position == 0 and _is_a_lead(body, len(texts)):
                # The first claim answers the question in one sentence (the prompt asks for
                # exactly that). Set as a plain line, it reads as the answer; as the first of
                # six identical bullets it read as one more fact.
                line = body + ("\n" if len(texts) > 1 else "")
            else:
                line = _grouped(_itemized(_led(body)))
            if used + len(line) > budget:
                if not rendered:
                    # A single claim over the whole budget: cut it rather than send over the cap.
                    line = line[: max(0, budget - 60)] + (
                        "\n```" if line[:budget].count("```") % 2 else ""
                    )
                    rendered.append(line)
                left = len(texts) - position - (0 if rendered else 1)
                if left > 0:
                    rendered.append(
                        f"\u2022 _{left} more claim(s) did not fit in one Slack message._"
                    )
                break
            rendered.append(line)
            used += len(line) + 1
        return "\n".join(rendered) + sources + tail

    # No claims is not one situation. A clarification is the assistant asking something
    # back, a partial means evidence was reachable but incomplete, and an abstention is a
    # deliberate refusal. Collapsing all three into a refusal loses the actual reply.
    if outcome == "clarification":
        return (
            _plain(message) if message else "What would you like to know about the Valkey project?"
        )
    if outcome == "partial":
        # The runtime's partial messages name internal conditions ("request lease is still
        # active", "retrieval evidence exceeds its byte bound"). Logged, not shown.
        log.warning("partial result: %s", message)
        return (
            "I couldn't finish that one: part of what I needed was not reachable just now. "
            "Please ask again in a moment."
        )
    if outcome == "error":
        log.warning("error result: %s", message)
        return "I couldn\u2019t produce a reliable answer. Please try again."
    if message:
        # An abstention is the one reply with nothing to click; it gets the pointer to people.
        return _bounded_reply(f"{_spoken(_plain(message))}\n\n_{_MORE_HELP}_")
    return f"I don't have grounded evidence for that ({outcome})."


# Closing pointers. Short by design: a paragraph of boilerplate under every answer trains people to
# stop reading the part that matters.
_MORE_HELP = "For more: the Valkey Slack help channels, GitHub Discussions, or valkey.io."

# Spoken forms for the phrases the model writes from its instructions. "The evidence does not
# include X" is true and reads like a form letter; "I couldn't find X" says the same thing the way
# a colleague would. Only exact stems are rewritten, so nothing about Valkey itself is reworded.
_SPOKEN: tuple[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]], ...] = (
    (
        # "Given these facts, the most likely cause is ..." says nothing before the comma that the
        # citation does not already say; the conclusion stands on its own. "Given that X, Y" keeps
        # its X, which is content.
        re.compile(
            r"^Given (?:these|those|the above|all this|this|the preceding|the|what I read)"
            r"(?: facts| recommendations| guidance| constraints| defaults| points"
            r"| behaviou?rs| documentation| evidence| sources)?,\s*"
            r"(\w)"
        ),
        lambda m: m.group(1).upper(),
    ),
    (
        re.compile(r"\b(?:are|is) (?:not )?listed in the (?:supplied |available )?evidence\b"),
        lambda m: (
            m.group(0)
            .replace("the evidence", "the results")
            .replace("supplied ", "")
            .replace("available ", "")
        ),
    ),
    (re.compile(r"\bAt (?:the )?observation(?: time)?\b,?\s*"), "When I checked, "),
    (re.compile(r"\bat (?:the )?observation(?: time)?\b"), "when I checked"),
    (
        re.compile(r"\b[Aa]s of the (?:latest |last |most recent )?observation\b(,?)(\s*)"),
        lambda m: "when I last checked" + (", " if m.group(1) or m.start() == 0 else m.group(2)),
    ),
    (re.compile(r"\b[Aa]s observed on (\d{4}-\d{2}-\d{2})\b"), r"when I checked on \1"),
    (
        re.compile(
            r"^The (?:supplied |available |retrieved |provided )?evidence "
            r"(?:does not|doesn\u2019t|doesn't) (?:include|contain|show|name|document|cover|list|"
            r"state|specify|mention|identify|record|say)\b"
        ),
        "I couldn't find",
    ),
    (
        re.compile(
            r"^The (?:supplied |available |retrieved |provided )?evidence "
            r"(?:contains|includes|shows|has|names|lists|offers|provides) no\b"
        ),
        "I found no",
    ),
    (
        re.compile(r"\b[Gg]iven (?:the |this |all the )?(?:supplied |available )?evidence,?\s*"),
        "From what I read, ",
    ),
    # Any other mention of "the evidence" is the model talking about its reading material; a
    # person says "what I read" at the start of a sentence (singular, so a following "it" agrees)
    # and "my reading" inside one ("a decision my reading cannot settle", "in my reading").
    (
        re.compile(
            r"\bThe (?:supplied |available |retrieved |provided |cited )?evidence\b(?: here)?"
        ),
        "What I read",
    ),
    (
        re.compile(
            r"\bthe (?:supplied |available |retrieved |provided |cited )?evidence\b(?: here)?"
        ),
        "my reading",
    ),
)
_ISO_INSTANT = re.compile(
    r"(?<![\w:/.-])(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?Z(?![\w:/-]|\.\w)"
)
_ISO_DAY = re.compile(r"(?<![\w:/.-])(\d{4})-(\d{2})-(\d{2})(?![\w:/-]|\.\w)")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _day(match: re.Match[str]) -> str:
    """A real calendar date (and, for an instant, a real time of day) as "15 Sep 2026"; anything
    that only looks like one ("2026-02-31T99:99:99Z" in a token) is left exactly as written."""
    from datetime import date

    try:
        when = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if match.re is _ISO_INSTANT:
            hour, minute, second = (int(match.group(i)) for i in (4, 5, 6))
            if not (hour < 24 and minute < 60 and second < 60):
                return match.group(0)
    except ValueError:
        return match.group(0)
    rendered = f"{when.day} {_MONTHS[when.month - 1]} {when.year}"
    if match.re is _ISO_INSTANT:
        # "merged on 15 Sep 2026" dropped 21:15:53Z; the time matters for a merge or a release.
        rendered += f", {int(match.group(4)):02d}:{int(match.group(5)):02d} UTC"
    return rendered


# Text the renderer must never rewrite or split inside: fenced code, inline code, and anything in
# double quotes (a quoted error message, a quoted command, quoted evidence). The prompt tells the
# model to keep quoted evidence exactly as spelled, and "CONFIG SET appendonly yes; CONFIG GET
# appendonly" inside quotes was split at its semicolon into two broken lines.
_PROTECTED = re.compile(r"```.*?```|``[^\n]*?``|`[^`\n]*`|\"[^\"\n]+\"", re.DOTALL)


def _protected_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _PROTECTED.finditer(text)]


def _is_protected(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _outside_protected(text: str, transform: Callable[[str], str]) -> str:
    """Apply `transform` to the stretches of `text` outside protected spans, in place."""
    spans = _protected_spans(text)
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        out.append(transform(text[cursor:start]))
        out.append(text[start:end])
        cursor = end
    out.append(transform(text[cursor:]))
    return "".join(out)


def _split_outside(text: str, delimiter: re.Pattern[str]) -> list[str]:
    """Split at delimiter matches that lie outside protected spans."""
    spans = _protected_spans(text)
    pieces: list[str] = []
    cursor = 0
    for m in delimiter.finditer(text):
        if _is_protected(m.start(), spans):
            continue
        # A delimiter inside parentheses belongs to the parenthetical: "(the offset it
        # processed; not the primary's)" was cut into a broken sentence.
        before = text[: m.start()]
        if before.count("(") > before.count(")"):
            continue
        pieces.append(text[cursor : m.start()])
        cursor = m.end()
    pieces.append(text[cursor:])
    return pieces


def _spoken(text: str) -> str:
    """Prose as a person would say it: spoken stems, and dates as "15 Sep 2026" rather than
    "2026-09-15T21:15:53Z". Code, inline code and quoted text are left exactly as written."""

    def prose(piece: str) -> str:
        for pattern, replacement in _SPOKEN:
            piece = pattern.sub(replacement, piece)
        piece = _ISO_INSTANT.sub(_day, piece)
        return _ISO_DAY.sub(_day, piece)

    return _outside_protected(text, prose)


def _is_a_lead(body: str, count: int) -> bool:
    """Whether the first claim can stand as the answer line: one sentence, not a list, short
    enough to read at a glance. Otherwise it is rendered like the rest."""
    if count == 1:
        return True
    visible = re.sub(r"<[^|>]+\|([^>]+)>", r"\1", body)
    if "\n" in body or len(visible) > LEAD_MAX_CHARS:
        return False
    if len(_ITEM_REFERENCE.findall(body)) >= MIN_ITEMS_TO_LIST:
        return False
    # "Before coding a major feature, open an issue..." is a prerequisite, not the answer.
    if re.match(r"(?:Before|After|If|When(?! I (?:last )?checked\b)|Unless|While)\b", visible):
        return False
    if visible.count(",") >= 6 and _enumerated(visible) is not None:
        return False  # a 21-item list reads down a list, not across a lead line
    return len(_SENTENCE_BREAK.split(body)) == 1


def _live_label(kind: str, name: str, url: str) -> str:
    """A live source named the way a person would point at it: "PR search: is:open fix test
    failure" rather than "issue (PR search: is:open fix test failure)", "valkey.conf (live)" rather
    than "file (valkey.conf)", "#3853" rather than "issue (#3853)"."""
    if kind in ("issue", "issue search"):
        if "/pull/" in url and name.startswith("#"):
            return f"PR {name}"
        if name.startswith(("PR search:", "issue search:")):
            return name.replace("PR search:", "PRs matching", 1).replace(
                "issue search:", "issues matching", 1
            )
        return name or kind
    if kind == "pull request":
        return f"PR {name}" if name else "pull request"
    if kind == "release":
        return f"release {name}" if name else "releases"
    if kind == "workflow run":
        if name.startswith("runs "):
            return f"workflow runs on {name.removeprefix('runs ').removeprefix('branch:')}"
        return name or "workflow run"
    if kind == "file":
        return f"{name} (live)" if name else "file"
    if kind == "directory":
        ref = re.search(r"/tree/([^/]+)", url)
        at = _at_ref(ref.group(1)) if ref else ""
        if name in ("", "/"):
            repository = re.search(r"github\.com/valkey-io/([^/]+)", url)
            return (f"{repository.group(1)} root listing" if repository else "root listing") + at
        return f"{name} listing{at}"
    if kind == "compare":
        return name or "compare"
    if kind == "controller status":
        return f"project board {name}" if name else "project board"
    if kind == "generic":
        if name.startswith("@"):
            return f"{name}'s profile"
        if name.endswith(" files"):
            return f"files changed by {name.removesuffix(' files')}"
        if "/commits/" in url:
            return f"commits on {url.rsplit('/commits/', 1)[1]}"
        return name or "GitHub"
    return f"{kind} ({name})" if name else kind


def _label_of(link: str) -> str:
    return link.rsplit("|", 1)[-1].rstrip(">")


def _names(url: str) -> str:
    """What a GitHub URL is ABOUT, for a citation label: a tag, a number, or a file path."""
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    path = parsed.path
    # Listings are labelled by their query, since the URL has no object in it: a search by its
    # terms and qualifiers, a runs page by its branch, a compare by its refs, a history by its path.
    if path.endswith(("/pulls", "/issues")) and query.get("q"):
        words = [
            w for w in query["q"][0].split() if not w.startswith(("is:pull-request", "is:issue"))
        ]
        kind = "PR search" if path.endswith("/pulls") else "issue search"
        return f"{kind}: {' '.join(words)[:60]}" if words else kind
    if path.endswith("/actions") and query.get("query"):
        return f"runs {query['query'][0]}"
    if "/actions/runs/" in path:
        return f"run {path.rsplit('/', 1)[-1]}"
    if "/compare/" in path:
        return path.split("/compare/", 1)[1]
    if "/commits/HEAD/" in path:
        return f"history of {path.split('/commits/HEAD/', 1)[1]}"
    if "/commit/" in path:
        return f"commit {path.rsplit('/', 1)[-1][:10]}"
    if "/orgs/valkey-io/projects/" in path:
        return path.rsplit("/", 1)[-1]
    if "/tree/" in path:
        # /valkey-io/valkey/tree/HEAD/src/commands -> src/commands
        tail = path.split("/tree/", 1)[1].split("/", 1)
        return tail[1] if len(tail) == 2 else "/"
    if path.endswith("/files") and "/pull/" in path:
        return f"#{path.split('/pull/', 1)[1].split('/', 1)[0]} files"
    if path.endswith("/graphs/contributors"):
        return "contributors"
    if path.count("/") == 1 and not path.startswith("/valkey-io") and not parsed.query:
        # A bare /login path is a user profile; github.com/search?q=... is not.
        if path[1:] not in {"search", "features", "about", "pricing", "login"}:
            return f"@{path[1:]}"
    parts = [part for part in path.split("/") if part]
    for marker, render in (
        ("tag", lambda rest: rest[0] if rest else ""),
        ("pull", lambda rest: f"#{rest[0]}" if rest else ""),
        ("issues", lambda rest: f"#{rest[0]}" if rest else ""),
        # A file URL carries the commit between blob and the path, which the link already pins.
        # A release tag or branch there is worth saying: valkey.conf at 9.0.0 and at 8.0.0 were
        # two identical labels.
        ("blob", lambda rest: "/".join(rest[1:]) + _at_ref(rest[0] if rest else "")),
        ("tree", lambda rest: "/".join(rest[1:])),
    ):
        if marker in parts:
            return render(parts[parts.index(marker) + 1 :])
    return ""


def _at_ref(ref: str) -> str:
    """ " at 9.0.0" for a tag or branch; nothing for a commit hash or HEAD, which the link pins."""
    if (
        not ref
        or ref in {"HEAD", "unstable", "main", "master"}
        or re.fullmatch(r"[0-9a-f]{40}", ref)
    ):
        return ""
    return f" at {ref}"


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
        label = _live_label(kind, _names(url), url)
    else:
        # repo/path@commit -> repo/path, since the link carries the commit already.
        label = label.split("@", 1)[0]
    return f"<{url}|{_plain(label)}>"


_REPOSITORY_IN_URL = re.compile(
    r"https://(?:api\.)?github\.com/(?:repos/)?valkey-io/([A-Za-z0-9._-]+)"
    r"|repo(?:%3A|:)valkey-io(?:%2F|/)([A-Za-z0-9._-]+)"
)
_REFERENCE = re.compile(
    r"(?<![\w/#])#(\d{1,6})\b"  # #4797
    # run 36943607261: GitHub run ids are nine digits or more; "run 10000 iterations" is not one.
    r"|\b(run) (\d{9,12})\b"
    r"|\b([0-9a-f]{40})\b"  # a full commit hash
    # "issue 4153", "PR 4795", "pull request 4795", with or without a hash: the model writes all
    # of these, and an unlinked number is a number the reader has to retype.
    r"|\b((?:issue|PR|pull request|pull-request)) #?(\d{1,6})\b",
    re.IGNORECASE,
)


_ITEM_REFERENCE = re.compile(r"(?<![\w/])(?:<[^|>]+\|)?#\d{1,6}\b")
# The same reference including a link's closing marker, for deciding whether an item is ONLY
# references.
_WHOLE_REFERENCE = re.compile(r"(?<![\w/])(?:<[^|>]+\|#\d{1,6}>|#\d{1,6}\b)")
# Text ending just before a reference that starts a list item ("include #", "fixes #", ", #").
_LIST_SEPARATOR_BEFORE = re.compile(
    r"(?:^|[,;:]|\band\b|\bor\b|\binclud(?:e|es|ing)|\bissues|\bfixes)\s*$"
)
# Text right after a reference that continues into another: "tracked in #4138, #4137" is a list
# even though its first reference follows a preposition.
_BEGINS_A_LIST = re.compile(r">?\s*(?:,|\band\b|\bor\b)\s*(?:<[^|>]+\|)?#\d")
# Text ending in a preposition the reference completes ("disabled in #", "the fix from PR #").
_COMPLETES_PHRASE_BEFORE = re.compile(
    r"\b(?:in|from|by|of|to|at|for|with|see|than|under|via)\s*(?:PR|pull request|issue)?\s*$",
    re.IGNORECASE,
)
MIN_ITEMS_TO_LIST = 4


def _itemized(text: str) -> str:
    """A claim that enumerates four or more numbered items becomes a bullet with sub-items.

    "#4797 Forkless Full-Sync, #4782 Make quicklist ..., #4754 ..." twenty times is a paragraph a
    person cannot scan. The text is cut where each reference starts: what comes before the first
    is the lead-in and stays the bullet, each reference with the words that follow it is one
    indented line, with the joining comma, semicolon or "and" trimmed off. Claims with fewer
    references are left exactly as written.
    """
    if not text.startswith("• ") or "\n" in text:
        return text
    text = text[2:]
    # A reference that completes the words before it ("disabled in #858", "the fix from #4795")
    # is part of the current item, not the start of the next one. An item starts at a reference
    # that follows a list separator (comma, semicolon, "and", "or") or begins the enumeration.
    protected = _protected_spans(text)
    starts = [
        m.start()
        for m in _ITEM_REFERENCE.finditer(text)
        if not _is_protected(m.start(), protected)
        and not text[: m.start()].rstrip().endswith("(")
        and (
            _LIST_SEPARATOR_BEFORE.search(text[: m.start()])
            or _BEGINS_A_LIST.match(text[m.end() :])
            or not _COMPLETES_PHRASE_BEFORE.search(text[: m.start()])
        )
    ]
    if len(starts) < MIN_ITEMS_TO_LIST:
        return f"• {text}"
    # "PR #4795 changes X, closing issues #1, #2, #3, #4": the first reference is the subject of
    # the sentence, not an item. A lead of one or two words takes the first piece into itself.
    lead_words = text[: starts[0]].split()
    if (
        1 <= len(lead_words) <= 2
        and lead_words[-1].lower().rstrip(":") in {"pr", "issue", "commit", "run", "request"}
        and len(starts) > MIN_ITEMS_TO_LIST
    ):
        starts = starts[1:]
    lead = text[: starts[0]].strip()
    pieces = [text[a:b] for a, b in zip(starts, [*starts[1:], len(text)], strict=True)]
    items = [re.sub(r"[\s,;]*(?:\band\b)?[\s,;]*$", "", piece).rstrip(".") for piece in pieces]

    # "#4442 and #4375 fixing the link failure" is two references sharing one description. Split
    # at the references, the first became a bare number. A bare reference joins the next item so
    # both numbers keep the words that were about them.
    def bare(item: str) -> bool:
        # Only references and separators: "#4138", "<url|#4138>", "#4138, #4137".
        return bool(item.strip()) and not _WHOLE_REFERENCE.sub("", item).strip(" ,").strip()

    merged: list[str] = []
    for item in items:
        if merged and bare(merged[-1]):
            # A bare reference followed by another bare reference is a list of numbers (nine fixed
            # issues); joined with commas they are one line. Followed by text, it shares that text.
            joiner = ", " if bare(item) else " and "
            merged[-1] = f"{merged[-1]}{joiner}{item}"
        else:
            merged.append(item)
    lines = [f"• {lead}" if lead else "• The items:"] + [f"    ◦ {item}" for item in merged]
    return "\n".join(lines)


# "9.0: ...; 9.1: ..." or "valkey-glide: ...; valkey-py: ...": two or more segments, each opening
# with a short label and a colon, joined by semicolons. A version, a repository or a client name.
_GROUP_LABEL = re.compile(r"^(?:[A-Za-z][\w.+-]{0,30}|\d+(?:\.\d+){1,2}(?:-rc\d+)?):\s")
MIN_GROUPS = 2


# A sentence end followed by a capital letter or a link: where a long claim can break.
# A sentence ends at . ! or ? followed by a capital, but not at an ellipsis: "'=== ... BUG REPORT
# START'" is one quoted log line, and splitting it put "BUG REPORT START" on its own row.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?<!\.\.\.)\s+(?=[A-Z<])")
# "; " joins clauses, except when the semicolon ends an HTML entity: Slack text carries < > & as
# &lt; &gt; &amp;, and "MAXBYTES &lt;bytes&gt; [LIMIT" split at "&gt;" left "&gt" broken on one row
# and "[LIMIT" on the next.
_CLAUSE_BREAK = re.compile(r"(?<!&lt)(?<!&gt)(?<!&amp)(?<!&quot)(?<!&#39); ")
LEAD_MAX_CHARS = 220


def _led(text: str) -> str:
    """A long claim of several sentences: the first sentence is the bullet, the rest is indented.

    A 300-character bullet is a paragraph; the eye finds nothing in it. The first sentence is what
    the claim asserts, the rest is how it knows, and setting them apart makes both readable while
    dropping nothing. Short claims and single sentences are left exactly as written.
    """
    if len(text) <= LEAD_MAX_CHARS:
        # A 214-character sentence listing 21 ACL categories is still a list to read down.
        enumerated = _enumerated(text) if text.count(",") >= 6 else None
        return enumerated if enumerated is not None else f"• {text}"
    sentences = _SENTENCE_BREAK.split(text, maxsplit=1)
    if len(sentences) >= 2:
        # Indented, the second sentence reads as part of the first. "The dictEntry no longer has a
        # next pointer." indented under a bullet about a pull request's goal read as a dangling
        # fragment; a sentence about something else is its own bullet.
        joint = "\n    " if _continues(sentences[0], sentences[1]) else "\n• "
        return f"• {sentences[0]}{joint}{sentences[1]}"
    # One long sentence: its semicolon-joined clauses are separate points and read as such.
    enumerated = _enumerated(text)
    if enumerated is not None:
        return enumerated
    clauses = [c.strip() for c in _split_outside(text, _CLAUSE_BREAK) if c.strip()]
    if len(clauses) >= 2:
        # The semicolon the split consumed becomes a period, and the continuation starts a
        # sentence: "...from its backlog\n    if that is not possible..." read as a fragment.
        # Lower-case names keep their spelling (_sentence_case leaves jemalloc alone).
        first, rest = _ended(clauses[0]), clauses[1:]
        return "\n".join([f"• {first}"] + [f"    {_ended(_sentence_case(c))}" for c in rest])
    # A long single sentence with a strong contrast is two thoughts: "X, but Y", "X, although Y".
    # "so" is left joined: a causal sentence reads as one and split as a non sequitur.
    for match in _STRONG_JOIN.finditer(text):
        head, tail = text[: match.start()].rstrip(), text[match.end() :].strip()
        if len(head) >= 60 and len(tail) >= 60:
            # The head is a sentence now and ends like one ("...nothing is evicted" ran into
            # "But the node..." with no period between them).
            return f"• {_ended(head)}\n    {_sentence_case(match.group(1))} {tail}"
    return f"• {text}"


def _continues(first: str, second: str) -> bool:
    """Whether the second sentence elaborates the first: it refers back to it (It, This, That,
    Otherwise, ...) or shares a substantive word with it. Two sentences with neither in common
    are two facts."""
    if _BACK_REFERENCE.match(second):
        return True

    def words(sentence: str) -> set[str]:
        found = re.findall(r"[A-Za-z][\w-]{4,}", sentence)
        return {w.casefold().strip("'`\"") for w in found} - _COMMON_WORDS

    return bool(words(first) & words(second))


_BACK_REFERENCE = re.compile(
    r"(?:It|Its|This|That|These|Those|Such|Here|There|Both|Each|Either|Neither|Otherwise|Instead|"
    r"So|Then|Hence|Thus|Therefore|However|Still|Also|In (?:that|this) case|If (?:so|not)|"
    r"The (?:same|former|latter|result|default|rest|fix|change|check))\b"
)
_COMMON_WORDS = frozenset(
    "about above after again against along among around because before being below between "
    "could during either every first found further having itself later might never often other "
    "others rather really right second should since still their there these those through under "
    "until using value values where which while whose would within without".split()
)


# "while" and "whereas" are left joined: a contrast reads as one thought, split it reads as two.
_STRONG_JOIN = re.compile(r",\s+(but|which means|although)\s+")
_LISTING_VERB = re.compile(
    r"\b(?:includes?|including|such as|are|were|adds?|added|lists?|supports?|provides?|"
    r"offers?|gained|"
    r"brings?|comprises?)(?:\s+the\s+following)?:?(?=\s)"
)


def _split_items(text: str) -> list[str]:
    """Split on commas outside brackets, quotes and code; a final ", and X" is one more item."""
    items: list[str] = []
    depth = 0
    current = []
    spans = _protected_spans(text)
    for position, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0 and not _is_protected(position, spans):
            items.append("".join(current))
            current = []
        else:
            current.append(ch)
    items.append("".join(current))
    cleaned = []
    for item in items:
        item = item.strip().rstrip(".")
        item = re.sub(r"^(?:and|or)\s+", "", item)
        if item:
            cleaned.append(item)
    return cleaned


def _enumerated(text: str) -> str | None:
    """ "New features in 9.0 include A, B, C, D, E, F, G and H." is a list a reader scans down,
    not across. Five or more short items after a listing verb become sub-bullets. A sentence
    whose items are long clauses, or that has a second sentence, is left alone."""
    text = text.rstrip()
    if _SENTENCE_BREAK.search(text) or ";" in text:
        return None
    # The first word that looks like a listing verb may be a noun ("Maintenance support and
    # security support end dates are: ..."); every candidate is tried and the first that yields a
    # clean list wins.
    for verb in _LISTING_VERB.finditer(text):
        lead, body = text[: verb.end()], text[verb.end() :].strip()
        if not 3 <= len(lead) <= 160 or len(body) < 80:
            continue
        items = _split_items(body)
        # "..., and 7.2 (16 Apr 2024), each with maintenance and security end dates" ends in a
        # qualifier about the whole list, not one more item; it follows the list on its own line.
        trailer = None
        if len(items) >= 2 and _TRAILER.match(items[-1]):
            trailer = items.pop()
        if len(items) < 5 or any(len(item) > 70 or len(item) < 2 for item in items):
            continue
        if any(
            item.lower() in _NOT_AN_ITEM or item.lower().split(" ", 1)[0] in _NOT_AN_ITEM
            for item in items
        ):
            continue  # "so", "but", "in practice" between commas are clauses, not items
        # "..., volatile-random, volatile-ttl so the server frees keys automatically": the last
        # item carries the sentence's closing clause. Listed, the clause hung off one policy.
        longest_other = max(len(item.split()) for item in items[:-1])
        if len(items[-1].split()) > 2 * longest_other + 1 and _CLAUSE_IN_ITEM.search(items[-1]):
            continue
        if _LISTING_VERB.search(items[0]):
            # The list starts at a later verb; this one was a noun ("security support end
            # dates are: ...").
            continue
        lines = [f"• {lead.rstrip(':')}"] + [f"    \u25e6 {item}" for item in items]
        if trailer:
            lines.append(f"    {_sentence_case(trailer)}.")
        return "\n".join(lines)
    return None


def _ended(clause: str) -> str:
    """A clause with a sentence ending: a period is added only when nothing ends it already
    ("?", "!", an ellipsis or a code span were given "?." and "```.")."""
    stripped = clause.rstrip()
    if stripped.endswith(("?", "!", "...", "`", ":")):
        return stripped
    return stripped.rstrip(".") + "."


def _sentence_case(text: str) -> str:
    """Capitalize a continuation's first word unless it is a name that is spelled lower-case:
    "mem_not_counted_for_evict in INFO memory" must not become "Mem_not_counted_for_evict"."""
    first = text.split(" ", 1)[0].rstrip(",;:")
    # Only a word that is plainly English prose is capitalized; a lower-case word that could be a
    # name (jemalloc, valkey-cli, appendonly) keeps its spelling, since a wrong capital changes
    # what it refers to and a missing one does not.
    # A verb form ("choosing", "enabling", "compared") is prose too: no setting or tool is
    # spelled that way, and "choosing an eviction policy..." began a line in lower case.
    verb_form = (
        first.isalpha() and first.islower() and len(first) > 5 and first.endswith(("ing", "ed"))
    )
    if first.lower() not in _PROSE_STARTERS and not verb_form:
        return text
    return text[0].upper() + text[1:]


_PROSE_STARTERS = frozenset(
    "a an the this that these those it its they there here if when while so but and or otherwise "
    "however therefore instead because since once after before until unless although though "
    "meanwhile it's that's there's don't doesn't isn't we're you're "
    "then thus hence also only even each every all both some any no not use set run check try "
    "small large other another with without for from to in on at by as of is are was were be "
    "being been has have had do does did can could will would should may might must you your we "
    "our i my he she his her which who what where how why".split()
)


_CLAUSE_IN_ITEM = re.compile(
    r"\s(?:so|which|because|since|unless|if|when|while|to|in order to|and then)\s", re.IGNORECASE
)
_NOT_AN_ITEM = frozenset(
    "so but and or then however therefore because although though whereas while yet".split()
)
_TRAILER = re.compile(
    r"^(?:each|all|both|plus|with|which|respectively|among others|and more|as well as|where|"
    r"though|although|but|while|whereas|none of|most of|some of|many of)\b",
    re.IGNORECASE,
)


def _grouped(text: str) -> str:
    """A comparison written as "label: text; label: text" becomes one indented line per label.

    Applied to a single bullet only (an itemized claim already has structure). The label is
    bolded so the eye finds the sides of the comparison; the text after it is untouched.
    """
    if "\n" in text or not text.startswith("• "):
        return text
    body = text[2:]
    segments = [s.strip() for s in _split_outside(body, _CLAUSE_BREAK) if s.strip()]
    if len(segments) < MIN_GROUPS or not all(_GROUP_LABEL.match(s) for s in segments):
        return text
    lines = ["•"]
    for segment in segments:
        label, _, rest = segment.partition(": ")
        lines.append(f"    ◦ *{label}:* {rest.rstrip('.')}")
    return "\n".join(lines)


def _sole_repository(citations: Sequence[str]) -> str | None:
    """The one valkey-io repository every source names, or None when they name none or several.

    Inline links are derived from the evidence's own repository, never guessed from the text: a
    "#4797" in an answer grounded in valkey-glide must not link into valkey. When the sources
    span repositories the numbers stay plain text, which is honest rather than wrong.
    """
    names: set[str] = set()
    for citation in citations:
        for match in _REPOSITORY_IN_URL.finditer(citation):
            names.add(match.group(1) or match.group(2))
    return names.pop() if len(names) == 1 else None


def _linked(text: str, repository: str | None) -> str:
    """Pull request and issue numbers, run ids and commit hashes as links into the repository.

    A number a maintainer cannot click is a number they have to retype. GitHub redirects
    /issues/N to /pull/N when N is a pull request, so one form serves both. A hash is shown
    short with the full one in the link. Applied after escaping, so it never touches the model's
    own text as markup; and only when the sources name exactly one repository.
    """
    if repository is None:
        return text
    base = f"https://github.com/valkey-io/{repository}"

    def link(match: re.Match[str]) -> str:
        if match.group(1):
            return f"<{base}/issues/{match.group(1)}|#{match.group(1)}>"
        if match.group(2):
            return f"run <{base}/actions/runs/{match.group(3)}|{match.group(3)}>"
        if match.group(4):
            sha = match.group(4)
            return f"<{base}/commit/{sha}|{sha[:10]}>"
        word, number = match.group(5), match.group(6)
        return f"{word} <{base}/issues/{number}|#{number}>"

    return _outside_protected(text, lambda piece: _REFERENCE.sub(link, piece))


def _plain(text: str) -> str:
    """Escape Slack's three control characters in model-authored text.

    Claims and messages are grounded in GitHub content anyone can edit. In mrkdwn, `<!channel>`
    pages the channel and `<@U…>` mentions a person; escaped, they are just the text they were.
    The bot's own link markup is added after escaping, so it is never affected.
    """
    # Decode Slack's three entities first so escaping is idempotent: text quoted back from a
    # thread (an earlier reply in the history) arrives already escaped, and "&lt;" became
    # "&amp;lt;".
    decoded = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return decoded.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    """Build the Slack app at startup so the answer path stays importable and testable."""
    # request_verification_enabled=False because that middleware is for HTTP mode: it
    # verifies a signing secret on inbound POSTs. Socket Mode has no inbound endpoint and
    # events arrive over a pre-authenticated WebSocket, so Bolt would otherwise demand a
    # signing secret that serves no purpose here. Token verification stays on, so a bad
    # bot token fails at startup rather than on the first mention.
    # Refuse to start without AWS credentials. A restart that exported the Slack tokens but not
    # AWS_PROFILE connected to Slack, said "running", and failed every question with
    # NoCredentialsError. Checking here turns that into a startup error with the cause in it.
    try:
        identity = boto3.client("sts", region_name="us-east-1").get_caller_identity()
    except Exception as error:  # noqa: BLE001 - name the cause and stop
        raise SystemExit(
            f"AWS credentials are not usable ({type(error).__name__}: {error}). Export AWS_PROFILE "
            "before starting the bot; see tools/run_bot.sh."
        ) from error
    log.info("answering as %s", identity.get("Arn"))
    missing = [name for name in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN") if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} must be set; see tools/run_bot.sh.")
    # Identity alone is not permission: credentials that resolve but cannot invoke the function
    # would connect to Slack and fail every question. A DryRun invoke checks the exact function
    # and version without running it.
    try:
        lambda_client.invoke(FunctionName=FUNCTION, Qualifier=QUALIFIER, InvocationType="DryRun")
    except Exception as error:  # noqa: BLE001 - name the cause and stop
        raise SystemExit(
            f"cannot invoke {FUNCTION}:{QUALIFIER} ({type(error).__name__}: {error})."
        ) from error
    app = App(token=os.environ["SLACK_BOT_TOKEN"], request_verification_enabled=False)
    app.event("app_mention")(answer_mention)
    # The assistant surface: Slack's AI panel and DMs, with no mention needed, a visible status
    # while the bot works, and suggested prompts for someone who has never used it. Same answer
    # path behind it; only the way in differs. Needs the manifest's Agents & AI Apps feature and
    # the assistant:write scope; without them Slack never sends these events and the mention
    # path continues to work alone.
    app.use(assistant)
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
        log.warning(
            "could not resolve the bot user id, feedback will be ignored (%s)", _brief(error)
        )
    log.info("connecting to Slack in Socket Mode, serving %s:%s", FUNCTION, QUALIFIER)
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
