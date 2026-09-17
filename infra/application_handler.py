"""Private Lambda entrypoint for one immutable qualified application artifact.

Accepts two shapes. A raw runtime event (``{"action": "answer", ...}``) is passed
straight through, which is what the evaluation runners, operator tooling, and the
tests use. A Lambda Function URL request is translated into one raw event first, so
the public URL cannot reach anything the raw path does not already allow.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs

from valkeyrie.application_runtime import AwsRuntimeServices, RuntimeServices, run_runtime_event

_MANIFEST = Path(__file__).resolve().parents[1] / "application-artifact.json"
_ROOT = Path(__file__).resolve().parents[1]

_KNOWLEDGE_BASE_ID = "ONVASJDDNX"
_MAX_QUESTION_BYTES = 2048
_LEASE_SECONDS = 300
# Questions about present project state must route live; everything else answers from the
# pinned corpus. The same list drives tools/ask.py so the URL and the operator UI agree.
_LIVE_HINTS = (
    "current",
    "currently",
    "latest",
    "right now",
    "upcoming",
    "recent",
    "status of",
)


def handler(
    event: object,
    _context: object,
    services: RuntimeServices | None = None,
) -> dict[str, object]:
    """Run one private health or answer request against the packaged identity."""
    try:
        value = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("application artifact manifest is unavailable") from error
    if not isinstance(value, dict):
        raise RuntimeError("application artifact manifest is malformed")
    manifest = cast(dict[str, object], value)

    if _is_http_request(event):
        return _handle_http(cast(Mapping[str, object], event), services, manifest)

    dependencies = services
    if dependencies is None and isinstance(event, Mapping) and event.get("action") == "answer":
        dependencies = AwsRuntimeServices()
    return run_runtime_event(
        event,
        dependencies,
        root=_ROOT,
        manifest=manifest,
        completion_clock=_completion_now,
    )


def _completion_now() -> str:
    """The instant a request finished, read at completion rather than supplied by the caller.

    The event's completed_at is built before the model runs, so it describes when the caller was
    preparing the request. This is the deployment's own clock, which is what makes the audit
    record duration truthful.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _is_http_request(event: object) -> bool:
    if not isinstance(event, Mapping):
        return False
    context = event.get("requestContext")
    return isinstance(context, Mapping) and isinstance(context.get("http"), Mapping)


def _handle_http(
    event: Mapping[str, object],
    services: RuntimeServices | None,
    manifest: dict[str, object],
) -> dict[str, object]:
    try:
        question = _question(event)
    except ValueError as error:
        return _response(400, {"error": str(error)})

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    live = any(hint in question.casefold() for hint in _LIVE_HINTS)
    request = {
        "action": "answer",
        # Generated here, never taken from the request. This is the idempotency and lease
        # key, so a caller supplying it could replay or collide with another request.
        "request_id": f"req_url-{uuid.uuid4().hex}",
        "question": question,
        "version_requirement": "current_state" if live else "none",
        "requested_version": None,
        "knowledge_base_id": _KNOWLEDGE_BASE_ID,
        "owner": "public-url",
        "lease_duration_seconds": _LEASE_SECONDS,
        "now": stamp,
        "completed_at": stamp,
    }
    dependencies = services if services is not None else AwsRuntimeServices()
    try:
        result = run_runtime_event(
            request,
            dependencies,
            root=_ROOT,
            manifest=manifest,
            completion_clock=_completion_now,
        )
    except Exception:
        # Never surface internals to an unauthenticated caller.
        return _response(500, {"error": "the request could not be completed"})
    return _response(200, result)


def _question(event: Mapping[str, object]) -> str:
    raw = ""
    query = event.get("rawQueryString")
    if isinstance(query, str) and query:
        raw = parse_qs(query).get("q", [""])[0]
    if not raw:
        body = event.get("body")
        if isinstance(body, str) and body:
            if event.get("isBase64Encoded") is True:
                try:
                    body = base64.b64decode(body, validate=True).decode("utf-8")
                except (ValueError, UnicodeError) as error:
                    raise ValueError("request body is not valid base64 UTF-8") from error
            try:
                decoded = json.loads(body)
            except json.JSONDecodeError as error:
                raise ValueError("request body is not JSON") from error
            if not isinstance(decoded, Mapping):
                raise ValueError("request body must be a JSON object")
            candidate = decoded.get("question")
            raw = candidate if isinstance(candidate, str) else ""
    question = raw.strip()
    if not question:
        raise ValueError('supply a question as ?q= or a JSON body {"question": "..."}')
    if len(question.encode("utf-8")) > _MAX_QUESTION_BYTES:
        raise ValueError(f"question exceeds its {_MAX_QUESTION_BYTES}-byte bound")
    return question


def _response(status: int, payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }
