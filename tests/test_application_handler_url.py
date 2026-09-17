"""The Function URL adapter must never widen what the raw runtime path already allows."""

from __future__ import annotations

import base64
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from infra.application_handler import handler
from valkeyrie.application_runtime import _TIMESTAMP


class _Recorder:
    """Captures the translated runtime event instead of calling AWS."""

    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.events: list[dict[str, object]] = []
        self.result = result or {"outcome": "answer", "claims": [], "citations": []}


def _http(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "version": "2.0",
        "rawQueryString": "q=How+do+Valkey+replication+and+failover+behave%3F",
        "requestContext": {"http": {"method": "GET", "path": "/"}},
        "isBase64Encoded": False,
    }
    event.update(overrides)
    return event


def _run(monkeypatch: pytest.MonkeyPatch, event: dict[str, object]) -> tuple[dict[str, Any], Any]:
    captured: dict[str, Any] = {}

    def fake_run(
        request: object,
        services: object,
        *,
        root: object,
        manifest: object,
        completion_clock: object = None,
    ) -> dict[str, object]:
        captured["request"] = request
        captured["completion_clock"] = completion_clock
        return {"outcome": "answer", "claims": [], "citations": [], "generation_id": None}

    monkeypatch.setattr("infra.application_handler.run_runtime_event", fake_run)
    monkeypatch.setattr("infra.application_handler.AwsRuntimeServices", lambda: object())
    _patch_manifest(monkeypatch)
    response = handler(event, None)
    # An invariant of every handler path, not one case: the completion instant must come from the
    # deployment's own clock. The event's completed_at is built before the model runs, so it
    # describes when the caller was preparing the request, not when it finished.
    # Rejected requests never reach the runtime, so the clock is only required where it ran.
    if "request" in captured:
        clock = captured.get("completion_clock")
        assert callable(clock), "the handler did not supply a completion clock"
        assert _TIMESTAMP.fullmatch(cast(Callable[[], str], clock)()) is not None
    return cast(dict[str, Any], response), captured.get("request")


def _patch_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest only exists inside the packaged artifact, not in the source tree."""
    path = Path(tempfile.gettempdir()) / "valkeyrie_url_test_manifest.json"
    path.write_text(json.dumps({"application_revision": f"sha256:{'a' * 64}"}), encoding="utf-8")
    monkeypatch.setattr("infra.application_handler._MANIFEST", path)


def test_get_query_is_translated_into_one_answer_event(monkeypatch: pytest.MonkeyPatch) -> None:
    response, request = _run(monkeypatch, _http())

    assert response["statusCode"] == 200
    assert cast(dict[str, str], response["headers"])["content-type"] == "application/json"
    assert json.loads(cast(str, response["body"]))["outcome"] == "answer"
    assert request["action"] == "answer"
    assert request["question"] == "How do Valkey replication and failover behave?"
    assert request["version_requirement"] == "none"
    assert request["knowledge_base_id"] == "ONVASJDDNX"


def test_post_json_body_and_base64_body_both_supply_the_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = _http(rawQueryString="", body=json.dumps({"question": "Who governs Valkey?"}))
    _, request = _run(monkeypatch, plain)
    assert request["question"] == "Who governs Valkey?"

    encoded = _http(
        rawQueryString="",
        body=base64.b64encode(json.dumps({"question": "Who governs Valkey?"}).encode()).decode(),
        isBase64Encoded=True,
    )
    _, request = _run(monkeypatch, encoded)
    assert request["question"] == "Who governs Valkey?"


def test_request_identity_and_timestamps_are_never_taken_from_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # request_id is the idempotency and lease key. A caller who could set it could replay
    # or collide with another request, so a supplied one must be ignored entirely.
    hostile = _http(
        rawQueryString="q=hello",
        body=json.dumps(
            {
                "question": "hello",
                "request_id": "req_attacker-owned",
                "now": "1999-01-01T00:00:00.000Z",
                "owner": "attacker",
            }
        ),
    )
    _, request = _run(monkeypatch, hostile)

    assert request["request_id"].startswith("req_url-")
    assert request["request_id"] != "req_attacker-owned"
    assert request["now"] != "1999-01-01T00:00:00.000Z"
    assert request["owner"] == "public-url"


def test_present_state_language_routes_live(monkeypatch: pytest.MonkeyPatch) -> None:
    _, request = _run(monkeypatch, _http(rawQueryString="q=What+is+the+latest+Valkey+release%3F"))
    assert request["version_requirement"] == "current_state"
    assert request["requested_version"] is None


@pytest.mark.parametrize(
    "event",
    [
        _http(rawQueryString=""),
        _http(rawQueryString="q=%20%20"),
        _http(rawQueryString="", body="not json"),
        _http(rawQueryString="", body=json.dumps(["not", "an", "object"])),
        _http(rawQueryString="", body="!!!", isBase64Encoded=True),
    ],
)
def test_unusable_requests_are_rejected_with_400(
    monkeypatch: pytest.MonkeyPatch, event: dict[str, object]
) -> None:
    response, request = _run(monkeypatch, event)
    assert response["statusCode"] == 400
    assert request is None, "a rejected request must never reach the runtime"


def test_oversized_question_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    response, request = _run(
        monkeypatch, _http(rawQueryString="", body=json.dumps({"question": "x" * 4096}))
    )
    assert response["statusCode"] == 400
    assert request is None


def test_runtime_failure_returns_500_without_leaking_internals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exploding(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("dynamodb table valkeyrie-development-state threw at line 42")

    monkeypatch.setattr("infra.application_handler.run_runtime_event", exploding)
    monkeypatch.setattr("infra.application_handler.AwsRuntimeServices", lambda: object())
    _patch_manifest(monkeypatch)
    response = cast(dict[str, Any], handler(_http(), None))

    assert response["statusCode"] == 500
    body = cast(str, response["body"])
    assert "dynamodb" not in body.casefold()
    assert "line 42" not in body


def test_a_raw_runtime_event_is_not_treated_as_http(monkeypatch: pytest.MonkeyPatch) -> None:
    # The evaluation runners and operator tooling depend on the raw path being untouched.
    captured: dict[str, Any] = {}

    def fake_run(
        request: object,
        services: object,
        *,
        root: object,
        manifest: object,
        completion_clock: object = None,
    ) -> dict[str, object]:
        captured["request"] = request
        captured["completion_clock"] = completion_clock
        return {"outcome": "answer"}

    monkeypatch.setattr("infra.application_handler.run_runtime_event", fake_run)
    _patch_manifest(monkeypatch)
    raw = {"action": "health", "application_revision": f"sha256:{'a' * 64}"}
    result = handler(raw, None)

    assert captured["request"] is raw, "the raw event must be passed through unchanged"
    assert "statusCode" not in result
