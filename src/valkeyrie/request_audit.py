"""Immutable executable request pins with lease/fence-bound terminal audit.

The first successful conditional claim stores the already verified A-04
``ModelInvocation`` itself, not merely identifiers describing it. Retries and
expired-lease recovery resolve that exact frozen model-visible input, model
target, inference settings, and application revision. The compact terminal
audit export is derived from the stored executable plan.

Claims carry an owner and bounded lease. Recovery requires the caller's exact
nonterminal record, an expired lease, and a different owner; one atomic
nonterminal compare-and-swap updates owner, lease, revision, and fence without
altering the pin. Completion uses the same store invariant, so a terminal
record can never be replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, cast
from urllib.parse import urlsplit


class RequestAuditError(ValueError):
    """A request pin, claim, live observation, or completion input is invalid."""


@dataclass(frozen=True)
class LiveObservation:
    """One content-addressed exact snapshot of a single live GitHub read."""

    observation_id: str
    observed_at: str
    source_url: str
    object_type: str
    canonical_payload: bytes
    payload_digest: str
    complete: bool
    truncated: bool


_OBSERVATION_API_VERSION: Final = "valkeyrie.io/live-observation/1"
_OBSERVATION_IDENTITY_VERSION: Final = "valkeyrie.io/live-observation-identity/1"
_OBSERVATION_ID: Final = re.compile(r"^obs_[0-9a-f]{64}$")
_TIMESTAMP: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
_OBSERVATION_TYPES: Final = frozenset(
    {
        "repository",
        "issue",
        "pull_request",
        "review",
        "check",
        "workflow_run",
        "tag",
        "release",
        "controller_status",
    }
)
_MAX_URL_CHARACTERS: Final = 2_048
_MAX_PAYLOAD_BYTES: Final = 256 * 1024


def create_live_observation(
    *,
    observed_at: str,
    source_url: str,
    object_type: str,
    payload: object,
) -> LiveObservation:
    """Create a bounded immutable exact JSON snapshot with internally derived identity."""
    _validated_timestamp(observed_at, "observation timestamp")
    _validated_https_url(source_url)
    if object_type not in _OBSERVATION_TYPES:
        raise RequestAuditError("live observation object type is unsupported")
    canonical_payload = _canonical_payload(payload)
    payload_digest = _sha256(canonical_payload)
    observation_id = _observation_id(
        observed_at=observed_at,
        source_url=source_url,
        object_type=object_type,
        payload_digest=payload_digest,
    )
    return LiveObservation(
        observation_id=observation_id,
        observed_at=observed_at,
        source_url=source_url,
        object_type=object_type,
        canonical_payload=canonical_payload,
        payload_digest=payload_digest,
        complete=True,
        truncated=False,
    )


def live_observation_value(observation: LiveObservation) -> dict[str, object]:
    """Return a fresh full payload value matching the live-observation schema."""
    _validate_observation(observation)
    return {
        "api_version": _OBSERVATION_API_VERSION,
        "kind": "LiveObservation",
        "observation_id": observation.observation_id,
        "observed_at": observation.observed_at,
        "source_url": observation.source_url,
        "object_type": observation.object_type,
        "payload": _decoded_payload(observation.canonical_payload),
        "payload_digest": observation.payload_digest,
        "complete": True,
        "truncated": False,
    }


def _validate_observation(observation: object) -> None:
    if not isinstance(observation, LiveObservation):
        raise RequestAuditError("live observation has the wrong runtime type")
    if (
        not isinstance(observation.observation_id, str)
        or _OBSERVATION_ID.fullmatch(observation.observation_id) is None
    ):
        raise RequestAuditError("live observation ID is malformed")
    _validated_timestamp(observation.observed_at, "observation timestamp")
    _validated_https_url(observation.source_url)
    if observation.object_type not in _OBSERVATION_TYPES:
        raise RequestAuditError("live observation object type is unsupported")
    payload = _decoded_payload(observation.canonical_payload)
    canonical_payload = _canonical_payload(payload)
    if canonical_payload != observation.canonical_payload:
        raise RequestAuditError("live observation payload is not canonical JSON")
    payload_digest = _sha256(canonical_payload)
    if observation.payload_digest != payload_digest:
        raise RequestAuditError("live observation payload digest does not match its payload")
    expected_id = _observation_id(
        observed_at=observation.observed_at,
        source_url=observation.source_url,
        object_type=observation.object_type,
        payload_digest=payload_digest,
    )
    if observation.observation_id != expected_id:
        raise RequestAuditError("live observation identity does not match its exact content")
    if observation.complete is not True or observation.truncated is not False:
        raise RequestAuditError("live observation must be a complete untruncated exact snapshot")


def _observation_id(
    *,
    observed_at: str,
    source_url: str,
    object_type: str,
    payload_digest: str,
) -> str:
    identity = {
        "api_version": _OBSERVATION_IDENTITY_VERSION,
        "kind": "LiveObservationIdentity",
        "observed_at": observed_at,
        "source_url": source_url,
        "object_type": object_type,
        "payload_digest": payload_digest,
        "complete": True,
        "truncated": False,
    }
    return f"obs_{hashlib.sha256(_canonical_json(identity)).hexdigest()}"


def _canonical_payload(value: object) -> bytes:
    _validated_json_value(value)
    try:
        encoded = _canonical_json(value)
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise RequestAuditError("live observation payload is not canonical JSON") from error
    if not 1 <= len(encoded) <= _MAX_PAYLOAD_BYTES:
        raise RequestAuditError(
            f"live observation payload exceeds its {_MAX_PAYLOAD_BYTES}-byte bound"
        )
    return encoded


def _decoded_payload(value: object) -> object:
    if not isinstance(value, bytes) or not 1 <= len(value) <= _MAX_PAYLOAD_BYTES:
        raise RequestAuditError("live observation canonical payload is outside its byte bound")
    try:
        text = value.decode("utf-8")
        decoded = cast(object, json.loads(text))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise RequestAuditError("live observation canonical payload is not valid JSON") from error
    _validated_json_value(decoded)
    return decoded


def _validated_json_value(value: object, ancestors: set[int] | None = None) -> None:
    if ancestors is None:
        ancestors = set()
    if value is None or type(value) is bool or type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RequestAuditError("live observation payload contains a non-finite number")
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise RequestAuditError("live observation payload contains invalid UTF-8") from error
        return
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in ancestors:
            raise RequestAuditError("live observation payload contains a cyclic JSON value")
        ancestors.add(identity)
        try:
            if isinstance(value, list):
                for item in value:
                    _validated_json_value(item, ancestors)
            else:
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise RequestAuditError("live observation JSON object key is not text")
                    _validated_json_value(key, ancestors)
                    _validated_json_value(item, ancestors)
        except RecursionError as error:
            raise RequestAuditError("live observation payload nesting is too deep") from error
        finally:
            ancestors.remove(identity)
        return
    raise RequestAuditError("live observation payload contains a non-JSON value")


def _validated_https_url(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= _MAX_URL_CHARACTERS:
        raise RequestAuditError("live observation URL is outside its bound")
    if any(ord(character) <= 32 or ord(character) == 127 for character in value):
        raise RequestAuditError("live observation URL contains whitespace or control characters")
    try:
        parts = urlsplit(value)
    except ValueError as error:
        raise RequestAuditError("live observation URL is malformed") from error
    if (
        parts.scheme != "https"
        or parts.username is not None
        or parts.password is not None
        or parts.port is not None
        or parts.fragment
        or "//" in parts.path
        or any(part in {".", ".."} for part in parts.path.split("/"))
    ):
        raise RequestAuditError("live observation URL must be canonical HTTPS")
    if parts.netloc == "api.github.com":
        allowed = (
            parts.path.startswith("/repos/valkey-io/")
            or parts.path == "/search/issues"
            or parts.path == "/graphql"
        )
    elif parts.netloc == "github.com":
        allowed = parts.path.startswith("/valkey-io/") or parts.path.startswith(
            "/orgs/valkey-io/projects/"
        )
    else:
        allowed = False
    if not allowed:
        raise RequestAuditError("live observation URL is outside the Valkey GitHub allowlist")
    return value


def _is_calendar_timestamp(value: str) -> bool:
    """Reject impossible dates and times the shape regex admits.

    The regex pins digit layout only, so 2026-99-99T99:99:99Z matches it. Parsing is what
    establishes the value names a real instant.
    """
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _validated_timestamp(value: object, label: str) -> datetime:
    if (
        not isinstance(value, str)
        or _TIMESTAMP.fullmatch(value) is None
        or not _is_calendar_timestamp(value)
    ):
        raise RequestAuditError(f"{label} is malformed")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise RequestAuditError(f"{label} is malformed") from error
    if parsed.tzinfo != UTC:
        raise RequestAuditError(f"{label} is malformed")
    return parsed


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"
