"""Typed fail-closed normalization for extracted Bedrock Converse final text.

Bedrock Converse returns the camel-case response field ``stopReason``. The values
observed and handled here are exact snake-case strings. Safety refusals are normal
successful HTTP response paths, but any accompanying partial text is untrusted and
must never reach JSON parsing or answer acceptance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Literal


class BedrockResponseError(ValueError):
    """An extracted Bedrock response cannot safely produce model output."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


NormalizationDisposition = Literal["unchanged", "safety_abstention"]


@dataclass(frozen=True)
class BedrockTextResponse:
    """Only bounded final text and the exact Converse ``stopReason`` value."""

    response_text: str
    stop_reason: str


@dataclass(frozen=True)
class NormalizedBedrockResponse:
    """Application-safe text plus its deterministic normalization identity."""

    response_text: str
    disposition: NormalizationDisposition
    policy_revision: str


_MAX_RESPONSE_BYTES: Final = 256 * 1024
_STRICT_ABSTENTION: Final = {
    "api_version": "valkeyrie.io/model-output/1",
    "kind": "ModelOutput",
    "outcome": "abstention",
    "reason": "Insufficient validated evidence.",
}
STRICT_A04_ABSTENTION_TEXT: Final = json.dumps(
    _STRICT_ABSTENTION,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
)
_POLICY: Final = {
    "api_version": "valkeyrie.io/bedrock-response-normalization-policy/1",
    "stop_reason_field": "stopReason",
    "stop_reason_value_format": "snake_case",
    "unchanged": ["end_turn"],
    "safety_abstention": ["content_filtered", "refusal"],
    "safety_response_text": "discard_without_validation",
    "fail_closed": ["max_tokens", "tool_use", "all_other_values"],
    "safety_abstention_text": STRICT_A04_ABSTENTION_TEXT,
    "maximum_response_bytes": _MAX_RESPONSE_BYTES,
}
BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION: Final = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            _POLICY,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
)


def normalize_bedrock_response(
    response_text: object,
    stop_reason: object,
) -> NormalizedBedrockResponse:
    """Normalize one extracted Converse final-text response or fail closed.

    ``content_filtered`` and ``refusal`` discard every raw byte and produce the
    strict A04 abstention. ``end_turn`` preserves the text byte-for-byte. Token
    exhaustion, tool use, and every unknown value are dependency failures rather
    than fabricated answers.
    """
    if not isinstance(stop_reason, str):
        raise BedrockResponseError("stop_reason_invalid")
    try:
        encoded_reason = stop_reason.encode("ascii")
    except UnicodeEncodeError as error:
        raise BedrockResponseError("stop_reason_invalid") from error
    if not encoded_reason or len(encoded_reason) > 64:
        raise BedrockResponseError("stop_reason_invalid")
    if stop_reason in {"content_filtered", "refusal"}:
        return NormalizedBedrockResponse(
            STRICT_A04_ABSTENTION_TEXT,
            "safety_abstention",
            BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION,
        )
    if stop_reason == "end_turn":
        return NormalizedBedrockResponse(
            _bounded_text(response_text),
            "unchanged",
            BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION,
        )
    if stop_reason in {"max_tokens", "tool_use"}:
        raise BedrockResponseError(f"unsupported_stop_reason:{stop_reason}")
    raise BedrockResponseError("unsupported_stop_reason:unknown")


def _bounded_text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BedrockResponseError("response_text_invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise BedrockResponseError("response_text_invalid") from error
    if len(encoded) > _MAX_RESPONSE_BYTES:
        raise BedrockResponseError("response_text_oversized")
    return value
