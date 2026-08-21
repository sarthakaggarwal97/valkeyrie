"""Deterministic question routing and evidence disposition for Valkeyrie."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from valkeyrie.structured import ExactIdentifier


class RoutingError(ValueError):
    """A routing request or evidence result is malformed."""


Route: TypeAlias = Literal["static_semantic", "exact_lookup", "live_read"]
DecisionOutcome: TypeAlias = Literal["route", "clarification", "abstention"]
VersionRequirement: TypeAlias = Literal["none", "required", "current_state"]

_MAX_QUESTION_BYTES = 8 * 1024
_MAX_VERSION_BYTES = 256
_CURRENT_STATE = re.compile(
    r"\b(current|currently|latest|now|upcoming|most\s+recent(?:ly)?|newest|"
    r"published\s+most\s+recently)\b",
    re.IGNORECASE,
)
_PROJECT_STATE = re.compile(
    r"\b(open|closed|merged|passing|failing|ready|status)\b",
    re.IGNORECASE,
)
_PROJECT_ENTITY = re.compile(
    r"\b(pull request|pr\s*#?\d+|issue|release|workflow|check(?: run)?|milestone|"
    r"github project|workstream|event|community meeting|repository|branch|commit)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuestionRequest:
    """Validated routing inputs derived before retrieval."""

    question: str
    version_requirement: VersionRequirement
    requested_version: str | None = None
    available_versions: tuple[str, ...] = ()
    exact_identifier: ExactIdentifier | None = None


@dataclass(frozen=True)
class RouteDecision:
    """One fail-closed routing decision."""

    outcome: DecisionOutcome
    routes: tuple[Route, ...]
    version_scope: str | None
    question: str | None = None
    reason: str | None = None
    exact_identifier: ExactIdentifier | None = None


def route_question(request: QuestionRequest) -> RouteDecision:
    """Resolve version scope and choose explicit retrieval routes."""
    _validate_request(request)
    question = request.question.strip()
    current_language = _CURRENT_STATE.search(question) is not None or (
        _PROJECT_STATE.search(question) is not None and _PROJECT_ENTITY.search(question) is not None
    )

    if request.version_requirement == "current_state" or current_language:
        if request.requested_version is not None:
            return RouteDecision(
                outcome="abstention",
                routes=(),
                version_scope=None,
                reason="I can’t verify current project state from a fixed static version.",
            )
        return RouteDecision(
            outcome="route",
            routes=("live_read",),
            version_scope=None,
            exact_identifier=request.exact_identifier,
        )

    version = request.requested_version
    if request.version_requirement == "required" and version is None:
        return RouteDecision(
            outcome="clarification",
            routes=(),
            version_scope=None,
            question="Which Valkey release or branch should I use?",
        )
    if version is not None and version not in request.available_versions:
        return RouteDecision(
            outcome="abstention",
            routes=(),
            version_scope=None,
            reason=f"the requested version scope is unavailable: {version}",
        )

    route: Route = "exact_lookup" if request.exact_identifier is not None else "static_semantic"
    return RouteDecision(
        outcome="route",
        routes=(route,),
        version_scope=version,
        exact_identifier=request.exact_identifier,
    )


def _validate_request(request: QuestionRequest) -> None:
    if not isinstance(request, QuestionRequest):
        raise RoutingError("request must be a QuestionRequest")
    _bounded_text(request.question, "question", _MAX_QUESTION_BYTES)
    if request.version_requirement not in {"none", "required", "current_state"}:
        raise RoutingError("unsupported version requirement")
    if not isinstance(request.available_versions, tuple):
        raise RoutingError("available versions must be an immutable tuple")
    _validate_versions(request.available_versions)
    if request.requested_version is not None:
        _bounded_text(request.requested_version, "requested version", _MAX_VERSION_BYTES)
    if request.version_requirement == "none" and request.requested_version is not None:
        raise RoutingError("a version-neutral request cannot select a version")
    if request.version_requirement == "current_state" and request.available_versions:
        raise RoutingError("a current-state request cannot use static available versions")


def _validate_versions(versions: Sequence[str]) -> None:
    if len(versions) != len(set(versions)):
        raise RoutingError("available versions must be unique")
    if tuple(sorted(versions)) != tuple(versions):
        raise RoutingError("available versions must use lexical order")
    for version in versions:
        _bounded_text(version, "available version", _MAX_VERSION_BYTES)


def _bounded_text(value: str, field: str, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RoutingError(f"{field} must be non-blank text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RoutingError(f"{field} must be valid UTF-8") from error
    if len(encoded) > maximum:
        raise RoutingError(f"{field} exceeds its {maximum}-byte bound")
    if any(
        (ord(character) < 32 and character not in "\t\n") or ord(character) == 127
        for character in value
    ):
        raise RoutingError(f"{field} contains a control character")
