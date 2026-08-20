"""Deterministic question routing and evidence disposition for Valkeyrie."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias
from urllib.parse import urlsplit

from valkeyrie.evidence import EvidencePackage, verify_evidence_package
from valkeyrie.generation import GenerationBundle
from valkeyrie.structured import ExactIdentifier, ExactLookup, StructuredRecord


class RoutingError(ValueError):
    """A routing request or evidence result is malformed."""


Route: TypeAlias = Literal["static_semantic", "exact_lookup", "live_read"]
DecisionOutcome: TypeAlias = Literal["route", "clarification", "abstention"]
AnswerOutcome: TypeAlias = Literal["answer", "partial", "abstention"]
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
_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_OBSERVATION_ID = re.compile(r"^obs_[a-z0-9-]+$")


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


@dataclass(frozen=True)
class LiveObservation:
    """One bounded, immutable request-time public observation."""

    observation_id: str
    observed_at: str
    source_url: str
    payload_digest: str
    complete: bool
    truncated: bool


@dataclass(frozen=True)
class EvidenceStatus:
    """Post-retrieval facts plus the exact evidence that proves them."""

    dependency_available: bool
    supported: bool
    conflicting: bool
    generation: GenerationBundle | None = None
    package: EvidencePackage | None = None
    live_observations: tuple[LiveObservation, ...] = ()
    exact_lookup: ExactLookup | None = None
    exact_record: StructuredRecord | None = None


@dataclass(frozen=True)
class AnswerDisposition:
    """Whether validated evidence permits drafting an answer."""

    outcome: AnswerOutcome
    reason: str | None


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


def resolve_evidence(decision: RouteDecision, evidence: EvidenceStatus) -> AnswerDisposition:
    """Accept only reconstructed, route-compatible evidence for drafting."""
    _validate_decision(decision)
    _validate_evidence(evidence)
    if decision.outcome != "route":
        raise RoutingError("evidence can be resolved only for a routing decision")
    if evidence.conflicting:
        return AnswerDisposition("abstention", "validated canonical evidence conflicts")
    if not evidence.supported:
        return AnswerDisposition("abstention", "validated evidence does not support an answer")
    if not evidence.dependency_available:
        return AnswerDisposition("partial", "a required retrieval dependency is unavailable")

    route = decision.routes[0]
    if route == "live_read":
        if evidence.generation is not None or evidence.package is not None:
            return AnswerDisposition(
                "abstention", "current-state routing cannot use static corpus evidence"
            )
        if not evidence.live_observations:
            return AnswerDisposition(
                "abstention", "I can’t verify the latest project state without a live observation."
            )
        if any(
            not observation.complete or observation.truncated
            for observation in evidence.live_observations
        ):
            return AnswerDisposition("partial", "the live observation is incomplete or truncated")
        return AnswerDisposition("answer", None)

    if evidence.live_observations:
        return AnswerDisposition("abstention", "unexpected live observations were supplied")
    if evidence.generation is None or evidence.package is None:
        return AnswerDisposition("abstention", "static retrieval produced no validated evidence")
    try:
        verified = verify_evidence_package(evidence.generation, evidence.package)
    except ValueError as error:
        raise RoutingError(f"static evidence package is invalid: {error}") from error
    if not verified.records:
        return AnswerDisposition("abstention", "static retrieval produced no validated evidence")

    if route == "exact_lookup":
        if (
            decision.exact_identifier is None
            or evidence.exact_lookup is None
            or evidence.exact_record is None
        ):
            return AnswerDisposition("abstention", "exact lookup produced no validated record")
        try:
            looked_up = evidence.exact_lookup.lookup(decision.exact_identifier)
        except ValueError as error:
            raise RoutingError(f"exact lookup result is invalid: {error}") from error
        if looked_up != evidence.exact_record:
            raise RoutingError("exact lookup record does not match the indexed identifier")
    return AnswerDisposition("answer", None)


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


def _validate_decision(decision: RouteDecision) -> None:
    if not isinstance(decision, RouteDecision):
        raise RoutingError("decision must be a RouteDecision")
    if decision.outcome == "route":
        if len(decision.routes) != 1 or decision.routes[0] not in {
            "static_semantic",
            "exact_lookup",
            "live_read",
        }:
            raise RoutingError("routing decisions require exactly one supported route")
        if decision.question is not None or decision.reason is not None:
            raise RoutingError("routing decisions cannot contain terminal text")
    elif decision.outcome == "clarification":
        if decision.routes or decision.question is None or decision.reason is not None:
            raise RoutingError("clarification decisions are malformed")
        _bounded_text(decision.question, "clarification question", 1024)
    elif decision.outcome == "abstention":
        if decision.routes or decision.reason is None or decision.question is not None:
            raise RoutingError("abstention decisions are malformed")
        _bounded_text(decision.reason, "abstention reason", 2048)
    else:
        raise RoutingError("unsupported decision outcome")


def _validate_evidence(evidence: EvidenceStatus) -> None:
    if not isinstance(evidence, EvidenceStatus):
        raise RoutingError("evidence must be an EvidenceStatus")
    for field in (evidence.dependency_available, evidence.supported, evidence.conflicting):
        if not isinstance(field, bool):
            raise RoutingError("evidence status flags must be booleans")
    if not isinstance(evidence.live_observations, tuple):
        raise RoutingError("live observations must be an immutable tuple")
    identifiers: list[str] = []
    for observation in evidence.live_observations:
        if not isinstance(observation, LiveObservation):
            raise RoutingError("live observation has an invalid type")
        if _OBSERVATION_ID.fullmatch(observation.observation_id) is None:
            raise RoutingError("live observation ID is malformed")
        _validate_timestamp(observation.observed_at)
        _validate_https_url(observation.source_url)
        if _DIGEST.fullmatch(observation.payload_digest) is None:
            raise RoutingError("live observation payload digest is malformed")
        if not isinstance(observation.complete, bool) or not isinstance(
            observation.truncated, bool
        ):
            raise RoutingError("live observation completeness flags must be booleans")
        identifiers.append(observation.observation_id)
    if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
        raise RoutingError("live observations must be unique and lexically ordered")


def _validate_timestamp(value: str) -> None:
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        raise RoutingError("live observation timestamp is malformed")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise RoutingError("live observation timestamp is invalid") from error


def _validate_https_url(value: str) -> None:
    if not isinstance(value, str):
        raise RoutingError("live observation source URL is malformed")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise RoutingError("live observation source URL must be canonical HTTPS")


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
