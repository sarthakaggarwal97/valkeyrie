"""Private Lambda orchestration for the qualified Valkeyrie application.

The runtime has two actions: an AWS-free deterministic ``health`` path and a
bounded ``answer`` path. AWS calls are isolated behind ``RuntimeServices``;
``AwsRuntimeServices`` lazily imports the boto3 SDK supplied by Lambda.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal, Protocol, TypeAlias, cast
from urllib.parse import quote, urlsplit

from valkeyrie.bedrock_response import (
    BedrockResponseError,
    BedrockTextResponse,
    normalize_bedrock_response,
)
from valkeyrie.drafting import DraftingError, _screened_model_text
from valkeyrie.github import fetch_public_github
from valkeyrie.live_github import (
    LiveGitHubError,
    LiveGitHubQuery,
    infer_live_query,
    infer_supplementary_search,
    read_live_github,
)
from valkeyrie.prompts import load_prompt_package
from valkeyrie.request_audit import LiveObservation, RequestAuditError, live_observation_value
from valkeyrie.retrieval import (
    RetrievalError,
    derive_retrieval_intent,
    reviewed_source_authority,
    verify_retrieval_results,
)
from valkeyrie.routing import QuestionRequest, route_question
from valkeyrie.structured import ReleaseArtifactIdentifier

RuntimeOutcome = Literal["answer", "clarification", "abstention", "partial", "error"]
_RUNTIME_OUTCOMES: Final = frozenset({"answer", "clarification", "abstention", "partial", "error"})


class ApplicationRuntimeError(ValueError):
    """A runtime request, dependency response, or persisted state is invalid."""


@dataclass(frozen=True)
class StaticRuntimeEvidence:
    evidence_id: str
    text: str
    generation_id: str
    repository: str
    path: str
    commit: str
    authority: str
    version_scope: str
    content_digest: str
    immutable_url: str


@dataclass(frozen=True)
class LiveRuntimeEvidence:
    evidence_id: str
    text: str
    observation_id: str
    observed_at: str
    object_type: str
    payload_digest: str
    source_url: str
    citation_url: str


RuntimeEvidence: TypeAlias = StaticRuntimeEvidence | LiveRuntimeEvidence


@dataclass(frozen=True)
class RuntimeResult:
    outcome: RuntimeOutcome
    request_id: str
    message: str | None = None
    claims: tuple[Mapping[str, object], ...] = ()
    citations: tuple[str, ...] = ()
    generation_id: str | None = None
    request_revision: int | None = None
    request_fence: int | None = None


# A refusal that only says evidence was insufficient leaves the asker with nothing to try.
# These clauses name the next move. They are presentation, deliberately not part of the
# prompt: what the model is allowed to claim must not depend on advice given to the user.
_STATIC_GUIDANCE: Final = (
    " I answer from indexed Valkey repositories, so naming the repository, command, or"
    " document usually helps."
)
_LIVE_GUIDANCE: Final = (
    " Ask without \u201ccurrent\u201d, \u201clatest\u201d, or \u201cstatus of\u201d and I will"
    " answer from the indexed corpus instead."
)
_LIVE_TARGET_GUIDANCE: Final = (
    " For live lookups, name an issue or pull request number, or ask for the latest release."
)


def _guided(message: str | None, guidance: str) -> str:
    """Append a next step to a refusal, without restating it if already present."""
    text = (message or "").strip()
    if not text:
        return guidance.strip()
    return text if guidance.strip() in text else f"{text}{guidance}"


class RuntimeServices(Protocol):
    """Exact deployed service operations used by one answer."""

    def read_controls(self) -> Mapping[str, str]: ...

    def read_generation(self, version_scope: str | None) -> Mapping[str, object] | None: ...

    def read_live(self, query: LiveGitHubQuery) -> LiveObservation: ...

    def retrieve(
        self, *, knowledge_base_id: str, generation_id: str, question: str
    ) -> tuple[Mapping[str, object], ...]: ...

    def read_structured_record(
        self, *, generation_id: str, record_id: str, expected_content_digest: str
    ) -> bytes | None: ...

    def get_request(self, request_id: str) -> Mapping[str, object] | None: ...

    def claim_request(self, item: Mapping[str, object]) -> bool: ...

    def recover_request(
        self,
        *,
        request_id: str,
        expected_revision: int,
        expected_fence: int,
        expected_owner: str,
        new_owner: str,
        now: str,
        lease_expires_at: str,
    ) -> Mapping[str, object] | None: ...

    def complete_request(
        self,
        *,
        request_id: str,
        revision: int,
        fence: int,
        outcome: str,
        completed_at: str,
        result: Mapping[str, object] | None = None,
    ) -> bool: ...

    def converse(
        self,
        *,
        model_id: str,
        system: tuple[str, ...],
        question: str,
        evidence: tuple[RuntimeEvidence, ...],
        maximum_output_tokens: int,
        reasoning_effort: str,
    ) -> BedrockTextResponse: ...


_REQUEST_ID: Final = re.compile(r"^req_[a-z0-9-]+$")
_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_KB_ID: Final = re.compile(r"^[A-Z0-9]{10}$")
_EVIDENCE_ID: Final = re.compile(r"^ev_[a-z0-9-]+$")
_TIMESTAMP: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
_MAX_QUESTION_BYTES: Final = 8 * 1024
# Distinguishes "not yet looked up" from "looked up and absent", so an absent token is
# not re-fetched on every question.
_UNSET: Final = object()
_MAX_EVIDENCE: Final = 10
_MAX_EVIDENCE_BYTES: Final = 64 * 1024
_RELEASE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_RELEASE_ARTIFACT: Final = re.compile(
    r"^valkey-(?P<release>[0-9]+\.[0-9]+\.[0-9]+(?:-rc[0-9]+)?)\.tar\.gz$"
)
_CONTROL_NAMES: Final = (
    "/valkeyrie-development/controls/model-processing-enabled",
    "/valkeyrie-development/controls/runtime-enabled",
)
_STATIC_METADATA: Final = frozenset(
    {
        "generation_id",
        "evidence_id",
        "repository",
        "path",
        "commit",
        "authority",
        "version_scope",
        "content_digest",
        "immutable_url",
    }
)
_LIVE_METADATA: Final = frozenset(
    {
        "evidence_id",
        "observation_id",
        "observed_at",
        "object_type",
        "payload_digest",
        "source_url",
        "citation_url",
    }
)
_LIVE_KINDS: Final[Mapping[str, frozenset[str]]] = {
    "pull_request": frozenset({"pull_request"}),
    "issue": frozenset({"issue", "issue_search", "milestone"}),
    "release": frozenset({"release"}),
    "workflow_run": frozenset({"workflow_run"}),
    "check": frozenset({"check_run", "commit_checks", "commit_status"}),
    "controller_status": frozenset({"project"}),
}


def run_runtime_event(
    event: object,
    services: RuntimeServices | None,
    *,
    root: Path,
    manifest: Mapping[str, object],
    completion_clock: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Run one exact private action and return a bounded JSON-compatible value.

    ``completion_clock`` supplies the instant a request finished. Callers that must reproduce a
    recorded run omit it, and the event's own ``completed_at`` is used, which is what every
    deterministic caller already does. A live deployment passes a real clock, because the event
    field is supplied by the caller before the model runs: it cannot describe when execution
    ended, and nothing stops it naming an arbitrary past or future instant.
    """
    _validate_manifest(manifest)
    if not isinstance(event, Mapping) or not isinstance(event.get("action"), str):
        raise ApplicationRuntimeError("runtime event has an unknown or missing action")
    if event["action"] == "health":
        return _health(event, manifest)
    if event["action"] != "answer":
        raise ApplicationRuntimeError("runtime action must be health or answer")
    request_id = event.get("request_id")
    if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
        raise ApplicationRuntimeError("runtime request ID is malformed")
    try:
        if services is None:
            raise ApplicationRuntimeError("runtime services are unavailable")
        result = _answer(
            event,
            services,
            root=root,
            manifest=manifest,
            completion_clock=completion_clock,
        )
    except (ApplicationRuntimeError, DraftingError) as error:
        result = RuntimeResult("error", request_id, str(error))
    return _result_value(result)


def _execution_authorization(manifest: Mapping[str, object]) -> str:
    authorization = manifest.get("authorization", "qualified_model_selection")
    if authorization == "qualified_model_selection":
        return authorization
    if authorization != "owner_directed_comparison":
        raise ApplicationRuntimeError("execution authorization is unsupported")
    if manifest.get("qualification_status") != "not_run_not_qualified":
        raise ApplicationRuntimeError("comparison qualification status is incompatible")
    if manifest.get("selected_model_revision") != "us.anthropic.claude-opus-5":
        raise ApplicationRuntimeError("comparison model identity is incompatible")
    return authorization


def _health(event: Mapping[str, object], manifest: Mapping[str, object]) -> dict[str, object]:
    if set(event) != {"action", "application_revision"}:
        raise ApplicationRuntimeError("health event has an unknown or missing field")
    if event["application_revision"] != manifest["application_revision"]:
        raise ApplicationRuntimeError("health application revision does not match")
    decision = route_question(QuestionRequest("Where are Valkey command docs written?", "none"))
    evidence = StaticRuntimeEvidence(
        "ev_health",
        "Valkey command documentation is stored in valkey-doc/commands.",
        "sha256:" + "1" * 64,
        "valkey-doc",
        "commands/get.md",
        "2" * 40,
        "canonical",
        "unstable",
        "sha256:" + "3" * 64,
        "https://github.com/valkey-io/valkey-doc/blob/" + "2" * 40 + "/commands/get.md",
    )
    raw = json.dumps(
        {
            "api_version": "valkeyrie.io/model-output/1",
            "kind": "ModelOutput",
            "outcome": "answer",
            "claims": [
                {
                    "claim_id": "health",
                    "text": "Valkey command documentation is stored in valkey-doc/commands.",
                    "evidence_ids": ["ev_health"],
                }
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    normalized = normalize_bedrock_response(raw, "end_turn")
    accepted = _accept_output(normalized.response_text, (evidence,))
    plan = {"request_id": "req_health", "question_digest": _question_digest(decision.routes[0])}
    state: dict[str, object] = {}
    claimed = not state and not state.update(plan)
    completed = claimed and state.get("request_id") == "req_health"
    if accepted[0] != "answer" or not completed:
        raise ApplicationRuntimeError("offline synthetic A03-A06 path failed")
    return {
        "status": "healthy",
        "application_revision": manifest["application_revision"],
        "prompt_revision": manifest["prompt_revision"],
        "selected_profile_revision": manifest["selected_profile_revision"],
        "selected_inference_config_revision": manifest["selected_inference_config_revision"],
        "selected_report_id": manifest["selected_report_id"],
        "selection_id": manifest["selection_id"],
        "selected_model_revision": manifest["selected_model_revision"],
        "selected_inference_profile_arn": manifest["selected_inference_profile_arn"],
        "execution_authorization": _execution_authorization(manifest),
        "qualification_status": manifest.get("qualification_status", "qualified"),
        "response_normalization_policy_revision": manifest[
            "response_normalization_policy_revision"
        ],
        "synthetic_path": {
            "a03_route": decision.routes[0],
            "a04_normalization": normalized.disposition,
            "a04_output": accepted[0],
            "a05_conditional_claim": claimed,
            "a05_completion": completed,
            "a06_outcome": "answer",
        },
    }


def _completion_timestamp(clock: Callable[[], str] | None, supplied: str) -> str:
    """Return the instant to record as completion, preferring a trusted clock.

    A clock that returns a malformed value is refused rather than silently falling back: a
    deployment that passes one is asserting the caller's value is not to be trusted, and quietly
    substituting it would defeat that.
    """
    if clock is None:
        return supplied
    produced = clock()
    return _timestamp(produced, "completion timestamp")


def _answer(
    event: Mapping[str, object],
    services: RuntimeServices,
    *,
    root: Path,
    manifest: Mapping[str, object],
    completion_clock: Callable[[], str] | None = None,
) -> RuntimeResult:
    expected = {
        "action",
        "request_id",
        "question",
        "version_requirement",
        "requested_version",
        "knowledge_base_id",
        "owner",
        "now",
        "completed_at",
        "lease_duration_seconds",
    }
    if set(event) not in (expected, expected | {"exact_identifier"}):
        raise ApplicationRuntimeError("answer event has an unknown or missing field")
    request_id = cast(str, event["request_id"])
    question = _bounded_text(event["question"], "question", _MAX_QUESTION_BYTES)
    requirement = event["version_requirement"]
    if requirement not in {"none", "required", "current_state"}:
        raise ApplicationRuntimeError("version requirement is unsupported")
    requested = event["requested_version"]
    if requested is not None and not isinstance(requested, str):
        raise ApplicationRuntimeError("requested version must be text or null")
    knowledge_base_id = event["knowledge_base_id"]
    if not isinstance(knowledge_base_id, str) or _KB_ID.fullmatch(knowledge_base_id) is None:
        raise ApplicationRuntimeError("knowledge base ID is malformed")
    owner = _bounded_text(event["owner"], "owner", 128)
    now = _timestamp(event["now"], "now")
    completed_at = _timestamp(event["completed_at"], "completed_at")
    lease = event["lease_duration_seconds"]
    if type(lease) is not int or not 1 <= lease <= 3600:
        raise ApplicationRuntimeError("lease duration must be from 1 through 3600 seconds")
    exact_identifier = _release_artifact_identifier(event.get("exact_identifier"))
    request_digest = _answer_request_digest(
        question, requirement, requested, knowledge_base_id, exact_identifier
    )
    lease_expires_at = _lease_expiry(now, lease)

    if not _controls_enabled(services):
        return RuntimeResult(
            "partial", request_id, "The answer service is temporarily unavailable."
        )

    existing = services.get_request(request_id)
    if existing is not None:
        return _resume_existing(
            services,
            existing,
            request_id=request_id,
            request_digest=request_digest,
            owner=owner,
            now=now,
            lease_expires_at=lease_expires_at,
            completed_at=completed_at,
            manifest=manifest,
            completion_clock=completion_clock,
        )

    provisional = route_question(
        QuestionRequest(
            question,
            cast(Any, requirement),
            requested,
            () if requested is None else (requested,),
            exact_identifier,
        )
    )
    if provisional.outcome == "clarification":
        return RuntimeResult("clarification", request_id, provisional.question)
    if provisional.outcome == "abstention":
        return RuntimeResult(
            "abstention", request_id, _guided(provisional.reason, _STATIC_GUIDANCE)
        )

    evidence: tuple[RuntimeEvidence, ...]
    generation_id: str | None
    plan_knowledge_base_id: str | None
    evidence_mode: Literal["static", "live"]
    if provisional.routes[0] == "live_read":
        try:
            query = infer_live_query(question)
        except LiveGitHubError:
            return RuntimeResult(
                "abstention",
                request_id,
                _guided(
                    "I couldn’t identify a supported live GitHub query.", _LIVE_TARGET_GUIDANCE
                ),
            )
        if query is None:
            return RuntimeResult(
                "abstention",
                request_id,
                _guided(
                    "I couldn’t identify a supported live GitHub query.", _LIVE_TARGET_GUIDANCE
                ),
            )
        try:
            observation = services.read_live(query)
        except Exception:
            return RuntimeResult(
                "partial",
                request_id,
                _guided("Live GitHub data is temporarily unavailable.", _LIVE_GUIDANCE),
            )
        try:
            evidence = (_live_evidence(observation),)
        except (ApplicationRuntimeError, RequestAuditError):
            return RuntimeResult(
                "abstention", request_id, "I couldn’t validate the live GitHub response."
            )
        generation_id = None
        plan_knowledge_base_id = None
        evidence_mode = "live"
    else:
        generation = services.read_generation(requested)
        available_versions = (
            () if requested is None else (requested,) if generation is not None else ()
        )
        decision = route_question(
            QuestionRequest(
                question,
                cast(Any, requirement),
                requested,
                available_versions,
                exact_identifier,
            )
        )
        if decision.outcome == "clarification":
            return RuntimeResult("clarification", request_id, decision.question)
        if decision.outcome == "abstention":
            return RuntimeResult(
                "abstention", request_id, _guided(decision.reason, _STATIC_GUIDANCE)
            )
        if generation is None:
            return RuntimeResult(
                "partial", request_id, "The selected corpus is temporarily unavailable."
            )
        generation_id = _generation_id(generation)
        if not all(
            generation.get(flag) is True
            for flag in ("sealed", "available", "ingested", "retrievable")
        ):
            return RuntimeResult(
                "abstention", request_id, "The selected corpus is not ready for reliable answers."
            )
        if decision.routes[0] == "exact_lookup":
            if exact_identifier is None:  # pragma: no cover - routing owns this invariant
                raise ApplicationRuntimeError("exact route lacks its identifier")
            record_id = _release_artifact_record_id(exact_identifier)
            expected_digest = _structured_record_digest(generation, record_id)
            record = (
                None
                if expected_digest is None
                else services.read_structured_record(
                    generation_id=generation_id,
                    record_id=record_id,
                    expected_content_digest=expected_digest,
                )
            )
            evidence = (
                ()
                if record is None
                else (_structured_record_evidence(record, generation_id, exact_identifier),)
            )
        else:
            retrieved = services.retrieve(
                knowledge_base_id=knowledge_base_id,
                generation_id=generation_id,
                question=question,
            )
            evidence = _bounded_evidence(
                _evidence(retrieved, generation_id)
                + _supplementary_live_evidence(services, question)
            )
        plan_knowledge_base_id = knowledge_base_id
        evidence_mode = "static"
    if not evidence:
        return RuntimeResult(
            "abstention",
            request_id,
            _guided(
                "I couldn’t find enough verified information to answer that.", _STATIC_GUIDANCE
            ),
        )
    package = load_prompt_package(root)
    prompt_by_name = {template.name: template.content for template in package.templates}
    prompt_order = ("system", "evidence-use", "citations", "clarification", "answer")
    if set(prompt_by_name) != set(prompt_order):  # pragma: no cover - loader owns this invariant
        raise ApplicationRuntimeError("prompt package roles are incompatible")
    selected_inference = manifest["selected_inference"]
    if not isinstance(selected_inference, Mapping):
        raise ApplicationRuntimeError("selected inference configuration is malformed")
    plan: dict[str, object] = {
        "request_id": request_id,
        "question_digest": request_digest,
        "question": question,
        "evidence_mode": evidence_mode,
        "generation_id": generation_id,
        "knowledge_base_id": plan_knowledge_base_id,
        "application_revision": manifest["application_revision"],
        "prompt_revision": package.prompt_revision,
        "selected_profile_revision": manifest["selected_profile_revision"],
        "selected_inference_config_revision": manifest["selected_inference_config_revision"],
        "selected_model_revision": manifest["selected_model_revision"],
        "execution_authorization": _execution_authorization(manifest),
        "model_id": manifest["selected_inference_profile_arn"],
        "maximum_output_tokens": selected_inference["maximum_output_tokens"],
        "reasoning_effort": selected_inference["reasoning_effort"],
        "system": [prompt_by_name[name] for name in prompt_order],
        "evidence": [_evidence_value(item) for item in evidence],
    }
    item = {
        "pk": f"request#{request_id}",
        "record_type": "request_audit",
        "revision": 1,
        "fence": 1,
        "owner": owner,
        "lease_expires_at": lease_expires_at,
        "started_at": now,
        "plan": plan,
    }
    if not services.claim_request(item):
        raced = services.get_request(request_id)
        if raced is None:
            return RuntimeResult("partial", request_id, "request claim raced without state")
        return _resume_existing(
            services,
            raced,
            request_id=request_id,
            request_digest=request_digest,
            owner=owner,
            now=now,
            lease_expires_at=lease_expires_at,
            completed_at=completed_at,
            manifest=manifest,
        )
    return _execute_plan(
        services,
        plan,
        request_id=request_id,
        revision=1,
        fence=1,
        completed_at=completed_at,
        completion_clock=completion_clock,
    )


def _controls_enabled(services: RuntimeServices) -> bool:
    return dict(services.read_controls()) == {name: "true" for name in _CONTROL_NAMES}


def _resume_existing(
    services: RuntimeServices,
    item: Mapping[str, object],
    *,
    request_id: str,
    request_digest: str,
    owner: str,
    now: str,
    lease_expires_at: str,
    completed_at: str,
    manifest: Mapping[str, object],
    completion_clock: Callable[[], str] | None = None,
) -> RuntimeResult:
    replayed = _replayed_result(item, request_digest, manifest, request_id=request_id)
    if replayed is not None:
        return replayed
    plan, revision, fence = _existing_plan(item, request_digest, manifest)
    expected_owner = _bounded_text(item.get("owner"), "request owner", 128)
    existing_expiry = _timestamp(item.get("lease_expires_at"), "request lease expiration")
    if _timestamp_value(now) < _timestamp_value(existing_expiry):
        return RuntimeResult("partial", request_id, "request lease is still active")
    recovered = services.recover_request(
        request_id=request_id,
        expected_revision=revision,
        expected_fence=fence,
        expected_owner=expected_owner,
        new_owner=owner,
        now=now,
        lease_expires_at=lease_expires_at,
    )
    if recovered is None:
        return RuntimeResult("partial", request_id, "request recovery condition failed")
    recovered_plan, recovered_revision, recovered_fence = _existing_plan(
        recovered, request_digest, manifest
    )
    return _execute_plan(
        services,
        recovered_plan,
        request_id=request_id,
        revision=recovered_revision,
        fence=recovered_fence,
        completed_at=completed_at,
        completion_clock=completion_clock,
    )


def _execute_plan(
    services: RuntimeServices,
    plan: Mapping[str, object],
    *,
    request_id: str,
    revision: int,
    fence: int,
    completed_at: str,
    completion_clock: Callable[[], str] | None = None,
) -> RuntimeResult:
    generation_id = _plan_generation_id(plan)
    if not _controls_enabled(services):
        return RuntimeResult(
            "partial",
            request_id,
            "The answer service is temporarily unavailable.",
            generation_id=generation_id,
            request_revision=revision,
            request_fence=fence,
        )
    evidence = tuple(
        _parse_evidence(item, generation_id=generation_id)
        for item in cast(list[object], plan["evidence"])
    )
    try:
        response = services.converse(
            model_id=cast(str, plan["model_id"]),
            system=tuple(cast(list[str], plan["system"])),
            question=cast(str, plan["question"]),
            evidence=evidence,
            maximum_output_tokens=cast(int, plan["maximum_output_tokens"]),
            reasoning_effort=cast(str, plan["reasoning_effort"]),
        )
    except Exception:
        return RuntimeResult(
            "partial",
            request_id,
            "The answer service is temporarily unavailable.",
            generation_id=generation_id,
            request_revision=revision,
            request_fence=fence,
        )
    # Generated here, after execution and immediately before the completing write, so the audit
    # records when the request actually finished rather than when its caller was preparing it.
    terminal_at = _completion_timestamp(completion_clock, completed_at)
    try:
        normalized = normalize_bedrock_response(response.response_text, response.stop_reason)
        outcome, claims, citations, message = _accept_output(normalized.response_text, evidence)
    except (ApplicationRuntimeError, BedrockResponseError, DraftingError):
        if not services.complete_request(
            request_id=request_id,
            revision=revision,
            fence=fence,
            outcome="error",
            completed_at=terminal_at,
        ):
            return RuntimeResult(
                "partial", request_id, "Request completion could not be confirmed."
            )
        return RuntimeResult(
            "error",
            request_id,
            "I couldn’t produce a reliable answer. Please try again.",
            generation_id=generation_id,
            request_revision=revision + 1,
            request_fence=fence,
        )
    if outcome == "abstention":
        # Applied here, at the single point where a parsed model outcome becomes a result,
        # rather than at each return site. The model writes its own reason, so wrapping the
        # sites individually left this one bare and would leave the next one bare too.
        # A clarification is deliberately excluded: it is already a question to the user.
        message = _guided(message, _STATIC_GUIDANCE)
    terminal = "answer" if outcome == "answer" else outcome
    completed = RuntimeResult(
        cast(RuntimeOutcome, terminal),
        request_id,
        message,
        claims,
        citations,
        generation_id=generation_id,
    )
    if not services.complete_request(
        request_id=request_id,
        revision=revision,
        fence=fence,
        outcome=terminal,
        completed_at=terminal_at,
        # Recorded with the outcome so a redelivery can be answered from the record rather than
        # charged for a second inference that could answer differently.
        result=_replayable_result(completed),
    ):
        return RuntimeResult("partial", request_id, "Request completion could not be confirmed.")
    return RuntimeResult(
        cast(RuntimeOutcome, outcome),
        request_id,
        message,
        claims,
        citations,
        generation_id,
        revision + 1,
        fence,
    )


def _accept_output(
    response_text: str,
    evidence: tuple[RuntimeEvidence, ...],
) -> tuple[str, tuple[Mapping[str, object], ...], tuple[str, ...], str | None]:
    try:
        value = json.loads(response_text, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ApplicationRuntimeError("normalized model response is invalid JSON") from error
    if not isinstance(value, Mapping) or set(value) not in (
        {"api_version", "kind", "outcome", "claims"},
        {"api_version", "kind", "outcome", "question"},
        {"api_version", "kind", "outcome", "reason"},
    ):
        raise ApplicationRuntimeError("model output has an unknown or missing field")
    if (value.get("api_version"), value.get("kind")) != (
        "valkeyrie.io/model-output/1",
        "ModelOutput",
    ):
        raise ApplicationRuntimeError("model output identity is incompatible")
    outcome = value.get("outcome")
    if outcome == "clarification":
        text = _bounded_text(value.get("question"), "clarification question", 1024)
        _screened_model_text(text, "clarification question", 1024)
        return "clarification", (), (), text
    if outcome == "abstention":
        text = _bounded_text(value.get("reason"), "abstention reason", 2048)
        _screened_model_text(text, "abstention reason", 2048)
        return "abstention", (), (), text
    if outcome != "answer" or not isinstance(value.get("claims"), list):
        raise ApplicationRuntimeError("model output outcome is unsupported")
    known = {item.evidence_id: item for item in evidence}
    claims: list[Mapping[str, object]] = []
    cited: set[str] = set()
    seen: set[str] = set()
    for raw in cast(list[object], value["claims"]):
        if not isinstance(raw, Mapping) or set(raw) != {"claim_id", "text", "evidence_ids"}:
            raise ApplicationRuntimeError("model claim has an unknown or missing field")
        claim_id = raw["claim_id"]
        text = _bounded_text(raw["text"], "claim text", 4096)
        ids = raw["evidence_ids"]
        if not isinstance(claim_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", claim_id):
            raise ApplicationRuntimeError("model claim ID is malformed")
        if claim_id in seen:
            raise ApplicationRuntimeError("model claim ID is duplicated")
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 20
            or any(item not in known for item in ids)
        ):
            raise ApplicationRuntimeError("model claim evidence is unknown or missing")
        if len(ids) != len(set(cast(list[str], ids))):
            raise ApplicationRuntimeError("model claim evidence is duplicated")
        _screened_model_text(text, "claim text", 4096)
        seen.add(claim_id)
        cited.update(cast(list[str], ids))
        claims.append({"claim_id": claim_id, "text": text, "evidence_ids": list(ids)})
    if not claims:
        raise ApplicationRuntimeError("model answer has no claim")
    static_targets: set[tuple[str, str, str, str]] = set()
    live_targets: set[tuple[str, str, str]] = set()
    for evidence_id in cited:
        item = known[evidence_id]
        if isinstance(item, StaticRuntimeEvidence):
            static_targets.add((item.repository, item.path, item.commit, item.immutable_url))
        else:
            live_targets.add((item.object_type, item.observed_at, item.citation_url))
    citations = tuple(
        f"{repository}/{path}@{commit}: {immutable_url}"
        for repository, path, commit, immutable_url in sorted(static_targets)
    ) + tuple(
        f"live GitHub {object_type} observed {observed_at}: {citation_url}"
        for object_type, observed_at, citation_url in sorted(live_targets)
    )
    return "answer", tuple(claims), citations, None


def _supplementary_live_evidence(
    services: RuntimeServices, question: str
) -> tuple[RuntimeEvidence, ...]:
    """Supplement corpus evidence with a GitHub search, when the question warrants one.

    The corpus cannot document a feature that has not shipped, so a question like "How does
    Valkey replication compression work?" is answerable only from the open pull requests that
    propose it. This adds that evidence beside the corpus evidence so one answer can say what
    the feature does AND that it is unmerged, rather than refusing.

    Best effort by design: anonymous GitHub reads are rate limited, and a corpus answer must
    not fail because a supplement was unavailable. Every failure yields no supplement.
    """
    # Both kinds, because GitHub requires an explicit is:issue or is:pull-request and the two
    # answer different halves of the same question: a pull request carries the design and
    # whether it merged, an issue carries discussion and current status.
    evidence: list[RuntimeEvidence] = []
    for kind in ("pull-request", "issue"):
        try:
            query = infer_supplementary_search(question, kind=kind)
        except LiveGitHubError:
            return ()
        if query is None:
            return ()
        # Each kind fails independently: one unavailable half must not discard the other.
        try:
            record: RuntimeEvidence | None = _live_evidence(services.read_live(query))
        except Exception:
            record = None
        if record is not None:
            evidence.append(record)
    return tuple(evidence)


def _bounded_evidence(values: tuple[RuntimeEvidence, ...]) -> tuple[RuntimeEvidence, ...]:
    """Re-apply the record and byte bounds to a combined static-plus-live package.

    _evidence bounds only the static tuple, so appending live supplements could carry the
    package past both limits: the model would receive more evidence than the bound admits.
    Live records are dropped rather than the request failing, because a supplement is an
    optional addition to an answer the static evidence can already support.
    """
    if len(values) <= _MAX_EVIDENCE and (
        sum(len(item.text.encode("utf-8")) for item in values) <= _MAX_EVIDENCE_BYTES
    ):
        return values
    kept: list[RuntimeEvidence] = []
    total = 0
    for item in values:
        size = len(item.text.encode("utf-8"))
        if len(kept) + 1 > _MAX_EVIDENCE or total + size > _MAX_EVIDENCE_BYTES:
            continue
        kept.append(item)
        total += size
    return tuple(kept)


def _evidence(
    values: tuple[Mapping[str, object], ...], generation_id: str
) -> tuple[RuntimeEvidence, ...]:
    if len(values) > _MAX_EVIDENCE:
        raise ApplicationRuntimeError("retrieval returned too many evidence records")
    result = tuple(_parse_evidence(value, generation_id=generation_id) for value in values)
    if sum(len(item.text.encode("utf-8")) for item in result) > _MAX_EVIDENCE_BYTES:
        raise ApplicationRuntimeError("retrieval evidence exceeds its byte bound")
    ids = [item.evidence_id for item in result]
    if len(ids) != len(set(ids)):
        raise ApplicationRuntimeError("retrieval evidence IDs are duplicated")
    return result


def _parse_evidence(value: object, *, generation_id: str | None = None) -> RuntimeEvidence:
    if not isinstance(value, Mapping) or set(value) != {"text", "metadata"}:
        raise ApplicationRuntimeError("retrieval evidence has an unknown or missing field")
    text = _bounded_text(value["text"], "retrieval evidence", _MAX_EVIDENCE_BYTES)
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping):
        raise ApplicationRuntimeError("retrieval metadata is malformed")
    fields = frozenset(metadata)
    if fields == _STATIC_METADATA:
        return _parse_static_evidence(text, metadata, generation_id)
    if fields == _LIVE_METADATA:
        # No generation check here, deliberately. `generation_id` is the PLAN's corpus
        # generation, which describes its static evidence. _LIVE_METADATA carries no
        # generation_id field at all and the two shapes are matched exactly, so a live
        # record structurally cannot claim a generation whatever the plan's is. The old
        # guard rejected any plan holding both kinds, which was unreachable until corpus
        # answers began being supplemented with GitHub, and which broke every such answer.
        return _parse_live_evidence(text, metadata)
    raise ApplicationRuntimeError("retrieval metadata is not an exact supported field set")


def _parse_static_evidence(
    text: str, metadata: Mapping[str, object], generation_id: str | None
) -> StaticRuntimeEvidence:
    observed_generation = metadata["generation_id"]
    if not isinstance(observed_generation, str) or _DIGEST.fullmatch(observed_generation) is None:
        raise ApplicationRuntimeError("retrieval generation identity is malformed")
    if generation_id is not None and observed_generation != generation_id:
        raise ApplicationRuntimeError("retrieval crossed the pinned generation")
    evidence_id = metadata["evidence_id"]
    if not isinstance(evidence_id, str) or _EVIDENCE_ID.fullmatch(evidence_id) is None:
        raise ApplicationRuntimeError("retrieval evidence identity is malformed")
    commit = metadata["commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ApplicationRuntimeError("retrieval commit is malformed")
    digest = metadata["content_digest"]
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ApplicationRuntimeError("retrieval content digest is malformed")
    url = metadata["immutable_url"]
    _immutable_url(url, commit)
    fields = ("repository", "path", "authority", "version_scope")
    if any(
        not isinstance(metadata[field], str) or not cast(str, metadata[field]) for field in fields
    ):
        raise ApplicationRuntimeError("retrieval provenance is malformed")
    return StaticRuntimeEvidence(
        evidence_id,
        text,
        observed_generation,
        cast(str, metadata["repository"]),
        cast(str, metadata["path"]),
        commit,
        cast(str, metadata["authority"]),
        cast(str, metadata["version_scope"]),
        digest,
        cast(str, url),
    )


def _parse_live_evidence(text: str, metadata: Mapping[str, object]) -> LiveRuntimeEvidence:
    evidence_id = metadata["evidence_id"]
    observation_id = metadata["observation_id"]
    observed_at = metadata["observed_at"]
    object_type = metadata["object_type"]
    payload_digest = metadata["payload_digest"]
    source_url = metadata["source_url"]
    citation_url = metadata["citation_url"]
    if not isinstance(evidence_id, str) or _EVIDENCE_ID.fullmatch(evidence_id) is None:
        raise ApplicationRuntimeError("live evidence identity is malformed")
    if (
        not isinstance(observation_id, str)
        or re.fullmatch(r"obs_[0-9a-f]{64}", observation_id) is None
    ):
        raise ApplicationRuntimeError("live observation identity is malformed")
    if evidence_id != "ev_" + observation_id.removeprefix("obs_"):
        raise ApplicationRuntimeError("live evidence identity conflicts with its observation")
    _timestamp(observed_at, "live observation timestamp")
    if object_type not in _LIVE_KINDS:
        raise ApplicationRuntimeError("live observation type is unsupported")
    if not isinstance(payload_digest, str) or _DIGEST.fullmatch(payload_digest) is None:
        raise ApplicationRuntimeError("live observation digest is malformed")
    _live_github_url(source_url)
    _live_github_url(citation_url)
    return LiveRuntimeEvidence(
        evidence_id,
        text,
        observation_id,
        cast(str, observed_at),
        cast(str, object_type),
        payload_digest,
        cast(str, source_url),
        cast(str, citation_url),
    )


def _evidence_value(value: RuntimeEvidence) -> dict[str, object]:
    if isinstance(value, StaticRuntimeEvidence):
        metadata: dict[str, object] = {
            "generation_id": value.generation_id,
            "evidence_id": value.evidence_id,
            "repository": value.repository,
            "path": value.path,
            "commit": value.commit,
            "authority": value.authority,
            "version_scope": value.version_scope,
            "content_digest": value.content_digest,
            "immutable_url": value.immutable_url,
        }
    else:
        metadata = {
            "evidence_id": value.evidence_id,
            "observation_id": value.observation_id,
            "observed_at": value.observed_at,
            "object_type": value.object_type,
            "payload_digest": value.payload_digest,
            "source_url": value.source_url,
            "citation_url": value.citation_url,
        }
    return {"text": value.text, "metadata": metadata}


def _live_evidence(observation: object) -> LiveRuntimeEvidence:
    if not isinstance(observation, LiveObservation):
        raise ApplicationRuntimeError("live reader returned the wrong observation type")
    value = live_observation_value(observation)
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ApplicationRuntimeError("live observation payload must be an object")
    if payload.get("api_version") != "valkeyrie.io/live-github/1":
        raise ApplicationRuntimeError("live observation payload identity is incompatible")
    kind = payload.get("kind")
    allowed_kinds = _LIVE_KINDS.get(observation.object_type)
    if not isinstance(kind, str) or allowed_kinds is None or kind not in allowed_kinds:
        raise ApplicationRuntimeError("live observation payload conflicts with its type")
    text = observation.canonical_payload.decode("utf-8")
    _bounded_text(text, "live observation payload", _MAX_EVIDENCE_BYTES)
    candidate = payload.get("url")
    citation_url = candidate if isinstance(candidate, str) else observation.source_url
    _live_github_url(observation.source_url)
    _live_github_url(citation_url)
    return LiveRuntimeEvidence(
        "ev_" + observation.observation_id.removeprefix("obs_"),
        text,
        observation.observation_id,
        observation.observed_at,
        observation.object_type,
        observation.payload_digest,
        observation.source_url,
        citation_url,
    )


def _replayable_result(value: RuntimeResult) -> dict[str, object]:
    """The parts of a result that a redelivery must reproduce.

    Revision and fence are deliberately excluded: they describe the write that completed the
    request, not the answer, and a replay is not that write.
    """
    return {
        "outcome": value.outcome,
        "message": value.message,
        "claims": [dict(item) for item in value.claims],
        "citations": list(value.citations),
        "generation_id": value.generation_id,
    }


def _dynamo_value(value: object) -> object:
    """Convert a result payload into DynamoDB-safe types.

    DynamoDB rejects float, which is what json numbers decode to, so numbers are carried as
    Decimal. Nothing else in a result needs converting.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (str, int, Decimal)):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, Mapping):
        return {str(key): _dynamo_value(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_dynamo_value(item) for item in value]
    raise ApplicationRuntimeError("result payload holds an unsupported type")


def _replayed_result(
    item: Mapping[str, object],
    request_digest: str,
    manifest: Mapping[str, object],
    *,
    request_id: str,
) -> RuntimeResult | None:
    """Return the recorded answer for an already-completed request, or None if it is still open.

    A redelivery is the normal case, not an error: Slack retries an event it did not see acked,
    and the request ID is derived from the event identity so the retry arrives as the same
    request. Recomputing would charge for a second inference and could answer differently, so the
    recorded result is returned instead.

    The digest is checked first. Without it, a caller reusing a request ID with a different
    question would receive the previous question's answer.
    """
    if item.get("outcome") is None:
        return None
    plan = item.get("plan")
    if not isinstance(plan, Mapping):
        raise ApplicationRuntimeError("request audit state is malformed")
    if plan.get("question_digest") != request_digest:
        raise ApplicationRuntimeError("request is pinned to different content")
    if plan.get("application_revision") != manifest["application_revision"]:
        raise ApplicationRuntimeError("request is pinned to a different application revision")
    stored = item.get("result")
    if not isinstance(stored, Mapping):
        # Completed before results were recorded. Replaying is impossible and recomputing would
        # break the single-completion guarantee, so this stays an error.
        raise ApplicationRuntimeError("request audit is already terminal")
    outcome = stored.get("outcome")
    if outcome not in _RUNTIME_OUTCOMES:
        raise ApplicationRuntimeError("recorded result outcome is unsupported")
    claims = stored.get("claims", ())
    citations = stored.get("citations", ())
    if not isinstance(claims, Sequence) or not isinstance(citations, Sequence):
        raise ApplicationRuntimeError("recorded result is malformed")
    message = stored.get("message")
    if message is not None and not isinstance(message, str):
        raise ApplicationRuntimeError("recorded result message is malformed")
    generation_id = stored.get("generation_id")
    if generation_id is not None and not isinstance(generation_id, str):
        raise ApplicationRuntimeError("recorded result generation is malformed")
    return RuntimeResult(
        cast(RuntimeOutcome, outcome),
        request_id,
        message,
        tuple(dict(cast(Mapping[str, object], claim)) for claim in claims),
        tuple(str(citation) for citation in citations),
        generation_id=generation_id,
    )


def _existing_plan(
    item: Mapping[str, object],
    request_digest: str,
    manifest: Mapping[str, object],
) -> tuple[Mapping[str, object], int, int]:
    if item.get("outcome") is not None:
        raise ApplicationRuntimeError("request audit is already terminal")
    plan = item.get("plan")
    revision, fence = item.get("revision"), item.get("fence")
    if not isinstance(plan, Mapping) or type(revision) is not int or type(fence) is not int:
        raise ApplicationRuntimeError("request audit state is malformed")
    if plan.get("question_digest") != request_digest:
        raise ApplicationRuntimeError("request is pinned to different content")
    if plan.get("application_revision") != manifest["application_revision"]:
        raise ApplicationRuntimeError("request is pinned to a different application revision")
    if plan.get("execution_authorization") != _execution_authorization(manifest):
        raise ApplicationRuntimeError("request is pinned to different execution authorization")
    required = {
        "request_id",
        "question_digest",
        "question",
        "evidence_mode",
        "generation_id",
        "knowledge_base_id",
        "application_revision",
        "prompt_revision",
        "selected_profile_revision",
        "selected_inference_config_revision",
        "selected_model_revision",
        "execution_authorization",
        "model_id",
        "maximum_output_tokens",
        "reasoning_effort",
        "system",
        "evidence",
    }
    if set(plan) != required:
        raise ApplicationRuntimeError("pinned execution has an unknown or missing field")
    _plan_generation_id(plan)
    return plan, revision, fence


def _plan_generation_id(plan: Mapping[str, object]) -> str | None:
    mode = plan.get("evidence_mode")
    generation_id = plan.get("generation_id")
    knowledge_base_id = plan.get("knowledge_base_id")
    if mode == "static":
        if not isinstance(generation_id, str) or _DIGEST.fullmatch(generation_id) is None:
            raise ApplicationRuntimeError("static plan generation identity is malformed")
        if not isinstance(knowledge_base_id, str) or _KB_ID.fullmatch(knowledge_base_id) is None:
            raise ApplicationRuntimeError("static plan knowledge base identity is malformed")
        return generation_id
    if mode == "live":
        if generation_id is not None or knowledge_base_id is not None:
            raise ApplicationRuntimeError("live plan cannot claim static provenance")
        return None
    raise ApplicationRuntimeError("pinned execution evidence mode is unsupported")


def _generation_id(value: Mapping[str, object]) -> str:
    generation_id = value.get("generation_id")
    if not isinstance(generation_id, str) or _DIGEST.fullmatch(generation_id) is None:
        raise ApplicationRuntimeError("generation state identity is malformed")
    revision = value.get("revision")
    if type(revision) is not int or revision < 1:
        raise ApplicationRuntimeError("generation state revision is malformed")
    return generation_id


def _validate_manifest(value: Mapping[str, object]) -> None:
    required = {
        "application_revision",
        "prompt_revision",
        "selected_model_revision",
        "selected_profile_revision",
        "selected_inference_config_revision",
        "selected_report_id",
        "selection_id",
        "selected_inference_profile_arn",
        "selected_foundation_model_arns",
        "selected_inference",
        "response_normalization_policy_revision",
    }
    if not required <= set(value):
        raise ApplicationRuntimeError("application manifest lacks qualification identity")
    for field in (
        "application_revision",
        "prompt_revision",
        "selected_profile_revision",
        "selected_inference_config_revision",
        "selection_id",
        "response_normalization_policy_revision",
    ):
        if not isinstance(value[field], str) or _DIGEST.fullmatch(cast(str, value[field])) is None:
            raise ApplicationRuntimeError(f"application manifest {field} is malformed")
    _execution_authorization(value)


def _result_value(value: RuntimeResult) -> dict[str, object]:
    return {
        "outcome": value.outcome,
        "request_id": value.request_id,
        "message": value.message,
        "claims": [dict(item) for item in value.claims],
        "citations": list(value.citations),
        "generation_id": value.generation_id,
        "request_revision": value.request_revision,
        "request_fence": value.request_fence,
    }


def _bounded_text(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApplicationRuntimeError(f"{label} must be non-blank text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ApplicationRuntimeError(f"{label} must be valid UTF-8") from error
    if len(encoded) > maximum:
        raise ApplicationRuntimeError(f"{label} exceeds its byte bound")
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


def _timestamp(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or _TIMESTAMP.fullmatch(value) is None
        or not _is_calendar_timestamp(value)
    ):
        raise ApplicationRuntimeError(f"{label} is malformed")
    _timestamp_value(value)
    return value


def _timestamp_value(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ApplicationRuntimeError("timestamp is invalid") from error
    if parsed.utcoffset() != timedelta(0):
        raise ApplicationRuntimeError("timestamp is not UTC")
    return parsed


def _lease_expiry(now: str, seconds: int) -> str:
    value = _timestamp_value(now) + timedelta(seconds=seconds)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _immutable_url(value: object, commit: str) -> None:
    if not isinstance(value, str):
        raise ApplicationRuntimeError("retrieval immutable URL is malformed")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or f"/blob/{commit}/" not in parsed.path
    ):
        raise ApplicationRuntimeError(
            "retrieval immutable URL is not commit-pinned GitHub evidence"
        )


def _live_github_url(value: object) -> None:
    if not isinstance(value, str):
        raise ApplicationRuntimeError("live GitHub URL is malformed")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.fragment
        or "//" in parsed.path
        or any(part in {".", ".."} for part in parsed.path.split("/"))
    ):
        raise ApplicationRuntimeError("live GitHub URL is not canonical")
    if parsed.netloc == "api.github.com":
        allowed = (
            parsed.path.startswith("/repos/valkey-io/")
            or parsed.path == "/search/issues"
            or parsed.path == "/graphql"
        )
    elif parsed.netloc == "github.com":
        allowed = parsed.path.startswith("/valkey-io/") or parsed.path.startswith(
            "/orgs/valkey-io/projects/"
        )
    else:
        allowed = False
    if not allowed:
        raise ApplicationRuntimeError("live GitHub URL is outside the Valkey allowlist")


def _answer_request_digest(
    question: str,
    version_requirement: str,
    requested_version: str | None,
    knowledge_base_id: str,
    exact_identifier: ReleaseArtifactIdentifier | None,
) -> str:
    exact = (
        None
        if exact_identifier is None
        else {
            "record_type": "release_artifact_digest",
            "release": exact_identifier.release,
            "artifact": exact_identifier.artifact,
        }
    )
    preimage = json.dumps(
        {
            "question": question,
            "version_requirement": version_requirement,
            "requested_version": requested_version,
            "knowledge_base_id": knowledge_base_id,
            "exact_identifier": exact,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def _question_digest(value: str, exact_identifier: ReleaseArtifactIdentifier | None = None) -> str:
    if exact_identifier is None:
        preimage = value
    else:
        preimage = json.dumps(
            [value, exact_identifier.release, exact_identifier.artifact],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return "sha256:" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def _release_artifact_identifier(value: object) -> ReleaseArtifactIdentifier | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"record_type", "release", "artifact"}:
        raise ApplicationRuntimeError("exact identifier has an unknown or missing field")
    if value["record_type"] != "release_artifact_digest":
        raise ApplicationRuntimeError("exact identifier record type is unsupported")
    release, artifact = value["release"], value["artifact"]
    if not isinstance(release, str) or _RELEASE.fullmatch(release) is None:
        raise ApplicationRuntimeError("exact release is malformed")
    if not isinstance(artifact, str):
        raise ApplicationRuntimeError("exact release artifact is malformed")
    match = _RELEASE_ARTIFACT.fullmatch(artifact)
    if match is None or match.group("release") != release:
        raise ApplicationRuntimeError("exact release artifact does not match its release")
    return ReleaseArtifactIdentifier(release, artifact)


def _release_artifact_record_id(identifier: ReleaseArtifactIdentifier) -> str:
    return f"release_artifact_digest:{identifier.release}:{identifier.artifact}"


def _structured_record_digest(generation: Mapping[str, object], record_id: str) -> str | None:
    values = generation.get("structured_records")
    root = generation.get("structured_index_sha256")
    if not isinstance(values, Mapping) or not isinstance(root, str):
        return None
    if not 1 <= len(values) <= 10_000 or any(
        not isinstance(key, str)
        or re.fullmatch(r"[0-9a-f]{64}", key) is None
        or not isinstance(value, str)
        or _DIGEST.fullmatch(value) is None
        for key, value in values.items()
    ):
        raise ApplicationRuntimeError("structured generation index is malformed")
    canonical = json.dumps(
        dict(sorted(values.items())), sort_keys=True, separators=(",", ":")
    ).encode()
    if root != "sha256:" + hashlib.sha256(canonical).hexdigest():
        raise ApplicationRuntimeError("structured generation index checksum is invalid")
    key = hashlib.sha256(record_id.encode()).hexdigest()
    return cast(str | None, values.get(key))


def _structured_record_evidence(
    content: bytes,
    generation_id: str,
    identifier: ReleaseArtifactIdentifier,
) -> StaticRuntimeEvidence:
    if not isinstance(content, bytes) or not 1 <= len(content) <= _MAX_EVIDENCE_BYTES:
        raise ApplicationRuntimeError("structured record content is outside its byte bound")
    try:
        record = json.loads(content, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ApplicationRuntimeError("structured record is invalid JSON") from error
    required = {
        "api_version",
        "generation_id",
        "identifier",
        "kind",
        "provenance",
        "record_id",
        "record_type",
        "source",
        "value",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise ApplicationRuntimeError("structured record has an unknown or missing field")
    record_id = _release_artifact_record_id(identifier)
    if (
        record["api_version"] != "valkeyrie.io/structured-record/1"
        or record["kind"] != "StructuredRecord"
        or record["record_type"] != "release_artifact_digest"
        or record["record_id"] != record_id
        or record["generation_id"] != generation_id
        or record["identifier"] != {"release": identifier.release, "artifact": identifier.artifact}
    ):
        raise ApplicationRuntimeError("structured record identity does not match the exact request")
    source, provenance, value = record["source"], record["provenance"], record["value"]
    source_fields = {
        "authority",
        "commit",
        "ref_kind",
        "repository",
        "repository_url",
        "requested_ref",
        "source_policy_digest",
        "version_scope",
    }
    if not isinstance(source, Mapping) or set(source) != source_fields:
        raise ApplicationRuntimeError("structured record source is malformed")
    if not isinstance(provenance, Mapping) or set(provenance) != {"commit", "path", "repository"}:
        raise ApplicationRuntimeError("structured record provenance is malformed")
    if not isinstance(value, Mapping) or set(value) != {"digest"}:
        raise ApplicationRuntimeError("structured record value is malformed")
    commit, repository, path = provenance["commit"], provenance["repository"], provenance["path"]
    if (
        source.get("repository") != "valkey-hashes"
        or source.get("repository_url") != "https://github.com/valkey-io/valkey-hashes"
        or source.get("authority") != "structured"
        or source.get("version_scope") != "release_artifacts"
        or source.get("ref_kind") != "branch"
        or source.get("requested_ref") != "main"
        or not isinstance(source.get("source_policy_digest"), str)
        or _DIGEST.fullmatch(cast(str, source["source_policy_digest"])) is None
        or repository != "valkey-hashes"
        or not isinstance(path, str)
        or (
            path != "README"
            and re.fullmatch(r"releases/[A-Za-z0-9][A-Za-z0-9._+-]*\.sha256", path) is None
        )
    ):
        raise ApplicationRuntimeError("structured record source is not authoritative")
    if (
        repository != source["repository"]
        or commit != source["commit"]
        or not isinstance(repository, str)
        or not isinstance(path, str)
        or not path
        or not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise ApplicationRuntimeError("structured record provenance conflicts with its source")
    authority, version_scope, digest = source["authority"], source["version_scope"], value["digest"]
    if (
        not isinstance(authority, str)
        or not authority
        or not isinstance(version_scope, str)
        or not version_scope
        or not isinstance(digest, str)
        or _DIGEST.fullmatch(digest) is None
    ):
        raise ApplicationRuntimeError("structured record exact value is malformed")
    evidence_id = "ev_" + hashlib.sha256(f"{generation_id}:{record_id}".encode()).hexdigest()
    immutable_url = (
        f"https://github.com/valkey-io/{quote(repository, safe='._-')}/blob/"
        f"{commit}/{quote(path, safe='/._-')}"
    )
    _immutable_url(immutable_url, commit)
    return StaticRuntimeEvidence(
        evidence_id,
        f"The SHA-256 digest for {identifier.artifact} is {digest}.",
        generation_id,
        repository,
        path,
        commit,
        authority,
        version_scope,
        "sha256:" + hashlib.sha256(content).hexdigest(),
        immutable_url,
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ApplicationRuntimeError(f"duplicate model output key: {key}")
        value[key] = item
    return value


def _bedrock_retrieval_text(content: object) -> str:
    if not isinstance(content, Mapping) or content.get("type") != "TEXT":
        raise ApplicationRuntimeError("Bedrock retrieval content is not text")
    text = content.get("text")
    if not isinstance(text, str) or not text:
        raise ApplicationRuntimeError("Bedrock retrieval text is empty")
    return text


def _normalize_dynamodb_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return {key: _normalize_dynamodb_value(item) for key, item in value.items()}


def _normalize_dynamodb_value(value: object) -> object:
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise ApplicationRuntimeError("DynamoDB state contains a non-integral number")
        return int(value)
    if isinstance(value, Mapping):
        return _normalize_dynamodb_mapping(cast(Mapping[str, object], value))
    if isinstance(value, list):
        return [_normalize_dynamodb_value(item) for item in value]
    return value


def _runtime_retrieval_metadata(metadata: object) -> Mapping[str, object]:
    if not isinstance(metadata, Mapping):
        raise ApplicationRuntimeError("Bedrock retrieval metadata is malformed")
    # Every field here comes from the metadata sidecar this project publishes, so it is
    # provenance we control and can insist on. `x-amz-bedrock-kb-chunk-id` is deliberately
    # NOT required: Bedrock omits it when a document produces a single chunk, so requiring
    # it rejected every short document. valkey/MAINTAINERS.md is 1,462 bytes, yields one
    # chunk, and names the TSC Chair, and this check was discarding it and failing the
    # whole request. A chunk identifier is a Bedrock implementation detail, not provenance.
    required = (
        "generation_id",
        "document_id",
        "repository",
        "path",
        "commit",
        "authority",
        "version_scope",
        "content_digest",
    )
    if any(
        not isinstance(metadata.get(field), str) or not metadata.get(field) for field in required
    ):
        raise ApplicationRuntimeError("Bedrock retrieval metadata lacks provenance")
    repository = cast(str, metadata["repository"])
    authority = cast(str, metadata["authority"])
    try:
        expected_authority = reviewed_source_authority(repository)
    except RetrievalError as error:
        raise ApplicationRuntimeError("Bedrock retrieval repository is not reviewed") from error
    if authority != expected_authority:
        raise ApplicationRuntimeError("Bedrock retrieval authority is invalid")
    path = cast(str, metadata["path"])
    if (
        path.startswith("/")
        or "\\" in path
        or "//" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        raise ApplicationRuntimeError("Bedrock retrieval path is unsafe")
    commit = cast(str, metadata["commit"])
    # The chunk id distinguishes several chunks of one document. Bedrock omits it when a
    # document yields a single chunk, in which case the document id already identifies the
    # chunk uniquely, so a fixed marker keeps evidence ids stable and distinct.
    chunk = metadata.get("x-amz-bedrock-kb-chunk-id") or "single-chunk"
    identity = json.dumps(
        [metadata["generation_id"], metadata["document_id"], chunk],
        separators=(",", ":"),
    )
    return {
        "generation_id": metadata["generation_id"],
        "evidence_id": f"ev_{hashlib.sha256(identity.encode('utf-8')).hexdigest()}",
        "repository": repository,
        "path": path,
        "commit": commit,
        "authority": metadata["authority"],
        "version_scope": metadata["version_scope"],
        "content_digest": metadata["content_digest"],
        "immutable_url": (
            f"https://github.com/valkey-io/{quote(repository, safe='._-')}/blob/"
            f"{commit}/{quote(path, safe='/._-')}"
        ),
    }


def _runtime_retrieve(
    client: Any, knowledge_base_id: str, query: str, retrieval_filter: Mapping[str, object]
) -> Mapping[str, object]:
    try:
        response = client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "filter": retrieval_filter,
                    "numberOfResults": 10,
                    "overrideSearchType": "HYBRID",
                }
            },
        )
    except RetrievalError as error:  # pragma: no cover - typed helper failures are preserved
        raise ApplicationRuntimeError(f"static retrieval intent is invalid: {error}") from error
    if not isinstance(response, Mapping):
        raise ApplicationRuntimeError("Bedrock retrieval response is malformed")
    return response


class AwsRuntimeServices:
    """Minimal native Lambda adapters; boto3 is imported only when an AWS action runs."""

    # Class-level default so an instance created without __init__ still reads anonymously.
    # Assignment in _github_token creates an instance attribute, so the cache never leaks
    # between instances.
    _github_token_cached: object = _UNSET

    def __init__(self) -> None:
        self._table_name = os.environ["STATE_TABLE_NAME"]
        self._model_id = os.environ["SELECTED_INFERENCE_PROFILE_ARN"]

    @staticmethod
    def _boto3() -> Any:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - Lambda provides boto3
            raise RuntimeError("Lambda boto3 runtime is unavailable") from error
        return boto3

    def _table(self) -> Any:
        return self._boto3().resource("dynamodb").Table(self._table_name)

    def read_controls(self) -> Mapping[str, str]:
        response = (
            self._boto3()
            .client("ssm")
            .get_parameters(Names=list(_CONTROL_NAMES), WithDecryption=False)
        )
        return {item["Name"]: item["Value"] for item in response.get("Parameters", [])}

    def read_generation(self, version_scope: str | None) -> Mapping[str, object] | None:
        pointer = "active_generation" if version_scope is None else f"version#{version_scope}"
        item = self._table().get_item(Key={"pk": pointer}, ConsistentRead=True).get("Item")
        if not isinstance(item, Mapping):
            return None
        generation_id = item.get("generation_id")
        if not isinstance(generation_id, str):
            return None
        state = (
            self._table()
            .get_item(
                Key={"pk": f"generation#{generation_id.removeprefix('sha256:')}"},
                ConsistentRead=True,
            )
            .get("Item")
        )
        if not isinstance(state, Mapping):
            return None
        return _normalize_dynamodb_mapping(state)

    def read_live(self, query: LiveGitHubQuery) -> LiveObservation:
        token = self._github_token()
        if token is None:
            return read_live_github(query)
        return read_live_github(
            query,
            fetch=partial(fetch_public_github, token=token),
        )

    def _github_token(self) -> str | None:
        """Return the read-only GitHub token, or None to read anonymously.

        Anonymous reads are limited to 60 an hour per IP, which the supplementary search
        exhausts quickly and then silently stops supplementing. A token raises that to 5,000
        an hour. Read once per container and cached, because a Secrets Manager call on every
        question would be its own rate limit.

        Every failure returns None rather than raising: an absent, empty, or unreadable secret
        must degrade to anonymous reads, not break answering.
        """
        if self._github_token_cached is not _UNSET:
            return cast("str | None", self._github_token_cached)
        token: str | None = None
        secret_id = os.environ.get("GITHUB_TOKEN_SECRET_ID", "")
        if secret_id:
            try:
                value = self._boto3().client("secretsmanager").get_secret_value(SecretId=secret_id)
                candidate = value.get("SecretString")
                if isinstance(candidate, str) and candidate.strip():
                    token = candidate.strip()
            except Exception:
                token = None
        self._github_token_cached = token
        return token

    def retrieve(
        self, *, knowledge_base_id: str, generation_id: str, question: str
    ) -> tuple[Mapping[str, object], ...]:
        try:
            intent = derive_retrieval_intent(question)
        except RetrievalError as error:
            raise ApplicationRuntimeError(f"static retrieval intent is invalid: {error}") from error
        generation_filter: Mapping[str, object] = {
            "equals": {"key": "generation_id", "value": generation_id}
        }
        repository_filters = [
            {"equals": {"key": "repository", "value": repository}}
            for repository in intent.repositories
        ]
        repository_filter: Mapping[str, object] = (
            repository_filters[0] if len(repository_filters) == 1 else {"orAll": repository_filters}
        )
        scoped_filter: Mapping[str, object] = {"andAll": [generation_filter, repository_filter]}
        client = self._boto3().client("bedrock-agent-runtime")
        response = _runtime_retrieve(client, knowledge_base_id, intent.query, scoped_filter)
        try:
            results = verify_retrieval_results(
                response, generation_id, "generation_id", 10, intent.repositories
            )
            if not results:
                response = _runtime_retrieve(
                    client, knowledge_base_id, intent.query, generation_filter
                )
                results = verify_retrieval_results(
                    response, generation_id, "generation_id", 10, intent.repositories
                )
        except RetrievalError as error:
            raise ApplicationRuntimeError(
                f"Bedrock retrieval response is not trusted: {error}"
            ) from error
        return tuple(
            {
                "text": result.text,
                "metadata": _runtime_retrieval_metadata(result.metadata),
            }
            for result in results
        )

    def read_structured_record(
        self, *, generation_id: str, record_id: str, expected_content_digest: str
    ) -> bytes | None:
        key_digest = hashlib.sha256(record_id.encode()).hexdigest()
        key = f"structured#{generation_id.removeprefix('sha256:')}#{key_digest}"
        item = self._table().get_item(Key={"pk": key}, ConsistentRead=True).get("Item")
        if not isinstance(item, Mapping):
            return None
        required = {
            "pk",
            "record_type",
            "generation_id",
            "record_id",
            "content_digest",
            "content",
        }
        if set(item) != required or (
            item.get("record_type") != "structured_record"
            or item.get("generation_id") != generation_id
            or item.get("record_id") != record_id
        ):
            raise ApplicationRuntimeError("structured record state identity is malformed")
        content = item.get("content")
        digest = item.get("content_digest")
        if not isinstance(content, str):
            raise ApplicationRuntimeError("structured record state content is malformed")
        encoded = content.encode("utf-8")
        if not 1 <= len(encoded) <= _MAX_EVIDENCE_BYTES:
            raise ApplicationRuntimeError("structured record state is outside its byte bound")
        if (
            digest != expected_content_digest
            or digest != "sha256:" + hashlib.sha256(encoded).hexdigest()
        ):
            raise ApplicationRuntimeError("structured record state checksum is invalid")
        return encoded

    def get_request(self, request_id: str) -> Mapping[str, object] | None:
        item = (
            self._table()
            .get_item(Key={"pk": f"request#{request_id}"}, ConsistentRead=True)
            .get("Item")
        )
        if not isinstance(item, Mapping):
            return None
        return _normalize_dynamodb_mapping(item)

    def claim_request(self, item: Mapping[str, object]) -> bool:
        try:
            self._table().put_item(Item=dict(item), ConditionExpression="attribute_not_exists(pk)")
        except Exception as error:
            if _aws_error_code(error) == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def recover_request(
        self,
        *,
        request_id: str,
        expected_revision: int,
        expected_fence: int,
        expected_owner: str,
        new_owner: str,
        now: str,
        lease_expires_at: str,
    ) -> Mapping[str, object] | None:
        try:
            response = self._table().update_item(
                Key={"pk": f"request#{request_id}"},
                UpdateExpression=(
                    "SET #owner = :new_owner, lease_expires_at = :lease, "
                    "revision = :next_revision, fence = :next_fence"
                ),
                ConditionExpression=(
                    "revision = :revision AND fence = :fence AND #owner = :owner "
                    "AND lease_expires_at <= :now AND attribute_not_exists(outcome)"
                ),
                ExpressionAttributeNames={"#owner": "owner"},
                ExpressionAttributeValues={
                    ":new_owner": new_owner,
                    ":lease": lease_expires_at,
                    ":next_revision": expected_revision + 1,
                    ":next_fence": expected_fence + 1,
                    ":revision": expected_revision,
                    ":fence": expected_fence,
                    ":owner": expected_owner,
                    ":now": now,
                },
                ReturnValues="ALL_NEW",
            )
        except Exception as error:
            if _aws_error_code(error) == "ConditionalCheckFailedException":
                return None
            raise
        attributes = response.get("Attributes")
        if not isinstance(attributes, Mapping):
            raise ApplicationRuntimeError("request recovery returned no state")
        return _normalize_dynamodb_mapping(attributes)

    def complete_request(
        self,
        *,
        request_id: str,
        revision: int,
        fence: int,
        outcome: str,
        completed_at: str,
        result: Mapping[str, object] | None = None,
    ) -> bool:
        # The result is written in the SAME conditional update as the outcome. A second write
        # could be lost, leaving a terminal record with no answer to replay, which is the state
        # that turns a Slack redelivery into an error.
        expression = "SET outcome = :outcome, completed_at = :completed, revision = :next"
        values: dict[str, object] = {
            ":outcome": outcome,
            ":completed": completed_at,
            ":next": revision + 1,
            ":revision": revision,
            ":fence": fence,
        }
        if result is not None:
            expression += ", #result = :result"
            values[":result"] = _dynamo_value(result)
        try:
            self._table().update_item(
                Key={"pk": f"request#{request_id}"},
                UpdateExpression=expression,
                ConditionExpression=(
                    "revision = :revision AND fence = :fence AND attribute_not_exists(outcome)"
                ),
                ExpressionAttributeNames={"#result": "result"},
                ExpressionAttributeValues=values,
            )
        except Exception as error:
            if _aws_error_code(error) == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def converse(
        self,
        *,
        model_id: str,
        system: tuple[str, ...],
        question: str,
        evidence: tuple[RuntimeEvidence, ...],
        maximum_output_tokens: int,
        reasoning_effort: str,
    ) -> BedrockTextResponse:
        if model_id != self._model_id:
            raise ApplicationRuntimeError("pinned model target differs from deployed selection")
        model_input = {
            "question": question,
            "evidence": [_evidence_value(item) for item in evidence],
        }
        response = (
            self._boto3()
            .client("bedrock-runtime")
            .converse(
                modelId=model_id,
                system=[{"text": text} for text in system],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": json.dumps(
                                    model_input,
                                    ensure_ascii=False,
                                    allow_nan=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                            },
                            {"text": system[-1]},
                        ],
                    }
                ],
                inferenceConfig={"maxTokens": maximum_output_tokens},
                additionalModelRequestFields={
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": reasoning_effort},
                },
            )
        )
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in blocks if isinstance(block, Mapping))
        return BedrockTextResponse(text, response.get("stopReason"))


def _aws_error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    return cast(str | None, details.get("Code") if isinstance(details, Mapping) else None)
