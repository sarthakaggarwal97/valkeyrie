from __future__ import annotations

from dataclasses import replace

import pytest

from tests.test_evidence import _bundle, _package
from valkeyrie.evidence import EvidencePackage
from valkeyrie.generation import GenerationBundle
from valkeyrie.retrieval_config import load_retrieval_config
from valkeyrie.routing import (
    EvidenceStatus,
    LiveObservation,
    QuestionRequest,
    RouteDecision,
    RoutingError,
    resolve_evidence,
    route_question,
)
from valkeyrie.structured import ExactLookup

VERSIONS = ("8.1", "9.0", "unstable")


@pytest.fixture(scope="module")
def generation() -> GenerationBundle:
    config = load_retrieval_config(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "retrieval-config.yaml"
    )
    return _bundle(config)


@pytest.fixture(scope="module")
def package(generation: GenerationBundle) -> EvidencePackage:
    return _package(generation)


def _static_request(**changes: object) -> QuestionRequest:
    values: dict[str, object] = {
        "question": "How is hash-field expiration implemented in 8.1?",
        "version_requirement": "required",
        "requested_version": "8.1",
        "available_versions": VERSIONS,
    }
    values.update(changes)
    return QuestionRequest(**values)  # type: ignore[arg-type]


def _evidence(
    bundle: GenerationBundle,
    evidence_package: EvidencePackage,
    **changes: object,
) -> EvidenceStatus:
    values: dict[str, object] = {
        "dependency_available": True,
        "supported": True,
        "conflicting": False,
        "generation": bundle,
        "package": evidence_package,
    }
    values.update(changes)
    return EvidenceStatus(**values)  # type: ignore[arg-type]


def _observation(**changes: object) -> LiveObservation:
    values: dict[str, object] = {
        "observation_id": "obs_pr-123",
        "observed_at": "2026-08-19T03:25:21Z",
        "source_url": "https://api.github.com/repos/valkey-io/valkey/pulls/123",
        "payload_digest": f"sha256:{'1' * 64}",
        "complete": True,
        "truncated": False,
    }
    values.update(changes)
    return LiveObservation(**values)  # type: ignore[arg-type]


def test_routes_versioned_questions_to_verified_static_evidence(
    generation: GenerationBundle, package: EvidencePackage
) -> None:
    decision = route_question(_static_request())
    assert decision == RouteDecision("route", ("static_semantic",), "8.1")
    assert resolve_evidence(decision, _evidence(generation, package)).outcome == "answer"


def test_exact_identifier_route_requires_the_validated_exact_index(
    generation: GenerationBundle, package: EvidencePackage
) -> None:
    record = generation.structured_record_templates[0]
    identifier = record.identifier
    decision = route_question(_static_request(exact_identifier=identifier))
    assert decision.routes == ("exact_lookup",)
    missing = resolve_evidence(decision, _evidence(generation, package))
    assert missing.outcome == "abstention"
    exact = _evidence(
        generation,
        package,
        exact_lookup=ExactLookup(generation.structured_record_templates),
        exact_record=record,
    )
    assert resolve_evidence(decision, exact).outcome == "answer"


@pytest.mark.parametrize(
    "question",
    [
        "Is pull request 123 currently open?",
        "What is the latest Valkey release?",
        "What is the status of pull request 123?",
        "Which upcoming Valkey events are announced?",
    ],
)
def test_current_latest_and_status_language_use_validated_live_observations(
    question: str,
) -> None:
    decision = route_question(QuestionRequest(question=question, version_requirement="none"))
    assert decision.routes == ("live_read",)
    status = EvidenceStatus(True, True, False, live_observations=(_observation(),))
    assert resolve_evidence(decision, status).outcome == "answer"


@pytest.mark.parametrize(
    "question",
    [
        "How do I open a Valkey connection?",
        "How are closed client connections cleaned up?",
        "How does a replica become ready?",
        "How does CLIENT LIST report status?",
    ],
)
def test_technical_state_words_do_not_trigger_live_project_routing(question: str) -> None:
    decision = route_question(QuestionRequest(question, "none"))

    assert decision == RouteDecision("route", ("static_semantic",), None)


def test_current_language_cannot_use_static_or_mixed_evidence(
    generation: GenerationBundle, package: EvidencePackage
) -> None:
    decision = route_question(
        QuestionRequest(question="What was published most recently?", version_requirement="none")
    )
    assert decision.routes == ("live_read",)
    stale = resolve_evidence(decision, _evidence(generation, package))
    assert stale.reason == "current-state routing cannot use static corpus evidence"
    mixed = resolve_evidence(
        decision,
        _evidence(generation, package, live_observations=(_observation(),)),
    )
    assert mixed.outcome == "abstention"


def test_current_state_rejects_a_static_version_scope() -> None:
    decision = route_question(_static_request(question="Is the 8.1 release currently ready?"))
    assert decision.outcome == "abstention"
    assert decision.routes == ()


@pytest.mark.parametrize(
    "question",
    [
        "How does SET work?",
        "Does this command support the new option?",
        "How is this feature implemented?",
    ],
)
def test_unqualified_valkey_feature_and_command_questions_default_to_core(
    question: str,
) -> None:
    decision = route_question(QuestionRequest(question, "none", available_versions=VERSIONS))
    assert decision == RouteDecision("route", ("static_semantic",), None)


def test_exact_identifier_routes_exact_before_non_material_ambiguity(
    generation: GenerationBundle,
) -> None:
    identifier = generation.structured_record_templates[0].identifier
    decision = route_question(
        QuestionRequest(
            "Does this command support the new option?",
            "none",
            available_versions=VERSIONS,
            exact_identifier=identifier,
        )
    )
    assert decision == RouteDecision("route", ("exact_lookup",), None, exact_identifier=identifier)


def test_explicit_missing_version_asks_one_bounded_clarification() -> None:
    decision = route_question(_static_request(requested_version=None))
    assert decision.outcome == "clarification"
    assert decision.question == "Which Valkey release or branch should I use?"
    assert decision.routes == ()


def test_unknown_version_abstains_without_retrieval() -> None:
    decision = route_question(_static_request(requested_version="7.2"))
    assert decision.outcome == "abstention"
    assert decision.reason == "the requested version scope is unavailable: 7.2"


@pytest.mark.parametrize(
    ("changes", "outcome", "reason"),
    [
        (
            {"dependency_available": False},
            "partial",
            "a required retrieval dependency is unavailable",
        ),
        ({"conflicting": True}, "abstention", "validated canonical evidence conflicts"),
        (
            {"supported": False},
            "abstention",
            "validated evidence does not support an answer",
        ),
        (
            {"generation": None, "package": None},
            "abstention",
            "static retrieval produced no validated evidence",
        ),
        (
            {"live_observations": (_observation(),)},
            "abstention",
            "unexpected live observations were supplied",
        ),
    ],
)
def test_insufficient_conflicting_or_unexpected_evidence_fails_closed(
    generation: GenerationBundle,
    package: EvidencePackage,
    changes: dict[str, object],
    outcome: str,
    reason: str,
) -> None:
    disposition = resolve_evidence(
        route_question(_static_request()), _evidence(generation, package, **changes)
    )
    assert disposition.outcome == outcome
    assert disposition.reason == reason


def test_conflict_or_unsupported_always_precedes_dependency_partial(
    generation: GenerationBundle, package: EvidencePackage
) -> None:
    decision = route_question(_static_request())
    conflict = resolve_evidence(
        decision,
        _evidence(generation, package, dependency_available=False, conflicting=True),
    )
    unsupported = resolve_evidence(
        decision,
        _evidence(generation, package, dependency_available=False, supported=False),
    )
    assert conflict.outcome == unsupported.outcome == "abstention"


def test_incomplete_live_observation_is_partial() -> None:
    decision = route_question(QuestionRequest("What is open now?", "none"))
    status = EvidenceStatus(
        True,
        True,
        False,
        live_observations=(_observation(complete=False),),
    )
    assert resolve_evidence(decision, status).outcome == "partial"


def test_tampered_static_package_is_reconstructed_and_rejected(
    generation: GenerationBundle, package: EvidencePackage
) -> None:
    tampered = replace(package, digest=f"sha256:{'0' * 64}")
    with pytest.raises(RoutingError, match="static evidence package is invalid"):
        resolve_evidence(route_question(_static_request()), _evidence(generation, tampered))


@pytest.mark.parametrize(
    "question_request",
    [
        QuestionRequest(" ", "none"),
        QuestionRequest("question", "none", requested_version="8.1"),
        QuestionRequest("question", "current_state", available_versions=("8.1",)),
        QuestionRequest("question", "required", available_versions=("unstable", "8.1")),
        QuestionRequest("question", "required", available_versions=("8.1", "8.1")),
        QuestionRequest("\ud800", "none"),
    ],
)
def test_malformed_requests_fail_closed(question_request: QuestionRequest) -> None:
    with pytest.raises(RoutingError):
        route_question(question_request)


@pytest.mark.parametrize(
    "observation",
    [
        _observation(observation_id="ev_wrong"),
        _observation(observed_at="2026-02-30T00:00:00Z"),
        _observation(source_url="http://api.github.com/object"),
        _observation(source_url="https://user:pass@api.github.com/object"),
        _observation(payload_digest="sha256:bad"),
    ],
)
def test_malformed_live_observations_fail_closed(observation: LiveObservation) -> None:
    decision = route_question(QuestionRequest("What is open now?", "none"))
    with pytest.raises(RoutingError):
        resolve_evidence(
            decision, EvidenceStatus(True, True, False, live_observations=(observation,))
        )


def test_terminal_and_runtime_tampered_decisions_fail_closed(
    generation: GenerationBundle, package: EvidencePackage
) -> None:
    terminal = route_question(_static_request(requested_version=None))
    with pytest.raises(RoutingError, match="only for a routing decision"):
        resolve_evidence(terminal, _evidence(generation, package))
    valid = route_question(_static_request())
    with pytest.raises(RoutingError, match="exactly one"):
        resolve_evidence(
            replace(valid, routes=("static_semantic", "live_read")),
            _evidence(generation, package),
        )
