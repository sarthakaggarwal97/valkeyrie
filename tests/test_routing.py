from __future__ import annotations

import pytest

from tests.test_evidence import _bundle, _package
from valkeyrie.evidence import EvidencePackage
from valkeyrie.generation import GenerationBundle
from valkeyrie.retrieval_config import load_retrieval_config
from valkeyrie.routing import (
    QuestionRequest,
    RouteDecision,
    RoutingError,
    route_question,
)

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


@pytest.mark.parametrize(
    "question",
    [
        "Is pull request 123 currently open?",
        "What is the latest Valkey release?",
        "What is the status of pull request 123?",
        "Which upcoming Valkey events are announced?",
    ],
)
def test_current_latest_and_status_language_routes_live(question: str) -> None:
    decision = route_question(QuestionRequest(question, "none"))
    assert decision.outcome == "route"
    assert decision.routes == ("live_read",)
    assert decision.version_scope is None


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
