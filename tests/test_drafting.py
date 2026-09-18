from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import cast

import pytest

from tests.test_evaluations import _passing_model_runs, _passing_retrieval_results
from tests.test_evidence import _bundle, _package
from valkeyrie.answer_models import (
    AnswerModelCandidate,
    AnswerModelProfile,
    AnswerModelSelection,
    create_answer_model_profile,
    create_candidate_profiles,
    load_answer_model_inventory,
    select_answer_model,
)
from valkeyrie.bedrock_response import (
    STRICT_A04_ABSTENTION_TEXT,
    BedrockTextResponse,
    normalize_bedrock_response,
)
from valkeyrie.drafting import (
    DraftClaim,
    DraftedAbstention,
    DraftedAnswer,
    DraftedClarification,
    DraftingError,
    ModelInput,
    ModelInvocation,
    accept_bedrock_response,
    accept_model_output,
    prepare_model_invocation,
)
from valkeyrie.evaluations import EvaluationSuite, evaluate_candidate, load_evaluation_suite
from valkeyrie.evidence import EvidencePackage, render_citations
from valkeyrie.generation import GenerationBundle
from valkeyrie.prompts import PromptPackage, load_prompt_package
from valkeyrie.retrieval_config import load_retrieval_config

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What does the GET command return in Valkey 9.0.0?"
STARTED = "2026-08-19T03:00:00Z"
COMPLETED = "2026-08-19T03:10:00Z"


@pytest.fixture(scope="module")
def bundle() -> GenerationBundle:
    return _bundle(load_retrieval_config(ROOT / "retrieval-config.yaml"))


@pytest.fixture(scope="module")
def package(bundle: GenerationBundle) -> EvidencePackage:
    return _package(bundle)


@pytest.fixture(scope="module")
def suite() -> EvaluationSuite:
    return load_evaluation_suite(ROOT)


@pytest.fixture(scope="module")
def prompt_package() -> PromptPackage:
    return load_prompt_package(ROOT)


@pytest.fixture(scope="module")
def candidates() -> tuple[AnswerModelCandidate, ...]:
    return load_answer_model_inventory(ROOT / "answer-models.yaml")


@pytest.fixture(scope="module")
def profiles(
    suite: EvaluationSuite,
    prompt_package: PromptPackage,
    bundle: GenerationBundle,
    candidates: tuple[AnswerModelCandidate, ...],
) -> tuple[AnswerModelProfile, ...]:
    return create_candidate_profiles(
        candidates,
        prompt_revision=prompt_package.prompt_revision,
        corpus_generation=bundle.generation_id,
        evaluation_suite_revision=suite.revision,
    )


@pytest.fixture(scope="module")
def reports(
    suite: EvaluationSuite,
    profiles: tuple[AnswerModelProfile, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(_report(suite, profile) for profile in profiles)


@pytest.fixture(scope="module")
def selection(
    suite: EvaluationSuite,
    profiles: tuple[AnswerModelProfile, ...],
    reports: tuple[dict[str, object], ...],
) -> AnswerModelSelection:
    return select_answer_model(suite, profiles, reports)


def _profile(
    suite: EvaluationSuite,
    model_revision: str,
    prompt_revision: str,
    corpus_generation: str,
) -> AnswerModelProfile:
    fable = model_revision == "us.anthropic.claude-fable-5"
    return create_answer_model_profile(
        model_revision,
        maximum_output_tokens=2048 if fable else 1200,
        temperature=None if fable else 0.0,
        top_p=None if fable else 1.0,
        reasoning_effort="low" if fable else None,
        prompt_revision=prompt_revision,
        corpus_generation=corpus_generation,
        evaluation_suite_revision=suite.revision,
    )


def _report(suite: EvaluationSuite, profile: AnswerModelProfile) -> dict[str, object]:
    return evaluate_candidate(
        suite,
        candidate_revision=profile.profile_revision,
        started_at=STARTED,
        completed_at=COMPLETED,
        model_runs=_passing_model_runs(suite),
        retrieval_results=_passing_retrieval_results(suite),
    )


def _answer(*claims: dict[str, object]) -> dict[str, object]:
    return {
        "api_version": "valkeyrie.io/model-output/1",
        "kind": "ModelOutput",
        "outcome": "answer",
        "claims": list(claims),
    }


def _claim(
    claim_id: str,
    evidence_ids: list[str],
    text: str = "GET returns the value stored at a key.",
) -> dict[str, object]:
    return {"claim_id": claim_id, "text": text, "evidence_ids": evidence_ids}


def test_model_input_carries_exactly_prompts_question_and_verified_evidence(
    suite: EvaluationSuite,
    prompt_package: PromptPackage,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    invocation = prepare_model_invocation(
        ROOT, QUESTION, suite, reports, selection, bundle, package
    )
    assert {field.name for field in fields(ModelInput)} == {
        "prompts",
        "question",
        "evidence",
    }
    assert invocation.input == ModelInput(
        prompts=prompt_package.templates,
        question=QUESTION,
        evidence=package,
    )


def test_model_target_and_inference_metadata_stay_outside_the_model_input(
    suite: EvaluationSuite,
    prompt_package: PromptPackage,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    invocation = prepare_model_invocation(
        ROOT, QUESTION, suite, reports, selection, bundle, package
    )
    assert {field.name for field in fields(ModelInvocation)} == {
        "input",
        "profile",
        "prompt_revision",
    }
    assert invocation.profile == selection.profile
    assert invocation.profile.inference.reasoning_effort == (
        "low" if invocation.profile.model_revision == "us.anthropic.claude-fable-5" else None
    )
    assert invocation.prompt_revision == prompt_package.prompt_revision
    input_fields = {field.name for field in fields(ModelInput)}
    metadata_fields = {field.name for field in fields(AnswerModelProfile)}
    assert input_fields.isdisjoint(metadata_fields)
    with pytest.raises(AttributeError):
        cast(object, invocation.input).profile = selection.profile  # type: ignore[attr-defined]


def test_candidate_set_is_the_approved_inventory_not_a_caller_argument(
    candidates: tuple[AnswerModelCandidate, ...],
    profiles: tuple[AnswerModelProfile, ...],
    selection: AnswerModelSelection,
) -> None:
    assert [candidate.model_revision for candidate in candidates] == [
        "us.anthropic.claude-fable-5",
        "amazon.nova-pro-v1:0",
    ]
    assert [profile.model_revision for profile in profiles] == [
        candidate.model_revision for candidate in candidates
    ]
    assert selection.profile in profiles


def test_each_inventory_candidate_requires_exactly_one_live_report(
    suite: EvaluationSuite,
    prompt_package: PromptPackage,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    outside = _report(
        suite,
        _profile(
            suite,
            "amazon.nova-lite-v1:0",
            prompt_package.prompt_revision,
            bundle.generation_id,
        ),
    )
    incomplete_or_excess: tuple[tuple[dict[str, object], ...], ...] = (
        (),
        reports[:1],
        (*reports, outside),
        (reports[0], outside),
        (reports[0], reports[0]),
    )
    for tampered in incomplete_or_excess:
        with pytest.raises(DraftingError, match="qualification is invalid"):
            prepare_model_invocation(ROOT, QUESTION, suite, tampered, selection, bundle, package)


def test_caller_forged_selection_is_rejected(
    suite: EvaluationSuite,
    prompt_package: PromptPackage,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    forged_report = replace(selection, evaluation_report_id=f"eval_{'0' * 64}")
    forged_cost = replace(selection, recorded_cost_usd=selection.recorded_cost_usd + 1.0)
    forged_profile = replace(
        selection,
        profile=_profile(
            suite,
            "amazon.nova-lite-v1:0",
            prompt_package.prompt_revision,
            bundle.generation_id,
        ),
    )
    for forged in (forged_report, forged_cost, forged_profile):
        with pytest.raises(DraftingError, match="selection does not match"):
            prepare_model_invocation(ROOT, QUESTION, suite, reports, forged, bundle, package)


def test_selection_recomputation_requires_content_verified_reports(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    tampered = dict(reports[0])
    tampered["report_id"] = f"eval_{'0' * 64}"
    with pytest.raises(DraftingError, match="qualification is invalid"):
        prepare_model_invocation(
            ROOT, QUESTION, suite, (tampered, reports[1]), selection, bundle, package
        )


def test_reports_must_be_qualified_with_this_exact_prompt_revision(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    stale = _report(
        suite,
        _profile(
            suite,
            "us.anthropic.claude-fable-5",
            f"sha256:{'1' * 64}",
            bundle.generation_id,
        ),
    )
    with pytest.raises(DraftingError, match="qualification is invalid"):
        prepare_model_invocation(
            ROOT, QUESTION, suite, (stale, reports[1]), selection, bundle, package
        )


def test_reports_must_be_qualified_with_this_exact_corpus_generation(
    suite: EvaluationSuite,
    prompt_package: PromptPackage,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    stale = _report(
        suite,
        _profile(
            suite,
            "us.anthropic.claude-fable-5",
            prompt_package.prompt_revision,
            f"sha256:{'9' * 64}",
        ),
    )
    with pytest.raises(DraftingError, match="qualification is invalid"):
        prepare_model_invocation(
            ROOT, QUESTION, suite, (stale, reports[1]), selection, bundle, package
        )


@pytest.mark.parametrize(
    "question",
    ["", "   ", "q" * (8 * 1024 + 1), "bad\x00question", "bad\x1bquestion", "bad\x9fquestion"],
)
def test_user_question_bounds_fail_closed(
    question: str,
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    with pytest.raises(DraftingError, match="user question"):
        prepare_model_invocation(ROOT, question, suite, reports, selection, bundle, package)


def test_unverified_evidence_package_is_rejected(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    tampered = replace(package, digest=f"sha256:{'0' * 64}")
    with pytest.raises(DraftingError, match="evidence package is not verified"):
        prepare_model_invocation(ROOT, QUESTION, suite, reports, selection, bundle, tampered)


def test_answer_uses_per_claim_structure_and_application_renders_citations(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    first, second = (record.evidence_id for record in package.records[:2])
    accepted = accept_model_output(
        _answer(
            _claim("z-claim", [second, first], "GET returns the stored value."),
            _claim("a-claim", [first], "The value is returned unmodified."),
        ),
        bundle,
        package,
    )
    assert isinstance(accepted, DraftedAnswer)
    assert accepted.claims == (
        DraftClaim("z-claim", "GET returns the stored value.", tuple(sorted((first, second)))),
        DraftClaim("a-claim", "The value is returned unmodified.", (first,)),
    )
    assert accepted.citations == render_citations(bundle, package, tuple(sorted({first, second})))
    assert all(
        citation.startswith("[") and "https://github.com/valkey-io/" in citation
        for citation in accepted.citations
    )


def test_bedrock_filter_discards_partial_text_before_json_parsing(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    accepted = accept_bedrock_response(
        BedrockTextResponse('{"api', "content_filtered"),
        bundle,
        package,
    )
    assert accepted == DraftedAbstention("Insufficient validated evidence.")
    assert accept_bedrock_response(
        BedrockTextResponse("", "content_filtered"),
        bundle,
        package,
    ) == DraftedAbstention("Insufficient validated evidence.")
    normalized = normalize_bedrock_response('{"api', "content_filtered")
    empty_normalized = normalize_bedrock_response("", "content_filtered")
    assert empty_normalized == normalized
    assert normalized.response_text == STRICT_A04_ABSTENTION_TEXT
    assert normalized.response_text != '{"api'
    assert normalized.disposition == "safety_abstention"


def test_bedrock_end_turn_preserves_and_accepts_exact_output(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    value = (
        '{"api_version":"valkeyrie.io/model-output/1","kind":"ModelOutput",'
        '"outcome":"abstention","reason":"Insufficient validated evidence."}'
    )
    normalized = normalize_bedrock_response(value, "end_turn")
    assert normalized.response_text == value
    assert normalized.disposition == "unchanged"
    assert accept_bedrock_response(
        BedrockTextResponse(value, "end_turn"),
        bundle,
        package,
    ) == DraftedAbstention("Insufficient validated evidence.")


@pytest.mark.parametrize("stop_reason", ["max_tokens", "tool_use", "stop_sequence", "END_TURN"])
def test_bedrock_unsupported_stop_reasons_fail_closed(
    stop_reason: str,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    with pytest.raises(DraftingError, match="Bedrock response failed closed"):
        accept_bedrock_response(
            BedrockTextResponse('{"api', stop_reason),
            bundle,
            package,
        )


def test_bedrock_backend_type_unknown_fields_and_bounds_fail_closed(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    with pytest.raises(DraftingError, match="unknown or missing field"):
        accept_bedrock_response(
            {"response_text": "{}", "stop_reason": "end_turn", "unknown": True},
            bundle,
            package,
        )
    with pytest.raises(DraftingError, match="response_text_oversized"):
        accept_bedrock_response(
            BedrockTextResponse("x" * (256 * 1024 + 1), "end_turn"),
            bundle,
            package,
        )


def test_global_evidence_ids_answer_shape_is_rejected(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    legacy = {
        "api_version": "valkeyrie.io/model-output/1",
        "kind": "ModelOutput",
        "outcome": "answer",
        "answer": "GET returns the value stored at a key.",
        "evidence_ids": [package.records[0].evidence_id],
    }
    with pytest.raises(DraftingError, match="schema validation failed"):
        accept_model_output(legacy, bundle, package)


def test_exact_fields_and_types_are_enforced(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    evidence_id = package.records[0].evidence_id
    valid = _answer(_claim("a-claim", [evidence_id]))

    unknown_field = dict(valid)
    unknown_field["unreviewed_extension"] = True
    wrong_version = dict(valid)
    wrong_version["api_version"] = "valkeyrie.io/incompatible/2"
    wrong_outcome = dict(valid)
    wrong_outcome["outcome"] = "write"
    wrong_claim_type = _answer({"claim_id": "a-claim", "text": "text", "evidence_ids": evidence_id})
    extra_claim_field = _answer(
        {
            "claim_id": "a-claim",
            "text": "text",
            "evidence_ids": [evidence_id],
            "source_url": "https://example.invalid",
        }
    )
    bad_claim_id = _answer(_claim("Bad-Claim", [evidence_id]))
    duplicate_evidence = _answer(_claim("a-claim", [evidence_id, evidence_id]))
    for invalid in (
        unknown_field,
        wrong_version,
        wrong_outcome,
        wrong_claim_type,
        extra_claim_field,
        bad_claim_id,
        duplicate_evidence,
    ):
        with pytest.raises(DraftingError, match="schema validation failed"):
            accept_model_output(invalid, bundle, package)


@pytest.mark.parametrize(
    "text",
    [
        "See ev_abc123 for details.",
        "Documented at https://example.invalid/page.",
        "Documented at WWW.example.invalid.",
        "See [the documentation](anywhere).",
        f"valkey/src/server.c@{'1' * 40} defines this behavior.",
        "The canonical source confirms this.",
        "This file is the source of truth.",
        "The release is ready.",
        "The build is ready to ship.",
        "Go/no-go: go.",
        "I merged the fix.",
        "We have deployed the change.",
    ],
)
def test_prohibited_model_authored_text_fails_closed(
    text: str,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    value = _answer(_claim("a-claim", [package.records[0].evidence_id], text))
    with pytest.raises(DraftingError, match="contains"):
        accept_model_output(value, bundle, package)


@pytest.mark.parametrize(
    "text",
    [
        "We deployed it successfully.",
        "I merged the fix.",
        "This release can ship now.",
        "It is safe to release.",
        "github.com/valkey-io/valkey",
        "[1]",
        "The docs are authoritative [1].",
        "According to the project maintainers, this is correct.",
    ],
)
def test_reproduced_screen_bypasses_now_fail_closed(
    text: str,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    value = _answer(_claim("a-claim", [package.records[0].evidence_id], text))
    with pytest.raises(DraftingError, match="contains"):
        accept_model_output(value, bundle, package)


def test_prohibitions_apply_to_clarification_and_abstention_text(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    clarification = {
        "api_version": "valkeyrie.io/model-output/1",
        "kind": "ModelOutput",
        "outcome": "clarification",
        "question": "Do you mean the version at https://example.invalid?",
    }
    abstention = {
        "api_version": "valkeyrie.io/model-output/1",
        "kind": "ModelOutput",
        "outcome": "abstention",
        "reason": "I merged the evidence and cannot answer.",
    }
    with pytest.raises(DraftingError, match="clarification question contains a link"):
        accept_model_output(clarification, bundle, package)
    with pytest.raises(DraftingError, match="contains a completed project-state write"):
        accept_model_output(abstention, bundle, package)


def test_strict_clarification_and_abstention_forms_remain(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    clarification = {
        "api_version": "valkeyrie.io/model-output/1",
        "kind": "ModelOutput",
        "outcome": "clarification",
        "question": "Which Valkey release should I use?",
    }
    abstention = {
        "api_version": "valkeyrie.io/model-output/1",
        "kind": "ModelOutput",
        "outcome": "abstention",
        "reason": "The reviewed evidence does not support an answer.",
    }
    assert accept_model_output(clarification, bundle, package) == DraftedClarification(
        "Which Valkey release should I use?"
    )
    assert accept_model_output(abstention, bundle, package) == DraftedAbstention(
        "The reviewed evidence does not support an answer."
    )

    blank = dict(clarification)
    blank["question"] = " "
    with pytest.raises(DraftingError, match="non-blank"):
        accept_model_output(blank, bundle, package)
    oversized = dict(clarification)
    oversized["question"] = "q" * 1025
    with pytest.raises(DraftingError, match="1024-byte bound"):
        accept_model_output(oversized, bundle, package)
    oversized_reason = dict(abstention)
    oversized_reason["reason"] = "r" * 2049
    with pytest.raises(DraftingError, match="2048-byte bound"):
        accept_model_output(oversized_reason, bundle, package)
    missing = {key: value for key, value in clarification.items() if key != "question"}
    with pytest.raises(DraftingError, match="schema validation failed"):
        accept_model_output(missing, bundle, package)


def test_claims_must_reference_known_evidence_only(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    value = _answer(_claim("a-claim", [f"ev_{'f' * 64}"]))
    with pytest.raises(DraftingError, match="claim support is invalid"):
        accept_model_output(value, bundle, package)


def test_duplicate_claim_ids_are_rejected(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    evidence_id = package.records[0].evidence_id
    value = _answer(_claim("a-claim", [evidence_id]), _claim("a-claim", [evidence_id]))
    with pytest.raises(DraftingError, match="duplicate claim ID"):
        accept_model_output(value, bundle, package)


def test_size_count_and_control_bounds_fail_closed(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    evidence_id = package.records[0].evidence_id

    with pytest.raises(DraftingError, match="must be a mapping"):
        accept_model_output([_claim("a-claim", [evidence_id])], bundle, package)
    with pytest.raises(DraftingError, match="not canonically encodable"):
        accept_model_output({"claims": {"a", "b"}}, bundle, package)
    with pytest.raises(DraftingError, match="262144-byte bound"):
        accept_model_output(
            _answer(_claim("a-claim", [evidence_id], "x" * 300_000)), bundle, package
        )
    with pytest.raises(DraftingError, match="4096-byte bound"):
        accept_model_output(_answer(_claim("a-claim", [evidence_id], "x" * 4097)), bundle, package)
    with pytest.raises(DraftingError, match="control character"):
        accept_model_output(
            _answer(_claim("a-claim", [evidence_id], "bad\x00text")), bundle, package
        )
    too_many_claims = _answer(
        *(_claim(f"claim-{index:03d}", [evidence_id]) for index in range(101))
    )
    with pytest.raises(DraftingError, match="schema validation failed"):
        accept_model_output(too_many_claims, bundle, package)
    too_many_ids = _answer(
        {
            "claim_id": "a-claim",
            "text": "text",
            "evidence_ids": [f"ev_{index:064d}" for index in range(21)],
        }
    )
    with pytest.raises(DraftingError, match="schema validation failed"):
        accept_model_output(too_many_ids, bundle, package)


def test_accepted_output_is_immutable(
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    accepted = accept_model_output(
        _answer(_claim("a-claim", [package.records[0].evidence_id])), bundle, package
    )
    assert isinstance(accepted, DraftedAnswer)
    with pytest.raises(AttributeError):
        cast(object, accepted).claims = ()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "text",
    [
        "The pull request was merged on 2026-09-15.",
        "Pull request 3853 is closed and has been merged.",
        "9.2.0-rc1 was published on 2026-09-16 as a prerelease.",
        "The tag was created by the release workflow.",
        "Issue 4413 was closed as completed.",
        # A disclaimer, a modal, a negation, or a recommendation is not a claim of having acted,
        # and the assistant must be free to say each of them.
        "I cannot merge anything; I can only report what the repository shows.",
        "I could not find when it was merged.",
        "We recommend you merge it after review.",
        "We did not merge it.",
    ],
)
def test_reporting_a_fact_about_project_state_is_not_claiming_to_have_written_it(
    text: str,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    """The guard stops the assistant claiming it acted; it must not stop it reporting.

    "I merged it" is a fabricated action and is refused. "It was merged on the 15th" is a fact from
    a live observation, and reporting it is the live route's purpose. The earlier pattern forbade
    every passive form, so correct answers about merged pull requests and published releases were
    rejected and the asker told no reliable answer existed.
    """
    value = _answer(_claim("a-claim", [package.records[0].evidence_id], text))
    accepted = accept_model_output(value, bundle, package)
    assert accepted is not None


@pytest.mark.parametrize(
    "text",
    [
        "I merged it.",
        "We have deployed the fix.",
        "I just released 9.2.",
        "We already tagged it.",
        "I have completed the merge.",
        "we finished the deployment",
        # Review reproduced these slipping past a fixed list of intervening words.
        "I've merged it.",
        "I successfully deployed it.",
        "We did publish the release.",
        "I'd already pushed the fix.",
    ],
)
def test_first_person_action_claims_remain_refused(
    text: str,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    value = _answer(_claim("a-claim", [package.records[0].evidence_id], text))
    with pytest.raises(DraftingError, match="completed project-state write"):
        accept_model_output(value, bundle, package)
