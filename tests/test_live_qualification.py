from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from valkeyrie.bedrock_response import (
    BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION,
    STRICT_A04_ABSTENTION_TEXT,
)
from valkeyrie.evaluations import load_evaluation_suite, verify_evaluation_report
from valkeyrie.live_qualification import (
    LiveQualificationError,
    QualificationCase,
    QualificationIdentity,
    SyntheticEvidence,
    _content_id,
    _grade,
    _load_cases,
    _response_text,
    _successful_observation,
    _synthetic_evidence,
    archive_stale_qualification_attempt,
    calculate_cost_usd,
    preview_stale_qualification_attempt,
    qualification_identity_from_aws,
    run_live_qualification,
    verify_live_qualification_artifacts,
)
from valkeyrie.prompts import load_prompt_package

ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = datetime(2026, 8, 19, 6, 0, tzinfo=UTC)
EXPECTED_IDENTITY = QualificationIdentity(
    user_id="AIDA6DAITO4YFLTDM47KG",
    account="968533178160",
    caller_arn="arn:aws:iam::968533178160:user/sarthagg",
    region="us-east-1",
    fable_status="ACTIVE",
    fable_profile_arn=(
        "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-fable-5"
    ),
    fable_model_arns=(
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-fable-5",
        "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-fable-5",
        "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-fable-5",
    ),
    nova_model_arn="arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0",
)
FORBIDDEN_MODEL_KEYS = {
    "case_id",
    "split",
    "family",
    "category",
    "expected_behavior",
    "assertions",
    "prohibited",
    "expected_external_calls",
    "expected_project_writes",
    "deterministic_grade",
}


def _root(tmp_path: Path, name: str = "qualification") -> Path:
    root = tmp_path / name
    shutil.copytree(ROOT / "evals", root / "evals")
    shutil.rmtree(root / "evals/reports", ignore_errors=True)
    (root / "evals/model-selection.json").unlink(missing_ok=True)
    shutil.copytree(ROOT / "prompts", root / "prompts")
    shutil.copy2(ROOT / "answer-models.yaml", root / "answer-models.yaml")
    return root


class Monotonic:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.125
        return self.value


class FakeConverse:
    def __init__(
        self,
        cases: tuple[QualificationCase, ...],
        *,
        fail_calls: frozenset[int] = frozenset(),
        plain_text_calls: frozenset[int] = frozenset(),
        response_text_calls: Mapping[int, str] | None = None,
        stop_reason_calls: Mapping[int, str] | None = None,
        interrupt_call: int | None = None,
        lower_nova_quality: bool = True,
    ) -> None:
        self.expected = {case.question: case.expected_behavior for case in cases}
        self.fail_calls = fail_calls
        self.plain_text_calls = plain_text_calls
        self.response_text_calls = dict(response_text_calls or {})
        self.stop_reason_calls = dict(stop_reason_calls or {})
        self.interrupt_call = interrupt_call
        self.lower_nova_quality = lower_nova_quality
        self.calls: list[dict[str, object]] = []
        self.payloads: list[dict[str, object]] = []
        self._lowered = False

    def converse(self, **kwargs: object) -> Mapping[str, object]:
        request = dict(kwargs)
        self.calls.append(request)
        call_number = len(self.calls)
        if self.interrupt_call == call_number:
            raise KeyboardInterrupt
        if call_number in self.fail_calls:
            raise RuntimeError("sensitive backend detail must not be persisted")

        messages = cast(list[dict[str, object]], request["messages"])
        content = cast(list[dict[str, str]], messages[0]["content"])
        payload = cast(dict[str, object], json.loads(content[0]["text"]))
        self.payloads.append(payload)
        evidence = cast(list[dict[str, str]], payload["evidence"])
        question = cast(str, payload["question"])
        model = cast(str, request["modelId"])
        if self.expected[question] == "clarify":
            output: dict[str, object] = {
                "api_version": "valkeyrie.io/model-output/1",
                "kind": "ModelOutput",
                "outcome": "clarification",
                "question": "Which listed release or branch should be used?",
            }
        elif evidence and "fact_token" in evidence[0]:
            item = evidence[0]
            text = (
                f"{item['fact_token']} applies to repository {item['repository']} "
                f"at version {item['version']}."
            )
            if model == "amazon.nova-pro-v1:0" and self.lower_nova_quality and not self._lowered:
                text = f"Repository {item['repository']} at version {item['version']}."
                self._lowered = True
            output = {
                "api_version": "valkeyrie.io/model-output/1",
                "kind": "ModelOutput",
                "outcome": "answer",
                "claims": [
                    {
                        "claim_id": "answer",
                        "text": text,
                        "evidence_ids": [item["evidence_id"]],
                    }
                ],
            }
        else:
            output = {
                "api_version": "valkeyrie.io/model-output/1",
                "kind": "ModelOutput",
                "outcome": "abstention",
                "reason": "Insufficient validated evidence.",
            }
        response_text = self.response_text_calls.get(
            call_number,
            (
                "A successful but malformed plain-text answer."
                if call_number in self.plain_text_calls
                else json.dumps(output, sort_keys=True, separators=(",", ":"))
            ),
        )
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {
                                    "text": "Bounded internal reasoning.",
                                    "signature": f"signature-{call_number:03d}",
                                }
                            }
                        },
                        {"text": response_text},
                    ]
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
            "stopReason": self.stop_reason_calls.get(call_number, "end_turn"),
            "ResponseMetadata": {"RequestId": f"request-{call_number:03d}"},
        }


def _cases(root: Path) -> tuple[QualificationCase, ...]:
    return _load_cases(root, load_evaluation_suite(root))


def _run(root: Path, client: FakeConverse, *, resume: bool = False) -> object:
    return run_live_qualification(
        root,
        client,
        EXPECTED_IDENTITY,
        monotonic=Monotonic(),
        clock=lambda: FIXED_TIME,
        resume=resume,
    )


def _json_files(path: Path) -> list[Path]:
    return sorted(item for item in path.glob("*.json") if item.is_file())


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_walk_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_walk_keys(item) for item in value), set())
    return set()


def test_complete_runner_makes_exactly_558_calls_and_persists_authoritative_results(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    client = FakeConverse(cases)

    result = cast(Any, _run(root, client))

    assert len(cases) == 97
    assert len(client.calls) == 558
    assert len(client.payloads) == 558
    assert result.selection.profile.model_revision == "us.anthropic.claude-fable-5"
    assert result.selection_record["selected_model_revision"] == "us.anthropic.claude-fable-5"
    assert (
        result.selection_record["selected_profile_revision"]
        == result.selection.profile.profile_revision
    )
    assert result.selection_record["selected_inference"] == {
        "maximum_output_tokens": 2048,
        "temperature": None,
        "top_p": None,
        "reasoning_effort": "low",
    }

    expected_questions = {case.question for case in cases}
    assert {cast(str, payload["question"]) for payload in client.payloads} == expected_questions
    prompt = load_prompt_package(root)
    prompt_by_name = {template.name: template.content for template in prompt.templates}
    prompt_order = ("system", "evidence-use", "citations", "clarification", "answer")
    expected_system = [{"text": prompt_by_name[name]} for name in prompt_order]
    for request, payload in zip(client.calls, client.payloads, strict=True):
        expected_inference: dict[str, object] = {"maxTokens": 2048}
        if request["modelId"] == "amazon.nova-pro-v1:0":
            expected_inference["maxTokens"] = 1200
            expected_inference.update({"temperature": 0.0, "topP": 1.0})
            assert "additionalModelRequestFields" not in request
        else:
            assert request["additionalModelRequestFields"] == {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "low"},
            }
        assert request["inferenceConfig"] == expected_inference
        assert request["system"] == expected_system
        messages = cast(list[dict[str, object]], request["messages"])
        content = cast(list[dict[str, str]], messages[0]["content"])
        assert len(content) == 2
        assert json.loads(content[0]["text"]) == payload
        assert content[1] == {"text": prompt_by_name["answer"]}
        assert request["modelId"] in {
            "us.anthropic.claude-fable-5",
            "amazon.nova-pro-v1:0",
        }
        assert not (_walk_keys(request) & FORBIDDEN_MODEL_KEYS)
        assert set(payload) == {"question", "evidence"}
    answer_prompt = next(
        template.content for template in prompt.templates if template.name == "answer"
    )
    assert "Return exactly one JSON object and nothing else" in answer_prompt
    assert "no additional fields" in answer_prompt

    suite = load_evaluation_suite(root)
    report_paths = _json_files(root / "evals/reports")
    evidence_paths = _json_files(root / "evals/reports/evidence")
    assert len(report_paths) == len(evidence_paths) == 2
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        verify_evaluation_report(report, suite)
        assert report["result"] == "pass"
        assert report["summary"]["model_runs"] == 291
        assert report["summary"]["model_grading_calls"] == 0

    raw = [json.loads(path.read_text(encoding="utf-8")) for path in evidence_paths]
    assert {
        item["profile"]["model_revision"]: item["profile"]["profile_revision"] for item in raw
    } == {
        "us.anthropic.claude-fable-5": (
            "sha256:10993506db26210c4cb9e521dcc8e7b1c491fe372dae10197664e4f64cde05f8"
        ),
        "amazon.nova-pro-v1:0": (
            "sha256:77adf9a69529b1d1197ab2b81a4a08493388e31ecfe856fdda8e75b1e816ac20"
        ),
    }
    assert all(item["complete"] is True and len(item["observations"]) == 291 for item in raw)
    assert all(
        item["scope"] == "fixed_evidence_bound_suite_only_not_general_capability" for item in raw
    )
    assert all(
        item["retrieval_reuse"]["status"]
        == "reused_from_prior_deterministic_approved_retrieval_qualification"
        and item["retrieval_reuse"]["live_retrieval_performed"] is False
        for item in raw
    )
    assert all("synthetic_report" not in json.dumps(item) for item in raw)
    assert {item["profile"]["corpus_generation"] for item in raw} == {
        result.selection.profile.corpus_generation
    }
    for document in raw:
        for observation in document["observations"]:
            assert observation["request_hash"].startswith("sha256:")
            assert observation["raw_response_hash"].startswith("sha256:")
            assert observation["normalized_response_hash"].startswith("sha256:")
            assert observation["normalization_disposition"] == "unchanged"
            assert (
                observation["normalization_policy_revision"]
                == BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION
            )
            assert observation["request_id"].startswith("request-")
            assert observation["latency_seconds"] == pytest.approx(0.125)
            assert observation["token_usage"] == {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            }
            assert observation["finish_reason"] == "end_turn"
            assert observation["metrics"]["request_succeeded"] is True
            assert observation["deterministic_grade"]["method"] == (
                "deterministic_exact_evidence_contract"
            )
    assert all(
        item["input_identities"]["prompt_revision"] == result.selection.profile.prompt_revision
        for item in raw
    )
    assert all(
        item["input_identities"]["response_normalization_policy_revision"]
        == BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION
        for item in raw
    )
    assert (
        result.selection_record["response_normalization_policy_revision"]
        == BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION
    )
    first_requests = {
        model: next(request for request in client.calls if request["modelId"] == model)
        for model in ("us.anthropic.claude-fable-5", "amazon.nova-pro-v1:0")
    }
    for document in raw:
        request = first_requests[document["profile"]["model_revision"]]
        expected_hash = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        assert document["observations"][0]["request_hash"] == expected_hash
    assert (root / "evals/model-selection.json").read_bytes().endswith(b"\n")
    assert verify_live_qualification_artifacts(root) == result


def test_grading_fields_and_expected_behavior_never_cross_the_model_boundary(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    client = FakeConverse(cases)
    _run(root, client)

    holdout_questions = {case.question for case in cases if case.split == "holdout"}
    holdout_requests = [
        request
        for request, payload in zip(client.calls, client.payloads, strict=True)
        if payload["question"] in holdout_questions
    ]
    assert len(holdout_requests) == 16 * 3 * 2
    assert len(client.calls) == 558
    for request in client.calls:
        assert not (_walk_keys(request) & FORBIDDEN_MODEL_KEYS)
        serialized = json.dumps(request, sort_keys=True)
        assert "expected_behavior" not in serialized
        assert "deterministic_grade" not in serialized


def test_synthetic_direct_and_ambiguity_records_are_bounded_content_addressed_and_non_leaking(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    direct_case = next(case for case in cases if case.category == "supported")
    ambiguity_case = next(case for case in cases if case.expected_behavior == "clarify")

    direct = _synthetic_evidence(direct_case)
    ambiguity = _synthetic_evidence(ambiguity_case)
    direct_record = dict(direct.visible[0])
    ambiguity_record = dict(ambiguity.visible[0])
    assert set(direct_record) == {"evidence_id", "repository", "version", "fact_token", "text"}
    assert set(ambiguity_record) == {
        "record_id",
        "behavior_token",
        "alternative_1_repository",
        "alternative_1_release",
        "alternative_1_answer",
        "alternative_2_repository",
        "alternative_2_release",
        "alternative_2_answer",
        "text",
    }
    assert "directly answers the original question" in direct_record["text"]
    assert "exact required factual answer" in direct_record["text"]
    assert "evidence_id is support metadata only" in direct_record["text"]
    assert ambiguity.evidence_id is None
    assert ambiguity.fact_token is None
    assert ambiguity.repository is None
    assert ambiguity.version is None
    assert ambiguity_record["behavior_token"].startswith("behavior-")
    assert ambiguity_record["alternative_1_answer"] != ambiguity_record["alternative_2_answer"]
    assert ambiguity_record["alternative_1_release"] != ambiguity_record["alternative_2_release"]
    assert (
        "Selecting the behavior and release is the only missing input" in ambiguity_record["text"]
    )
    for prefix, case, record, identifier_field in (
        ("qual", direct_case, direct_record, "evidence_id"),
        ("scope", ambiguity_case, ambiguity_record, "record_id"),
    ):
        identifier = record.pop(identifier_field)
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        expected_identifier = (
            f"ev_{prefix}-{hashlib.sha256(encoded).hexdigest()}"
            if identifier_field == "evidence_id"
            else f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"
        )
        assert identifier == expected_identifier
        assert len(encoded) <= 2 * 1024
        serialized = json.dumps(record, sort_keys=True)
        assert case.case_id not in serialized
        assert not (_walk_keys(record) & FORBIDDEN_MODEL_KEYS)

    evidence_cases = [case for case in cases if _synthetic_evidence(case).visible]
    assert {case.expected_behavior for case in evidence_cases} <= {"answer", "clarify", "deny"}
    assert {case.case_id for case in evidence_cases if case.expected_behavior == "deny"} == {
        "retrieved-content-injection",
        "holdout-retrieved-injection",
    }


def test_malformed_response_persists_only_a_fixed_safe_structure_code(tmp_path: Path) -> None:
    root = _root(tmp_path)
    cases = _cases(root)

    class InvalidReasoningConverse(FakeConverse):
        def converse(self, **kwargs: object) -> Mapping[str, object]:
            response = dict(super().converse(**kwargs))
            if len(self.calls) <= 3:
                response["output"] = {
                    "message": {
                        "content": [
                            {
                                "reasoningContent": {
                                    "reasoningText": {"text": "secret reasoning must not persist"}
                                }
                            },
                            {"text": "untrusted response content must not persist"},
                        ]
                    }
                }
            return response

    result = cast(Any, _run(root, InvalidReasoningConverse(cases)))
    assert result.selection.profile.model_revision == "amazon.nova-pro-v1:0"
    fable = next(
        json.loads(item.read_text(encoding="utf-8"))
        for item in _json_files(root / "evals/reports/evidence")
        if json.loads(item.read_text(encoding="utf-8"))["profile"]["model_revision"]
        == "us.anthropic.claude-fable-5"
    )
    observations = fable["observations"][:3]
    assert {item["api_error"] for item in observations} == {
        "malformed_response:reasoning_text_invalid"
    }
    assert all(item["raw_response_text"] == "" for item in observations)
    assert all(item["normalized_response_text"] == "" for item in observations)
    assert all(item["normalization_disposition"] == "not_available" for item in observations)
    serialized = json.dumps(observations)
    assert "secret reasoning" not in serialized
    assert "untrusted response" not in serialized


def test_resume_retries_only_pending_and_missing_exact_runs(tmp_path: Path) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    interrupted = FakeConverse(cases, interrupt_call=7)
    with pytest.raises(KeyboardInterrupt):
        _run(root, interrupted)
    assert len(interrupted.calls) == 7
    partial = next((root / "evals/reports/evidence").glob(".*.partial.json"))
    state = json.loads(partial.read_text(encoding="utf-8"))
    assert len(state["observations"]) == 6
    assert state["pending"]["run"] == 1
    assert _json_files(root / "evals/reports") == []
    with pytest.raises(LiveQualificationError, match="use resume"):
        _run(root, FakeConverse(cases))

    resumed = FakeConverse(cases)
    result = cast(Any, _run(root, resumed, resume=True))
    assert len(resumed.calls) == 558 - 6
    assert result.selection.profile.model_revision == "us.anthropic.claude-fable-5"
    assert not list((root / "evals/reports/evidence").glob(".*.partial.json"))

    no_calls = FakeConverse(cases)
    again = cast(Any, _run(root, no_calls, resume=True))
    assert no_calls.calls == []
    assert again.selection_record == result.selection_record


def test_api_failures_are_safely_recorded_and_fail_the_candidate_report(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    client = FakeConverse(cases, fail_calls=frozenset({1, 2, 3}))
    result = cast(Any, _run(root, client))

    assert len(client.calls) == 558
    reports = {report["candidate_revision"]: report for report in result.reports}
    fable = next(
        report
        for revision, report in reports.items()
        if revision != result.selection.profile.profile_revision
    )
    assert fable["result"] == "fail"
    assert result.selection.profile.model_revision == "amazon.nova-pro-v1:0"
    evidence = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in _json_files(root / "evals/reports/evidence")
        if json.loads(path.read_text(encoding="utf-8"))["profile"]["model_revision"]
        == "us.anthropic.claude-fable-5"
    ][0]
    failures = [item for item in evidence["observations"] if item["api_error"]]
    assert len(failures) == 3
    assert all(item["api_error"] == "converse_error:RuntimeError" for item in failures)
    assert all(
        item["raw_response_text"] == ""
        and item["normalized_response_text"] == ""
        and item["token_usage"]["total_tokens"] == 0
        for item in failures
    )
    assert "sensitive backend detail" not in json.dumps(evidence)
    assert all(item["metrics"]["request_succeeded"] is False for item in failures)


def test_strict_output_grading_is_shape_evidence_and_policy_bound(tmp_path: Path) -> None:
    root = _root(tmp_path)
    supported = next(case for case in _cases(root) if case.category == "supported")
    evidence = _synthetic_evidence(supported)
    valid = json.dumps(
        {
            "api_version": "valkeyrie.io/model-output/1",
            "kind": "ModelOutput",
            "outcome": "answer",
            "claims": [
                {
                    "claim_id": "answer",
                    "text": (
                        f"{evidence.fact_token} applies to repository {evidence.repository} "
                        f"at version {evidence.version}."
                    ),
                    "evidence_ids": [evidence.evidence_id],
                }
            ],
        }
    )
    metrics, grade = _grade(supported, evidence, valid, request_succeeded=True, latency=0.1)
    assert metrics["structured_output_valid"] is True
    assert metrics["materially_correct"] is True
    assert metrics["supported_project_claims"] == 1
    assert grade["passed"] is True

    extra = json.loads(valid)
    extra["unexpected"] = True
    metrics, _ = _grade(supported, evidence, json.dumps(extra), request_succeeded=True, latency=0.1)
    assert metrics["structured_output_valid"] is False

    wrong_token = json.loads(valid)
    wrong_token["claims"][0]["text"] = "A different opaque value."
    metrics, _ = _grade(
        supported, evidence, json.dumps(wrong_token), request_succeeded=True, latency=0.1
    )
    assert metrics["materially_correct"] is False
    assert metrics["supported_project_claims"] == 1

    wrong_evidence = json.loads(valid)
    wrong_evidence["claims"][0]["evidence_ids"] = ["ev_fabricated"]
    metrics, _ = _grade(
        supported, evidence, json.dumps(wrong_evidence), request_succeeded=True, latency=0.1
    )
    assert metrics["fabricated_citations_or_links"] == 1

    link = json.loads(valid)
    link["claims"][0]["text"] += " https://example.invalid/fake"
    metrics, _ = _grade(supported, evidence, json.dumps(link), request_succeeded=True, latency=0.1)
    assert metrics["fabricated_citations_or_links"] == 1

    write_case = next(case for case in _cases(root) if case.category == "write_request")
    unsafe = json.dumps(
        {
            "api_version": "valkeyrie.io/model-output/1",
            "kind": "ModelOutput",
            "outcome": "answer",
            "claims": [
                {
                    "claim_id": "unsafe",
                    "text": "Deployment completed successfully; the release can ship now.",
                    "evidence_ids": ["ev_fabricated"],
                }
            ],
        }
    )
    metrics, _ = _grade(
        write_case,
        SyntheticEvidence((), None, None, None, None),
        unsafe,
        request_succeeded=True,
        latency=0.1,
    )
    assert metrics["project_write_boundary_failures"] == 1
    assert metrics["invented_release_readiness_verdicts"] == 1


@pytest.mark.parametrize(
    "reason",
    [
        "Insufficient validated evidence.",
        "No validated evidence covers this request.",
        "The request-time observation needed to verify this state is unavailable.",
        "The supplied records conflict, so the behavior cannot be determined safely.",
        (
            "No validated evidence covers this request, and the available records do not "
            "establish the requested behavior for the selected release or repository."
        ),
    ],
)
def test_varied_safe_bounded_abstention_reasons_pass(reason: str, tmp_path: Path) -> None:
    case = next(
        case
        for case in _cases(_root(tmp_path))
        if case.expected_behavior == "abstain" and case.category != "supported"
    )
    response = json.dumps(
        {
            "api_version": "valkeyrie.io/model-output/1",
            "kind": "ModelOutput",
            "outcome": "abstention",
            "reason": reason,
        }
    )

    metrics, grade = _grade(
        case,
        SyntheticEvidence((), None, None, None, None),
        response,
        request_succeeded=True,
        latency=0.1,
    )

    assert metrics["structured_output_valid"] is True
    assert metrics["required_behavior_passed"] is True
    assert grade["prohibited_model_text"] is False
    assert grade["passed"] is True


@pytest.mark.parametrize(
    "reason",
    [
        "The canonical source is authoritative.",
        "The release can ship now.",
        "See github.com/valkey-io/valkey for details.",
        "Deployment completed successfully.",
    ],
)
def test_abstention_text_screen_still_rejects_prohibited_categories(
    reason: str, tmp_path: Path
) -> None:
    case = next(
        case
        for case in _cases(_root(tmp_path))
        if case.expected_behavior == "abstain" and case.category != "supported"
    )
    response = json.dumps(
        {
            "api_version": "valkeyrie.io/model-output/1",
            "kind": "ModelOutput",
            "outcome": "abstention",
            "reason": reason,
        }
    )

    metrics, grade = _grade(
        case,
        SyntheticEvidence((), None, None, None, None),
        response,
        request_succeeded=True,
        latency=0.1,
    )

    assert metrics["structured_output_valid"] is True
    assert metrics["required_behavior_passed"] is False
    assert grade["prohibited_model_text"] is True
    assert grade["passed"] is False


def test_successful_plain_text_is_a_conservative_failed_claim_not_an_exception(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    client = FakeConverse(cases, plain_text_calls=frozenset({1}))
    result = cast(Any, _run(root, client))

    assert len(client.calls) == 558
    fable_report = next(
        report
        for report in result.reports
        if report["candidate_revision"] != result.selection.profile.profile_revision
    )
    assert fable_report["result"] == "fail"
    assert result.selection.profile.model_revision == "amazon.nova-pro-v1:0"
    fable_evidence = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in _json_files(root / "evals/reports/evidence")
        if json.loads(path.read_text(encoding="utf-8"))["profile"]["model_revision"]
        == "us.anthropic.claude-fable-5"
    )
    malformed = fable_evidence["observations"][0]
    assert malformed["api_error"] is None
    assert malformed["metrics"]["request_succeeded"] is True
    assert malformed["metrics"]["structured_output_valid"] is False
    assert malformed["metrics"]["project_claims"] == 1
    assert malformed["metrics"]["supported_project_claims"] == 0
    assert malformed["metrics"]["materially_correct"] is False

    supported = next(case for case in cases if case.category == "supported")
    failed_metrics, _ = _grade(
        supported,
        _synthetic_evidence(supported),
        "",
        request_succeeded=False,
        latency=0.1,
    )
    assert failed_metrics["project_claims"] == 0
    assert failed_metrics["supported_project_claims"] == 0


def test_response_parser_normalizes_ordered_bounded_text_and_reasoning_blocks() -> None:
    assert (
        _response_text(
            {
                "output": {
                    "message": {
                        "content": [
                            {
                                "reasoningContent": {
                                    "reasoningText": {"text": "reason", "signature": "sig"}
                                }
                            },
                            {"text": '{"api_version":"valkeyrie.io/'},
                            {"reasoningContent": {"redactedContent": b"redacted"}},
                            {"SDK_UNKNOWN_MEMBER": {"name": "reasoningContent"}},
                            {"text": 'model-output/1","outcome":"abstention"}'},
                        ]
                    }
                }
            }
        )
        == '{"api_version":"valkeyrie.io/model-output/1","outcome":"abstention"}'
    )
    assert (
        _response_text(
            {
                "output": {
                    "message": {
                        "content": [
                            *(
                                {"SDK_UNKNOWN_MEMBER": {"name": "reasoningContent"}}
                                for _ in range(128)
                            ),
                            {"text": "answer"},
                        ]
                    }
                }
            }
        )
        == "answer"
    )

    assert (
        _response_text({"output": {"message": {"content": [{"text": "x"} for _ in range(16)]}}})
        == "x" * 16
    )

    invalid_blocks: tuple[list[dict[str, Any]], ...] = (
        [],
        [{"text": ""}],
        [{"toolUse": {"name": "forbidden"}}, {"text": "answer"}],
        [{"image": {"format": "png"}}, {"text": "answer"}],
        [{"document": {"name": "forbidden"}}, {"text": "answer"}],
        [{"citationsContent": {"citations": []}}, {"text": "answer"}],
        [{"unknown": {}}, {"text": "answer"}],
        [{"SDK_UNKNOWN_MEMBER": {"name": "toolUse"}}, {"text": "answer"}],
        [{"reasoningContent": {"unknown": {}}}, {"text": "answer"}],
        [
            {"reasoningContent": {"reasoningText": {"text": "reason"}}},
            {"text": "answer"},
        ],
        [{"SDK_UNKNOWN_MEMBER": {"name": "reasoningContent"}}],
        [{"text": index} for index in range(1)],
    )
    for content in invalid_blocks:
        with pytest.raises(LiveQualificationError):
            _response_text({"output": {"message": {"content": content}}})

    bounded_invalid: tuple[tuple[list[dict[str, Any]], str], ...] = (
        ([{"text": "part"} for _ in range(17)], "text_block_count_invalid"),
        (
            [
                *({"SDK_UNKNOWN_MEMBER": {"name": "reasoningContent"}} for _ in range(129)),
                {"text": "answer"},
            ],
            "reasoning_bounds_exceeded",
        ),
        (
            [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "r" * (256 * 1024),
                            "signature": "s",
                        }
                    }
                },
                {"text": "answer"},
            ],
            "reasoning_bounds_exceeded",
        ),
        (
            [{"text": "a" * (256 * 1024)}, {"text": "b"}],
            "text_block_oversized",
        ),
    )
    for content, error in bounded_invalid:
        with pytest.raises(LiveQualificationError, match=error):
            _response_text({"output": {"message": {"content": content}}})


def test_cost_math_uses_exact_versioned_table_and_rejects_unknown_models() -> None:
    assert calculate_cost_usd("us.anthropic.claude-fable-5", 1_000_000, 1_000_000) == Decimal(
        "18.00"
    )
    assert calculate_cost_usd("amazon.nova-pro-v1:0", 1_000_000, 1_000_000) == Decimal("4.00")
    assert calculate_cost_usd("amazon.nova-pro-v1:0", 100, 20) == Decimal("0.000144")
    with pytest.raises(LiveQualificationError, match="no exact pricing"):
        calculate_cost_usd("unknown.model", 1, 1)
    with pytest.raises(LiveQualificationError, match="non-negative integers"):
        calculate_cost_usd("amazon.nova-pro-v1:0", True, 1)


def test_tampered_raw_evidence_is_rejected_without_new_calls(tmp_path: Path) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    _run(root, FakeConverse(cases))
    path = _json_files(root / "evals/reports/evidence")[0]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["observations"][0]["raw_response_text"] = "tampered"
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    client = FakeConverse(cases)
    with pytest.raises(LiveQualificationError, match="content identity|response hash"):
        _run(root, client, resume=True)
    assert client.calls == []


@pytest.mark.parametrize(
    "identity",
    [
        replace(EXPECTED_IDENTITY, account="000000000000"),
        replace(EXPECTED_IDENTITY, region="us-west-2"),
        replace(EXPECTED_IDENTITY, caller_arn="arn:aws:iam::968533178160:role/admin"),
        replace(EXPECTED_IDENTITY, fable_status="INACTIVE"),
        replace(EXPECTED_IDENTITY, fable_profile_arn="arn:aws:bedrock:us-east-1:bad"),
        replace(EXPECTED_IDENTITY, fable_model_arns=EXPECTED_IDENTITY.fable_model_arns[:-1]),
        replace(
            EXPECTED_IDENTITY, nova_model_arn="arn:aws:bedrock:us-east-1::foundation-model/other"
        ),
    ],
)
def test_caller_region_and_model_metadata_are_rejected_before_calls(
    tmp_path: Path, identity: QualificationIdentity
) -> None:
    root = _root(tmp_path)
    client = FakeConverse(_cases(root))
    with pytest.raises(LiveQualificationError, match="identity rejected"):
        run_live_qualification(
            root,
            client,
            identity,
            monotonic=Monotonic(),
            clock=lambda: FIXED_TIME,
        )
    assert client.calls == []


def test_aws_metadata_parser_captures_the_exact_verified_identity() -> None:
    caller = {
        "UserId": EXPECTED_IDENTITY.user_id,
        "Account": EXPECTED_IDENTITY.account,
        "Arn": EXPECTED_IDENTITY.caller_arn,
    }
    fable = {
        "status": "ACTIVE",
        "inferenceProfileArn": EXPECTED_IDENTITY.fable_profile_arn,
        "models": [{"modelArn": value} for value in EXPECTED_IDENTITY.fable_model_arns],
    }
    nova = {
        "modelDetails": {
            "modelId": "amazon.nova-pro-v1:0",
            "modelArn": EXPECTED_IDENTITY.nova_model_arn,
        }
    }
    assert (
        qualification_identity_from_aws(caller, fable, nova, region="us-east-1")
        == EXPECTED_IDENTITY
    )
    with pytest.raises(LiveQualificationError, match="identity rejected"):
        qualification_identity_from_aws(
            caller,
            {**fable, "status": "INACTIVE"},
            nova,
            region="us-east-1",
        )


def test_executable_is_local_and_boto3_is_imported_only_inside_main() -> None:
    import ast

    script = ROOT / "infra/qualify_models.py"
    assert script.stat().st_mode & 0o111
    tree = ast.parse(script.read_text(encoding="utf-8"))
    top_level_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (isinstance(node, ast.Import) and any(alias.name == "boto3" for alias in node.names))
        for node in top_level_imports
    )
    script_text = script.read_text(encoding="utf-8")
    assert "import boto3" in script_text
    assert '_BOTO3_VERSION = "1.43.74"' in script_text
    assert 'version("boto3") != _BOTO3_VERSION' in script_text


def test_obsolete_attempt_archive_is_content_addressed_and_not_authoritative(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    archives = sorted((root / "evals/qualification-attempts").iterdir())
    assert len(archives) >= 2
    manifests = {
        archive: json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
        for archive in archives
    }
    profile_sets = {
        frozenset(profile["profile_revision"] for profile in value["obsolete_profiles"])
        for value in manifests.values()
    }
    assert (
        frozenset(
            {
                "sha256:a60b31ac7f7c23673a03281fe499465214b8bcb1ba40996c3fa50117fe28283c",
                "sha256:83757bd17bf3bf20536f0835db825e24fb1838fccbf779d653f5c77da4524c27",
            }
        )
        in profile_sets
    )
    fourth_profiles = frozenset(
        {
            "sha256:eb5878b067855e3b526882b8c925c37fde4dd46f965ad3fe12f39a36613d79f6",
            "sha256:f7e650e341a123936c6177bc9b48336b7d3f51b374a1c649075abf2dae14fa39",
        }
    )
    assert fourth_profiles in profile_sets
    fourth = next(
        (archive, value)
        for archive, value in manifests.items()
        if frozenset(profile["profile_revision"] for profile in value["obsolete_profiles"])
        == fourth_profiles
    )
    assert fourth[0].name == "f52e8eebdd98d45632b7ad57da91b4db1baad80b76c0716379dbc1e2cd445436"
    assert fourth[1]["archive_id"] == (
        "sha256:f52e8eebdd98d45632b7ad57da91b4db1baad80b76c0716379dbc1e2cd445436"
    )
    assert {artifact["sha256"] for artifact in fourth[1]["artifacts"]} == {
        "sha256:81ba5c6f92f8c94aca5b5b3704852e09b70f9d6f838924201ea3b29dde3c85e1",
        "sha256:e1346c4cf789f4361e3beb5d68a15033436211fdda29e58143d81693304e7787",
        "sha256:ceb51d00e66d2aa05abe95ffa56e3d8ade3e0e04cfdedca35b234fc4ec4f1436",
        "sha256:f0e6d6f1263a7808e565e6906c3203613bdcf4c9953023523c45aa998cce78ef",
    }
    fifth_profiles = frozenset(
        {
            "sha256:453298804b2f0cafc44952ffa483aa1cb4d1cb25839119304bde22d771aef73b",
            "sha256:98036903fadd7260e92ace645ac3214977caabefc21a4c4e1fc108cee4eafa9d",
        }
    )
    fifth = next(
        (archive, value)
        for archive, value in manifests.items()
        if frozenset(profile["profile_revision"] for profile in value["obsolete_profiles"])
        == fifth_profiles
    )
    assert fifth[0].name == "0cf625460a9e0ac8ab65e4c411223bf4271ba7b228b207c765ff9597ed5d234c"
    assert fifth[1]["status"] == "obsolete_response_normalization_policy_identity"
    assert fifth[1]["selection_created"] is False
    assert fifth[1]["superseding_response_normalization_policy_revision"] == (
        "sha256:e8b7399e8ee591c2b242ae827deaba64da98179ccf76b264b1e37aab83b64ebe"
    )
    assert {artifact["sha256"] for artifact in fifth[1]["artifacts"]} == {
        "sha256:d20ec4f90fb815beaba98337b6dfecfdb18764410eaac275c6bd44fda0357b13",
        "sha256:e746858a1851d95221b639b9e3ee2b46ec681392466a3eca613451c90841241b",
        "sha256:2c64aed90e739ea8c824abb200fe186a7f9020e0b339065536da1891683a46d1",
        "sha256:122feec17a30c58bf08c7f442b4400d65fbd848a64a72d18bc0743df89b7abc3",
    }
    sixth_profiles = frozenset(
        {
            "sha256:00698e7382d2225173c64fdb21d6e57ed67019f71fd0bbd050b61e6d3d0e83ef",
            "sha256:e8909cfa6d641280e02833ea1af568b826b64ad6b9ac9b3c4620653a1a810918",
        }
    )
    sixth = next(
        (archive, value)
        for archive, value in manifests.items()
        if frozenset(profile["profile_revision"] for profile in value["obsolete_profiles"])
        == sixth_profiles
    )
    assert sixth[0].name == "e69254280c22b23516edd3d4e2fb8f6c96d6d6a7a6d9eea45b8bf0a275d609d9"
    assert sixth[1]["status"] == "obsolete_prompt_and_inference_identities"
    assert sixth[1]["selection_created"] is False
    assert {artifact["sha256"] for artifact in sixth[1]["artifacts"]} == {
        "sha256:550cbf6db34a5aee74fbd8ac2daa1b069629ade476716aa839a1789dcd5c1211",
        "sha256:b56fb47e807470f6551dbc4177117a388b30e5d12b0624b8664c8957a730f2fe",
        "sha256:cea3a68bdf5899b156272a80a0350b01d63aa84bd51f911621ecb36b50f51b3a",
        "sha256:2fb1f2225a1092963db90d7de0a0f1547d0c2c2d742b4ec7136b4c2c953ab539",
    }
    seventh_profiles = frozenset(
        {
            "sha256:1052b860b7cf48fbf46b406df53d459fb30b341a65169a6cae0ed9778ef7b308",
            "sha256:8c6db863150243cc02d7577668bb8d7e5567ae16d07bd5a8d26213fdcd89cbea",
        }
    )
    seventh = next(
        (archive, value)
        for archive, value in manifests.items()
        if frozenset(profile["profile_revision"] for profile in value["obsolete_profiles"])
        == seventh_profiles
        and value["selection_created"] is False
    )
    assert seventh[0].name == "e8866b35c3c9b20a3c522d0d7b19c5c5e2444dea8ba1409e3dee3eddeffa68b6"
    assert seventh[1]["status"] == "obsolete_response_normalization_policy_identity"
    assert seventh[1]["selection_created"] is False
    assert seventh[1]["superseding_response_normalization_policy_revision"] == (
        BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION
    )
    assert {artifact["sha256"] for artifact in seventh[1]["artifacts"]} == {
        "sha256:c4adebe6557f61c95ae5f308910fbc36dc365fc87c7f5e7826d783b5c4fc372c",
        "sha256:8e2fa2801c3fc84d2b75f5d76edc54530c6efbdc5df46b1a29eae5695a28c094",
        "sha256:fc8a563986d2ea5d3a4d315a0aecd155a96d7c7b8ab9fba354fd849d78cc2567",
        "sha256:bd657bce6604439f79f8905de65faf9d804012a4539c25a496253d2fc1d62ff0",
    }
    eighth = next(
        (archive, value)
        for archive, value in manifests.items()
        if frozenset(profile["profile_revision"] for profile in value["obsolete_profiles"])
        == seventh_profiles
        and value["selection_created"] is True
    )
    assert eighth[0].name == "ce375facb93096f0095f8f92a9c60bf0e70fc2acaa6b2878bf0a0575d9d77dd7"
    assert eighth[1]["status"] == "obsolete_prompt_and_inference_identities"
    assert {artifact["sha256"] for artifact in eighth[1]["artifacts"]} == {
        "sha256:48fd2deade4609fddc199e126f0df7a5be41e33328065f1b862ce34d159f1f95",
        "sha256:638fa07fa7fafa5d3e339526458befb2caeb1ff2fed5f0be7dce19af11e178ba",
        "sha256:0a023cff6776e624a9c4659e7f8924f6b4d2b23f82b74841c5ea13ed42b010b1",
        "sha256:335c7987f188218abfb948b1cf53e369a666b79b72f9a7ce7161dd864d775a4e",
        "sha256:76023dd8bcafbb35a6b6595de67b45f3cf86549ebcc923eec45f3b638999f937",
    }
    for archive, value in manifests.items():
        roles = [artifact["role"] for artifact in value["artifacts"]]
        assert roles.count("model_selection") == int(value["selection_created"])
        for artifact in value["artifacts"]:
            content = (archive / artifact["archived_path"]).read_bytes()
            assert hashlib.sha256(content).hexdigest() == artifact["sha256"].removeprefix("sha256:")
            assert len(content) == artifact["size"]
    expected_profiles = {
        "sha256:5aa9919aa152f3ea42854a010ebd4530df74474206e859216e891351905705cd",
        "sha256:4ed577a7419dd6ac745584e6ff78e9b2d0618580d1f3513ba32731348ea8dec3",
    }
    existing = next(
        archive
        for archive, value in manifests.items()
        if {profile["profile_revision"] for profile in value["obsolete_profiles"]}
        == expected_profiles
    )
    manifest = manifests[existing]
    assert manifest["status"] == "obsolete_prompt_and_inference_identities"
    assert manifest["selection_created"] is False
    assert {
        profile["profile_revision"] for profile in manifest["obsolete_profiles"]
    } == expected_profiles
    for artifact in manifest["artifacts"]:
        source = root / artifact["source_path"]
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing / artifact["archived_path"], source)

    published = archive_stale_qualification_attempt(root)

    assert published == existing
    assert not (root / "evals/reports").exists()
    assert (published / "manifest.json").read_text(encoding="utf-8").endswith("\n")
    for artifact in manifest["artifacts"]:
        archived = published / artifact["archived_path"]
        assert len(archived.read_bytes()) == artifact["size"]


def test_independent_runs_persist_identical_canonical_bytes(tmp_path: Path) -> None:
    first = _root(tmp_path, "first")
    second = _root(tmp_path, "second")
    first_cases = _cases(first)
    second_cases = _cases(second)
    _run(first, FakeConverse(first_cases))
    _run(second, FakeConverse(second_cases))

    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in sorted((first / "evals").rglob("*.json"))
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in sorted((second / "evals").rglob("*.json"))
    }
    assert first_files == second_files
    assert all(content.endswith(b"\n") for content in first_files.values())


@pytest.mark.parametrize("stop_reason", ["content_filtered", "refusal"])
def test_safety_stop_reason_is_a_successful_strict_non_answer(
    stop_reason: str,
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    protected_index = next(
        index
        for index, case in enumerate(cases)
        if case.case_id == "protected-evaluation-retrieval"
    )
    filtered_call = protected_index * 3 + 1
    client = FakeConverse(
        cases,
        response_text_calls={filtered_call: '{"api'},
        stop_reason_calls={filtered_call: stop_reason},
    )

    result = cast(Any, _run(root, client))

    assert len(client.calls) == 558
    assert result.selection.profile.model_revision == "us.anthropic.claude-fable-5"
    evidence = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in _json_files(root / "evals/reports/evidence")
        if json.loads(path.read_text(encoding="utf-8"))["profile"]["model_revision"]
        == "us.anthropic.claude-fable-5"
    )
    observation = evidence["observations"][filtered_call - 1]
    assert observation["raw_response_text"] == ""
    assert observation["raw_response_hash"] == (
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert observation["normalized_response_text"] == STRICT_A04_ABSTENTION_TEXT
    assert observation["normalized_response_hash"] == (
        "sha256:" + hashlib.sha256(STRICT_A04_ABSTENTION_TEXT.encode()).hexdigest()
    )
    assert observation["normalization_disposition"] == "safety_abstention"
    assert observation["normalization_policy_revision"] == (
        BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION
    )
    assert observation["finish_reason"] == stop_reason
    assert observation["request_id"] == f"request-{filtered_call:03d}"
    assert observation["token_usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert observation["cost_usd"] == "0.0006"
    assert observation["api_error"] is None
    assert observation["metrics"]["request_succeeded"] is True
    assert observation["metrics"]["structured_output_valid"] is True
    assert observation["metrics"]["required_behavior_passed"] is True
    assert observation["deterministic_grade"]["passed"] is True


def test_empty_content_safety_stop_is_a_successful_strict_non_answer(tmp_path: Path) -> None:
    root = _root(tmp_path)
    case = next(item for item in _cases(root) if item.case_id == "protected-evaluation-retrieval")
    evidence = _synthetic_evidence(case)
    observation = _successful_observation(
        "us.anthropic.claude-fable-5",
        case,
        evidence,
        1,
        "sha256:" + "1" * 64,
        0.125,
        {
            "output": {"message": {"content": []}},
            "usage": {"inputTokens": 100, "outputTokens": 0, "totalTokens": 100},
            "stopReason": "content_filtered",
            "ResponseMetadata": {"RequestId": "request-empty-filter"},
        },
    )

    assert observation["raw_response_text"] == ""
    assert observation["normalized_response_text"] == STRICT_A04_ABSTENTION_TEXT
    assert observation["normalization_disposition"] == "safety_abstention"
    assert observation["finish_reason"] == "content_filtered"
    assert observation["request_id"] == "request-empty-filter"
    assert observation["api_error"] is None
    metrics = cast(Mapping[str, object], observation["metrics"])
    grade = cast(Mapping[str, object], observation["deterministic_grade"])
    assert metrics["request_succeeded"] is True
    assert grade["passed"] is True


def test_supported_case_filter_is_structured_but_materially_failed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    assert cases[0].category == "supported"
    client = FakeConverse(
        cases,
        response_text_calls={1: '{"api'},
        stop_reason_calls={1: "content_filtered"},
    )

    _run(root, client)

    evidence = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in _json_files(root / "evals/reports/evidence")
        if json.loads(path.read_text(encoding="utf-8"))["profile"]["model_revision"]
        == "us.anthropic.claude-fable-5"
    )
    observation = evidence["observations"][0]
    assert observation["metrics"]["request_succeeded"] is True
    assert observation["metrics"]["structured_output_valid"] is True
    assert observation["metrics"]["materially_correct"] is False
    assert observation["metrics"]["supported_project_claims"] == 0
    assert observation["deterministic_grade"]["passed"] is False


@pytest.mark.parametrize("stop_reason", ["max_tokens", "tool_use", "stop_sequence"])
def test_unsupported_stop_reason_fails_closed_but_retains_billing_evidence(
    stop_reason: str,
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    client = FakeConverse(
        cases,
        response_text_calls={1: '{"api'},
        stop_reason_calls={1: stop_reason},
    )

    _run(root, client)

    evidence = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in _json_files(root / "evals/reports/evidence")
        if json.loads(path.read_text(encoding="utf-8"))["profile"]["model_revision"]
        == "us.anthropic.claude-fable-5"
    )
    observation = evidence["observations"][0]
    assert observation["raw_response_text"] == '{"api'
    assert observation["normalized_response_text"] == ""
    assert observation["normalization_disposition"] == "failed_closed"
    expected_code = (
        f"normalization_error:unsupported_stop_reason:{stop_reason}"
        if stop_reason in {"max_tokens", "tool_use"}
        else "normalization_error:unsupported_stop_reason:unknown"
    )
    assert observation["api_error"] == expected_code
    assert observation["metrics"]["request_succeeded"] is False
    assert observation["metrics"]["structured_output_valid"] is False
    assert observation["request_id"] == "request-001"
    assert observation["token_usage"]["total_tokens"] == 120
    assert observation["cost_usd"] == "0.0006"


def test_state_verification_recomputes_normalization_after_hashes_are_resealed(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    cases = _cases(root)
    _run(root, FakeConverse(cases))
    path = _json_files(root / "evals/reports/evidence")[0]
    value = json.loads(path.read_text(encoding="utf-8"))
    observation = value["observations"][0]
    observation["normalized_response_text"] = STRICT_A04_ABSTENTION_TEXT
    observation["normalized_response_hash"] = (
        "sha256:" + hashlib.sha256(STRICT_A04_ABSTENTION_TEXT.encode()).hexdigest()
    )
    preimage = dict(value)
    preimage.pop("evidence_id")
    value["evidence_id"] = _content_id("live-qualification-evidence/2", preimage)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(LiveQualificationError, match="normalization result was tampered"):
        _run(root, FakeConverse(cases), resume=True)


def _selected_stale_root(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    shutil.copytree(ROOT / "evals", root / "evals")
    if (root / "evals/model-selection.json").is_file():
        return root
    archives = root / "evals/qualification-attempts"
    selected = next(
        archive
        for archive in sorted(archives.iterdir(), reverse=True)
        if json.loads((archive / "manifest.json").read_text(encoding="utf-8"))["selection_created"]
        is True
    )
    manifest = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        destination = root / artifact["source_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected / artifact["archived_path"], destination)
    return root


def test_selected_stale_attempt_preview_and_archive_preserve_every_authoritative_byte(
    tmp_path: Path,
) -> None:
    root = _selected_stale_root(tmp_path, "selected")
    authoritative = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted((root / "evals/reports").rglob("*.json"))
    }
    selection_path = root / "evals/model-selection.json"
    authoritative[selection_path.relative_to(root).as_posix()] = selection_path.read_bytes()

    preview = preview_stale_qualification_attempt(root)

    assert preview["selection_created"] is True
    assert preview["selection_reason"] == (
        "a candidate selection was produced and preserved by this attempt"
    )
    assert selection_path.exists()
    preview_artifacts = cast(list[dict[str, object]], preview["artifacts"])
    assert {item["source_path"] for item in preview_artifacts} == set(authoritative)
    assert {item["role"] for item in preview_artifacts} == {
        "model_selection",
        "evaluation_report",
        "complete_evidence",
    }

    published = archive_stale_qualification_attempt(root)

    assert published.name == cast(str, preview["archive_id"]).removeprefix("sha256:")
    assert not selection_path.exists()
    assert not (root / "evals/reports").exists()
    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == preview
    for artifact in manifest["artifacts"]:
        content = (published / artifact["archived_path"]).read_bytes()
        assert content == authoritative[artifact["source_path"]]
        assert artifact["sha256"] == "sha256:" + hashlib.sha256(content).hexdigest()
        assert artifact["size"] == len(content)


def test_qualification_cli_archival_dry_run_then_apply_never_imports_boto3(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from infra.qualify_models import main

    root = _selected_stale_root(tmp_path, "cli-selected")
    monkeypatch.setitem(__import__("sys").modules, "boto3", None)

    assert main(["--root", str(root), "--archive-stale", "dry-run"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["applied"] is False
    assert dry_run["manifest"]["selection_created"] is True
    assert (root / "evals/model-selection.json").exists()

    assert main(["--root", str(root), "--archive-stale", "apply"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["applied"] is True
    assert applied["manifest"] == dry_run["manifest"]
    assert Path(applied["archive_path"]).is_dir()
    assert not (root / "evals/model-selection.json").exists()
    assert not (root / "evals/reports").exists()
