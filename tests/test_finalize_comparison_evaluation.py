import base64
import json
import shutil
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from infra.finalize_comparison_evaluation import finalize_comparison_evaluation
from infra.run_comparison_case import (
    ComparisonEvaluationIdentity,
    comparison_run_identity,
    run_comparison_case_once,
)
from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedInvocation,
    _real_cases,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATION = "sha256:6760b22e3e09ac0e2ac64c9c23b3a77733f5bd1f8f8c9704c66d2c69d1362b60"
FIXED_TIME = datetime(2026, 8, 20, 17, 0, tzinfo=UTC)


def _identity() -> ComparisonEvaluationIdentity:
    return ComparisonEvaluationIdentity(
        "valkeyrie-development-application",
        "8",
        "sha256:fca8f94a37959e127240239526448855ca9a93fc16a910b0294ebfc800725a54",
        "sha256:d0cfac9167b343eb75f3875e688d6ef3bf596f21c2f0a1588f5de49f2d6ae303",
        "sha256:394c725de332cb66cff5808d50c39f19a66979d27e377ac89c0eb4e62295ef89",
        GENERATION,
        "ONVASJDDNX",
        "us.anthropic.claude-opus-5",
        ("arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-opus-5"),
        "sha256:5bfca020eed7a24b06c6e92c10cb2ae99b99b1a82a257e679eba40bd8faba8d5",
        "owner_directed_comparison",
        "not_run_not_qualified",
    )


class CaseInvoker:
    def __init__(
        self, *, mismatch_index: int | None = None, timeout_index: int | None = None
    ) -> None:
        self.mismatch_index = mismatch_index
        self.timeout_index = timeout_index
        self.calls = 0

    def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
        self.calls += 1
        if self.calls == self.timeout_index:
            raise TimeoutError("read timed out")
        request = json.loads(invocation.payload)
        live = request["version_requirement"] == "current_state"
        abstain = "upcoming Valkey events" in request["question"] or (
            "most recent Valkey community meeting" in request["question"]
        )
        outcome = "abstention" if abstain else "answer"
        if self.calls == self.mismatch_index:
            outcome = "abstention" if outcome == "answer" else "answer"
        return {
            "outcome": outcome,
            "request_id": request["request_id"],
            "message": None if outcome == "answer" else "Insufficient validated live evidence.",
            "claims": (
                []
                if outcome != "answer"
                else [
                    {
                        "claim_id": "one",
                        "text": "Grounded synthetic answer.",
                        "evidence_ids": ["ev_one"],
                    }
                ]
            ),
            "citations": [] if outcome != "answer" else ["valkey/README.md@" + "a" * 40],
            "generation_id": None if live else GENERATION,
            "request_revision": 2,
            "request_fence": 1,
        }


class FakePreflight:
    def verify(self, identity: ComparisonEvaluationIdentity) -> Mapping[str, object]:
        return {
            "configuration": {
                "FunctionName": identity.function_name,
                "Version": identity.function_qualifier,
                "CodeSha256": base64.b64encode(
                    bytes.fromhex(identity.artifact_sha256[7:])
                ).decode(),
                "Timeout": 240,
                "APPLICATION_REVISION": identity.application_revision,
                "SELECTED_INFERENCE_PROFILE_ARN": identity.selected_inference_profile_arn,
            },
            "health": {
                "status": "healthy",
                "application_revision": identity.application_revision,
                "selection_id": identity.selection_id,
                "selected_model_revision": identity.selected_model_revision,
                "selected_inference_profile_arn": identity.selected_inference_profile_arn,
                "execution_authorization": identity.execution_authorization,
                "qualification_status": identity.qualification_status,
            },
        }


def _monotonic(start: int, latency_ms: int) -> Callable[[], int]:
    values = iter((start, start + latency_ms * 1_000_000))
    return lambda: next(values)


def _run_ten(
    tmp_path: Path,
    invoker: CaseInvoker,
) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    shutil.copytree(
        ROOT / "evals",
        project / "evals",
        ignore=shutil.ignore_patterns("comparison-attempts", "comparison-reports"),
    )
    _, run_id = comparison_run_identity(_identity(), _real_cases(project))
    ledger = project / "evals/comparison-attempts" / run_id.removeprefix("sha256:")
    cases = ledger / "cases"
    output = project / "evals/comparison-reports"
    cases.mkdir(parents=True)
    output.mkdir()
    for index in range(1, 11):
        case_id = cast(str, _real_cases(project)[index - 1]["id"])
        run_comparison_case_once(
            project,
            ledger,
            cases / f"{index:02d}-{case_id}",
            _identity(),
            index,
            invoker,
            FakePreflight(),
            clock=lambda: FIXED_TIME,
            monotonic_ns=_monotonic(index * 1_000_000_000, index),
        )
    return project, ledger, output


def test_finalizes_all_pass_with_content_addressed_latency_and_fable_baseline(
    tmp_path: Path,
) -> None:
    project, ledger, output = _run_ten(tmp_path, CaseInvoker())
    path, manifest = finalize_comparison_evaluation(project, ledger, output, _identity())

    assert json.loads(path.read_text()) == manifest
    assert path.stem == str(manifest["evidence_id"]).removeprefix("sha256:")
    assert manifest["evaluation_result"] == "evidence_collected_pending_semantic_review"
    assert manifest["summary"] == {
        "case_count": 10,
        "attempt_count": 10,
        "result_count": 10,
        "terminal_count": 0,
        "protocol_passed": 10,
        "protocol_failed": 0,
        "semantic_review_status": "not_evaluated",
        "timeouts": 0,
        "automatic_retries": 0,
        "client_observed_latency_ms": {
            "count": 10,
            "minimum": 1.0,
            "maximum": 10.0,
            "mean": 5.5,
            "median": 5.5,
            "p95_nearest_rank": 10.0,
            "measurement": "client_monotonic_single_lambda_invoke",
        },
    }
    baseline = manifest["baseline_fable"]
    assert isinstance(baseline, Mapping)
    assert baseline["evidence_id"] == (
        "sha256:2fe6adb72c33008ab8a6b9ea07d251b3937de87dc14fcbe60762629eab83d547"
    )
    assert baseline["summary"]["passed"] == 6
    assert baseline["summary"]["failed"] == 4
    assert baseline["latency_measurement_available"] is False
    observations = manifest["observations"]
    assert isinstance(observations, list)
    assert [item["latency_ms"] for item in observations] == [float(i) for i in range(1, 11)]


def test_finalizer_handles_semantic_failure_and_terminal_timeout_generically(
    tmp_path: Path,
) -> None:
    project, ledger, output = _run_ten(tmp_path, CaseInvoker(mismatch_index=2, timeout_index=9))
    _, manifest = finalize_comparison_evaluation(project, ledger, output, _identity())

    summary = cast(Mapping[str, object], manifest["summary"])
    assert summary["result_count"] == 9
    assert summary["terminal_count"] == 1
    assert summary["protocol_passed"] == 8
    assert summary["protocol_failed"] == 2
    assert summary["semantic_review_status"] == "not_evaluated"
    assert summary["timeouts"] == 1
    observations = cast(list[Mapping[str, object]], manifest["observations"])
    assert observations[1]["protocol_verdict"] == "fail"
    assert observations[1]["semantic_verdict"] == "not_evaluated"
    assert observations[1]["outcome"] == "abstention"
    assert observations[8]["protocol_verdict"] == "fail"
    assert observations[8]["outcome"] == "timeout"
    terminal = cast(Mapping[str, object], observations[8]["terminal"])
    assert terminal["retry_permitted"] is False


def test_finalizer_rejects_tampered_fable_baseline_with_preserved_id(
    tmp_path: Path,
) -> None:
    project, ledger, output = _run_ten(tmp_path, CaseInvoker())
    baseline = (
        project / "evals/deployed-reports/"
        "2fe6adb72c33008ab8a6b9ea07d251b3937de87dc14fcbe60762629eab83d547.json"
    )
    value = json.loads(baseline.read_text())
    value["summary"]["passed"] = 10
    baseline.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(DeployedEvaluationError, match="baseline file hash"):
        finalize_comparison_evaluation(project, ledger, output, _identity())
