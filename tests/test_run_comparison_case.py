import base64
import json
import shutil
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
        "sha256:64f9a99faefc432eca78ed0bbf68d3fb6da3a0867e558fc07111ced476e66193",
        GENERATION,
        "ONVASJDDNX",
        "us.anthropic.claude-opus-5",
        ("arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-opus-5"),
        "sha256:5bfca020eed7a24b06c6e92c10cb2ae99b99b1a82a257e679eba40bd8faba8d5",
        "owner_directed_comparison",
        "not_run_not_qualified",
    )


class RecordingInvoker:
    def __init__(self, *, outcome: str = "answer", error: Exception | None = None) -> None:
        self.outcome = outcome
        self.error = error
        self.calls: list[DeployedInvocation] = []

    def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
        self.calls.append(invocation)
        if self.error is not None:
            raise self.error
        request = json.loads(invocation.payload)
        return {
            "outcome": self.outcome,
            "request_id": request["request_id"],
            "message": None if self.outcome == "answer" else "safe non-answer",
            "claims": [] if self.outcome != "answer" else [{"claim_id": "one"}],
            "citations": [],
            "generation_id": GENERATION,
            "request_revision": 2,
            "request_fence": 1,
        }


class FakePreflight:
    def __init__(self) -> None:
        self.calls: list[ComparisonEvaluationIdentity] = []

    def verify(self, identity: ComparisonEvaluationIdentity) -> Mapping[str, object]:
        self.calls.append(identity)
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


def _monotonic(*values: int) -> Callable[[], int]:
    remaining = iter(values)
    return lambda: next(remaining)


def _ledger(tmp_path: Path) -> tuple[Path, Path, Path]:
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
    cases.mkdir(parents=True)
    return project, ledger, cases


def test_success_records_finite_client_latency_and_one_call(tmp_path: Path) -> None:
    project, ledger, cases = _ledger(tmp_path)
    invoker = RecordingInvoker()
    preflight = FakePreflight()
    result = run_comparison_case_once(
        project,
        ledger,
        cases / "01-real-governance",
        _identity(),
        1,
        invoker,
        preflight,
        clock=lambda: FIXED_TIME,
        monotonic_ns=_monotonic(1_000_000_000, 1_012_345_678),
    )

    assert len(invoker.calls) == 1
    assert preflight.calls == [_identity()]
    assert (result.journal_directory / "preflight.json").is_file()
    assert result.latency_ms == 12.345678
    assert result.response is not None and result.terminal is None
    records = list((result.journal_directory / "results").glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["latency_ms"] == 12.345678
    assert list((result.journal_directory / "terminal").glob("*.json")) == []
    assert len(list((result.journal_directory / "attempts").glob("*.json"))) == 1


def test_timeout_writes_one_terminal_record_and_never_redispatches(tmp_path: Path) -> None:
    project, ledger, cases = _ledger(tmp_path)
    target = cases / "03-real-tsc-process"
    failing = RecordingInvoker(error=TimeoutError("read timed out"))
    result = run_comparison_case_once(
        project,
        ledger,
        target,
        _identity(),
        3,
        failing,
        FakePreflight(),
        clock=lambda: FIXED_TIME,
        monotonic_ns=_monotonic(2_000_000_000, 242_000_000_000),
    )

    assert len(failing.calls) == 1
    assert result.response is None
    assert result.terminal is not None
    assert result.terminal["failure_class"] == "timeout"
    assert result.terminal["retry_permitted"] is False
    assert result.latency_ms == 240_000.0
    assert len(list((target / "terminal").glob("*.json"))) == 1
    assert list((target / "results").glob("*.json")) == []

    second = RecordingInvoker()
    with pytest.raises(DeployedEvaluationError, match="resume or repeat"):
        run_comparison_case_once(
            project,
            ledger,
            target,
            _identity(),
            3,
            second,
            FakePreflight(),
            clock=lambda: FIXED_TIME,
            monotonic_ns=_monotonic(1, 2),
        )
    assert second.calls == []


def test_semantic_mismatch_is_persisted_without_blocking_later_case(tmp_path: Path) -> None:
    project, ledger, cases = _ledger(tmp_path)
    mismatch = RecordingInvoker(outcome="abstention")
    first = run_comparison_case_once(
        project,
        ledger,
        cases / "01-real-governance",
        _identity(),
        1,
        mismatch,
        FakePreflight(),
        clock=lambda: FIXED_TIME,
        monotonic_ns=_monotonic(10, 20),
    )
    later = RecordingInvoker()
    second = run_comparison_case_once(
        project,
        ledger,
        cases / "02-real-getting-started",
        _identity(),
        2,
        later,
        FakePreflight(),
        clock=lambda: FIXED_TIME,
        monotonic_ns=_monotonic(30, 50),
    )

    assert first.response is not None and first.response["outcome"] == "abstention"
    assert first.terminal is None
    assert second.response is not None and second.response["outcome"] == "answer"
    assert len(mismatch.calls) == len(later.calls) == 1


def test_rejects_alternate_ledger_root_before_preflight_or_dispatch(tmp_path: Path) -> None:
    project, _, _ = _ledger(tmp_path)
    ledger = project / "evals/comparison-attempts/not-the-run-id"
    cases = ledger / "cases"
    cases.mkdir(parents=True)
    preflight = FakePreflight()
    invoker = RecordingInvoker()

    with pytest.raises(DeployedEvaluationError, match="canonical run directory"):
        run_comparison_case_once(
            project,
            ledger,
            cases / "01-real-governance",
            _identity(),
            1,
            invoker,
            preflight,
            clock=lambda: FIXED_TIME,
            monotonic_ns=_monotonic(1, 2),
        )

    assert preflight.calls == []
    assert invoker.calls == []


def test_atomic_case_claim_allows_only_one_concurrent_dispatch(tmp_path: Path) -> None:
    project, ledger, cases = _ledger(tmp_path)
    target = cases / "01-real-governance"
    barrier = threading.Barrier(2)
    invoker = RecordingInvoker()

    class BarrierPreflight(FakePreflight):
        def verify(self, identity: ComparisonEvaluationIdentity) -> Mapping[str, object]:
            proof = super().verify(identity)
            barrier.wait(timeout=5)
            return proof

    preflight = BarrierPreflight()

    def run() -> str:
        try:
            run_comparison_case_once(
                project,
                ledger,
                target,
                _identity(),
                1,
                invoker,
                preflight,
                clock=lambda: FIXED_TIME,
                monotonic_ns=_monotonic(1, 2),
            )
        except DeployedEvaluationError as error:
            return str(error)
        return "dispatched"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: run(), range(2)))

    assert outcomes.count("dispatched") == 1
    assert sum("resume or repeat" in outcome for outcome in outcomes) == 1
    assert len(invoker.calls) == 1
    assert len(list((target / "attempts").glob("*.json"))) == 1
