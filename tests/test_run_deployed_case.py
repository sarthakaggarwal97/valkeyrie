import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infra.run_deployed_case import run_deployed_case_once
from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedEvaluationIdentity,
    DeployedInvocation,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATION = "sha256:6760b22e3e09ac0e2ac64c9c23b3a77733f5bd1f8f8c9704c66d2c69d1362b60"
FIXED_TIME = datetime(2026, 8, 20, 17, 0, tzinfo=UTC)


def _identity() -> DeployedEvaluationIdentity:
    return DeployedEvaluationIdentity(
        "valkeyrie-development-application",
        "7",
        "sha256:08079d56dcbde9f2c44f421ae0b3dd7e1b157d3ba457d84d971acf3a5efb7f7e",
        "sha256:64f9a99faefc432eca78ed0bbf68d3fb6da3a0867e558fc07111ced476e66193",
        GENERATION,
        "ONVASJDDNX",
    )


class RecordingInvoker:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[DeployedInvocation] = []

    def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
        self.calls.append(invocation)
        if self.fail:
            raise RuntimeError("uncertain transport")
        request = json.loads(invocation.payload)
        return {
            "outcome": "abstention",
            "request_id": request["request_id"],
            "message": "Insufficient validated live evidence.",
            "claims": [],
            "citations": [],
            "generation_id": None,
            "request_revision": 2,
            "request_fence": 1,
        }


def test_runs_one_unattempted_case_once(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    cases = ledger / "cases"
    cases.mkdir(parents=True)
    invoker = RecordingInvoker()

    result = run_deployed_case_once(
        ROOT,
        ledger,
        cases / "04-real-upcoming-events",
        _identity(),
        4,
        invoker,
        clock=lambda: FIXED_TIME,
    )

    assert len(invoker.calls) == 1
    assert json.loads(invoker.calls[0].payload)["question"] == (
        "What upcoming Valkey events are announced right now?"
    )
    assert result.response["outcome"] == "abstention"
    assert len(list((result.journal_directory / "attempts").glob("*.json"))) == 1
    assert len(list((result.journal_directory / "results").glob("*.json"))) == 1


def test_failed_attempt_is_never_dispatched_again(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    cases = ledger / "cases"
    cases.mkdir(parents=True)
    target = cases / "05-real-contributor-onboarding"
    failing = RecordingInvoker(fail=True)
    with pytest.raises(DeployedEvaluationError, match="retry is forbidden"):
        run_deployed_case_once(
            ROOT,
            ledger,
            target,
            _identity(),
            5,
            failing,
            clock=lambda: FIXED_TIME,
        )
    assert len(failing.calls) == 1
    assert len(list((target / "attempts").glob("*.json"))) == 1
    assert list((target / "results").glob("*.json")) == []

    second = RecordingInvoker()
    with pytest.raises(DeployedEvaluationError, match="already has a journaled attempt"):
        run_deployed_case_once(
            ROOT,
            ledger,
            cases / "second-target",
            _identity(),
            5,
            second,
            clock=lambda: FIXED_TIME,
        )
    assert second.calls == []
    assert not (cases / "second-target").exists()


def test_existing_attempt_in_other_journal_blocks_before_target_creation(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger"
    attempts = ledger / "prior/attempts"
    attempts.mkdir(parents=True)
    (attempts / "03-real-tsc-process.json").write_text(
        json.dumps(
            {
                "index": 3,
                "case_id": "real-tsc-process",
                "request_id": "req_existing",
            }
        )
    )
    cases = ledger / "cases"
    cases.mkdir()
    invoker = RecordingInvoker()
    target = cases / "03-real-tsc-process"
    with pytest.raises(DeployedEvaluationError, match="already has a journaled attempt"):
        run_deployed_case_once(
            ROOT,
            ledger,
            target,
            _identity(),
            3,
            invoker,
            clock=lambda: FIXED_TIME,
        )
    assert invoker.calls == []
    assert not target.exists()
