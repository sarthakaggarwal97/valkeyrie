from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedEvaluationIdentity,
    DeployedInvocation,
    run_deployed_evaluation,
)
from valkeyrie.evaluations import load_evaluation_suite

ROOT = Path(__file__).resolve().parents[1]
APPLICATION = "sha256:" + "a" * 64
GENERATION = "sha256:" + "b" * 64
FIXED_TIME = datetime(2026, 8, 20, 5, 30, tzinfo=UTC)
REAL_IDS = (
    "real-governance",
    "real-getting-started",
    "real-tsc-process",
    "real-upcoming-events",
    "real-contributor-onboarding",
    "real-workstream-status",
    "real-named-person",
    "real-recent-community-meeting",
    "real-replication-failover",
    "real-leaderboard",
)


def _identity() -> DeployedEvaluationIdentity:
    return DeployedEvaluationIdentity(
        function_name="valkeyrie-development-application",
        function_qualifier="7",
        application_revision=APPLICATION,
        evaluation_suite_revision=load_evaluation_suite(ROOT).revision,
        generation_id=GENERATION,
        knowledge_base_id="ABCDEFGHIJ",
    )


class RecordingInvoker:
    def __init__(self, journal: Path | None = None) -> None:
        self.journal = journal
        self.calls: list[DeployedInvocation] = []

    def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
        self.calls.append(invocation)
        payload = json.loads(invocation.payload)
        if self.journal is not None:
            attempt = (
                self.journal
                / "attempts"
                / f"{len(self.calls):02d}-{REAL_IDS[len(self.calls) - 1]}.json"
            )
            assert attempt.is_file()
            assert json.loads(attempt.read_text())["attempted"] is True
        live = payload["version_requirement"] == "current_state"
        abstention = payload["question"] in {
            "What upcoming Valkey events are announced right now?",
            "What happened in the most recent Valkey community meeting?",
        }
        return {
            "outcome": "abstention" if abstention else "answer",
            "request_id": payload["request_id"],
            "message": "Insufficient validated live evidence."
            if abstention
            else "Verified answer.",
            "claims": [] if abstention else [{"claim_id": "answer", "text": "Verified answer."}],
            "citations": [] if abstention else ["https://example.invalid/immutable"],
            "generation_id": None if live else invocation.generation_id,
            "request_revision": "sha256:" + "c" * 64,
            "request_fence": 1,
        }


class FailingInvoker:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
        del invocation
        self.calls += 1
        raise RuntimeError("backend detail")


def _content_id(domain: str, value: object) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    digest = hashlib.sha256()
    for part in (domain.encode(), canonical):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return "sha256:" + digest.hexdigest()


def test_runs_exactly_ten_real_cases_once_with_pinned_identities_and_durable_journals(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal"
    invoker = RecordingInvoker(journal)

    result = run_deployed_evaluation(ROOT, journal, _identity(), invoker, clock=lambda: FIXED_TIME)

    assert len(invoker.calls) == 10
    payloads = [json.loads(call.payload) for call in invoker.calls]
    observations = cast(list[dict[str, object]], result.manifest["observations"])
    assert tuple(observation["case_id"] for observation in observations) == REAL_IDS
    assert len({payload["request_id"] for payload in payloads}) == 10
    assert all(payload["request_id"].startswith("req_deployed-eval-") for payload in payloads)
    assert [payload["version_requirement"] for payload in payloads].count("current_state") == 3
    assert all(
        call.function_name == _identity().function_name
        and call.function_qualifier == _identity().function_qualifier
        and call.application_revision == APPLICATION
        and call.evaluation_suite_revision == _identity().evaluation_suite_revision
        and call.generation_id == GENERATION
        for call in invoker.calls
    )
    assert len(list((journal / "attempts").glob("*.json"))) == 10
    assert len(list((journal / "results").glob("*.json"))) == 10
    assert stat.S_IMODE(journal.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in journal.rglob("*.json"))
    assert result.manifest["case_count"] == 10
    assert result.manifest["automatic_retries"] == 0
    assert result.manifest["identity"] == {
        "function_name": _identity().function_name,
        "function_qualifier": "7",
        "application_revision": APPLICATION,
        "evaluation_suite_revision": _identity().evaluation_suite_revision,
        "generation_id": GENERATION,
        "knowledge_base_id": "ABCDEFGHIJ",
    }
    preimage = dict(result.manifest)
    evidence_id = preimage.pop("evidence_id")
    assert evidence_id == _content_id("deployed-evaluation-evidence/1", preimage)
    assert result.manifest_path.name == f"evidence-{evidence_id.removeprefix('sha256:')}.json"
    assert json.loads(result.manifest_path.read_text()) == result.manifest


def test_request_ids_and_final_evidence_are_deterministic_for_the_same_pins(
    tmp_path: Path,
) -> None:
    first_invoker = RecordingInvoker()
    second_invoker = RecordingInvoker()
    first = run_deployed_evaluation(
        ROOT, tmp_path / "first", _identity(), first_invoker, clock=lambda: FIXED_TIME
    )
    second = run_deployed_evaluation(
        ROOT, tmp_path / "second", _identity(), second_invoker, clock=lambda: FIXED_TIME
    )

    assert [call.request_id for call in first_invoker.calls] == [
        call.request_id for call in second_invoker.calls
    ]
    assert first.run_id == second.run_id
    assert first.evidence_id == second.evidence_id
    assert first.manifest == second.manifest


def test_pre_attempt_record_is_fsynced_before_each_injected_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = tmp_path / "fsync"
    real_fsync = os.fsync
    fsync_calls: list[int] = []

    def recording_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    class CheckingInvoker(RecordingInvoker):
        def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
            assert len(fsync_calls) >= 4 + len(self.calls) * 2
            return super().invoke(invocation)

    monkeypatch.setattr("valkeyrie.deployed_evaluation.os.fsync", recording_fsync)
    invoker = CheckingInvoker(journal)
    run_deployed_evaluation(ROOT, journal, _identity(), invoker, clock=lambda: FIXED_TIME)
    assert len(invoker.calls) == 10


def test_failure_after_journaled_attempt_is_never_retried_or_resumed(tmp_path: Path) -> None:
    journal = tmp_path / "failed"
    invoker = FailingInvoker()

    with pytest.raises(DeployedEvaluationError, match="retry is forbidden"):
        run_deployed_evaluation(ROOT, journal, _identity(), invoker, clock=lambda: FIXED_TIME)

    assert invoker.calls == 1
    attempts = list((journal / "attempts").glob("*.json"))
    assert len(attempts) == 1
    assert list((journal / "results").glob("*.json")) == []
    assert list(journal.glob("evidence-*.json")) == []
    with pytest.raises(DeployedEvaluationError, match="refusing to resume or repeat"):
        run_deployed_evaluation(ROOT, journal, _identity(), invoker, clock=lambda: FIXED_TIME)
    assert invoker.calls == 1


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        (replace(_identity(), function_qualifier="$LATEST"), "immutable numeric version"),
        (replace(_identity(), application_revision="latest"), "application revision"),
        (replace(_identity(), generation_id="active"), "generation ID"),
        (replace(_identity(), evaluation_suite_revision="sha256:" + "0" * 64), "suite revision"),
        (replace(_identity(), knowledge_base_id="missing"), "knowledge base ID"),
    ],
)
def test_rejects_unpinned_or_stale_identity_before_any_attempt(
    tmp_path: Path, identity: DeployedEvaluationIdentity, message: str
) -> None:
    invoker = RecordingInvoker()
    with pytest.raises(DeployedEvaluationError, match=message):
        run_deployed_evaluation(
            ROOT, tmp_path / message.replace(" ", "-"), identity, invoker, clock=lambda: FIXED_TIME
        )
    assert invoker.calls == []


@pytest.mark.parametrize("fault", ["request", "generation", "live-generation", "outcome"])
def test_invalid_deployed_result_stops_after_one_attempt_without_retry(
    tmp_path: Path, fault: str
) -> None:
    class InvalidInvoker(RecordingInvoker):
        def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
            value = dict(super().invoke(invocation))
            if fault == "request":
                value["request_id"] = "req_wrong"
            elif fault == "generation":
                value["generation_id"] = "sha256:" + "0" * 64
            elif fault == "live-generation":
                payload = json.loads(invocation.payload)
                if payload["version_requirement"] == "current_state":
                    value["generation_id"] = GENERATION
            else:
                value["outcome"] = "abstention"
            return value

    invoker = InvalidInvoker()
    journal = tmp_path / fault
    if fault == "live-generation":
        # The first live case is fourth, so preceding completed attempts are expected.
        expected_calls = 4
    else:
        expected_calls = 1
    with pytest.raises(DeployedEvaluationError):
        run_deployed_evaluation(ROOT, journal, _identity(), invoker, clock=lambda: FIXED_TIME)
    assert len(invoker.calls) == expected_calls
    assert len(list((journal / "attempts").glob("*.json"))) == expected_calls
    assert list(journal.glob("evidence-*.json")) == []


def test_accepts_nullable_message_for_successful_answers(tmp_path: Path) -> None:
    class NullableAnswerInvoker(RecordingInvoker):
        def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
            value = dict(super().invoke(invocation))
            if value["outcome"] == "answer":
                value["message"] = None
            return value

    result = run_deployed_evaluation(
        ROOT,
        tmp_path / "nullable-answer",
        _identity(),
        NullableAnswerInvoker(),
        clock=lambda: FIXED_TIME,
    )
    assert result.manifest["case_count"] == 10


def test_rejects_nullable_message_for_non_answer(tmp_path: Path) -> None:
    class NullableAbstentionInvoker(RecordingInvoker):
        def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
            value = dict(super().invoke(invocation))
            payload = json.loads(invocation.payload)
            if payload["question"] == "What upcoming Valkey events are announced right now?":
                value["message"] = None
            return value

    with pytest.raises(DeployedEvaluationError, match="requires a message"):
        run_deployed_evaluation(
            ROOT,
            tmp_path / "nullable-abstention",
            _identity(),
            NullableAbstentionInvoker(),
            clock=lambda: FIXED_TIME,
        )
