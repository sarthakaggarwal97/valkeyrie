import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infra.continue_deployed_evaluation import (
    AdoptedJournalHashes,
    continue_after_nullable_message_rejection,
)
from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedEvaluationIdentity,
    DeployedInvocation,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "624d06a0fde7265214b80cdf0eac3dcc9635fd8441d80eb2af816f19efd2cf2f"
SOURCE = ROOT / "evals/deployed-attempts" / RUN_ID / "initial"
FIXED_TIME = datetime(2026, 8, 20, 16, 0, tzinfo=UTC)
GENERATION = "sha256:6760b22e3e09ac0e2ac64c9c23b3a77733f5bd1f8f8c9704c66d2c69d1362b60"
EXPECTED_REMAINING = (
    "How do I get started with Valkey?",
    "How does the Valkey Technical Steering Committee work?",
    "What upcoming Valkey events are announced right now?",
    "How can I become a Valkey contributor?",
    "What is the current status of the connection-storm workstream in valkey issue #1688?",
    "Who is Madelyn Olson in the Valkey project?",
    "What happened in the most recent Valkey community meeting?",
    "How do Valkey replication and failover behave?",
    "How do I build a leaderboard with Valkey?",
)


def _identity() -> DeployedEvaluationIdentity:
    return DeployedEvaluationIdentity(
        "valkeyrie-development-application",
        "7",
        "sha256:08079d56dcbde9f2c44f421ae0b3dd7e1b157d3ba457d84d971acf3a5efb7f7e",
        "sha256:64f9a99faefc432eca78ed0bbf68d3fb6da3a0867e558fc07111ced476e66193",
        GENERATION,
        "ONVASJDDNX",
    )


def _hashes() -> AdoptedJournalHashes:
    return AdoptedJournalHashes(
        "sha256:9052d4664256188556a82645a9c778a6ba3b449bb85b9eaabd2be9fcea7a6d1f",
        "sha256:cbf879cfc182b3bc1d7eeee73c88faf27c0b665fcb84b6ba27f7977f31175bc4",
        "sha256:116cfd7276bff324a5c26f98148701e112abf33cdffe1cc8ea16d877803b1d83",
    )


class RecordingInvoker:
    def __init__(self) -> None:
        self.calls: list[DeployedInvocation] = []

    def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
        self.calls.append(invocation)
        request = json.loads(invocation.payload)
        question = request["question"]
        live = request["version_requirement"] == "current_state"
        abstain = question in {
            "What upcoming Valkey events are announced right now?",
            "What happened in the most recent Valkey community meeting?",
        }
        return {
            "outcome": "abstention" if abstain else "answer",
            "request_id": request["request_id"],
            "message": "Insufficient validated live evidence." if abstain else None,
            "claims": [] if abstain else [{"claim_id": "verified", "text": "Verified."}],
            "citations": [] if abstain else ["immutable evidence"],
            "generation_id": None if live else GENERATION,
            "request_revision": 2,
            "request_fence": 1,
        }


def test_adopts_first_result_and_invokes_exactly_remaining_nine_once(tmp_path: Path) -> None:
    invoker = RecordingInvoker()
    result = continue_after_nullable_message_rejection(
        ROOT,
        SOURCE,
        tmp_path / "continuation",
        _identity(),
        _hashes(),
        invoker,
        clock=lambda: FIXED_TIME,
    )

    questions = tuple(json.loads(call.payload)["question"] for call in invoker.calls)
    assert questions == EXPECTED_REMAINING
    assert "Who governs Valkey?" not in questions
    assert len(list((result.journal_directory / "attempts").glob("*.json"))) == 9
    assert len(list((result.journal_directory / "results").glob("*.json"))) == 9
    assert not (result.journal_directory / "attempts/01-real-governance.json").exists()
    observations = result.manifest["observations"]
    assert isinstance(observations, list)
    assert len(observations) == 10
    assert observations[0]["journal_role"] == "adopted_initial"
    assert all(item["journal_role"] == "continuation" for item in observations[1:])
    assert result.manifest["adopted_journaled_results"] == 1
    assert result.manifest["new_invocations"] == 9
    assert result.manifest["automatic_retries"] == 0


def test_rejects_source_hash_mismatch_before_creating_journal_or_invoking(
    tmp_path: Path,
) -> None:
    invoker = RecordingInvoker()
    hashes = AdoptedJournalHashes(
        "sha256:" + "0" * 64,
        _hashes().attempt_sha256,
        _hashes().result_sha256,
    )
    target = tmp_path / "rejected"
    with pytest.raises(DeployedEvaluationError, match="explicit pin"):
        continue_after_nullable_message_rejection(
            ROOT,
            SOURCE,
            target,
            _identity(),
            hashes,
            invoker,
            clock=lambda: FIXED_TIME,
        )
    assert invoker.calls == []
    assert not target.exists()


def test_refuses_existing_continuation_without_invocation(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    invoker = RecordingInvoker()
    with pytest.raises(DeployedEvaluationError, match="refusing to resume or repeat"):
        continue_after_nullable_message_rejection(
            ROOT,
            SOURCE,
            target,
            _identity(),
            _hashes(),
            invoker,
            clock=lambda: FIXED_TIME,
        )
    assert invoker.calls == []
