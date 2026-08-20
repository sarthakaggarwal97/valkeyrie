from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from tests.helpers import load_yaml
from tests.test_generation import _bundle
from tests.test_publication import MemoryPublicationStore
from valkeyrie.generation import GenerationBundle
from valkeyrie.ingestion import (
    CandidateBusyError,
    CandidateCoordinator,
    CandidateState,
    CandidateStatus,
    IngestionError,
    IngestionPendingError,
    StaleFenceError,
    run_ingestion,
)
from valkeyrie.publication import PublicationError, publish_generation
from valkeyrie.retrieval_config import load_retrieval_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = cast(dict[str, Any], load_yaml(ROOT / "src/valkeyrie/schemas/contracts.schema.json"))
KB_ID = "ABCDEFGHIJ"
DATA_SOURCE_ID = "KLMNOPQRST"
JOB_ID = "UVWXYZ1234"
NOW = 1_800_000_000


class MemoryCandidateStore:
    def __init__(self, state: CandidateState | None = None) -> None:
        self.state = state
        self.history: list[CandidateState] = []
        self.reject_writes = 0

    def read_candidate(self) -> CandidateState | None:
        return self.state

    def compare_and_swap(
        self,
        expected_revision: int | None,
        replacement: CandidateState,
    ) -> bool:
        if self.reject_writes:
            self.reject_writes -= 1
            return False
        current_revision = None if self.state is None else self.state.revision
        if current_revision != expected_revision:
            return False
        self.state = replacement
        self.history.append(replacement)
        return True


class FakeBedrock:
    def __init__(self, polls: list[object]) -> None:
        self.polls = list(polls)
        self.start_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []

    def start_ingestion_job(self, **kwargs: object) -> dict[str, object]:
        self.start_calls.append(kwargs)
        return {"ingestionJob": {"ingestionJobId": JOB_ID, "status": "STARTING"}}

    def get_ingestion_job(self, **kwargs: object) -> Mapping[str, object]:
        self.get_calls.append(kwargs)
        if not self.polls:
            raise AssertionError("unexpected ingestion poll")
        return cast(Mapping[str, object], self.polls.pop(0))


def _poll(status: str, *, scanned: int = 0, failed: int = 0) -> dict[str, object]:
    job: dict[str, object] = {"ingestionJobId": JOB_ID, "status": status}
    if status == "COMPLETE":
        job["statistics"] = {
            "numberOfDocumentsScanned": scanned,
            "numberOfDocumentsFailed": failed,
        }
    return {"ingestionJob": job}


@pytest.fixture
def bundle() -> GenerationBundle:
    return _bundle(load_retrieval_config(ROOT / "retrieval-config.yaml"))


def _sealed(bundle: GenerationBundle) -> MemoryPublicationStore:
    store = MemoryPublicationStore()
    publish_generation(store, bundle)
    store.put_calls.clear()
    return store


def test_sealed_generation_runs_one_idempotent_ingestion_to_success(
    bundle: GenerationBundle,
) -> None:
    publication = _sealed(bundle)
    candidate = MemoryCandidateStore()
    bedrock = FakeBedrock([_poll("IN_PROGRESS"), _poll("COMPLETE", scanned=7)])
    sleeps: list[float] = []

    result = run_ingestion(
        publication,
        candidate,
        bedrock,
        bundle,
        knowledge_base_id=KB_ID,
        data_source_id=DATA_SOURCE_ID,
        owner="worker-1",
        now_epoch=lambda: NOW,
        sleep=sleeps.append,
        poll_interval_seconds=2.0,
    )

    assert result.generation_id == bundle.generation_id
    assert result.ingestion_job_id == JOB_ID
    assert result.documents_scanned == 7
    assert result.already_ingested is False
    assert sleeps == [2.0]
    assert bedrock.start_calls == [
        {
            "knowledgeBaseId": KB_ID,
            "dataSourceId": DATA_SOURCE_ID,
            "clientToken": f"{bundle.generation_id.removeprefix('sha256:')}-1",
            "description": f"Valkeyrie sealed generation {bundle.generation_id}",
        }
    ]
    assert all(call["ingestionJobId"] == JOB_ID for call in bedrock.get_calls)
    assert candidate.state is not None
    assert candidate.state.status is CandidateStatus.INGESTED
    assert candidate.state.documents_scanned == 7
    assert candidate.state.failure is None
    assert publication.put_calls == []


def test_unsealed_generation_never_claims_or_starts_ingestion(bundle: GenerationBundle) -> None:
    publication = MemoryPublicationStore()
    candidate = MemoryCandidateStore()
    bedrock = FakeBedrock([])

    with pytest.raises(PublicationError, match="object set mismatch"):
        run_ingestion(
            publication,
            candidate,
            bedrock,
            bundle,
            knowledge_base_id=KB_ID,
            data_source_id=DATA_SOURCE_ID,
            owner="worker-1",
            now_epoch=lambda: NOW,
            sleep=lambda _: None,
        )
    assert candidate.state is None
    assert bedrock.start_calls == []


def test_active_lease_serializes_competing_generation(bundle: GenerationBundle) -> None:
    store = MemoryCandidateStore()
    coordinator = CandidateCoordinator(store)
    first = coordinator.claim(
        bundle.generation_id,
        "worker-1",
        now_epoch=NOW,
        lease_seconds=30,
    )

    with pytest.raises(CandidateBusyError, match="cross-generation takeover"):
        coordinator.claim(
            "sha256:" + "f" * 64,
            "worker-2",
            now_epoch=NOW + 1,
            lease_seconds=30,
        )
    assert store.state is not None
    assert store.state.fence == first.fence == 1


def test_expired_worker_cannot_write_after_new_fence(bundle: GenerationBundle) -> None:
    store = MemoryCandidateStore()
    coordinator = CandidateCoordinator(store)
    first = coordinator.claim(
        bundle.generation_id,
        "worker-1",
        now_epoch=NOW,
        lease_seconds=30,
    )
    second = coordinator.claim(
        bundle.generation_id,
        "worker-2",
        now_epoch=NOW + 30,
        lease_seconds=30,
    )

    assert second.fence == first.fence + 1
    with pytest.raises(StaleFenceError, match="stale"):
        coordinator.bind_job(first, JOB_ID, now_epoch=NOW + 30)
    assert store.state is not None
    assert store.state.owner == "worker-2"
    assert store.state.ingestion_job_id is None


@pytest.mark.parametrize("job_id", [None, JOB_ID])
def test_expired_inflight_attempt_still_blocks_cross_generation_takeover(
    bundle: GenerationBundle,
    job_id: str | None,
) -> None:
    store = MemoryCandidateStore(
        CandidateState(
            bundle.generation_id,
            CandidateStatus.INGESTING,
            revision=3,
            fence=2,
            attempt=1,
            owner="expired-worker",
            lease_expires_at=NOW,
            ingestion_job_id=job_id,
        )
    )
    coordinator = CandidateCoordinator(store)

    with pytest.raises(CandidateBusyError, match="cross-generation takeover"):
        coordinator.claim(
            "sha256:" + "f" * 64,
            "other-generation-worker",
            now_epoch=NOW + 1,
            lease_seconds=30,
        )
    assert store.history == []
    assert store.state is not None
    assert store.state.generation_id == bundle.generation_id
    assert store.state.attempt == 1


def test_attempt_is_persisted_before_start_and_blocks_mid_call_takeover(
    bundle: GenerationBundle,
) -> None:
    publication = _sealed(bundle)
    candidate = MemoryCandidateStore()
    coordinator = CandidateCoordinator(candidate)
    clock = [NOW]

    class TakeoverBedrock(FakeBedrock):
        def start_ingestion_job(self, **kwargs: object) -> dict[str, object]:
            assert candidate.state is not None
            assert candidate.state.attempt == 1
            assert kwargs["clientToken"] == (f"{bundle.generation_id.removeprefix('sha256:')}-1")
            clock[0] = NOW + 300
            with pytest.raises(CandidateBusyError, match="cross-generation takeover"):
                coordinator.claim(
                    "sha256:" + "f" * 64,
                    "other-generation-worker",
                    now_epoch=clock[0],
                    lease_seconds=30,
                )
            return super().start_ingestion_job(**kwargs)

    bedrock = TakeoverBedrock([])
    with pytest.raises(StaleFenceError, match="stale"):
        run_ingestion(
            publication,
            candidate,
            bedrock,
            bundle,
            knowledge_base_id=KB_ID,
            data_source_id=DATA_SOURCE_ID,
            owner="worker-1",
            now_epoch=lambda: clock[0],
            sleep=lambda _: None,
        )
    assert len(bedrock.start_calls) == 1
    assert candidate.state is not None
    assert candidate.state.generation_id == bundle.generation_id
    assert candidate.state.attempt == 1
    assert candidate.state.ingestion_job_id is None


def test_recovery_reuses_existing_job_without_duplicate_start(bundle: GenerationBundle) -> None:
    publication = _sealed(bundle)
    candidate = MemoryCandidateStore(
        CandidateState(
            bundle.generation_id,
            CandidateStatus.INGESTING,
            revision=4,
            fence=2,
            attempt=1,
            owner="expired-worker",
            lease_expires_at=NOW,
            ingestion_job_id=JOB_ID,
        )
    )
    bedrock = FakeBedrock([_poll("COMPLETE", scanned=3)])

    result = run_ingestion(
        publication,
        candidate,
        bedrock,
        bundle,
        knowledge_base_id=KB_ID,
        data_source_id=DATA_SOURCE_ID,
        owner="recovery-worker",
        now_epoch=lambda: NOW,
        sleep=lambda _: None,
    )

    assert result.documents_scanned == 3
    assert bedrock.start_calls == []
    assert candidate.state is not None
    assert candidate.state.fence == 3
    assert candidate.state.status is CandidateStatus.INGESTED


def test_poll_bound_leaves_recoverable_ingesting_candidate(bundle: GenerationBundle) -> None:
    publication = _sealed(bundle)
    candidate = MemoryCandidateStore()
    bedrock = FakeBedrock([_poll("IN_PROGRESS"), _poll("IN_PROGRESS")])

    with pytest.raises(IngestionPendingError, match="did not reach a terminal state"):
        run_ingestion(
            publication,
            candidate,
            bedrock,
            bundle,
            knowledge_base_id=KB_ID,
            data_source_id=DATA_SOURCE_ID,
            owner="worker-1",
            now_epoch=lambda: NOW,
            sleep=lambda _: None,
            max_polls=2,
        )
    assert candidate.state is not None
    assert candidate.state.status is CandidateStatus.INGESTING
    assert candidate.state.ingestion_job_id == JOB_ID


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_poll("FAILED"), "ended with FAILED"),
        (_poll("COMPLETE", scanned=2, failed=1), "reported 1 failed documents"),
        ({"ingestionJob": {"ingestionJobId": JOB_ID, "status": "UNKNOWN"}}, "unknown"),
        ({"malformed": True}, "has no job"),
    ],
)
def test_terminal_or_malformed_results_fail_candidate(
    bundle: GenerationBundle,
    response: object,
    message: str,
) -> None:
    candidate = MemoryCandidateStore()

    with pytest.raises(IngestionError, match=message):
        run_ingestion(
            _sealed(bundle),
            candidate,
            FakeBedrock([response]),
            bundle,
            knowledge_base_id=KB_ID,
            data_source_id=DATA_SOURCE_ID,
            owner="worker-1",
            now_epoch=lambda: NOW,
            sleep=lambda _: None,
        )
    assert candidate.state is not None
    assert candidate.state.status is CandidateStatus.FAILED
    assert candidate.state.failure is not None


def test_failed_generation_is_explicitly_non_retryable(bundle: GenerationBundle) -> None:
    publication = _sealed(bundle)
    candidate = MemoryCandidateStore(
        CandidateState(
            bundle.generation_id,
            CandidateStatus.FAILED,
            revision=6,
            fence=2,
            attempt=1,
            owner="failed-worker",
            lease_expires_at=0,
            ingestion_job_id=JOB_ID,
            failure="Bedrock ingestion ended with FAILED",
        )
    )
    bedrock = FakeBedrock([])

    with pytest.raises(IngestionError, match="explicitly non-retryable"):
        run_ingestion(
            publication,
            candidate,
            bedrock,
            bundle,
            knowledge_base_id=KB_ID,
            data_source_id=DATA_SOURCE_ID,
            owner="retry-worker",
            now_epoch=lambda: NOW,
            sleep=lambda _: None,
        )
    assert bedrock.start_calls == []
    assert candidate.history == []


def test_ingested_result_uses_exact_claim_snapshot_under_replacement_race(
    bundle: GenerationBundle,
) -> None:
    claimed = CandidateState(
        bundle.generation_id,
        CandidateStatus.INGESTED,
        revision=9,
        fence=4,
        attempt=1,
        owner="prior-worker",
        lease_expires_at=0,
        ingestion_job_id=JOB_ID,
        documents_scanned=5,
    )
    replacement = CandidateState(
        "sha256:" + "f" * 64,
        CandidateStatus.INGESTED,
        revision=10,
        fence=5,
        attempt=1,
        owner="other-worker",
        lease_expires_at=0,
        ingestion_job_id="ZZZZZZZZZZ",
        documents_scanned=99,
    )

    class RacingStore(MemoryCandidateStore):
        def __init__(self) -> None:
            super().__init__(claimed)
            self.reads = 0

        def read_candidate(self) -> CandidateState | None:
            self.reads += 1
            return claimed if self.reads == 1 else replacement

    candidate = RacingStore()
    result = run_ingestion(
        _sealed(bundle),
        candidate,
        FakeBedrock([]),
        bundle,
        knowledge_base_id=KB_ID,
        data_source_id=DATA_SOURCE_ID,
        owner="retry-worker",
        now_epoch=lambda: NOW,
        sleep=lambda _: None,
    )

    assert candidate.reads == 1
    assert result.generation_id == bundle.generation_id
    assert result.ingestion_job_id == JOB_ID
    assert result.documents_scanned == 5


def test_ingested_retry_is_read_only_and_returns_pinned_result(bundle: GenerationBundle) -> None:
    publication = _sealed(bundle)
    candidate = MemoryCandidateStore(
        CandidateState(
            bundle.generation_id,
            CandidateStatus.INGESTED,
            revision=9,
            fence=4,
            attempt=1,
            owner="prior-worker",
            lease_expires_at=0,
            ingestion_job_id=JOB_ID,
            documents_scanned=5,
        )
    )
    bedrock = FakeBedrock([])

    result = run_ingestion(
        publication,
        candidate,
        bedrock,
        bundle,
        knowledge_base_id=KB_ID,
        data_source_id=DATA_SOURCE_ID,
        owner="retry-worker",
        now_epoch=lambda: NOW,
        sleep=lambda _: None,
    )

    assert result.already_ingested is True
    assert result.documents_scanned == 5
    assert bedrock.start_calls == []
    assert bedrock.get_calls == []
    assert candidate.history == []
    assert publication.put_calls == []


def test_candidate_state_matches_shared_contract(bundle: GenerationBundle) -> None:
    state = CandidateState(
        bundle.generation_id,
        CandidateStatus.INGESTING,
        revision=1,
        fence=1,
        attempt=1,
        owner="worker-1",
        lease_expires_at=NOW + 300,
    )
    document = asdict(state)
    document["status"] = state.status.value

    Draft202012Validator(
        {
            "$schema": SCHEMA["$schema"],
            "$defs": SCHEMA["$defs"],
            "$ref": "#/$defs/candidate_state",
        }
    ).validate(document)
