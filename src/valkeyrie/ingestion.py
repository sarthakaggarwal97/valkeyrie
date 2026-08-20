"""Fenced candidate state and serialized Bedrock ingestion orchestration."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final, Protocol, cast

from valkeyrie.generation import GenerationBundle
from valkeyrie.publication import PublicationStore, verify_sealed_generation


class IngestionError(RuntimeError):
    """Ingestion cannot advance safely."""


class CandidateBusyError(IngestionError):
    """Another worker holds the unexpired serialized ingestion lease."""


class StaleFenceError(IngestionError):
    """A worker no longer owns the candidate fencing token."""


class IngestionPendingError(IngestionError):
    """The bounded poll window ended while Bedrock still reported progress."""


class CandidateStatus(StrEnum):
    INGESTING = "INGESTING"
    INGESTED = "INGESTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class CandidateState:
    """One strongly consistent serialized-candidate item."""

    generation_id: str
    status: CandidateStatus
    revision: int
    fence: int
    attempt: int
    owner: str
    lease_expires_at: int
    ingestion_job_id: str | None = None
    documents_scanned: int | None = None
    failure: str | None = None


@dataclass(frozen=True)
class CandidateLease:
    """Worker capability bound to one generation and monotonic fence."""

    generation_id: str
    owner: str
    fence: int
    attempt: int
    lease_expires_at: int
    ingestion_job_id: str | None
    documents_scanned: int | None
    already_ingested: bool = False


class ConditionalCandidateStore(Protocol):
    """DynamoDB port for one candidate item.

    Reads must be strongly consistent and writes conditional.
    """

    def read_candidate(self) -> CandidateState | None: ...

    def compare_and_swap(
        self,
        expected_revision: int | None,
        replacement: CandidateState,
    ) -> bool:
        """Atomically replace only when revision equals expected, or the item is absent for None."""
        ...


class BedrockIngestionClient(Protocol):
    def start_ingestion_job(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_ingestion_job(self, **kwargs: object) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class IngestionResult:
    generation_id: str
    ingestion_job_id: str
    documents_scanned: int
    already_ingested: bool


_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_JOB_ID: Final = re.compile(r"^[A-Z0-9]{10}$")
_PENDING_STATUSES: Final = {"STARTING", "IN_PROGRESS", "STOPPING"}
_TERMINAL_FAILURE_STATUSES: Final = {"FAILED", "STOPPED"}
_MAX_CAS_ATTEMPTS: Final = 8


class CandidateCoordinator:
    """Own candidate transitions and enforce lease/fence checks around every write."""

    def __init__(self, store: ConditionalCandidateStore) -> None:
        if store is None or any(
            not callable(getattr(store, method, None))
            for method in ("read_candidate", "compare_and_swap")
        ):
            raise IngestionError("candidate store does not implement conditional state operations")
        self._store = store

    def claim(
        self,
        generation_id: str,
        owner: str,
        *,
        now_epoch: int,
        lease_seconds: int,
    ) -> CandidateLease:
        _validate_claim_inputs(generation_id, owner, now_epoch, lease_seconds)
        for _ in range(_MAX_CAS_ATTEMPTS):
            current = self._read_validated()
            if current is not None:
                same_generation = current.generation_id == generation_id
                if same_generation and current.status is CandidateStatus.INGESTED:
                    return _lease(current, already_ingested=True)
                if same_generation and current.status is CandidateStatus.FAILED:
                    raise IngestionError("failed candidate generation is explicitly non-retryable")
                if current.status is CandidateStatus.INGESTING:
                    if not same_generation:
                        raise CandidateBusyError(
                            "in-flight ingestion blocks cross-generation takeover until terminal"
                        )
                    if current.lease_expires_at > now_epoch:
                        if current.owner == owner:
                            return _lease(current)
                        raise CandidateBusyError(
                            "another worker holds the serialized ingestion lease"
                        )
                    attempt = current.attempt
                    job_id = current.ingestion_job_id
                else:
                    attempt = 1
                    job_id = None
                revision = current.revision + 1
                fence = current.fence + 1
                expected_revision: int | None = current.revision
            else:
                revision = 1
                fence = 1
                attempt = 1
                job_id = None
                expected_revision = None
            replacement_state = CandidateState(
                generation_id=generation_id,
                status=CandidateStatus.INGESTING,
                revision=revision,
                fence=fence,
                attempt=attempt,
                owner=owner,
                lease_expires_at=now_epoch + lease_seconds,
                ingestion_job_id=job_id,
            )
            if self._store.compare_and_swap(expected_revision, replacement_state):
                return _lease(replacement_state)
        raise IngestionError("candidate claim exceeded the conditional-write retry bound")

    def renew(
        self,
        lease: CandidateLease,
        *,
        now_epoch: int,
        lease_seconds: int,
    ) -> CandidateLease:
        _validate_epoch_and_lease(now_epoch, lease_seconds)
        for _ in range(_MAX_CAS_ATTEMPTS):
            current = self._owned_state(lease, now_epoch)
            replacement_state = replace(
                current,
                revision=current.revision + 1,
                lease_expires_at=now_epoch + lease_seconds,
            )
            if self._store.compare_and_swap(current.revision, replacement_state):
                return _lease(replacement_state)
        raise IngestionError("candidate renewal exceeded the conditional-write retry bound")

    def bind_job(
        self,
        lease: CandidateLease,
        job_id: str,
        *,
        now_epoch: int,
    ) -> CandidateLease:
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise IngestionError("Bedrock returned a malformed ingestion job ID")
        for _ in range(_MAX_CAS_ATTEMPTS):
            current = self._owned_state(lease, now_epoch)
            if current.ingestion_job_id is not None:
                if current.ingestion_job_id != job_id:
                    raise IngestionError("candidate is already bound to a different ingestion job")
                return _lease(current)
            replacement_state = replace(
                current,
                revision=current.revision + 1,
                ingestion_job_id=job_id,
            )
            if self._store.compare_and_swap(current.revision, replacement_state):
                return _lease(replacement_state)
        raise IngestionError("job binding exceeded the conditional-write retry bound")

    def complete(
        self,
        lease: CandidateLease,
        *,
        now_epoch: int,
        documents_scanned: int,
    ) -> None:
        if not isinstance(documents_scanned, int) or isinstance(documents_scanned, bool):
            raise IngestionError("document statistics are malformed")
        if documents_scanned < 0:
            raise IngestionError("document statistics cannot be negative")
        current = self._owned_state(lease, now_epoch)
        if current.ingestion_job_id is None:
            raise IngestionError("candidate cannot complete without an ingestion job")
        replacement_state = replace(
            current,
            status=CandidateStatus.INGESTED,
            revision=current.revision + 1,
            lease_expires_at=0,
            documents_scanned=documents_scanned,
            failure=None,
        )
        if not self._store.compare_and_swap(current.revision, replacement_state):
            raise StaleFenceError("candidate changed before successful completion")

    def fail(self, lease: CandidateLease, reason: str, *, now_epoch: int) -> None:
        if not isinstance(reason, str) or not reason or len(reason) > 500:
            raise IngestionError("candidate failure reason is invalid")
        current = self._owned_state(lease, now_epoch)
        replacement_state = replace(
            current,
            status=CandidateStatus.FAILED,
            revision=current.revision + 1,
            lease_expires_at=0,
            failure=reason,
        )
        if not self._store.compare_and_swap(current.revision, replacement_state):
            raise StaleFenceError("candidate changed before failure recording")

    def _read_validated(self) -> CandidateState | None:
        value = self._store.read_candidate()
        if value is not None:
            _validate_state(value)
        return value

    def _owned_state(self, lease: CandidateLease, now_epoch: int) -> CandidateState:
        if not isinstance(now_epoch, int) or isinstance(now_epoch, bool) or now_epoch < 0:
            raise IngestionError("current epoch is invalid")
        current = self._read_validated()
        if (
            current is None
            or current.generation_id != lease.generation_id
            or current.owner != lease.owner
            or current.fence != lease.fence
            or current.attempt != lease.attempt
            or current.status is not CandidateStatus.INGESTING
            or current.lease_expires_at <= now_epoch
        ):
            raise StaleFenceError("candidate lease or fencing token is stale")
        return current


def run_ingestion(
    publication_store: PublicationStore,
    candidate_store: ConditionalCandidateStore,
    bedrock: BedrockIngestionClient,
    bundle: GenerationBundle,
    *,
    knowledge_base_id: str,
    data_source_id: str,
    owner: str,
    now_epoch: Callable[[], int],
    sleep: Callable[[float], None],
    lease_seconds: int = 300,
    poll_interval_seconds: float = 5.0,
    max_polls: int = 120,
) -> IngestionResult:
    """Verify sealing, serialize the candidate, and poll Bedrock to successful completion."""
    verify_sealed_generation(publication_store, bundle)
    _validate_service_inputs(
        bedrock,
        knowledge_base_id,
        data_source_id,
        now_epoch,
        sleep,
        poll_interval_seconds,
        max_polls,
    )
    coordinator = CandidateCoordinator(candidate_store)
    lease = coordinator.claim(
        bundle.generation_id,
        owner,
        now_epoch=now_epoch(),
        lease_seconds=lease_seconds,
    )
    if lease.already_ingested:
        if lease.ingestion_job_id is None or lease.documents_scanned is None:
            raise IngestionError("ingested candidate state is incomplete")
        return IngestionResult(
            bundle.generation_id,
            lease.ingestion_job_id,
            lease.documents_scanned,
            True,
        )

    if lease.ingestion_job_id is None:
        response = bedrock.start_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            clientToken=(f"{bundle.generation_id.removeprefix('sha256:')}-{lease.attempt}"),
            description=f"Valkeyrie sealed generation {bundle.generation_id}",
        )
        job = _job_from_response(response)
        started_job_id = _job_id(job)
        lease = coordinator.bind_job(lease, started_job_id, now_epoch=now_epoch())
    job_id = lease.ingestion_job_id
    if job_id is None:
        raise IngestionError("candidate ingestion job binding is absent")

    for poll in range(max_polls):
        lease = coordinator.renew(
            lease,
            now_epoch=now_epoch(),
            lease_seconds=lease_seconds,
        )
        response = bedrock.get_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            ingestionJobId=job_id,
        )
        status: object = None
        try:
            job = _job_from_response(response)
            if _job_id(job) != job_id:
                raise IngestionError("Bedrock returned a different ingestion job")
            status = job.get("status")
            if status in _PENDING_STATUSES:
                if poll + 1 < max_polls:
                    sleep(poll_interval_seconds)
                continue
            if status in _TERMINAL_FAILURE_STATUSES:
                reason = f"Bedrock ingestion ended with {status}"
                coordinator.fail(lease, reason, now_epoch=now_epoch())
                raise IngestionError(reason)
            if status != "COMPLETE":
                raise IngestionError("Bedrock returned an unknown ingestion status")
            scanned, failed = _statistics(job)
            if failed != 0:
                reason = f"Bedrock ingestion reported {failed} failed documents"
                coordinator.fail(lease, reason, now_epoch=now_epoch())
                raise IngestionError(reason)
            coordinator.complete(
                lease,
                now_epoch=now_epoch(),
                documents_scanned=scanned,
            )
            return IngestionResult(bundle.generation_id, job_id, scanned, False)
        except IngestionError as error:
            if status not in _TERMINAL_FAILURE_STATUSES and status != "COMPLETE":
                _record_validation_failure(coordinator, lease, str(error), now_epoch())
            raise
    raise IngestionPendingError("Bedrock ingestion did not reach a terminal state within the bound")


def _record_validation_failure(
    coordinator: CandidateCoordinator,
    lease: CandidateLease,
    reason: str,
    now_epoch: int,
) -> None:
    coordinator.fail(lease, reason[:500], now_epoch=now_epoch)


def _job_from_response(response: object) -> Mapping[str, object]:
    if not isinstance(response, Mapping):
        raise IngestionError("Bedrock ingestion response is malformed")
    job = response.get("ingestionJob")
    if not isinstance(job, Mapping):
        raise IngestionError("Bedrock ingestion response has no job")
    return cast(Mapping[str, object], job)


def _job_id(job: Mapping[str, object]) -> str:
    value = job.get("ingestionJobId")
    if not isinstance(value, str) or not _JOB_ID.fullmatch(value):
        raise IngestionError("Bedrock returned a malformed ingestion job ID")
    return value


def _statistics(job: Mapping[str, object]) -> tuple[int, int]:
    statistics = job.get("statistics")
    if not isinstance(statistics, Mapping):
        raise IngestionError("completed ingestion has no statistics")
    scanned = statistics.get("numberOfDocumentsScanned")
    failed = statistics.get("numberOfDocumentsFailed")
    for value in (scanned, failed):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise IngestionError("completed ingestion statistics are malformed")
    return cast(int, scanned), cast(int, failed)


def _validate_claim_inputs(
    generation_id: str,
    owner: str,
    now_epoch: int,
    lease_seconds: int,
) -> None:
    if not isinstance(generation_id, str) or not _DIGEST.fullmatch(generation_id):
        raise IngestionError("candidate generation ID is malformed")
    if not isinstance(owner, str) or not owner or len(owner) > 128:
        raise IngestionError("candidate owner is invalid")
    _validate_epoch_and_lease(now_epoch, lease_seconds)


def _validate_epoch_and_lease(now_epoch: int, lease_seconds: int) -> None:
    if not isinstance(now_epoch, int) or isinstance(now_epoch, bool) or now_epoch < 0:
        raise IngestionError("current epoch is invalid")
    if (
        not isinstance(lease_seconds, int)
        or isinstance(lease_seconds, bool)
        or not 30 <= lease_seconds <= 3600
    ):
        raise IngestionError("lease duration is outside its bound")


def _validate_state(state: CandidateState) -> None:
    if not isinstance(state, CandidateState):
        raise IngestionError("candidate store returned the wrong runtime type")
    _validate_claim_inputs(state.generation_id, state.owner, max(state.lease_expires_at, 0), 30)
    if not isinstance(state.status, CandidateStatus):
        raise IngestionError("candidate status is invalid")
    if state.revision < 1 or state.fence < 1 or state.attempt < 1:
        raise IngestionError("candidate revision, fence, or attempt is invalid")
    if state.ingestion_job_id is not None and not _JOB_ID.fullmatch(state.ingestion_job_id):
        raise IngestionError("candidate ingestion job ID is malformed")
    if state.documents_scanned is not None and (
        not isinstance(state.documents_scanned, int)
        or isinstance(state.documents_scanned, bool)
        or state.documents_scanned < 0
    ):
        raise IngestionError("candidate document statistics are malformed")
    if state.failure is not None and (not state.failure or len(state.failure) > 500):
        raise IngestionError("candidate failure is malformed")


def _validate_service_inputs(
    bedrock: object,
    knowledge_base_id: str,
    data_source_id: str,
    now_epoch: object,
    sleep: object,
    poll_interval_seconds: float,
    max_polls: int,
) -> None:
    if bedrock is None or any(
        not callable(getattr(bedrock, method, None))
        for method in ("start_ingestion_job", "get_ingestion_job")
    ):
        raise IngestionError("Bedrock client does not implement ingestion operations")
    if not isinstance(knowledge_base_id, str) or not _JOB_ID.fullmatch(knowledge_base_id):
        raise IngestionError("knowledge base ID is malformed")
    if not isinstance(data_source_id, str) or not _JOB_ID.fullmatch(data_source_id):
        raise IngestionError("data source ID is malformed")
    if not callable(now_epoch) or not callable(sleep):
        raise IngestionError("clock and sleeper must be callable")
    if (
        not isinstance(poll_interval_seconds, (int, float))
        or isinstance(poll_interval_seconds, bool)
        or not 0 <= poll_interval_seconds <= 60
    ):
        raise IngestionError("poll interval is outside its bound")
    if not isinstance(max_polls, int) or isinstance(max_polls, bool) or not 1 <= max_polls <= 1000:
        raise IngestionError("poll count is outside its bound")


def _lease(state: CandidateState, *, already_ingested: bool = False) -> CandidateLease:
    return CandidateLease(
        generation_id=state.generation_id,
        owner=state.owner,
        fence=state.fence,
        attempt=state.attempt,
        lease_expires_at=state.lease_expires_at,
        ingestion_job_id=state.ingestion_job_id,
        documents_scanned=state.documents_scanned,
        already_ingested=already_ingested,
    )
