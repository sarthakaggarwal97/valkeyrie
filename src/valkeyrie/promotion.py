"""Conditional generation activation, request pinning, and retained rollback."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, Protocol

from valkeyrie.evaluations import (
    EvaluationError,
    EvaluationSuite,
    verify_evaluation_report,
)
from valkeyrie.retrieval import (
    BedrockRetrievalClient,
    GenerationAvailability,
    GenerationRegistry,
    PinnedGeneration,
    RetrievalError,
    pin_generation,
    run_candidate_retrieval_smoke,
)
from valkeyrie.retrieval_config import FrozenRetrievalConfiguration


class PromotionError(RuntimeError):
    """Activation or rollback cannot satisfy its approval, state, or CAS contract."""


@dataclass(frozen=True)
class ActiveGeneration:
    """The singleton active-generation item updated only through compare-and-swap."""

    generation_id: str
    revision: int
    evaluation_report_id: str
    activated_at: str


@dataclass(frozen=True)
class ProtectedApproval:
    """Human approval loaded from a trusted protected-workflow repository."""

    approval_id: str
    action: Literal["activate", "rollback"]
    generation_id: str
    expected_active_generation: str | None
    evaluation_report_id: str
    approver: str
    approved_at: str


class ApprovalRegistry(Protocol):
    def consume_approval(self, approval_id: str) -> ProtectedApproval | None:
        """Atomically return and permanently consume one protected approval, or return None."""
        ...


class PromotionStore(GenerationRegistry, Protocol):
    def read_active(self) -> ActiveGeneration | None: ...

    def compare_and_swap_active(
        self,
        expected_active_revision: int | None,
        expected_generation: GenerationAvailability,
        replacement: ActiveGeneration,
    ) -> bool:
        """Atomically match active revision and exact generation state before replacement."""
        ...


_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPORT_ID: Final = re.compile(r"^eval_[0-9a-f]{64}$")
_TIMESTAMP: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)


def pin_active_generation(store: PromotionStore) -> PinnedGeneration:
    """Capture the active identity once; later active changes cannot alter this pin."""
    _validate_store(store)
    active = store.read_active()
    if active is None:
        raise PromotionError("active generation is absent")
    _validate_active(active)
    try:
        return pin_generation(store, active.generation_id)
    except RetrievalError as error:
        raise PromotionError(f"active generation cannot be pinned: {error}") from error


def activate_candidate(
    store: PromotionStore,
    suite: EvaluationSuite,
    report: Mapping[str, object],
    approval_registry: ApprovalRegistry | None,
    *,
    approval_id: str | None,
    retrieval_client: BedrockRetrievalClient,
    retrieval_config: FrozenRetrievalConfiguration,
    knowledge_base_id: str,
    expected_active_generation: str | None,
    activated_at: str,
) -> ActiveGeneration:
    """Smoke and CAS-activate one exact qualified generation state.

    ``approval_id`` is optional so an unattended corpus refresh can promote a
    generation without a human token. The quality gates are unchanged and all still
    apply: the evaluation report must pass, the stored candidate must record that
    exact passing report, a live generation-filtered retrieval smoke must succeed,
    and the active pointer still moves only through compare-and-swap. Supplying an
    approval adds the one-time human token on top of those.
    """
    _validate_store(store)
    generation_id, report_id = _passing_report(suite, report)
    record = _generation(store, generation_id)
    _require_activation_candidate(record, report_id)
    _validate_timestamp(activated_at, "activation timestamp")
    current = _expected_active(store, expected_active_generation)
    try:
        run_candidate_retrieval_smoke(
            store,
            retrieval_client,
            retrieval_config,
            suite,
            knowledge_base_id=knowledge_base_id,
            generation_id=generation_id,
        )
    except RetrievalError as error:
        raise PromotionError(f"candidate generation-filtered smoke failed: {error}") from error
    # Only an explicitly supplied approval id requests the human gate. The release path
    # always constructs an approval registry, so keying off the registry would force
    # consumption of a None id and fail every unattended activation.
    if approval_id is not None:
        _consume_approval(
            approval_registry,
            approval_id,
            "activate",
            generation_id,
            expected_active_generation,
            report_id,
        )
    replacement = ActiveGeneration(
        generation_id,
        1 if current is None else current.revision + 1,
        report_id,
        activated_at,
    )
    expected_revision = None if current is None else current.revision
    if not store.compare_and_swap_active(expected_revision, record, replacement):
        raise PromotionError("active or candidate generation state changed during activation")
    _require_active_ownership(store, replacement, "activation")
    return replacement


def _passing_report(
    suite: EvaluationSuite,
    report: Mapping[str, object],
) -> tuple[str, str]:
    if not isinstance(report, Mapping):
        raise PromotionError("candidate evaluation report is malformed")
    try:
        verify_evaluation_report(report, suite)
    except EvaluationError as error:
        raise PromotionError(f"candidate evaluation report is invalid: {error}") from error
    generation_id = report.get("candidate_revision")
    report_id = report.get("report_id")
    if not isinstance(generation_id, str) or not _DIGEST.fullmatch(generation_id):
        raise PromotionError("candidate report generation is malformed")
    if not isinstance(report_id, str) or not _REPORT_ID.fullmatch(report_id):
        raise PromotionError("candidate report identity is malformed")
    if report.get("result") != "pass":
        raise PromotionError("candidate evaluation did not pass")
    return generation_id, report_id


def _generation(store: PromotionStore, generation_id: object) -> GenerationAvailability:
    if not isinstance(generation_id, str) or not _DIGEST.fullmatch(generation_id):
        raise PromotionError("generation ID is malformed")
    record = store.get_generation(generation_id)
    if record is None:
        raise PromotionError("generation is unknown")
    if not isinstance(record, GenerationAvailability) or record.generation_id != generation_id:
        raise PromotionError("generation registry returned a malformed or mismatched record")
    try:
        pin_generation(store, generation_id)
    except RetrievalError as error:
        raise PromotionError(f"generation is not safely retrievable: {error}") from error
    return record


def _require_activation_candidate(record: GenerationAvailability, report_id: str) -> None:
    if not record.evaluation_passed:
        raise PromotionError("candidate generation has not passed evaluation")
    if record.evaluation_report_id != report_id:
        raise PromotionError("candidate state and evaluation report do not match")


def _expected_active(
    store: PromotionStore,
    expected_generation_id: str | None,
) -> ActiveGeneration | None:
    if expected_generation_id is not None and (
        not isinstance(expected_generation_id, str) or not _DIGEST.fullmatch(expected_generation_id)
    ):
        raise PromotionError("expected active generation is malformed")
    current = store.read_active()
    if current is not None:
        _validate_active(current)
    observed = None if current is None else current.generation_id
    if observed != expected_generation_id:
        raise PromotionError("active generation does not match the caller's expected state")
    return current


def _validate_active(active: object) -> None:
    if not isinstance(active, ActiveGeneration):
        raise PromotionError("active generation state is malformed")
    if not _DIGEST.fullmatch(active.generation_id):
        raise PromotionError("active generation ID is malformed")
    if (
        not isinstance(active.revision, int)
        or isinstance(active.revision, bool)
        or active.revision < 1
    ):
        raise PromotionError("active generation revision is malformed")
    if not _REPORT_ID.fullmatch(active.evaluation_report_id):
        raise PromotionError("active generation report ID is malformed")
    _validate_timestamp(active.activated_at, "active generation timestamp")


def _consume_approval(
    registry: ApprovalRegistry | None,
    approval_id: object,
    action: Literal["activate", "rollback"],
    generation_id: str,
    expected_active_generation: str | None,
    report_id: str,
) -> ProtectedApproval:
    if registry is None or not callable(getattr(registry, "consume_approval", None)):
        raise PromotionError("trusted approval registry does not implement consumption")
    if not isinstance(approval_id, str) or not re.fullmatch(r"approval_[a-z0-9-]+", approval_id):
        raise PromotionError("protected approval ID is malformed")
    approval = registry.consume_approval(approval_id)
    if approval is None:
        raise PromotionError("protected approval is absent, invalid, or already consumed")
    if not isinstance(approval, ProtectedApproval) or approval.approval_id != approval_id:
        raise PromotionError("trusted approval registry returned malformed evidence")
    if (
        approval.action != action
        or approval.generation_id != generation_id
        or approval.expected_active_generation != expected_active_generation
        or approval.evaluation_report_id != report_id
    ):
        raise PromotionError("protected approval does not match the exact transition")
    if (
        not isinstance(approval.approver, str)
        or not approval.approver
        or len(approval.approver) > 128
    ):
        raise PromotionError("protected approval has an invalid approver")
    _validate_timestamp(approval.approved_at, "approval timestamp")
    return approval


def _require_active_ownership(
    store: PromotionStore,
    expected: ActiveGeneration,
    action: str,
) -> None:
    observed = store.read_active()
    if observed != expected:
        raise PromotionError(f"active generation changed after successful {action}")


def _is_calendar_timestamp(value: str) -> bool:
    """Reject impossible dates and times the shape regex admits.

    The regex pins digit layout only, so 2026-99-99T99:99:99Z matches it. Parsing is what
    establishes the value names a real instant.
    """
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _validate_timestamp(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not _TIMESTAMP.fullmatch(value)
        or not _is_calendar_timestamp(value)
    ):
        raise PromotionError(f"{label} is malformed")


def _validate_store(store: object) -> None:
    required = ("get_generation", "read_active", "compare_and_swap_active")
    if store is None or any(not callable(getattr(store, method, None)) for method in required):
        raise PromotionError("promotion store does not implement conditional generation state")
