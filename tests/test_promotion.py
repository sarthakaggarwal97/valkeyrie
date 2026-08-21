from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from tests.helpers import load_yaml
from tests.test_evaluations import _passing_model_runs, _passing_retrieval_results
from valkeyrie.evaluations import EvaluationSuite, evaluate_candidate, load_evaluation_suite
from valkeyrie.promotion import (
    ActiveGeneration,
    PromotionError,
    ProtectedApproval,
    pin_active_generation,
)
from valkeyrie.promotion import (
    activate_candidate as activate_candidate_impl,
)
from valkeyrie.retrieval import GenerationAvailability
from valkeyrie.retrieval_config import FrozenRetrievalConfiguration, load_retrieval_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = cast(dict[str, Any], load_yaml(ROOT / "src/valkeyrie/schemas/contracts.schema.json"))


def _filter_values(value: object, key: str) -> list[str]:
    if not isinstance(value, dict):
        return []
    equals = value.get("equals")
    if isinstance(equals, dict) and equals.get("key") == key:
        candidate = equals.get("value")
        return [candidate] if isinstance(candidate, str) else []
    values: list[str] = []
    for operation in ("andAll", "orAll"):
        children = value.get(operation)
        if isinstance(children, list):
            for child in children:
                values.extend(_filter_values(child, key))
    return values


GEN_A = "sha256:" + "a" * 64
GEN_B = "sha256:" + "b" * 64
GEN_C = "sha256:" + "c" * 64
NOW = "2026-08-19T03:00:00Z"
LATER = "2026-08-19T03:05:00Z"
CONFIG: FrozenRetrievalConfiguration = load_retrieval_config(ROOT / "retrieval-config.yaml")


class MemoryPromotionStore:
    def __init__(
        self,
        generations: list[GenerationAvailability],
        active: ActiveGeneration | None = None,
    ) -> None:
        self.generations = {record.generation_id: record for record in generations}
        self.active = active
        self.writes: list[ActiveGeneration] = []
        self.reject_writes = 0
        self.before_cas: Any = None
        self.after_cas: Any = None

    def get_generation(self, generation_id: str) -> GenerationAvailability | None:
        return self.generations.get(generation_id)

    def read_active(self) -> ActiveGeneration | None:
        return self.active

    def compare_and_swap_active(
        self,
        expected_revision: int | None,
        expected_generation: GenerationAvailability,
        replacement: ActiveGeneration,
    ) -> bool:
        if self.before_cas is not None:
            callback = self.before_cas
            self.before_cas = None
            callback()
        if self.reject_writes:
            self.reject_writes -= 1
            return False
        current_revision = None if self.active is None else self.active.revision
        if (
            current_revision != expected_revision
            or self.generations.get(expected_generation.generation_id) != expected_generation
        ):
            return False
        self.active = replacement
        self.writes.append(replacement)
        if self.after_cas is not None:
            callback = self.after_cas
            self.after_cas = None
            callback()
        return True


@pytest.fixture(scope="module")
def suite() -> EvaluationSuite:
    return load_evaluation_suite(ROOT)


def _report(suite: EvaluationSuite, generation_id: str) -> dict[str, object]:
    return evaluate_candidate(
        suite,
        candidate_revision=generation_id,
        started_at="2026-08-19T02:00:00Z",
        completed_at="2026-08-19T02:05:00Z",
        model_runs=_passing_model_runs(suite),
        retrieval_results=_passing_retrieval_results(suite),
    )


def _record(
    generation_id: str,
    report_id: str,
    **changes: object,
) -> GenerationAvailability:
    values: dict[str, object] = {
        "generation_id": generation_id,
        "revision": 1,
        "sealed": True,
        "available": True,
        "ingested": True,
        "retrievable": True,
        "evaluation_passed": True,
        "retained": True,
        "evaluation_report_id": report_id,
    }
    values.update(changes)
    return GenerationAvailability(**values)  # type: ignore[arg-type]


class MemoryApprovalRegistry:
    def __init__(self, approvals: tuple[ProtectedApproval, ...]) -> None:
        self.approvals = {approval.approval_id: approval for approval in approvals}
        self.consumed: list[str] = []

    def consume_approval(self, approval_id: str) -> ProtectedApproval | None:
        approval = self.approvals.pop(approval_id, None)
        if approval is not None:
            self.consumed.append(approval_id)
        return approval


class SmokeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"retrievalResults": []}


def _approval(action: str, generation_id: str, report_id: str) -> ProtectedApproval:
    return ProtectedApproval(
        approval_id=f"approval_{action}-{generation_id[-8:]}",
        action=cast(Any, action),
        generation_id=generation_id,
        expected_active_generation=None,
        evaluation_report_id=report_id,
        approver="sarthagg",
        approved_at=NOW,
    )


def activate_candidate(
    store: MemoryPromotionStore,
    suite: EvaluationSuite,
    report: dict[str, object],
    approval: ProtectedApproval,
    *,
    expected_active_generation: str | None,
    activated_at: str,
) -> ActiveGeneration:
    bound = replace(approval, expected_active_generation=expected_active_generation)
    return activate_candidate_impl(
        store,
        suite,
        report,
        MemoryApprovalRegistry((bound,)),
        approval_id=bound.approval_id,
        retrieval_client=SmokeClient(),
        retrieval_config=CONFIG,
        knowledge_base_id="ABCDEFGHIJ",
        expected_active_generation=expected_active_generation,
        activated_at=activated_at,
    )


def activate_without_approval(
    store: MemoryPromotionStore,
    suite: EvaluationSuite,
    report: dict[str, object],
    *,
    expected_active_generation: str | None,
    activated_at: str,
) -> ActiveGeneration:
    """Activate the way an unattended weekly refresh does, with no human token."""
    return activate_candidate_impl(
        store,
        suite,
        report,
        None,
        approval_id=None,
        retrieval_client=SmokeClient(),
        retrieval_config=CONFIG,
        knowledge_base_id="ABCDEFGHIJ",
        expected_active_generation=expected_active_generation,
        activated_at=activated_at,
    )


def test_unattended_activation_needs_no_approval_but_still_requires_the_cas(
    suite: EvaluationSuite,
) -> None:
    report = _report(suite, GEN_A)
    report_id = cast(str, report["report_id"])
    store = MemoryPromotionStore([_record(GEN_A, report_id)])

    active = activate_without_approval(
        store, suite, report, expected_active_generation=None, activated_at=NOW
    )

    assert active == ActiveGeneration(GEN_A, 1, report_id, NOW)
    assert store.active == active
    assert store.writes == [active]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"retrievable": False}, "not safely retrievable"),
        ({"evaluation_passed": False}, "has not passed evaluation"),
    ],
)
def test_unattended_activation_still_refuses_an_unqualified_generation(
    suite: EvaluationSuite, changes: dict[str, object], message: str
) -> None:
    # Removing the human token must not remove the quality gates: without an approval the
    # evaluation result is the only thing standing between a bad corpus and live traffic.
    report = _report(suite, GEN_A)
    report_id = cast(str, report["report_id"])
    store = MemoryPromotionStore([_record(GEN_A, report_id, **changes)])

    with pytest.raises(PromotionError, match=message):
        activate_without_approval(
            store, suite, report, expected_active_generation=None, activated_at=NOW
        )

    assert store.active is None
    assert store.writes == []


def test_passing_candidate_activates_only_through_expected_cas_and_approval(
    suite: EvaluationSuite,
) -> None:
    report = _report(suite, GEN_A)
    report_id = cast(str, report["report_id"])
    store = MemoryPromotionStore([_record(GEN_A, report_id)])

    active = activate_candidate(
        store,
        suite,
        report,
        _approval("activate", GEN_A, report_id),
        expected_active_generation=None,
        activated_at=NOW,
    )

    assert active == ActiveGeneration(GEN_A, 1, report_id, NOW)
    assert store.active == active
    assert store.writes == [active]
    assert pin_active_generation(store).generation_id == GEN_A


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sealed": False}, "not safely retrievable"),
        ({"available": False}, "not safely retrievable"),
        ({"ingested": False}, "not safely retrievable"),
        ({"retrievable": False}, "not safely retrievable"),
        ({"evaluation_passed": False}, "has not passed evaluation"),
    ],
)
def test_unqualified_candidate_never_activates(
    suite: EvaluationSuite,
    changes: dict[str, object],
    message: str,
) -> None:
    report = _report(suite, GEN_A)
    report_id = cast(str, report["report_id"])
    store = MemoryPromotionStore([_record(GEN_A, report_id, **changes)])

    with pytest.raises(PromotionError, match=message):
        activate_candidate(
            store,
            suite,
            report,
            _approval("activate", GEN_A, report_id),
            expected_active_generation=None,
            activated_at=NOW,
        )
    assert store.writes == []


def test_report_approval_and_candidate_state_must_match(suite: EvaluationSuite) -> None:
    report = _report(suite, GEN_A)
    report_id = cast(str, report["report_id"])

    mismatched_state = MemoryPromotionStore([_record(GEN_A, "eval_" + "f" * 64)])
    with pytest.raises(PromotionError, match="state and evaluation report"):
        activate_candidate(
            mismatched_state,
            suite,
            report,
            _approval("activate", GEN_A, report_id),
            expected_active_generation=None,
            activated_at=NOW,
        )

    valid_state = MemoryPromotionStore([_record(GEN_A, report_id)])
    with pytest.raises(PromotionError, match="exact transition"):
        activate_candidate(
            valid_state,
            suite,
            report,
            _approval("rollback", GEN_A, report_id),
            expected_active_generation=None,
            activated_at=NOW,
        )
    assert mismatched_state.writes == valid_state.writes == []


def test_stale_or_racing_activation_fails_without_substitution(suite: EvaluationSuite) -> None:
    report = _report(suite, GEN_B)
    report_id = cast(str, report["report_id"])
    current = ActiveGeneration(GEN_A, 4, "eval_" + "a" * 64, NOW)
    store = MemoryPromotionStore([_record(GEN_B, report_id)], current)

    with pytest.raises(PromotionError, match="expected state"):
        activate_candidate(
            store,
            suite,
            report,
            _approval("activate", GEN_B, report_id),
            expected_active_generation=GEN_C,
            activated_at=LATER,
        )
    store.reject_writes = 1
    with pytest.raises(PromotionError, match="changed during"):
        activate_candidate(
            store,
            suite,
            report,
            _approval("activate", GEN_B, report_id),
            expected_active_generation=GEN_A,
            activated_at=LATER,
        )
    assert store.active == current
    assert store.writes == []


def test_inflight_pin_remains_original_after_activation(suite: EvaluationSuite) -> None:
    report_a = _report(suite, GEN_A)
    report_b = _report(suite, GEN_B)
    report_a_id = cast(str, report_a["report_id"])
    report_b_id = cast(str, report_b["report_id"])
    store = MemoryPromotionStore(
        [_record(GEN_A, report_a_id), _record(GEN_B, report_b_id)],
        ActiveGeneration(GEN_A, 1, report_a_id, NOW),
    )
    inflight = pin_active_generation(store)

    activate_candidate(
        store,
        suite,
        report_b,
        _approval("activate", GEN_B, report_b_id),
        expected_active_generation=GEN_A,
        activated_at=LATER,
    )

    assert inflight.generation_id == GEN_A
    assert pin_active_generation(store).generation_id == GEN_B
    assert not hasattr(store, "delete_generation")


def test_state_and_approval_values_match_shared_contracts(suite: EvaluationSuite) -> None:
    report = _report(suite, GEN_A)
    report_id = cast(str, report["report_id"])
    availability = _record(GEN_A, report_id)
    active = ActiveGeneration(GEN_A, 1, report_id, NOW)
    approval = _approval("activate", GEN_A, report_id)
    values = {
        "generation_availability": asdict(availability),
        "active_generation": asdict(active),
        "protected_approval": asdict(approval),
    }

    for contract, value in values.items():
        Draft202012Validator(
            {
                "$schema": SCHEMA["$schema"],
                "$defs": SCHEMA["$defs"],
                "$ref": f"#/$defs/{contract}",
            }
        ).validate(value)


def test_activation_runs_generation_filtered_smoke_before_consuming_approval(
    suite: EvaluationSuite,
) -> None:
    report = _report(suite, GEN_A)
    report_id = cast(str, report["report_id"])
    store = MemoryPromotionStore([_record(GEN_A, report_id)])
    approval = _approval("activate", GEN_A, report_id)
    approvals = MemoryApprovalRegistry((approval,))

    class LeakingSmokeClient(SmokeClient):
        def retrieve(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(kwargs)
            return {
                "retrievalResults": [
                    {
                        "content": {"text": "wrong generation"},
                        "location": {"type": "S3"},
                        "metadata": {"generation_id": GEN_B},
                        "score": 0.5,
                    }
                ]
            }

    client = LeakingSmokeClient()
    with pytest.raises(PromotionError, match="generation-filtered smoke failed"):
        activate_candidate_impl(
            store,
            suite,
            report,
            approvals,
            approval_id=approval.approval_id,
            retrieval_client=client,
            retrieval_config=CONFIG,
            knowledge_base_id="ABCDEFGHIJ",
            expected_active_generation=None,
            activated_at=NOW,
        )
    assert client.calls
    assert all(
        _filter_values(
            cast(dict[str, Any], call["retrievalConfiguration"])["vectorSearchConfiguration"][
                "filter"
            ],
            "generation_id",
        )
        == [GEN_A]
        for call in client.calls
    )
    assert approvals.consumed == []
    assert store.writes == []


def test_protected_approval_is_trusted_one_time_evidence(suite: EvaluationSuite) -> None:
    report = _report(suite, GEN_A)
    report_id = cast(str, report["report_id"])
    store = MemoryPromotionStore([_record(GEN_A, report_id)])
    approval = _approval("activate", GEN_A, report_id)
    approvals = MemoryApprovalRegistry((approval,))

    activate_candidate_impl(
        store,
        suite,
        report,
        approvals,
        approval_id=approval.approval_id,
        retrieval_client=SmokeClient(),
        retrieval_config=CONFIG,
        knowledge_base_id="ABCDEFGHIJ",
        expected_active_generation=None,
        activated_at=NOW,
    )
    assert approvals.consumed == [approval.approval_id]

    with pytest.raises(PromotionError, match="already consumed"):
        activate_candidate_impl(
            store,
            suite,
            report,
            approvals,
            approval_id=approval.approval_id,
            retrieval_client=SmokeClient(),
            retrieval_config=CONFIG,
            knowledge_base_id="ABCDEFGHIJ",
            expected_active_generation=GEN_A,
            activated_at=LATER,
        )


def test_generation_lifecycle_change_before_cas_blocks_activation(
    suite: EvaluationSuite,
) -> None:
    report = _report(suite, GEN_A)
    report_id = cast(str, report["report_id"])
    original = _record(GEN_A, report_id)
    store = MemoryPromotionStore([original])

    def make_unavailable() -> None:
        store.generations[GEN_A] = replace(original, revision=2, retrievable=False)

    store.before_cas = make_unavailable
    with pytest.raises(PromotionError, match="candidate generation state changed"):
        activate_candidate(
            store,
            suite,
            report,
            _approval("activate", GEN_A, report_id),
            expected_active_generation=None,
            activated_at=NOW,
        )
    assert store.active is None
