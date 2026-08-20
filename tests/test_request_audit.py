from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from tests.helpers import load_yaml
from tests.test_evaluations import _passing_model_runs, _passing_retrieval_results
from tests.test_evidence import _bundle, _package
from valkeyrie.answer_models import (
    AnswerModelCandidate,
    AnswerModelProfile,
    AnswerModelSelection,
    InferenceConfiguration,
    create_candidate_profiles,
    load_answer_model_inventory,
    select_answer_model,
)
from valkeyrie.evaluations import EvaluationSuite, evaluate_candidate, load_evaluation_suite
from valkeyrie.evidence import EvidencePackage
from valkeyrie.generation import GenerationBundle
from valkeyrie.prompts import PromptPackage, load_prompt_package
from valkeyrie.request_audit import (
    LiveObservation,
    RequestAuditError,
    RequestAuditRecord,
    RequestClaim,
    claim_request,
    complete_request,
    create_live_observation,
    live_observation_value,
    recover_request,
    request_audit_value,
    request_content_digest,
    resolve_pinned_execution,
)
from valkeyrie.retrieval_config import load_retrieval_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = cast(dict[str, Any], load_yaml(ROOT / "src/valkeyrie/schemas/contracts.schema.json"))
REQUEST = "req_a-05-example"
QUESTION = "What does the GET command return in Valkey 9.0.0?"
APPLICATION = f"sha256:{'ab' * 32}"
CURRENT_APPLICATION = f"sha256:{'cd' * 32}"
STARTED = "2026-08-19T04:00:00Z"
EARLY = "2026-08-19T04:04:59Z"
RETRIED = "2026-08-19T04:05:00Z"
COMPLETED = "2026-08-19T04:10:00Z"
OBSERVED = "2026-08-19T03:59:00Z"
OBSERVATION_URL = "https://api.github.com/repos/valkey-io/valkey/releases/latest"
PAYLOAD: dict[str, object] = {
    "tag_name": "9.0.0",
    "assets": [{"name": "valkey-9.0.0.tar.gz", "size": 1234}],
}
EVALUATION_STARTED = "2026-08-19T03:00:00Z"
EVALUATION_COMPLETED = "2026-08-19T03:10:00Z"


class InMemoryRequestAuditStore:
    """Conditional store whose implementation enforces terminal immutability."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, RequestAuditRecord] = {}

    def get_request(self, request_id: str) -> RequestAuditRecord | None:
        with self._lock:
            return self._records.get(request_id)

    def put_request_if_absent(self, record: RequestAuditRecord) -> RequestAuditRecord | None:
        with self._lock:
            existing = self._records.get(record.pin.request_id)
            if existing is not None:
                return existing
            self._records[record.pin.request_id] = record
            return None

    def compare_and_swap_nonterminal(
        self,
        expected: RequestAuditRecord,
        replacement: RequestAuditRecord,
    ) -> bool:
        with self._lock:
            current = self._records.get(expected.pin.request_id)
            if current != expected or current.outcome is not None or expected.outcome is not None:
                return False
            if replacement.pin != expected.pin or replacement.revision != expected.revision + 1:
                return False
            if replacement.outcome is None:
                valid = (
                    replacement.fence == expected.fence + 1
                    and replacement.owner != expected.owner
                    and replacement.completed_at is None
                )
            else:
                valid = (
                    replacement.fence == expected.fence
                    and replacement.owner == expected.owner
                    and replacement.lease_expires_at == expected.lease_expires_at
                    and replacement.completed_at is not None
                )
            if not valid:
                return False
            self._records[expected.pin.request_id] = replacement
            return True

    def force(self, record: RequestAuditRecord) -> None:
        """Test-only corruption injection bypassing every conditional write."""
        with self._lock:
            self._records[record.pin.request_id] = record


class AbsentReadStore(InMemoryRequestAuditStore):
    """Force every claim through the conditional-insert race path."""

    def get_request(self, request_id: str) -> RequestAuditRecord | None:
        return None


@pytest.fixture(scope="module")
def bundle() -> GenerationBundle:
    return _bundle(load_retrieval_config(ROOT / "retrieval-config.yaml"))


@pytest.fixture(scope="module")
def package(bundle: GenerationBundle) -> EvidencePackage:
    return _package(bundle)


@pytest.fixture(scope="module")
def suite() -> EvaluationSuite:
    return load_evaluation_suite(ROOT)


@pytest.fixture(scope="module")
def prompt_package() -> PromptPackage:
    return load_prompt_package(ROOT)


@pytest.fixture(scope="module")
def candidates() -> tuple[AnswerModelCandidate, ...]:
    return load_answer_model_inventory(ROOT / "answer-models.yaml")


@pytest.fixture(scope="module")
def profiles(
    suite: EvaluationSuite,
    prompt_package: PromptPackage,
    bundle: GenerationBundle,
    candidates: tuple[AnswerModelCandidate, ...],
) -> tuple[AnswerModelProfile, ...]:
    return create_candidate_profiles(
        candidates,
        prompt_revision=prompt_package.prompt_revision,
        corpus_generation=bundle.generation_id,
        evaluation_suite_revision=suite.revision,
    )


@pytest.fixture(scope="module")
def reports(
    suite: EvaluationSuite,
    profiles: tuple[AnswerModelProfile, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(_report(suite, profile) for profile in profiles)


@pytest.fixture(scope="module")
def selection(
    suite: EvaluationSuite,
    profiles: tuple[AnswerModelProfile, ...],
    reports: tuple[dict[str, object], ...],
) -> AnswerModelSelection:
    return select_answer_model(suite, profiles, reports)


@pytest.fixture(scope="module")
def current_bundle() -> GenerationBundle:
    return _bundle(load_retrieval_config(ROOT / "retrieval-config.yaml"), commit="4" * 40)


@pytest.fixture(scope="module")
def current_package(current_bundle: GenerationBundle) -> EvidencePackage:
    return _package(current_bundle)


def _report(suite: EvaluationSuite, profile: AnswerModelProfile) -> dict[str, object]:
    return evaluate_candidate(
        suite,
        candidate_revision=profile.profile_revision,
        started_at=EVALUATION_STARTED,
        completed_at=EVALUATION_COMPLETED,
        model_runs=_passing_model_runs(suite),
        retrieval_results=_passing_retrieval_results(suite),
    )


def _observation(payload: object = PAYLOAD) -> LiveObservation:
    return create_live_observation(
        observed_at=OBSERVED,
        source_url=OBSERVATION_URL,
        object_type="release",
        payload=payload,
    )


def _claim(
    store: InMemoryRequestAuditStore,
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
    *,
    root: Path = ROOT,
    request_id: str = REQUEST,
    question: str = QUESTION,
    application_revision: str = APPLICATION,
    owner: str = "worker-1",
    now: str = STARTED,
    lease_duration_seconds: int = 300,
    live_observations: tuple[LiveObservation, ...] = (),
) -> RequestClaim:
    return claim_request(
        store,
        root,
        suite,
        reports,
        selection,
        bundle,
        package,
        request_id=request_id,
        question=question,
        application_revision=application_revision,
        owner=owner,
        now=now,
        lease_duration_seconds=lease_duration_seconds,
        live_observations=live_observations,
    )


def _validator(contract: str) -> Draft202012Validator:
    return Draft202012Validator(
        {
            "$schema": SCHEMA["$schema"],
            "$defs": SCHEMA["$defs"],
            "$ref": f"#/$defs/{contract}",
        },
        format_checker=FormatChecker(),
    )


def test_first_claim_pins_the_exact_executable_plan_and_lease(
    suite: EvaluationSuite,
    prompt_package: PromptPackage,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    observation = _observation()
    claim = _claim(
        store, suite, reports, selection, bundle, package, live_observations=(observation,)
    )
    assert claim.pinned
    pin = claim.record.pin
    execution = pin.execution
    assert pin.request_id == REQUEST
    assert pin.request_content_digest == request_content_digest(QUESTION)
    assert execution.application_revision == APPLICATION
    assert execution.invocation.input.prompts == prompt_package.templates
    assert execution.invocation.input.question == QUESTION
    assert execution.invocation.input.evidence == package
    assert execution.invocation.profile == selection.profile
    assert execution.invocation.profile.inference == selection.profile.inference
    assert pin.generation_id == bundle.generation_id
    assert pin.static_evidence_ids == tuple(record.evidence_id for record in package.records)
    assert pin.live_observations == (observation,)
    assert pin.started_at == STARTED
    assert (claim.record.revision, claim.record.fence) == (1, 1)
    assert claim.record.owner == "worker-1"
    assert claim.record.lease_expires_at == RETRIED
    assert claim.record.outcome is None
    assert claim.record.completed_at is None
    assert store.get_request(REQUEST) == claim.record


def test_retry_resolves_exact_stored_model_input_target_and_settings_despite_current_changes(
    tmp_path: Path,
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
    current_bundle: GenerationBundle,
    current_package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    first = _claim(store, suite, reports, selection, bundle, package)
    different_inference = InferenceConfiguration(1, 1.0, 0.1, None)
    different_profile = replace(
        selection.profile,
        model_revision="different.model:1",
        inference=different_inference,
        inference_config_revision=f"sha256:{'1' * 64}",
        profile_revision=f"sha256:{'2' * 64}",
    )
    different_selection = replace(selection, profile=different_profile)

    retry = _claim(
        store,
        suite,
        reports,
        different_selection,
        current_bundle,
        current_package,
        root=tmp_path,
        application_revision=CURRENT_APPLICATION,
        owner="worker-current",
        now=RETRIED,
    )
    execution = resolve_pinned_execution(store, request_id=REQUEST, question=QUESTION)

    assert not retry.pinned
    assert retry.record == first.record
    assert execution is first.record.pin.execution
    assert execution.invocation.input.prompts == first.record.pin.execution.invocation.input.prompts
    assert execution.invocation.input.question == QUESTION
    assert execution.invocation.input.evidence == package
    assert execution.invocation.input.evidence != current_package
    assert execution.invocation.profile == selection.profile
    assert execution.invocation.profile.model_revision != different_profile.model_revision
    assert execution.invocation.profile.inference != different_inference
    assert execution.application_revision == APPLICATION
    assert execution.application_revision != CURRENT_APPLICATION

    with pytest.raises(RequestAuditError, match="different request content"):
        resolve_pinned_execution(store, request_id=REQUEST, question="What does SET do?")


def test_matching_retry_returns_record_without_mutation(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    first = _claim(store, suite, reports, selection, bundle, package)
    retry = _claim(
        store,
        suite,
        reports,
        selection,
        bundle,
        package,
        owner="worker-2",
        now=RETRIED,
    )
    assert not retry.pinned
    assert retry.record == first.record
    assert store.get_request(REQUEST) == first.record


def test_concurrent_first_claims_yield_exactly_one_winner(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    workers = 4
    barrier = threading.Barrier(workers)
    results: list[RequestClaim] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            barrier.wait()
            results.append(_claim(store, suite, reports, selection, bundle, package))
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == workers
    assert sum(claim.pinned for claim in results) == 1
    assert len({claim.record for claim in results}) == 1


def test_conditional_insert_loser_gets_winner_and_content_conflict_fails(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = AbsentReadStore()
    winner = _claim(store, suite, reports, selection, bundle, package)
    loser = _claim(store, suite, reports, selection, bundle, package, now=RETRIED)
    assert winner.pinned
    assert not loser.pinned
    assert loser.record == winner.record
    with pytest.raises(RequestAuditError, match="different request content"):
        _claim(store, suite, reports, selection, bundle, package, question="What does SET do?")


def test_invalid_evidence_and_selection_fail_before_first_pin(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    with pytest.raises(RequestAuditError, match="request pin inputs are invalid"):
        _claim(
            store, suite, reports, selection, bundle, replace(package, digest="sha256:" + "0" * 64)
        )
    with pytest.raises(RequestAuditError, match="request pin inputs are invalid"):
        _claim(
            store,
            suite,
            reports,
            replace(selection, recorded_cost_usd=selection.recorded_cost_usd + 1.0),
            bundle,
            package,
        )
    assert store.get_request(REQUEST) is None


def test_live_observation_retains_exact_canonical_payload_and_fresh_exports() -> None:
    observation = _observation({"z": [1, {"ok": True}], "a": "value"})
    same = _observation({"a": "value", "z": [1, {"ok": True}]})
    assert observation == same
    assert observation.canonical_payload == b'{"a":"value","z":[1,{"ok":true}]}'
    assert (
        observation.payload_digest
        == "sha256:" + hashlib.sha256(observation.canonical_payload).hexdigest()
    )
    assert observation.observation_id.startswith("obs_")
    assert len(observation.observation_id) == len("obs_") + 64

    first = live_observation_value(observation)
    _validator("live_observation").validate(first)
    payload = cast(dict[str, object], first["payload"])
    cast(list[object], payload["z"]).append("mutated export")
    second = live_observation_value(observation)
    assert second["payload"] == {"a": "value", "z": [1, {"ok": True}]}
    assert observation.canonical_payload == b'{"a":"value","z":[1,{"ok":true}]}'


def test_live_observation_payload_has_exact_positive_256_kib_bound() -> None:
    exact = _observation("x" * (256 * 1024 - 2))
    assert len(exact.canonical_payload) == 256 * 1024
    with pytest.raises(RequestAuditError, match="262144-byte bound"):
        _observation("x" * (256 * 1024 - 1))


@pytest.mark.parametrize(
    "payload",
    [
        ("tuple",),
        {"bytes": b"no"},
        {1: "non-text key"},
        {"set": {1}},
        {"number": float("nan")},
        {"number": float("inf")},
        {"text": "\ud800"},
    ],
)
def test_non_json_non_utf8_and_non_finite_payloads_fail_closed(payload: object) -> None:
    with pytest.raises(RequestAuditError, match="live observation"):
        _observation(payload)


def test_cyclic_payload_fails_closed() -> None:
    payload: list[object] = []
    payload.append(payload)
    with pytest.raises(RequestAuditError, match="cyclic JSON"):
        _observation(payload)


def test_tampered_observation_snapshot_fails_closed(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    observation = _observation()
    tampered_cases = (
        replace(observation, canonical_payload=b'{"tag_name":"forged"}'),
        replace(observation, canonical_payload=b'{ "noncanonical": true }'),
        replace(observation, canonical_payload=b'"\xff"'),
        replace(observation, payload_digest=f"sha256:{'e' * 64}"),
        replace(observation, observed_at="2026-08-19T04:01:00Z"),
        replace(observation, source_url="https://api.github.com/repos/valkey-io/valkey"),
        replace(observation, object_type="tag"),
        replace(observation, observation_id=f"obs_{'0' * 64}"),
        replace(observation, observation_id="obs_caller-prefix"),
        replace(observation, complete=False),
        replace(observation, truncated=True),
    )
    for tampered in tampered_cases:
        with pytest.raises(RequestAuditError, match="live observation"):
            _claim(
                store,
                suite,
                reports,
                selection,
                bundle,
                package,
                live_observations=(tampered,),
            )
    assert store.get_request(REQUEST) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_url", "http://api.github.com/repos/valkey-io/valkey"),
        ("source_url", "https://"),
        ("source_url", "https://api.github.com/a b"),
        ("source_url", "https://example.com/repos/valkey-io/valkey/releases/latest"),
        ("source_url", "https://api.github.com/repos/other/valkey/releases/latest"),
        ("source_url", "https://user@api.github.com/repos/valkey-io/valkey/releases/latest"),
        ("source_url", "https://api.github.com/repos/valkey-io/../other/releases/latest"),
        ("source_url", "https://github.com/other/valkey/issues/1"),
        ("observed_at", "2026-08-19 04:00:00Z"),
        ("observed_at", "2026-02-30T04:00:00Z"),
        ("object_type", "gist"),
    ],
)
def test_malformed_observation_metadata_fails_closed(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "observed_at": OBSERVED,
        "source_url": OBSERVATION_URL,
        "object_type": "release",
        "payload": PAYLOAD,
    }
    kwargs[field] = value
    with pytest.raises(RequestAuditError):
        create_live_observation(**cast(dict[str, Any], kwargs))


def test_observation_container_bounds_fail_closed(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    observation = _observation()
    with pytest.raises(RequestAuditError, match="immutable tuple"):
        _claim(
            store,
            suite,
            reports,
            selection,
            bundle,
            package,
            live_observations=cast(tuple[LiveObservation, ...], [observation]),
        )
    with pytest.raises(RequestAuditError, match="duplicate live observation"):
        _claim(
            store,
            suite,
            reports,
            selection,
            bundle,
            package,
            live_observations=(observation, observation),
        )
    oversized = tuple(_observation({"index": index}) for index in range(51))
    with pytest.raises(RequestAuditError, match="outside its bound"):
        _claim(store, suite, reports, selection, bundle, package, live_observations=oversized)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_id", "req_", "request ID"),
        ("question", "", "user question"),
        ("question", "bad\x00question", "user question"),
        ("question", "q" * (8 * 1024 + 1), "user question"),
        ("application_revision", f"sha1:{'a' * 40}", "application revision"),
        ("owner", "", "request owner"),
        ("owner", "bad owner", "request owner"),
        ("owner", "x" * 129, "request owner"),
        ("owner", "\ud800", "request owner"),
        ("now", "yesterday", "claim timestamp"),
        ("now", "2026-02-30T04:00:00Z", "claim timestamp"),
        ("lease_duration_seconds", 0, "lease duration"),
        ("lease_duration_seconds", 3601, "lease duration"),
        ("lease_duration_seconds", True, "lease duration"),
    ],
)
def test_claim_identity_owner_time_and_lease_bounds_fail_closed(
    field: str,
    value: object,
    message: str,
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    kwargs: dict[str, object] = {
        "request_id": REQUEST,
        "question": QUESTION,
        "application_revision": APPLICATION,
        "owner": "worker-1",
        "now": STARTED,
        "lease_duration_seconds": 300,
    }
    kwargs[field] = value
    with pytest.raises(RequestAuditError, match=message):
        _claim(store, suite, reports, selection, bundle, package, **cast(dict[str, Any], kwargs))
    assert store.get_request(REQUEST) is None


def test_active_takeover_fails_and_expired_exact_recovery_updates_only_lifecycle(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    first = _claim(store, suite, reports, selection, bundle, package)

    with pytest.raises(RequestAuditError, match="still active"):
        recover_request(
            store,
            first.record,
            new_owner="worker-2",
            now=EARLY,
            lease_duration_seconds=300,
        )
    with pytest.raises(RequestAuditError, match="new request owner"):
        recover_request(
            store,
            first.record,
            new_owner="worker-1",
            now=RETRIED,
            lease_duration_seconds=300,
        )
    assert store.get_request(REQUEST) == first.record

    recovered = recover_request(
        store,
        first.record,
        new_owner="worker-2",
        now=RETRIED,
        lease_duration_seconds=300,
    )
    assert recovered.pin is first.record.pin
    assert (recovered.revision, recovered.fence) == (2, 2)
    assert recovered.owner == "worker-2"
    assert recovered.lease_expires_at == COMPLETED
    assert recovered.outcome is None

    with pytest.raises(RequestAuditError, match="state changed"):
        recover_request(
            store,
            first.record,
            new_owner="worker-3",
            now=RETRIED,
            lease_duration_seconds=300,
        )
    with pytest.raises(RequestAuditError, match="state changed"):
        complete_request(store, first.record, outcome="answer", completed_at=COMPLETED)


def test_recovery_validates_new_owner_time_and_duration(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    first = _claim(store, suite, reports, selection, bundle, package)
    cases = (
        ({"new_owner": "", "now": RETRIED, "lease_duration_seconds": 300}, "owner"),
        ({"new_owner": "worker-2", "now": "bad", "lease_duration_seconds": 300}, "timestamp"),
        ({"new_owner": "worker-2", "now": RETRIED, "lease_duration_seconds": 0}, "duration"),
    )
    for kwargs, message in cases:
        with pytest.raises(RequestAuditError, match=message):
            recover_request(store, first.record, **cast(dict[str, Any], kwargs))
    assert store.get_request(REQUEST) == first.record


def test_completion_is_exact_nonterminal_cas_and_terminal_is_store_immutable(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    first = _claim(store, suite, reports, selection, bundle, package)
    terminal = complete_request(store, first.record, outcome="answer", completed_at=COMPLETED)
    assert terminal.pin is first.record.pin
    assert (terminal.revision, terminal.fence) == (2, 1)
    assert terminal.owner == first.record.owner
    assert terminal.lease_expires_at == first.record.lease_expires_at
    assert terminal.outcome == "answer"
    assert store.get_request(REQUEST) == terminal

    attempted_replacement = replace(terminal, revision=3, outcome="error")
    assert not store.compare_and_swap_nonterminal(terminal, attempted_replacement)
    assert store.get_request(REQUEST) == terminal
    with pytest.raises(RequestAuditError, match="state changed"):
        complete_request(store, first.record, outcome="error", completed_at=COMPLETED)
    with pytest.raises(RequestAuditError, match="non-terminal claimed record"):
        complete_request(store, terminal, outcome="error", completed_at=COMPLETED)
    with pytest.raises(RequestAuditError, match="already terminal"):
        recover_request(
            store,
            terminal,
            new_owner="worker-2",
            now=COMPLETED,
            lease_duration_seconds=300,
        )


def test_terminal_audit_is_compact_derived_schema_evidence(
    suite: EvaluationSuite,
    prompt_package: PromptPackage,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    observation = _observation()
    first = _claim(
        store, suite, reports, selection, bundle, package, live_observations=(observation,)
    )
    with pytest.raises(RequestAuditError, match="terminal record"):
        request_audit_value(first.record)
    terminal = complete_request(store, first.record, outcome="answer", completed_at=COMPLETED)
    value = request_audit_value(terminal)
    _validator("request_audit").validate(value)
    assert value == {
        "api_version": "valkeyrie.io/request-audit/1",
        "kind": "RequestAudit",
        "request_id": REQUEST,
        "request_content_digest": request_content_digest(QUESTION),
        "generation_id": bundle.generation_id,
        "prompt_revision": prompt_package.prompt_revision,
        "application_revision": APPLICATION,
        "model_revision": selection.profile.model_revision,
        "inference_config_revision": selection.profile.inference_config_revision,
        "static_evidence_ids": [record.evidence_id for record in package.records],
        "live_observations": [live_observation_value(observation)],
        "started_at": STARTED,
        "outcome": "answer",
    }


def test_malformed_stored_records_and_executable_plans_fail_closed(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    store = InMemoryRequestAuditStore()
    first = _claim(store, suite, reports, selection, bundle, package)
    record = first.record
    pin = record.pin
    execution = pin.execution
    invocation = execution.invocation
    tampered_cases = (
        replace(record, owner=""),
        replace(record, lease_expires_at="bad"),
        replace(record, revision=0),
        replace(record, fence=0),
        replace(record, revision=1, fence=2),
        replace(record, completed_at=COMPLETED),
        replace(
            record,
            pin=replace(pin, execution=replace(execution, application_revision="sha256:short")),
        ),
        replace(
            record,
            pin=replace(
                pin,
                execution=replace(
                    execution,
                    invocation=replace(
                        invocation,
                        input=replace(invocation.input, question="different question"),
                    ),
                ),
            ),
        ),
        replace(
            record,
            pin=replace(
                pin,
                execution=replace(
                    execution,
                    invocation=replace(
                        invocation,
                        profile=replace(
                            invocation.profile,
                            inference=InferenceConfiguration(1, 0.0, 1.0, None),
                        ),
                    ),
                ),
            ),
        ),
    )
    for tampered in tampered_cases:
        store.force(tampered)
        with pytest.raises(RequestAuditError):
            resolve_pinned_execution(store, request_id=REQUEST, question=QUESTION)


def test_store_must_implement_dedicated_nonterminal_cas(
    suite: EvaluationSuite,
    reports: tuple[dict[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    package: EvidencePackage,
) -> None:
    with pytest.raises(RequestAuditError, match="does not implement conditional state"):
        _claim(
            cast(InMemoryRequestAuditStore, object()), suite, reports, selection, bundle, package
        )
    with pytest.raises(RequestAuditError, match="does not implement conditional state"):
        resolve_pinned_execution(
            cast(InMemoryRequestAuditStore, None), request_id=REQUEST, question=QUESTION
        )
