from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from infra import release_corpus
from infra.release_corpus import (
    CorpusReleaseError,
    PreparedRelease,
    ReleaseResources,
    ReleaseServices,
    execute_release,
    main,
    preflight_result,
    prepare_release,
)
from tests.test_evaluations import _passing_model_runs, _passing_retrieval_results
from tests.test_generation import _bundle
from tests.test_ingestion import FakeBedrock, MemoryCandidateStore, _poll
from tests.test_promotion import SmokeClient
from tests.test_publication import MemoryPublicationStore
from valkeyrie.evaluations import EvaluationSuite, evaluate_candidate, load_evaluation_suite
from valkeyrie.generation import GenerationBundle
from valkeyrie.ingestion import BedrockIngestionClient, IngestionError
from valkeyrie.promotion import (
    ActiveGeneration,
    ProtectedApproval,
)
from valkeyrie.promotion import (
    ApprovalRegistry as ApprovalRegistryProtocol,
)
from valkeyrie.publication import PublicationError, publish_generation
from valkeyrie.retrieval import BedrockRetrievalClient, GenerationAvailability
from valkeyrie.retrieval_config import load_retrieval_config

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-08-20T04:00:00Z"
EPOCH = 1_800_000_000
RESOURCES = ReleaseResources(
    account_id="968533178160",
    region="us-east-1",
    bucket="valkeyrie-development-corpus-968533178160-us-east-1",
    state_table="valkeyrie-development-state",
    knowledge_base_id="ONVASJDDNX",
    data_source_id="X9BEGFO9IQ",
)


class RecordingPublicationStore(MemoryPublicationStore):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.started = False

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        if not self.started:
            self.started = True
            self.events.append("publish")
        return super().list_keys(prefix)


class LifecycleStore:
    def __init__(
        self,
        events: list[str],
        generation: GenerationAvailability | None = None,
    ) -> None:
        self.events = events
        self.generation = generation
        self.active: ActiveGeneration | None = None
        self.reject_transition = False
        self.structured_records: dict[str, str] | None = None

    def get_generation(self, generation_id: str) -> GenerationAvailability | None:
        if self.generation is None or self.generation.generation_id != generation_id:
            return None
        return self.generation

    def create_generation(
        self,
        record: GenerationAvailability,
        structured_records: Mapping[str, str],
    ) -> bool:
        self.events.append("create")
        if self.generation is not None:
            return False
        self.generation = record
        self.structured_records = dict(structured_records)
        return True

    def compare_and_swap_generation(
        self,
        expected: GenerationAvailability,
        replacement: GenerationAvailability,
    ) -> bool:
        phase = "ingested" if replacement.revision == 2 else "evaluated"
        self.events.append(phase)
        if self.reject_transition:
            return False
        if self.generation != expected:
            return False
        self.generation = replacement
        return True

    def read_active(self) -> ActiveGeneration | None:
        return self.active

    def compare_and_swap_active(
        self,
        expected_active_revision: int | None,
        expected_generation: GenerationAvailability,
        replacement: ActiveGeneration,
    ) -> bool:
        current_revision = None if self.active is None else self.active.revision
        if current_revision != expected_active_revision or self.generation != expected_generation:
            return False
        self.events.append("activate")
        self.active = replacement
        return True


class RecordingBedrock(FakeBedrock):
    def __init__(self, events: list[str]) -> None:
        super().__init__([_poll("COMPLETE", scanned=7)])
        self.events = events

    def start_ingestion_job(self, **kwargs: object) -> dict[str, object]:
        self.events.append("ingest")
        return super().start_ingestion_job(**kwargs)


class ApprovalRegistry:
    def __init__(self, events: list[str], approval: ProtectedApproval) -> None:
        self.events = events
        self.approval = approval

    def consume_approval(self, approval_id: str) -> ProtectedApproval | None:
        self.events.append("approval")
        if self.approval.approval_id != approval_id:
            return None
        approval = self.approval
        self.approval = replace(approval, approval_id="consumed")
        return approval


@pytest.fixture(scope="module")
def suite() -> EvaluationSuite:
    return load_evaluation_suite(ROOT)


@pytest.fixture
def bundle() -> GenerationBundle:
    return _bundle(load_retrieval_config(ROOT / "retrieval-config.yaml"))


def _report(suite: EvaluationSuite, generation_id: str) -> dict[str, object]:
    return evaluate_candidate(
        suite,
        candidate_revision=generation_id,
        started_at="2026-08-20T02:00:00Z",
        completed_at="2026-08-20T02:05:00Z",
        model_runs=_passing_model_runs(suite),
        retrieval_results=_passing_retrieval_results(suite),
    )


def _prepared(
    bundle: GenerationBundle,
    suite: EvaluationSuite,
    *,
    report: Mapping[str, object] | None = None,
    activate: bool = False,
) -> tuple[PreparedRelease, str | None]:
    report_id = None if report is None else cast(str, report["report_id"])
    return PreparedRelease(
        resources=RESOURCES,
        bundle=bundle,
        suite=suite,
        lock_id="sha256:" + "1" * 64,
        source_commits={"valkey": "a" * 40},
        structured_index=release_corpus._structured_index(bundle),
        report=report,
        approval_id="approval_activate-generation" if activate else None,
        expected_active_generation=None,
        activated_at=NOW if activate else None,
    ), report_id


def _services(
    publication: MemoryPublicationStore,
    lifecycle: LifecycleStore,
    bedrock: BedrockIngestionClient,
    *,
    retrieval: BedrockRetrievalClient | None = None,
    approvals: ApprovalRegistryProtocol | None = None,
) -> ReleaseServices:
    return ReleaseServices(
        publication_store=publication,
        lifecycle_store=lifecycle,
        candidate_store=MemoryCandidateStore(),
        ingestion_client=bedrock,
        retrieval_client=retrieval,
        approval_registry=approvals,
    )


def test_release_orders_existing_components_and_persists_structured_index(
    bundle: GenerationBundle,
    suite: EvaluationSuite,
) -> None:
    events: list[str] = []
    report = _report(suite, bundle.generation_id)
    prepared, report_id = _prepared(bundle, suite, report=report, activate=True)
    lifecycle = LifecycleStore(events)
    publication = RecordingPublicationStore(events)
    approval = ProtectedApproval(
        "approval_activate-generation",
        "activate",
        bundle.generation_id,
        None,
        cast(str, report_id),
        "maintainer",
        NOW,
    )

    result = execute_release(
        prepared,
        _services(
            publication,
            lifecycle,
            RecordingBedrock(events),
            retrieval=SmokeClient(),
            approvals=ApprovalRegistry(events, approval),
        ),
        owner="release-worker",
        now_epoch=lambda: EPOCH,
        sleep=lambda _: None,
    )

    assert events == [
        "publish",
        "create",
        "ingest",
        "ingested",
        "evaluated",
        "approval",
        "activate",
    ]
    assert lifecycle.structured_records == prepared.structured_index
    assert lifecycle.generation is not None
    assert lifecycle.generation.evaluation_passed
    assert lifecycle.generation.evaluation_report_id == report_id
    assert cast(dict[str, object], result["active_generation"])["generation_id"] == (
        bundle.generation_id
    )


def test_release_without_report_stops_after_retrievable_state(
    bundle: GenerationBundle,
    suite: EvaluationSuite,
) -> None:
    events: list[str] = []
    prepared, _ = _prepared(bundle, suite)
    lifecycle = LifecycleStore(events)

    result = execute_release(
        prepared,
        _services(RecordingPublicationStore(events), lifecycle, RecordingBedrock(events)),
        owner="release-worker",
        now_epoch=lambda: EPOCH,
        sleep=lambda _: None,
    )

    assert events == ["publish", "create", "ingest", "ingested"]
    assert lifecycle.generation is not None
    assert lifecycle.generation.retrievable
    assert not lifecycle.generation.evaluation_passed
    assert result["active_generation"] is None


def test_completed_release_is_idempotently_verified_without_new_writes(
    bundle: GenerationBundle,
    suite: EvaluationSuite,
) -> None:
    events: list[str] = []
    report = _report(suite, bundle.generation_id)
    prepared, report_id = _prepared(bundle, suite, report=report)
    publication = RecordingPublicationStore(events)
    first_publication = publish_generation(publication, bundle)
    events.clear()
    publication.started = False
    evaluated = GenerationAvailability(
        bundle.generation_id,
        revision=3,
        sealed=True,
        available=True,
        ingested=True,
        retrievable=True,
        evaluation_passed=True,
        retained=True,
        evaluation_report_id=cast(str, report_id),
    )
    lifecycle = LifecycleStore(events, evaluated)

    result = execute_release(
        prepared,
        _services(publication, lifecycle, FakeBedrock([])),
        owner="release-worker",
        now_epoch=lambda: EPOCH,
        sleep=lambda _: None,
    )

    assert events == ["publish"]
    assert cast(dict[str, object], result["publication"])["created_keys"] == []
    assert first_publication.created_keys
    assert result["ingestion"] is None


def test_publication_ingestion_and_lifecycle_failures_stop_later_phases(
    bundle: GenerationBundle,
    suite: EvaluationSuite,
) -> None:
    prepared, _ = _prepared(bundle, suite, report=_report(suite, bundle.generation_id))

    class BrokenPublication(MemoryPublicationStore):
        def list_keys(self, prefix: str) -> tuple[str, ...]:
            raise PublicationError("publication failed")

    lifecycle = LifecycleStore([])
    with pytest.raises(PublicationError, match="publication failed"):
        execute_release(
            prepared,
            _services(BrokenPublication(), lifecycle, FakeBedrock([])),
            owner="worker",
            now_epoch=lambda: EPOCH,
            sleep=lambda _: None,
        )
    assert lifecycle.generation is None

    lifecycle = LifecycleStore([])
    with pytest.raises(IngestionError, match="Bedrock ingestion ended with FAILED"):
        execute_release(
            prepared,
            _services(
                MemoryPublicationStore(),
                lifecycle,
                FakeBedrock([_poll("FAILED")]),
            ),
            owner="worker",
            now_epoch=lambda: EPOCH,
            sleep=lambda _: None,
        )
    assert lifecycle.generation is not None
    assert lifecycle.generation.revision == 1

    lifecycle = LifecycleStore([])
    lifecycle.reject_transition = True
    with pytest.raises(CorpusReleaseError, match="ingested lifecycle transition raced"):
        execute_release(
            prepared,
            _services(MemoryPublicationStore(), lifecycle, FakeBedrock([_poll("COMPLETE")])),
            owner="worker",
            now_epoch=lambda: EPOCH,
            sleep=lambda _: None,
        )
    assert lifecycle.generation is not None
    assert lifecycle.generation.revision == 1


def test_prepare_rejects_candidate_mismatch_before_any_client_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle: GenerationBundle,
    suite: EvaluationSuite,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(_report(suite, "sha256:" + "f" * 64)),
        encoding="utf-8",
    )
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    locked = SimpleNamespace(
        lock_id="sha256:" + "1" * 64,
        revisions=(SimpleNamespace(repository="valkey", commit="a" * 40),),
    )
    monkeypatch.setattr(release_corpus, "load_source_inventory", lambda _: object())
    monkeypatch.setattr(release_corpus, "load_source_lock", lambda *_: locked)
    monkeypatch.setattr(
        release_corpus,
        "load_retrieval_config",
        lambda *_args, **_kwargs: bundle.retrieval_config,
    )
    monkeypatch.setattr(release_corpus, "build_locked_corpus", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(release_corpus, "verify_generation_bundle", lambda value: value)
    monkeypatch.setattr(release_corpus, "load_evaluation_suite", lambda _: suite)

    with pytest.raises(CorpusReleaseError, match="candidate does not match"):
        prepare_release(
            root=ROOT,
            lock_path=lock_path,
            cache=cache,
            created_at=bundle.created_at,
            resources=RESOURCES,
            evaluation_report_path=report_path,
        )


def test_dry_run_constructs_no_aws_clients(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bundle: GenerationBundle,
    suite: EvaluationSuite,
) -> None:
    prepared, _ = _prepared(bundle, suite)
    monkeypatch.setattr(release_corpus, "prepare_release", lambda **_: prepared)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("AWS client factory called during dry-run")

    monkeypatch.setattr(release_corpus, "create_s3_client", forbidden)
    monkeypatch.setattr(release_corpus, "create_dynamodb_client", forbidden)
    monkeypatch.setattr(release_corpus, "create_bedrock_ingestion_client", forbidden)
    monkeypatch.setattr(release_corpus, "create_bedrock_retrieval_client", forbidden)

    result = main(
        [
            "--root",
            str(ROOT),
            "--lock",
            "/tmp/lock.json",
            "--cache",
            "/tmp/cache",
            "--created-at",
            bundle.created_at,
            "--account-id",
            RESOURCES.account_id,
            "--region",
            RESOURCES.region,
            "--bucket",
            RESOURCES.bucket,
            "--state-table",
            RESOURCES.state_table,
            "--knowledge-base-id",
            RESOURCES.knowledge_base_id,
            "--data-source-id",
            RESOURCES.data_source_id,
            "--owner",
            "worker",
            "--dry-run",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["operation"] == "preflight"
    assert output["aws_calls"] is False
    assert preflight_result(prepared, owner="worker")["generation_id"] == bundle.generation_id


def test_resources_and_activation_inputs_are_explicit_and_fail_closed(
    bundle: GenerationBundle,
) -> None:
    with pytest.raises(CorpusReleaseError, match="account ID"):
        prepare_release(
            root=ROOT,
            lock_path=Path("/not/read"),
            cache=Path("/not/read"),
            created_at=bundle.created_at,
            resources=replace(RESOURCES, account_id="123"),
        )
    with pytest.raises(CorpusReleaseError, match="explicit expected active generation"):
        prepare_release(
            root=ROOT,
            lock_path=Path("/not/read"),
            cache=Path("/not/read"),
            created_at=bundle.created_at,
            resources=RESOURCES,
            evaluation_report_path=Path("/not/read"),
            approval_id="approval_activate-generation",
            activated_at=NOW,
        )
