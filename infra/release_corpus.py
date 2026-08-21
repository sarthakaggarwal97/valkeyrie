from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, cast

from valkeyrie.aws_adapters import (
    DynamoApprovalRegistry,
    DynamoCandidateStore,
    DynamoPromotionStore,
    S3PublicationStore,
    create_bedrock_ingestion_client,
    create_bedrock_retrieval_client,
    create_dynamodb_client,
    create_s3_client,
)
from valkeyrie.evaluations import (
    EvaluationError,
    EvaluationSuite,
    load_evaluation_suite,
    verify_evaluation_report,
)
from valkeyrie.generation import GenerationBundle, GenerationError, verify_generation_bundle
from valkeyrie.git_acquisition import GitAcquisitionError, build_locked_corpus, load_source_lock
from valkeyrie.ingestion import (
    BedrockIngestionClient,
    ConditionalCandidateStore,
    IngestionError,
    IngestionResult,
    run_ingestion,
)
from valkeyrie.promotion import (
    ActiveGeneration,
    ApprovalRegistry,
    PromotionError,
    PromotionStore,
    activate_candidate,
)
from valkeyrie.publication import (
    PublicationError,
    PublicationResult,
    PublicationStore,
    publish_generation,
)
from valkeyrie.retrieval import BedrockRetrievalClient, GenerationAvailability
from valkeyrie.retrieval_config import RetrievalConfigError, load_retrieval_config
from valkeyrie.sources import SourceInventoryError, load_source_inventory


class CorpusReleaseError(RuntimeError):
    """A release cannot proceed without violating its exact local plan."""


class GenerationLifecycleStore(PromotionStore, Protocol):
    def create_generation(
        self,
        record: GenerationAvailability,
        structured_records: Mapping[str, str],
    ) -> bool: ...

    def compare_and_swap_generation(
        self,
        expected: GenerationAvailability,
        replacement: GenerationAvailability,
    ) -> bool: ...


@dataclass(frozen=True)
class ReleaseResources:
    account_id: str
    region: str
    bucket: str
    state_table: str
    knowledge_base_id: str
    data_source_id: str


@dataclass(frozen=True)
class PreparedRelease:
    resources: ReleaseResources
    bundle: GenerationBundle
    suite: EvaluationSuite
    lock_id: str
    source_commits: Mapping[str, str]
    structured_index: Mapping[str, str]
    report: Mapping[str, object] | None
    approval_id: str | None
    expected_active_generation: str | None
    activated_at: str | None


@dataclass(frozen=True)
class ReleaseServices:
    publication_store: PublicationStore
    lifecycle_store: GenerationLifecycleStore
    candidate_store: ConditionalCandidateStore
    ingestion_client: BedrockIngestionClient
    retrieval_client: BedrockRetrievalClient | None = None
    approval_registry: ApprovalRegistry | None = None


_ACCOUNT = re.compile(r"^[0-9]{12}$")
_REGION = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]$")
_SERVICE_ID = re.compile(r"^[A-Z0-9]{10}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_APPROVAL_ID = re.compile(r"^approval_[a-z0-9-]+$")
_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$")


def prepare_release(
    *,
    root: Path,
    lock_path: Path,
    cache: Path,
    created_at: str,
    resources: ReleaseResources,
    evaluation_report_path: Path | None = None,
    approval_id: str | None = None,
    expected_active_generation: str | None = None,
    activated_at: str | None = None,
) -> PreparedRelease:
    """Build and verify every local input before any AWS client is constructed."""
    _validate_resources(resources)
    project_root = _project_root(root)
    activation_expected = _validate_activation_inputs(
        evaluation_report_path=evaluation_report_path,
        approval_id=approval_id,
        expected_active_generation=expected_active_generation,
        activated_at=activated_at,
    )

    sources_yaml = (project_root / "sources.yaml").read_bytes()
    inventory = load_source_inventory(project_root / "sources.yaml")
    lock_document = lock_path.expanduser().resolve(strict=True).read_bytes()
    locked = load_source_lock(inventory, lock_document)
    retrieval_config = load_retrieval_config(
        project_root / "retrieval-config.yaml",
        project_root=project_root,
    )
    bundle = verify_generation_bundle(
        build_locked_corpus(
            sources_yaml,
            inventory,
            retrieval_config,
            lock_document,
            cache.expanduser().resolve(),
            created_at=created_at,
        )
    )
    suite = load_evaluation_suite(project_root)
    report = (
        None
        if evaluation_report_path is None
        else _load_verified_report(evaluation_report_path, suite, bundle.generation_id)
    )
    structured_index = _structured_index(bundle)
    return PreparedRelease(
        resources=resources,
        bundle=bundle,
        suite=suite,
        lock_id=locked.lock_id,
        source_commits={item.repository: item.commit for item in locked.revisions},
        structured_index=structured_index,
        report=report,
        approval_id=approval_id,
        expected_active_generation=activation_expected,
        activated_at=activated_at,
    )


def execute_release(
    prepared: PreparedRelease,
    services: ReleaseServices,
    *,
    owner: str,
    now_epoch: Callable[[], int],
    sleep: Callable[[float], None],
    lease_seconds: int = 300,
    poll_interval_seconds: float = 15.0,
    max_polls: int = 480,
) -> Mapping[str, object]:
    """Publish, ingest, qualify, and optionally activate one prepared generation."""
    _validate_execution_inputs(
        owner=owner,
        lease_seconds=lease_seconds,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
    )
    bundle = verify_generation_bundle(prepared.bundle)
    publication = publish_generation(services.publication_store, bundle)

    sealed = GenerationAvailability(
        bundle.generation_id,
        revision=1,
        sealed=True,
        available=False,
        ingested=False,
        retrievable=False,
        evaluation_passed=False,
        retained=True,
    )
    ingested = replace(
        sealed,
        revision=2,
        available=True,
        ingested=True,
        retrievable=True,
    )
    evaluated = _evaluated_state(ingested, prepared.report)

    current = services.lifecycle_store.get_generation(bundle.generation_id)
    lifecycle_created = False
    if current is None:
        lifecycle_created = services.lifecycle_store.create_generation(
            sealed,
            prepared.structured_index,
        )
        if not lifecycle_created:
            raise CorpusReleaseError("generation lifecycle creation raced")
        current = sealed
    if current not in {sealed, ingested, evaluated}:
        raise CorpusReleaseError("generation lifecycle does not match a releasable exact state")

    ingestion: IngestionResult | None = None
    if current == sealed:
        ingestion = run_ingestion(
            services.publication_store,
            services.candidate_store,
            services.ingestion_client,
            bundle,
            knowledge_base_id=prepared.resources.knowledge_base_id,
            data_source_id=prepared.resources.data_source_id,
            owner=owner,
            now_epoch=now_epoch,
            sleep=sleep,
            lease_seconds=lease_seconds,
            poll_interval_seconds=poll_interval_seconds,
            max_polls=max_polls,
        )
        if not services.lifecycle_store.compare_and_swap_generation(sealed, ingested):
            raise CorpusReleaseError("ingested lifecycle transition raced")
        current = ingested

    if prepared.report is not None and current == ingested:
        if not services.lifecycle_store.compare_and_swap_generation(ingested, evaluated):
            raise CorpusReleaseError("evaluated lifecycle transition raced")
        current = evaluated
    if prepared.report is not None and current != evaluated:
        raise CorpusReleaseError("generation evaluation state does not match the supplied report")

    active: ActiveGeneration | None = None
    if prepared.approval_id is not None:
        if prepared.report is None or prepared.activated_at is None:
            raise CorpusReleaseError("activation requires a prepared passing report and timestamp")
        if services.retrieval_client is None or services.approval_registry is None:
            raise CorpusReleaseError("activation services are absent")
        active = activate_candidate(
            services.lifecycle_store,
            prepared.suite,
            prepared.report,
            services.approval_registry,
            approval_id=prepared.approval_id,
            retrieval_client=services.retrieval_client,
            retrieval_config=bundle.retrieval_config,
            knowledge_base_id=prepared.resources.knowledge_base_id,
            expected_active_generation=prepared.expected_active_generation,
            activated_at=prepared.activated_at,
        )

    return _release_result(
        prepared,
        publication,
        current,
        lifecycle_created=lifecycle_created,
        ingestion=ingestion,
        active=active,
    )


def preflight_result(
    prepared: PreparedRelease,
    *,
    owner: str,
    lease_seconds: int = 300,
    poll_interval_seconds: float = 15.0,
    max_polls: int = 480,
) -> Mapping[str, object]:
    """Return the canonical, local-only release plan."""
    _validate_execution_inputs(
        owner=owner,
        lease_seconds=lease_seconds,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
    )
    return {
        **_base_result(prepared),
        "operation": "preflight",
        "aws_calls": False,
        "ingestion": {
            "owner": owner,
            "lease_seconds": lease_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "max_polls": max_polls,
        },
        "phases": [
            "publish",
            "create_sealed_lifecycle",
            "ingest",
            "mark_retrievable",
            *(["apply_verified_evaluation"] if prepared.report is not None else []),
            *(["consume_approval_and_activate"] if prepared.approval_id is not None else []),
        ],
    }


def _evaluated_state(
    ingested: GenerationAvailability,
    report: Mapping[str, object] | None,
) -> GenerationAvailability:
    if report is None:
        return ingested
    return replace(
        ingested,
        revision=3,
        evaluation_passed=True,
        evaluation_report_id=cast(str, report["report_id"]),
    )


def _release_result(
    prepared: PreparedRelease,
    publication: PublicationResult,
    lifecycle: GenerationAvailability,
    *,
    lifecycle_created: bool,
    ingestion: IngestionResult | None,
    active: ActiveGeneration | None,
) -> Mapping[str, object]:
    return {
        **_base_result(prepared),
        "operation": "release",
        "publication": {
            "completion_key": publication.completion_key,
            "created_keys": list(publication.created_keys),
            "object_count": publication.object_count,
        },
        "lifecycle_created": lifecycle_created,
        "lifecycle": asdict(lifecycle),
        "ingestion": None if ingestion is None else asdict(ingestion),
        "active_generation": None if active is None else asdict(active),
    }


def _base_result(prepared: PreparedRelease) -> dict[str, object]:
    bundle = prepared.bundle
    resources = prepared.resources
    return {
        "account_id": resources.account_id,
        "region": resources.region,
        "bucket": resources.bucket,
        "state_table": resources.state_table,
        "knowledge_base_id": resources.knowledge_base_id,
        "data_source_id": resources.data_source_id,
        "lock_id": prepared.lock_id,
        "generation_id": bundle.generation_id,
        "manifest_digest": bundle.manifest.digest,
        "documents": len(bundle.documents),
        "metadata_sidecars": len(bundle.metadata_sidecars),
        "structured_records": len(bundle.structured_records),
        "structured_index_sha256": _structured_index_root(prepared.structured_index),
        "source_commits": dict(prepared.source_commits),
        "evaluation_report_id": (None if prepared.report is None else prepared.report["report_id"]),
        "activation_requested": prepared.approval_id is not None,
        "approval_id": prepared.approval_id,
        "expected_active_generation": prepared.expected_active_generation,
        "activated_at": prepared.activated_at,
    }


def _structured_index(bundle: GenerationBundle) -> Mapping[str, str]:
    index = {
        hashlib.sha256(record.object_id.encode("utf-8")).hexdigest(): record.digest
        for record in bundle.structured_records
    }
    if not index or len(index) != len(bundle.structured_records) or len(index) > 10_000:
        raise CorpusReleaseError("generation structured record index is outside its bound")
    return index


def _structured_index_root(index: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(index), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _load_verified_report(
    path: Path,
    suite: EvaluationSuite,
    generation_id: str,
) -> Mapping[str, object]:
    try:
        value = json.loads(
            path.expanduser().resolve(strict=True).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as error:
        raise CorpusReleaseError(f"evaluation report cannot be loaded: {error}") from error
    if not isinstance(value, Mapping):
        raise CorpusReleaseError("evaluation report must be a JSON object")
    report = cast(Mapping[str, object], value)
    verify_evaluation_report(report, suite)
    if report.get("candidate_revision") != generation_id:
        raise CorpusReleaseError("evaluation report candidate does not match the built generation")
    if report.get("result") != "pass":
        raise CorpusReleaseError("supplied candidate evaluation did not pass")
    return report


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _project_root(value: Path) -> Path:
    root = value.expanduser().resolve(strict=True)
    required = ("sources.yaml", "retrieval-config.yaml", "evals")
    if not root.is_dir() or any(not (root / name).exists() for name in required):
        raise CorpusReleaseError("operator root is not a Valkeyrie project")
    return root


def _validate_resources(resources: ReleaseResources) -> None:
    if _ACCOUNT.fullmatch(resources.account_id) is None:
        raise CorpusReleaseError("AWS account ID must be exactly 12 digits")
    if _REGION.fullmatch(resources.region) is None:
        raise CorpusReleaseError("AWS region is malformed")
    match = re.fullmatch(
        rf"valkeyrie-([a-z0-9-]+)-corpus-{resources.account_id}-{resources.region}",
        resources.bucket,
    )
    if match is None:
        raise CorpusReleaseError("corpus bucket is not bound to the explicit account and region")
    if resources.state_table != f"valkeyrie-{match.group(1)}-state":
        raise CorpusReleaseError("state table and corpus bucket environments do not match")
    if _SERVICE_ID.fullmatch(resources.knowledge_base_id) is None:
        raise CorpusReleaseError("Knowledge Base ID is malformed")
    if _SERVICE_ID.fullmatch(resources.data_source_id) is None:
        raise CorpusReleaseError("data source ID is malformed")


def _validate_activation_inputs(
    *,
    evaluation_report_path: Path | None,
    approval_id: str | None,
    expected_active_generation: str | None,
    activated_at: str | None,
) -> str | None:
    activation_values = (approval_id, expected_active_generation, activated_at)
    if approval_id is None:
        if any(value is not None for value in activation_values[1:]):
            raise CorpusReleaseError("activation inputs require a protected approval ID")
        return None
    if evaluation_report_path is None:
        raise CorpusReleaseError("activation requires a supplied evaluation report")
    if _APPROVAL_ID.fullmatch(approval_id) is None:
        raise CorpusReleaseError("protected approval ID is malformed")
    if expected_active_generation is None:
        raise CorpusReleaseError("activation requires an explicit expected active generation")
    expected = None if expected_active_generation == "none" else expected_active_generation
    if expected is not None and _DIGEST.fullmatch(expected) is None:
        raise CorpusReleaseError("expected active generation is malformed")
    if activated_at is None or _TIMESTAMP.fullmatch(activated_at) is None:
        raise CorpusReleaseError("activation timestamp is malformed")
    return expected


def _validate_execution_inputs(
    *,
    owner: object,
    lease_seconds: object,
    poll_interval_seconds: object,
    max_polls: object,
) -> None:
    if not isinstance(owner, str) or not owner or len(owner) > 128:
        raise CorpusReleaseError("ingestion owner is malformed")
    if not isinstance(lease_seconds, int) or isinstance(lease_seconds, bool) or lease_seconds < 1:
        raise CorpusReleaseError("ingestion lease seconds must be a positive integer")
    if (
        not isinstance(poll_interval_seconds, (int, float))
        or isinstance(poll_interval_seconds, bool)
        or poll_interval_seconds < 0
    ):
        raise CorpusReleaseError("ingestion poll interval must be nonnegative")
    if not isinstance(max_polls, int) or isinstance(max_polls, bool) or max_polls < 1:
        raise CorpusReleaseError("ingestion max polls must be a positive integer")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, publish, ingest, qualify, and optionally activate one exact corpus."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--state-table", required=True)
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--data-source-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--evaluation-report", type=Path)
    parser.add_argument("--approval-id")
    parser.add_argument("--expected-active-generation")
    parser.add_argument("--activated-at")
    parser.add_argument("--lease-seconds", type=int, default=300)
    # 3,774 documents took ~35 minutes to ingest, so the former 120 x 5s (10 minute) bound
    # made every successful refresh report failure. 480 x 15s allows 2 hours.
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--max-polls", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _canonical(value: object) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def main(arguments: Sequence[str] | None = None) -> int:
    values = _parser().parse_args(arguments)
    try:
        resources = ReleaseResources(
            values.account_id,
            values.region,
            values.bucket,
            values.state_table,
            values.knowledge_base_id,
            values.data_source_id,
        )
        _validate_execution_inputs(
            owner=values.owner,
            lease_seconds=values.lease_seconds,
            poll_interval_seconds=values.poll_interval_seconds,
            max_polls=values.max_polls,
        )
        prepared = prepare_release(
            root=values.root,
            lock_path=values.lock,
            cache=values.cache,
            created_at=values.created_at,
            resources=resources,
            evaluation_report_path=values.evaluation_report,
            approval_id=values.approval_id,
            expected_active_generation=values.expected_active_generation,
            activated_at=values.activated_at,
        )
        if values.dry_run:
            result = preflight_result(
                prepared,
                owner=values.owner,
                lease_seconds=values.lease_seconds,
                poll_interval_seconds=values.poll_interval_seconds,
                max_polls=values.max_polls,
            )
        else:
            s3 = create_s3_client(resources.region)
            dynamodb = create_dynamodb_client(resources.region)
            activation = prepared.approval_id is not None
            services = ReleaseServices(
                publication_store=S3PublicationStore(resources.bucket, s3),
                lifecycle_store=DynamoPromotionStore(resources.state_table, dynamodb),
                candidate_store=DynamoCandidateStore(resources.state_table, dynamodb),
                ingestion_client=create_bedrock_ingestion_client(resources.region),
                retrieval_client=(
                    create_bedrock_retrieval_client(resources.region) if activation else None
                ),
                approval_registry=(
                    DynamoApprovalRegistry(resources.state_table, dynamodb) if activation else None
                ),
            )
            result = execute_release(
                prepared,
                services,
                owner=values.owner,
                now_epoch=lambda: int(time.time()),
                sleep=time.sleep,
                lease_seconds=values.lease_seconds,
                poll_interval_seconds=values.poll_interval_seconds,
                max_polls=values.max_polls,
            )
    except (
        CorpusReleaseError,
        EvaluationError,
        GenerationError,
        GitAcquisitionError,
        IngestionError,
        OSError,
        PromotionError,
        PublicationError,
        RetrievalConfigError,
        SourceInventoryError,
        UnicodeError,
        ValueError,
    ) as error:
        print(f"corpus release failed: {error}", file=sys.stderr)
        return 1
    print(_canonical(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
