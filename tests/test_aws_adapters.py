from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from tests.test_generation import _bundle
from valkeyrie.aws_adapters import (
    DynamoApprovalRegistry,
    DynamoCandidateStore,
    DynamoPromotionStore,
    S3PublicationStore,
)
from valkeyrie.generation import GenerationBundle
from valkeyrie.ingestion import CandidateState, CandidateStatus, IngestionError
from valkeyrie.promotion import ActiveGeneration, PromotionError, ProtectedApproval
from valkeyrie.publication import PublicationError, publish_generation
from valkeyrie.retrieval import GenerationAvailability
from valkeyrie.retrieval_config import load_retrieval_config

ROOT = Path(__file__).resolve().parents[1]
GEN_A = "sha256:" + "a" * 64
REPORT_A = "eval_" + "a" * 64
NOW = "2026-08-20T04:00:00Z"


class AwsError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class Body:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def read(self) -> bytes:
        return self.content


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.put_calls: list[dict[str, object]] = []
        self.denied = False

    def list_objects_v2(self, **kwargs: object) -> Mapping[str, object]:
        prefix = cast(str, kwargs["Prefix"])
        return {
            "Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(prefix)],
            "IsTruncated": False,
        }

    def head_object(self, **kwargs: object) -> Mapping[str, object]:
        key = cast(str, kwargs["Key"])
        if key not in self.objects:
            raise AwsError("404")
        content, metadata = self.objects[key]
        return {"ContentLength": len(content), "Metadata": dict(metadata)}

    def get_object(self, **kwargs: object) -> Mapping[str, object]:
        key = cast(str, kwargs["Key"])
        return {"Body": Body(self.objects[key][0])}

    def put_object(self, **kwargs: object) -> Mapping[str, object]:
        self.put_calls.append(dict(kwargs))
        if self.denied:
            raise AwsError("AccessDenied")
        key = cast(str, kwargs["Key"])
        if key in self.objects:
            raise AwsError("PreconditionFailed")
        self.objects[key] = (
            cast(bytes, kwargs["Body"]),
            dict(cast(dict[str, str], kwargs["Metadata"])),
        )
        return {"ETag": "etag"}

    def get_item(self, **kwargs: object) -> Mapping[str, object]:
        raise AssertionError(kwargs)

    def put_item(self, **kwargs: object) -> Mapping[str, object]:
        raise AssertionError(kwargs)

    def update_item(self, **kwargs: object) -> Mapping[str, object]:
        raise AssertionError(kwargs)

    def transact_write_items(self, **kwargs: object) -> Mapping[str, object]:
        raise AssertionError(kwargs)


class PagedS3(FakeS3):
    def __init__(self) -> None:
        super().__init__()
        self.list_calls: list[dict[str, object]] = []

    def list_objects_v2(self, **kwargs: object) -> Mapping[str, object]:
        self.list_calls.append(dict(kwargs))
        if "ContinuationToken" not in kwargs:
            return {
                "Contents": [{"Key": "prefix/a"}],
                "IsTruncated": True,
                "NextContinuationToken": "page-2",
            }
        return {"Contents": [{"Key": "prefix/b"}], "IsTruncated": False}


class FakeDynamo:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, object]] = {}
        self.get_calls: list[dict[str, object]] = []
        self.put_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []
        self.transaction_calls: list[dict[str, object]] = []
        self.reject_puts = 0
        self.reject_transaction = False

    def list_objects_v2(self, **kwargs: object) -> Mapping[str, object]:
        raise AssertionError(kwargs)

    def head_object(self, **kwargs: object) -> Mapping[str, object]:
        raise AssertionError(kwargs)

    def get_object(self, **kwargs: object) -> Mapping[str, object]:
        raise AssertionError(kwargs)

    def put_object(self, **kwargs: object) -> Mapping[str, object]:
        raise AssertionError(kwargs)

    def get_item(self, **kwargs: object) -> Mapping[str, object]:
        self.get_calls.append(dict(kwargs))
        key = _key(cast(Mapping[str, object], kwargs["Key"]))
        item = self.items.get(key)
        return {} if item is None else {"Item": dict(item)}

    def put_item(self, **kwargs: object) -> Mapping[str, object]:
        self.put_calls.append(dict(kwargs))
        item = dict(cast(Mapping[str, object], kwargs["Item"]))
        key = _key(item)
        existing = self.items.get(key)
        condition = kwargs["ConditionExpression"]
        if self.reject_puts:
            self.reject_puts -= 1
            raise AwsError("ConditionalCheckFailedException")
        if condition == "attribute_not_exists(pk)":
            if existing is not None:
                raise AwsError("ConditionalCheckFailedException")
        elif "ExpressionAttributeNames" in kwargs:
            if existing is None:
                raise AwsError("ConditionalCheckFailedException")
            names = cast(Mapping[str, str], kwargs["ExpressionAttributeNames"])
            expected_values = cast(Mapping[str, object], kwargs["ExpressionAttributeValues"])
            for index, name_key in enumerate(sorted(names, key=lambda value: int(value[2:]))):
                if existing.get(names[name_key]) != expected_values[f":v{index}"]:
                    raise AwsError("ConditionalCheckFailedException")
        else:
            expected = cast(Mapping[str, object], kwargs["ExpressionAttributeValues"])[":revision"]
            if existing is None or existing.get("revision") != expected:
                raise AwsError("ConditionalCheckFailedException")
        self.items[key] = item
        return {}

    def update_item(self, **kwargs: object) -> Mapping[str, object]:
        self.update_calls.append(dict(kwargs))
        key = _key(cast(Mapping[str, object], kwargs["Key"]))
        existing = self.items.get(key)
        if existing is None or existing.get("consumed") == {"BOOL": True}:
            raise AwsError("ConditionalCheckFailedException")
        old = dict(existing)
        existing["consumed"] = {"BOOL": True}
        return {"Attributes": old}

    def transact_write_items(self, **kwargs: object) -> Mapping[str, object]:
        self.transaction_calls.append(dict(kwargs))
        if self.reject_transaction:
            raise AwsError("TransactionCanceledException")
        transaction = cast(list[dict[str, object]], kwargs["TransactItems"])
        put = cast(dict[str, object], transaction[1]["Put"])
        item = dict(cast(Mapping[str, object], put["Item"]))
        key = _key(item)
        condition = put["ConditionExpression"]
        if condition == "attribute_not_exists(pk)" and key in self.items:
            raise AwsError("TransactionCanceledException")
        if condition != "attribute_not_exists(pk)":
            expected = cast(Mapping[str, object], put["ExpressionAttributeValues"])[":revision"]
            current = self.items.get(key)
            if current is None or current.get("revision") != expected:
                raise AwsError("TransactionCanceledException")
        self.items[key] = item
        return {}


def _key(item: Mapping[str, object]) -> str:
    return cast(str, cast(Mapping[str, object], item["pk"])["S"])


def _av(value: object) -> dict[str, object]:
    if value is None:
        return {"NULL": True}
    if isinstance(value, bool):
        return {"BOOL": value}
    if isinstance(value, int):
        return {"N": str(value)}
    if isinstance(value, str):
        return {"S": value}
    if isinstance(value, Mapping):
        return {"M": {str(key): _av(item) for key, item in value.items()}}
    raise AssertionError(value)


def _item(values: Mapping[str, object]) -> dict[str, object]:
    return {key: _av(value) for key, value in values.items()}


@pytest.fixture
def bundle() -> GenerationBundle:
    return _bundle(load_retrieval_config(ROOT / "retrieval-config.yaml"))


def _generation_values(*, structured: bool = False) -> dict[str, object]:
    values: dict[str, object] = {
        "pk": f"generation#{GEN_A.removeprefix('sha256:')}",
        "record_type": "generation_availability",
        "generation_id": GEN_A,
        "revision": 3,
        "sealed": True,
        "available": True,
        "ingested": True,
        "retrievable": True,
        "evaluation_passed": True,
        "retained": True,
        "evaluation_report_id": REPORT_A,
    }
    if structured:
        records = {"b" * 64: "sha256:" + "c" * 64}
        content = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        values["structured_records"] = records
        values["structured_index_sha256"] = "sha256:" + hashlib.sha256(content).hexdigest()
    return values


def _availability() -> GenerationAvailability:
    return GenerationAvailability(
        GEN_A,
        revision=3,
        sealed=True,
        available=True,
        ingested=True,
        retrievable=True,
        evaluation_passed=True,
        retained=True,
        evaluation_report_id=REPORT_A,
    )


def _approval_values() -> dict[str, object]:
    return {
        "pk": "approval#approval_activate-a",
        "record_type": "protected_approval",
        "approval_id": "approval_activate-a",
        "action": "activate",
        "generation_id": GEN_A,
        "expected_active_generation": None,
        "evaluation_report_id": REPORT_A,
        "approver": "maintainer",
        "approved_at": NOW,
    }


def test_s3_adapter_publishes_write_once_and_verifies_exact_objects(
    bundle: GenerationBundle,
) -> None:
    client = FakeS3()
    store = S3PublicationStore("corpus-bucket", client)

    first = publish_generation(store, bundle)
    first_call_count = len(client.put_calls)
    second = publish_generation(store, bundle)

    assert first.created_keys
    assert second.created_keys == ()
    assert len(client.put_calls) == first_call_count
    assert all(call["Bucket"] == "corpus-bucket" for call in client.put_calls)
    assert all(call["IfNoneMatch"] == "*" for call in client.put_calls)
    assert client.put_calls[-1]["Key"] == first.completion_key


def test_s3_adapter_paginates_and_fails_closed_on_non_conflict_errors() -> None:
    paged = PagedS3()
    store = S3PublicationStore("corpus-bucket", paged)
    assert store.list_keys("prefix/") == ("prefix/a", "prefix/b")
    assert paged.list_calls[1]["ContinuationToken"] == "page-2"

    existing = FakeS3()
    existing.objects["prefix/object"] = (b"content", {})
    adapter = S3PublicationStore("corpus-bucket", existing)
    assert adapter.put_object_if_absent("prefix/object", b"content", {}) is False
    existing.denied = True
    with pytest.raises(AwsError, match="AccessDenied"):
        adapter.put_object_if_absent("prefix/denied", b"content", {})


def test_candidate_adapter_uses_strong_reads_and_conditional_revision_writes() -> None:
    client = FakeDynamo()
    store = DynamoCandidateStore("state-table", client)
    state = CandidateState(
        GEN_A,
        CandidateStatus.INGESTING,
        revision=1,
        fence=1,
        attempt=1,
        owner="worker-1",
        lease_expires_at=1_800_000_300,
    )

    assert store.read_candidate() is None
    assert store.compare_and_swap(None, state)
    assert store.read_candidate() == state
    assert not store.compare_and_swap(None, state)
    renewed = replace(state, revision=2, lease_expires_at=1_800_000_600)
    assert store.compare_and_swap(1, renewed)
    assert store.read_candidate() == renewed

    assert all(call["ConsistentRead"] is True for call in client.get_calls)
    assert client.put_calls[0]["ConditionExpression"] == "attribute_not_exists(pk)"
    assert client.put_calls[-1]["ConditionExpression"] == (
        "revision = :revision AND record_type = :record_type"
    )


def test_candidate_adapter_rejects_non_exact_state() -> None:
    client = FakeDynamo()
    values = {
        "pk": "candidate_generation",
        "record_type": "candidate_state",
        "generation_id": GEN_A,
        "status": "INGESTING",
        "revision": 1,
        "fence": 1,
        "attempt": 1,
        "owner": "worker",
        "lease_expires_at": 1_800_000_300,
        "ingestion_job_id": None,
        "documents_scanned": None,
        "failure": None,
        "unexpected": "value",
    }
    client.items["candidate_generation"] = _item(values)

    with pytest.raises(IngestionError, match="fields are not exact"):
        DynamoCandidateStore("state-table", client).read_candidate()


def test_generation_lifecycle_adapter_creates_and_transitions_exact_state() -> None:
    client = FakeDynamo()
    store = DynamoPromotionStore("state-table", client)
    record_id = "release_artifact_digest:9.0.0:valkey-9.0.0.tar.gz"
    structured = {
        hashlib.sha256(record_id.encode()).hexdigest(): "sha256:" + "c" * 64,
    }
    sealed = replace(
        _availability(),
        revision=1,
        available=False,
        ingested=False,
        retrievable=False,
        evaluation_passed=False,
        evaluation_report_id=None,
    )

    assert store.create_generation(sealed, structured)
    assert not store.create_generation(sealed, structured)
    assert store.get_generation(GEN_A) == sealed
    key = f"generation#{'a' * 64}"
    root = client.items[key]["structured_index_sha256"]

    ingested = replace(
        sealed,
        revision=2,
        available=True,
        ingested=True,
        retrievable=True,
    )
    assert store.compare_and_swap_generation(sealed, ingested)
    assert store.get_generation(GEN_A) == ingested
    assert client.items[key]["structured_index_sha256"] == root
    assert client.items[key]["structured_records"] == _av(structured)

    transition = client.put_calls[-1]
    names = cast(dict[str, str], transition["ExpressionAttributeNames"])
    assert set(names.values()) == set(_generation_values(structured=True)) - {"pk"}
    assert not store.compare_and_swap_generation(sealed, ingested)


def test_generation_lifecycle_adapter_fails_closed_on_invalid_index_and_race() -> None:
    client = FakeDynamo()
    store = DynamoPromotionStore("state-table", client)
    sealed = replace(
        _availability(),
        revision=1,
        available=False,
        ingested=False,
        retrievable=False,
        evaluation_passed=False,
        evaluation_report_id=None,
    )
    with pytest.raises(PromotionError, match="structured record index is malformed"):
        store.create_generation(sealed, {})
    with pytest.raises(PromotionError, match="structured record index is malformed"):
        store.create_generation(sealed, {"not-a-hash": "sha256:" + "c" * 64})

    structured = {"b" * 64: "sha256:" + "c" * 64}
    assert store.create_generation(sealed, structured)
    client.reject_puts = 1
    ingested = replace(
        sealed,
        revision=2,
        available=True,
        ingested=True,
        retrievable=True,
    )
    assert not store.compare_and_swap_generation(sealed, ingested)
    assert store.get_generation(GEN_A) == sealed
    with pytest.raises(PromotionError, match="not the next revision"):
        store.compare_and_swap_generation(sealed, replace(ingested, revision=3))


def test_promotion_adapter_validates_structured_index_and_swaps_transactionally() -> None:
    client = FakeDynamo()
    client.items[f"generation#{'a' * 64}"] = _item(_generation_values(structured=True))
    store = DynamoPromotionStore("state-table", client)

    assert store.get_generation(GEN_A) == _availability()
    active = ActiveGeneration(GEN_A, 1, REPORT_A, NOW)
    assert store.compare_and_swap_active(None, _availability(), active)
    assert store.read_active() == active

    assert len(client.transaction_calls) == 1
    transaction = cast(list[dict[str, object]], client.transaction_calls[0]["TransactItems"])
    condition = cast(dict[str, object], transaction[0]["ConditionCheck"])
    names = cast(dict[str, str], condition["ExpressionAttributeNames"])
    assert set(names.values()) == set(_generation_values(structured=True)) - {"pk"}
    assert cast(dict[str, object], transaction[1]["Put"])["ConditionExpression"] == (
        "attribute_not_exists(pk)"
    )


def test_promotion_adapter_rejects_corrupt_structured_index_and_transaction_races() -> None:
    client = FakeDynamo()
    values = _generation_values(structured=True)
    values["structured_index_sha256"] = "sha256:" + "0" * 64
    client.items[f"generation#{'a' * 64}"] = _item(values)
    store = DynamoPromotionStore("state-table", client)

    with pytest.raises(PromotionError, match="checksum is invalid"):
        store.get_generation(GEN_A)

    client.items[f"generation#{'a' * 64}"] = _item(_generation_values())
    client.reject_transaction = True
    assert not store.compare_and_swap_active(
        None,
        _availability(),
        ActiveGeneration(GEN_A, 1, REPORT_A, NOW),
    )


def test_approval_adapter_conditionally_consumes_exact_evidence_once() -> None:
    client = FakeDynamo()
    client.items["approval#approval_activate-a"] = _item(_approval_values())
    registry = DynamoApprovalRegistry("state-table", client)

    approval = registry.consume_approval("approval_activate-a")
    assert approval == ProtectedApproval(
        "approval_activate-a",
        "activate",
        GEN_A,
        None,
        REPORT_A,
        "maintainer",
        NOW,
    )
    assert registry.consume_approval("approval_activate-a") is None

    assert all(call["ConsistentRead"] is True for call in client.get_calls)
    assert len(client.update_calls) == 1
    call = client.update_calls[0]
    assert call["UpdateExpression"] == "SET #consumed = :consumed"
    assert cast(str, call["ConditionExpression"]).startswith("attribute_not_exists(#consumed) AND ")
    assert cast(Mapping[str, str], call["ExpressionAttributeNames"])["#consumed"] == ("consumed")
    assert call["ReturnValues"] == "ALL_OLD"


def test_approval_adapter_rejects_malformed_evidence_before_consumption() -> None:
    client = FakeDynamo()
    malformed = _approval_values()
    malformed["generation_id"] = "forged"
    client.items["approval#approval_activate-a"] = _item(malformed)

    with pytest.raises(PromotionError, match="generation ID is malformed"):
        DynamoApprovalRegistry("state-table", client).consume_approval("approval_activate-a")
    assert client.update_calls == []


def test_adapter_constructors_and_responses_fail_closed() -> None:
    with pytest.raises(PublicationError, match="bucket name"):
        S3PublicationStore("", FakeS3())
    with pytest.raises(IngestionError, match="table name"):
        DynamoCandidateStore("", FakeDynamo())
    with pytest.raises(PromotionError, match="table name"):
        DynamoPromotionStore("", FakeDynamo())
