"""Native boto3 adapters for corpus publication, ingestion, and promotion state."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedInvocation,
)
from valkeyrie.ingestion import (
    BedrockIngestionClient,
    CandidateState,
    CandidateStatus,
    ConditionalCandidateStore,
    IngestionError,
)
from valkeyrie.promotion import (
    ActiveGeneration,
    ApprovalRegistry,
    PromotionError,
    PromotionStore,
    ProtectedApproval,
)
from valkeyrie.publication import PublicationError, PublicationStore, StoredObject
from valkeyrie.retrieval import BedrockRetrievalClient, GenerationAvailability


class AwsClient(Protocol):
    """Subset of botocore's dynamic client surface used by these adapters."""

    def list_objects_v2(self, **kwargs: object) -> Mapping[str, object]: ...

    def head_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def put_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_item(self, **kwargs: object) -> Mapping[str, object]: ...

    def put_item(self, **kwargs: object) -> Mapping[str, object]: ...

    def update_item(self, **kwargs: object) -> Mapping[str, object]: ...

    def transact_write_items(self, **kwargs: object) -> Mapping[str, object]: ...


class LambdaClient(Protocol):
    """Single-operation Lambda client with SDK retries disabled by construction."""

    def invoke(self, **kwargs: object) -> Mapping[str, object]: ...


_MAX_LAMBDA_RESULT_BYTES = 1024 * 1024


class AwsLambdaInvoker:
    """Invoke one immutable Lambda version and return one bounded JSON result."""

    def __init__(self, client: LambdaClient) -> None:
        self._client = client

    def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]:
        if not isinstance(invocation, DeployedInvocation):
            raise DeployedEvaluationError("deployed invocation is malformed")
        raw_response = self._client.invoke(
            FunctionName=invocation.function_name,
            Qualifier=invocation.function_qualifier,
            InvocationType="RequestResponse",
            Payload=invocation.payload,
        )
        if not isinstance(raw_response, Mapping):
            raise DeployedEvaluationError("AWS returned a malformed response")
        response = raw_response
        status = response.get("StatusCode")
        if not isinstance(status, int) or isinstance(status, bool) or status != 200:
            raise DeployedEvaluationError("Lambda invocation did not return status 200")
        if "FunctionError" in response:
            raise DeployedEvaluationError("Lambda invocation returned a function error")
        if response.get("ExecutedVersion") != invocation.function_qualifier:
            raise DeployedEvaluationError("Lambda executed version differs from the immutable pin")
        body = response.get("Payload")
        read = getattr(body, "read", None)
        if not callable(read):
            raise DeployedEvaluationError("Lambda response payload is malformed")
        try:
            content = read(_MAX_LAMBDA_RESULT_BYTES + 1)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if not isinstance(content, bytes) or not 1 <= len(content) <= _MAX_LAMBDA_RESULT_BYTES:
            raise DeployedEvaluationError("Lambda response payload exceeds its byte bound")

        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate key {key!r}")
                value[key] = item
            return value

        def reject_constant(value: str) -> object:
            raise ValueError(f"non-finite JSON constant {value!r}")

        try:
            result = json.loads(
                content.decode("utf-8"),
                object_pairs_hook=unique,
                parse_constant=reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise DeployedEvaluationError("Lambda response payload is not strict JSON") from error
        if not isinstance(result, Mapping):
            raise DeployedEvaluationError("Lambda response payload root is malformed")
        return cast(Mapping[str, object], result)


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPORT_ID = re.compile(r"^eval_[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[A-Z0-9]{10}$")
_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$")
_APPROVAL_ID = re.compile(r"^approval_[a-z0-9-]+$")
_CANDIDATE_PK = "candidate_generation"
_ACTIVE_PK = "active_generation"


class S3PublicationStore(PublicationStore):
    """Strongly verified, conditional S3 storage for immutable generation objects."""

    def __init__(self, bucket_name: str, client: AwsClient | None = None) -> None:
        if not isinstance(bucket_name, str) or not bucket_name:
            raise PublicationError("publication bucket name is invalid")
        self._bucket_name = bucket_name
        self._client = client if client is not None else _default_client("s3")

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        if not isinstance(prefix, str) or not prefix:
            raise PublicationError("publication prefix is invalid")
        keys: list[str] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            request: dict[str, object] = {"Bucket": self._bucket_name, "Prefix": prefix}
            if token is not None:
                request["ContinuationToken"] = token
            response = _response(self._client.list_objects_v2(**request), PublicationError)
            contents = response.get("Contents", [])
            if not isinstance(contents, list):
                raise PublicationError("S3 object listing is malformed")
            for value in contents:
                if not isinstance(value, Mapping):
                    raise PublicationError("S3 object listing contains a malformed entry")
                key = value.get("Key")
                if not isinstance(key, str) or not key.startswith(prefix):
                    raise PublicationError("S3 object listing contains an invalid key")
                keys.append(key)
            truncated = response.get("IsTruncated", False)
            if not isinstance(truncated, bool):
                raise PublicationError("S3 object listing truncation state is malformed")
            if not truncated:
                return tuple(keys)
            next_token = response.get("NextContinuationToken")
            if not isinstance(next_token, str) or not next_token or next_token in seen_tokens:
                raise PublicationError("S3 object listing continuation is malformed")
            seen_tokens.add(next_token)
            token = next_token

    def head_object(self, key: str) -> StoredObject | None:
        _validate_key(key, PublicationError)
        try:
            response = _response(
                self._client.head_object(Bucket=self._bucket_name, Key=key),
                PublicationError,
            )
        except Exception as error:
            if _error_code(error) in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        size = response.get("ContentLength")
        metadata = response.get("Metadata")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(metadata, Mapping)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())
        ):
            raise PublicationError("S3 object metadata is malformed")
        return StoredObject(size, dict(cast(Mapping[str, str], metadata)))

    def get_object(self, key: str) -> bytes:
        _validate_key(key, PublicationError)
        response = _response(
            self._client.get_object(Bucket=self._bucket_name, Key=key),
            PublicationError,
        )
        body = response.get("Body")
        read = getattr(body, "read", None)
        if not callable(read):
            raise PublicationError("S3 object body is malformed")
        try:
            content = read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if not isinstance(content, bytes):
            raise PublicationError("S3 object body is not bytes")
        return content

    def put_object_if_absent(
        self,
        key: str,
        content: bytes,
        metadata: Mapping[str, str],
    ) -> bool:
        _validate_key(key, PublicationError)
        if not isinstance(content, bytes) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
        ):
            raise PublicationError("conditional S3 object input is malformed")
        try:
            _response(
                self._client.put_object(
                    Bucket=self._bucket_name,
                    Key=key,
                    Body=content,
                    Metadata=dict(metadata),
                    IfNoneMatch="*",
                ),
                PublicationError,
            )
        except Exception as error:
            if _error_code(error) in {"PreconditionFailed", "ConditionalRequestConflict"}:
                return False
            raise
        return True


class DynamoCandidateStore(ConditionalCandidateStore):
    """Strongly consistent singleton candidate state with revision CAS writes."""

    def __init__(self, table_name: str, client: AwsClient | None = None) -> None:
        self._table_name = _table_name(table_name, IngestionError)
        self._client = client if client is not None else _default_client("dynamodb")

    def read_candidate(self) -> CandidateState | None:
        response = _response(
            self._client.get_item(
                TableName=self._table_name,
                Key=_item({"pk": _CANDIDATE_PK}),
                ConsistentRead=True,
            ),
            IngestionError,
        )
        raw = response.get("Item")
        if raw is None:
            return None
        try:
            return _candidate_from_item(_decode_item(raw))
        except (TypeError, ValueError) as error:
            raise IngestionError(f"candidate state item is malformed: {error}") from error

    def compare_and_swap(
        self,
        expected_revision: int | None,
        replacement: CandidateState,
    ) -> bool:
        values = _candidate_values(replacement)
        if expected_revision is None:
            condition = "attribute_not_exists(pk)"
            expression_values: dict[str, dict[str, object]] | None = None
        else:
            _positive_int(expected_revision, "expected candidate revision")
            condition = "revision = :revision AND record_type = :record_type"
            expression_values = _item(
                {":revision": expected_revision, ":record_type": "candidate_state"}
            )
        request: dict[str, object] = {
            "TableName": self._table_name,
            "Item": _item({"pk": _CANDIDATE_PK, **values}),
            "ConditionExpression": condition,
        }
        if expression_values is not None:
            request["ExpressionAttributeValues"] = expression_values
        try:
            _response(self._client.put_item(**request), IngestionError)
        except Exception as error:
            if _error_code(error) == "ConditionalCheckFailedException":
                return False
            raise
        return True


@dataclass(frozen=True)
class _GenerationSnapshot:
    record: GenerationAvailability
    values: Mapping[str, object]


class DynamoPromotionStore(PromotionStore):
    """Strong generation reads and transactional active-pointer compare-and-swap."""

    def __init__(self, table_name: str, client: AwsClient | None = None) -> None:
        self._table_name = _table_name(table_name, PromotionError)
        self._client = client if client is not None else _default_client("dynamodb")

    def get_generation(self, generation_id: str) -> GenerationAvailability | None:
        snapshot = self._read_generation(generation_id)
        return None if snapshot is None else snapshot.record

    def create_generation(
        self,
        record: GenerationAvailability,
        structured_records: Mapping[str, str],
    ) -> bool:
        values = {
            **_generation_values(record),
            **_structured_index_values(structured_records),
        }
        try:
            _response(
                self._client.put_item(
                    TableName=self._table_name,
                    Item=_item({"pk": _generation_pk(record.generation_id), **values}),
                    ConditionExpression="attribute_not_exists(pk)",
                ),
                PromotionError,
            )
        except Exception as error:
            if _error_code(error) == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def compare_and_swap_generation(
        self,
        expected: GenerationAvailability,
        replacement: GenerationAvailability,
    ) -> bool:
        if (
            replacement.generation_id != expected.generation_id
            or replacement.revision != expected.revision + 1
        ):
            raise PromotionError("generation lifecycle replacement is not the next revision")
        replacement_values = _generation_values(replacement)
        snapshot = self._read_generation(expected.generation_id)
        if snapshot is None or snapshot.record != expected:
            return False
        structured_values = {
            name: value
            for name, value in snapshot.values.items()
            if name in {"structured_records", "structured_index_sha256"}
        }
        names, expression_values, condition = _exact_condition(snapshot.values)
        try:
            _response(
                self._client.put_item(
                    TableName=self._table_name,
                    Item=_item(
                        {
                            "pk": _generation_pk(replacement.generation_id),
                            **replacement_values,
                            **structured_values,
                        }
                    ),
                    ConditionExpression=condition,
                    ExpressionAttributeNames=names,
                    ExpressionAttributeValues=expression_values,
                ),
                PromotionError,
            )
        except Exception as error:
            if _error_code(error) == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def read_active(self) -> ActiveGeneration | None:
        response = _response(
            self._client.get_item(
                TableName=self._table_name,
                Key=_item({"pk": _ACTIVE_PK}),
                ConsistentRead=True,
            ),
            PromotionError,
        )
        raw = response.get("Item")
        if raw is None:
            return None
        try:
            return _active_from_item(_decode_item(raw))
        except (TypeError, ValueError) as error:
            raise PromotionError(f"active generation item is malformed: {error}") from error

    def compare_and_swap_active(
        self,
        expected_active_revision: int | None,
        expected_generation: GenerationAvailability,
        replacement: ActiveGeneration,
    ) -> bool:
        replacement_values = _active_values(replacement)
        snapshot = self._read_generation(expected_generation.generation_id)
        if snapshot is None or snapshot.record != expected_generation:
            return False
        generation_names, generation_values, generation_condition = _exact_condition(
            snapshot.values
        )
        if expected_active_revision is None:
            active_condition = "attribute_not_exists(pk)"
            active_values: dict[str, dict[str, object]] | None = None
        else:
            _positive_int(expected_active_revision, "expected active revision")
            active_condition = "revision = :revision AND record_type = :record_type"
            active_values = _item(
                {
                    ":revision": expected_active_revision,
                    ":record_type": "active_generation",
                }
            )
        put: dict[str, object] = {
            "TableName": self._table_name,
            "Item": _item({"pk": _ACTIVE_PK, **replacement_values}),
            "ConditionExpression": active_condition,
        }
        if active_values is not None:
            put["ExpressionAttributeValues"] = active_values
        transaction = [
            {
                "ConditionCheck": {
                    "TableName": self._table_name,
                    "Key": _item({"pk": _generation_pk(expected_generation.generation_id)}),
                    "ConditionExpression": generation_condition,
                    "ExpressionAttributeNames": generation_names,
                    "ExpressionAttributeValues": generation_values,
                }
            },
            {"Put": put},
        ]
        try:
            _response(
                self._client.transact_write_items(TransactItems=transaction),
                PromotionError,
            )
        except Exception as error:
            if _error_code(error) in {
                "ConditionalCheckFailedException",
                "TransactionCanceledException",
            }:
                return False
            raise
        return True

    def _read_generation(self, generation_id: str) -> _GenerationSnapshot | None:
        _digest(generation_id, "generation ID")
        response = _response(
            self._client.get_item(
                TableName=self._table_name,
                Key=_item({"pk": _generation_pk(generation_id)}),
                ConsistentRead=True,
            ),
            PromotionError,
        )
        raw = response.get("Item")
        if raw is None:
            return None
        try:
            return _generation_from_item(_decode_item(raw), generation_id)
        except (TypeError, ValueError) as error:
            raise PromotionError(f"generation state item is malformed: {error}") from error


class DynamoApprovalRegistry(ApprovalRegistry):
    """Conditionally and permanently consume protected approvals in DynamoDB."""

    def __init__(self, table_name: str, client: AwsClient | None = None) -> None:
        self._table_name = _table_name(table_name, PromotionError)
        self._client = client if client is not None else _default_client("dynamodb")

    def consume_approval(self, approval_id: str) -> ProtectedApproval | None:
        if not isinstance(approval_id, str) or not _APPROVAL_ID.fullmatch(approval_id):
            raise PromotionError("protected approval ID is malformed")
        key = _item({"pk": f"approval#{approval_id}"})
        response = _response(
            self._client.get_item(
                TableName=self._table_name,
                Key=key,
                ConsistentRead=True,
            ),
            PromotionError,
        )
        raw = response.get("Item")
        if raw is None:
            return None
        try:
            values = _decode_item(raw)
            if values.get("consumed") is True:
                return None
            approval = _approval_from_item(values, approval_id)
            condition_names, condition_values, condition = _exact_condition(
                {key: value for key, value in values.items() if key != "pk"}
            )
        except (TypeError, ValueError) as error:
            raise PromotionError(f"protected approval item is malformed: {error}") from error
        condition_names["#consumed"] = "consumed"
        condition = f"attribute_not_exists(#consumed) AND {condition}"
        try:
            updated = _response(
                self._client.update_item(
                    TableName=self._table_name,
                    Key=key,
                    UpdateExpression="SET #consumed = :consumed",
                    ConditionExpression=condition,
                    ExpressionAttributeNames=condition_names,
                    ExpressionAttributeValues={
                        **condition_values,
                        ":consumed": {"BOOL": True},
                    },
                    ReturnValues="ALL_OLD",
                ),
                PromotionError,
            )
        except Exception as error:
            if _error_code(error) == "ConditionalCheckFailedException":
                return None
            raise
        try:
            returned = _approval_from_item(_decode_item(updated.get("Attributes")), approval_id)
        except (TypeError, ValueError) as error:
            raise PromotionError(f"consumed approval response is malformed: {error}") from error
        if returned != approval:
            raise PromotionError("consumed approval differs from its strongly consistent read")
        return approval


def create_lambda_client(region_name: str) -> LambdaClient:
    """Create a regional Lambda client with exactly one SDK attempt."""
    if (
        not isinstance(region_name, str)
        or re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]", region_name) is None
    ):
        raise ValueError("AWS region is malformed")
    try:
        import boto3  # type: ignore[import-not-found]
        from botocore.config import Config  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - operator environment supplies boto3
        raise RuntimeError("boto3 is required for AWS adapters") from error
    config = Config(
        retries={"mode": "standard", "total_max_attempts": 1},
        connect_timeout=10,
        read_timeout=900,
    )
    return cast(
        LambdaClient,
        boto3.client("lambda", region_name=region_name, config=config),
    )


def create_s3_client(region_name: str) -> AwsClient:
    """Create an S3 client pinned to one explicit AWS region."""
    return _default_client("s3", region_name)


def create_dynamodb_client(region_name: str) -> AwsClient:
    """Create a DynamoDB client pinned to one explicit AWS region."""
    return _default_client("dynamodb", region_name)


def create_bedrock_ingestion_client(
    region_name: str | None = None,
) -> BedrockIngestionClient:
    """Create the native Bedrock Agent client accepted by ``run_ingestion``."""
    return cast(BedrockIngestionClient, _default_client("bedrock-agent", region_name))


def create_bedrock_retrieval_client(
    region_name: str | None = None,
) -> BedrockRetrievalClient:
    """Create the native Bedrock Agent Runtime client accepted by promotion smoke tests."""
    return cast(
        BedrockRetrievalClient,
        _default_client("bedrock-agent-runtime", region_name),
    )


def _default_client(service_name: str, region_name: str | None = None) -> AwsClient:
    if region_name is not None and (
        not isinstance(region_name, str)
        or re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]", region_name) is None
    ):
        raise ValueError("AWS region is malformed")
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - production runtime supplies boto3
        raise RuntimeError("boto3 is required for AWS adapters") from error
    options = {} if region_name is None else {"region_name": region_name}
    return cast(AwsClient, boto3.client(service_name, **options))


def _response(value: object, error_type: type[RuntimeError]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise error_type("AWS returned a malformed response")
    return cast(Mapping[str, object], value)


def _error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return code if isinstance(code, str) else None


def _validate_key(key: object, error_type: type[RuntimeError]) -> None:
    if (
        not isinstance(key, str)
        or not key
        or key.startswith("/")
        or "//" in key
        or ".." in key.split("/")
    ):
        raise error_type("AWS object key is invalid")


def _table_name(value: object, error_type: type[RuntimeError]) -> str:
    if not isinstance(value, str) or not value:
        raise error_type("DynamoDB table name is invalid")
    return value


def _attribute(value: object) -> dict[str, object]:
    if value is None:
        return {"NULL": True}
    if isinstance(value, bool):
        return {"BOOL": value}
    if isinstance(value, int):
        return {"N": str(value)}
    if isinstance(value, str):
        return {"S": value}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("DynamoDB map key is not a string")
        return {
            "M": {key: _attribute(item) for key, item in cast(Mapping[str, object], value).items()}
        }
    raise ValueError("unsupported DynamoDB attribute type")


def _item(values: Mapping[str, object]) -> dict[str, dict[str, object]]:
    return {key: _attribute(value) for key, value in values.items()}


def _decode_attribute(value: object) -> object:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise ValueError("DynamoDB attribute is malformed")
    kind, raw = next(iter(value.items()))
    if kind == "NULL" and raw is True:
        return None
    if kind == "BOOL" and isinstance(raw, bool):
        return raw
    if kind == "S" and isinstance(raw, str):
        return raw
    if kind == "N" and isinstance(raw, str) and re.fullmatch(r"0|[1-9][0-9]*", raw):
        return int(raw)
    if kind == "M" and isinstance(raw, Mapping):
        if any(not isinstance(key, str) for key in raw):
            raise ValueError("DynamoDB map key is malformed")
        return {
            key: _decode_attribute(item) for key, item in cast(Mapping[str, object], raw).items()
        }
    raise ValueError("DynamoDB attribute type or value is malformed")


def _decode_item(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("DynamoDB item is malformed")
    return {
        key: _decode_attribute(attribute)
        for key, attribute in cast(Mapping[str, object], value).items()
    }


def _exact_keys(values: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(values) != expected:
        raise ValueError(f"{label} fields are not exact")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is malformed")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} is malformed")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} is malformed")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} is malformed")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} is malformed")
    return value


def _report_id(value: object, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _REPORT_ID.fullmatch(value):
        raise ValueError(f"{label} is malformed")
    return value


def _candidate_values(state: object) -> dict[str, object]:
    if not isinstance(state, CandidateState):
        raise IngestionError("candidate replacement has the wrong runtime type")
    generation_id = _digest(state.generation_id, "candidate generation ID")
    if not isinstance(state.status, CandidateStatus):
        raise IngestionError("candidate status is malformed")
    owner = _string(state.owner, "candidate owner")
    if len(owner) > 128:
        raise IngestionError("candidate owner is malformed")
    job_id = state.ingestion_job_id
    if job_id is not None and (not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id)):
        raise IngestionError("candidate ingestion job ID is malformed")
    documents = state.documents_scanned
    if documents is not None:
        documents = _nonnegative_int(documents, "candidate document statistics")
    failure = state.failure
    if failure is not None and (not isinstance(failure, str) or not failure or len(failure) > 500):
        raise IngestionError("candidate failure is malformed")
    return {
        "record_type": "candidate_state",
        "generation_id": generation_id,
        "status": state.status.value,
        "revision": _positive_int(state.revision, "candidate revision"),
        "fence": _positive_int(state.fence, "candidate fence"),
        "attempt": _positive_int(state.attempt, "candidate attempt"),
        "owner": owner,
        "lease_expires_at": _nonnegative_int(state.lease_expires_at, "candidate lease expiry"),
        "ingestion_job_id": job_id,
        "documents_scanned": documents,
        "failure": failure,
    }


def _candidate_from_item(values: Mapping[str, object]) -> CandidateState:
    fields = {
        "pk",
        "record_type",
        "generation_id",
        "status",
        "revision",
        "fence",
        "attempt",
        "owner",
        "lease_expires_at",
        "ingestion_job_id",
        "documents_scanned",
        "failure",
    }
    _exact_keys(values, fields, "candidate state")
    if values["pk"] != _CANDIDATE_PK or values["record_type"] != "candidate_state":
        raise ValueError("candidate state identity is malformed")
    try:
        status = CandidateStatus(_string(values["status"], "candidate status"))
    except ValueError as error:
        raise ValueError("candidate status is malformed") from error
    state = CandidateState(
        generation_id=_digest(values["generation_id"], "candidate generation ID"),
        status=status,
        revision=_positive_int(values["revision"], "candidate revision"),
        fence=_positive_int(values["fence"], "candidate fence"),
        attempt=_positive_int(values["attempt"], "candidate attempt"),
        owner=_string(values["owner"], "candidate owner"),
        lease_expires_at=_nonnegative_int(values["lease_expires_at"], "candidate lease expiry"),
        ingestion_job_id=cast(str | None, values["ingestion_job_id"]),
        documents_scanned=cast(int | None, values["documents_scanned"]),
        failure=cast(str | None, values["failure"]),
    )
    _candidate_values(state)
    return state


def _generation_pk(generation_id: str) -> str:
    return f"generation#{generation_id.removeprefix('sha256:')}"


def _generation_values(record: object) -> dict[str, object]:
    if not isinstance(record, GenerationAvailability):
        raise PromotionError("generation state has the wrong runtime type")
    generation_id = _digest(record.generation_id, "generation ID")
    return {
        "record_type": "generation_availability",
        "generation_id": generation_id,
        "revision": _positive_int(record.revision, "generation revision"),
        "sealed": _boolean(record.sealed, "generation sealed flag"),
        "available": _boolean(record.available, "generation available flag"),
        "ingested": _boolean(record.ingested, "generation ingested flag"),
        "retrievable": _boolean(record.retrievable, "generation retrievable flag"),
        "evaluation_passed": _boolean(record.evaluation_passed, "generation evaluation flag"),
        "retained": _boolean(record.retained, "generation retained flag"),
        "evaluation_report_id": _report_id(
            record.evaluation_report_id, "generation report ID", optional=True
        ),
    }


def _structured_index_values(structured_records: object) -> dict[str, object]:
    if not isinstance(structured_records, Mapping) or not 1 <= len(structured_records) <= 10_000:
        raise PromotionError("structured record index is malformed")
    if any(
        not isinstance(key, str)
        or re.fullmatch(r"[0-9a-f]{64}", key) is None
        or not isinstance(value, str)
        or _DIGEST.fullmatch(value) is None
        for key, value in structured_records.items()
    ):
        raise PromotionError("structured record index is malformed")
    normalized = dict(cast(Mapping[str, str], structured_records))
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "structured_records": normalized,
        "structured_index_sha256": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    }


def _generation_from_item(
    values: Mapping[str, object], expected_generation_id: str
) -> _GenerationSnapshot:
    required = {
        "pk",
        "record_type",
        "generation_id",
        "revision",
        "sealed",
        "available",
        "ingested",
        "retrievable",
        "evaluation_passed",
        "retained",
        "evaluation_report_id",
    }
    optional = {"structured_records", "structured_index_sha256"}
    if set(values) not in {frozenset(required), frozenset(required | optional)}:
        raise ValueError("generation state fields are not exact")
    if (
        values["pk"] != _generation_pk(expected_generation_id)
        or values["record_type"] != "generation_availability"
        or values["generation_id"] != expected_generation_id
    ):
        raise ValueError("generation state identity is malformed")
    record = GenerationAvailability(
        generation_id=_digest(values["generation_id"], "generation ID"),
        revision=_positive_int(values["revision"], "generation revision"),
        sealed=_boolean(values["sealed"], "generation sealed flag"),
        available=_boolean(values["available"], "generation available flag"),
        ingested=_boolean(values["ingested"], "generation ingested flag"),
        retrievable=_boolean(values["retrievable"], "generation retrievable flag"),
        evaluation_passed=_boolean(values["evaluation_passed"], "generation evaluation flag"),
        retained=_boolean(values["retained"], "generation retained flag"),
        evaluation_report_id=_report_id(
            values["evaluation_report_id"], "generation report ID", optional=True
        ),
    )
    condition_values = _generation_values(record)
    if optional <= set(values):
        structured = values["structured_records"]
        if (
            not isinstance(structured, Mapping)
            or not 1 <= len(structured) <= 10_000
            or any(
                not isinstance(key, str)
                or not re.fullmatch(r"[0-9a-f]{64}", key)
                or not isinstance(value, str)
                or not _DIGEST.fullmatch(value)
                for key, value in structured.items()
            )
        ):
            raise ValueError("structured record index is malformed")
        normalized = dict(cast(Mapping[str, str], structured))
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected_root = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        if values["structured_index_sha256"] != expected_root:
            raise ValueError("structured record index checksum is invalid")
        condition_values.update(
            {
                "structured_records": normalized,
                "structured_index_sha256": expected_root,
            }
        )
    return _GenerationSnapshot(record, condition_values)


def _active_values(active: object) -> dict[str, object]:
    if not isinstance(active, ActiveGeneration):
        raise PromotionError("active generation has the wrong runtime type")
    timestamp = _string(active.activated_at, "active generation timestamp")
    if not _TIMESTAMP.fullmatch(timestamp):
        raise PromotionError("active generation timestamp is malformed")
    return {
        "record_type": "active_generation",
        "generation_id": _digest(active.generation_id, "active generation ID"),
        "revision": _positive_int(active.revision, "active generation revision"),
        "evaluation_report_id": _report_id(
            active.evaluation_report_id, "active generation report ID"
        ),
        "activated_at": timestamp,
    }


def _active_from_item(values: Mapping[str, object]) -> ActiveGeneration:
    fields = {
        "pk",
        "record_type",
        "generation_id",
        "revision",
        "evaluation_report_id",
        "activated_at",
    }
    _exact_keys(values, fields, "active generation")
    if values["pk"] != _ACTIVE_PK or values["record_type"] != "active_generation":
        raise ValueError("active generation identity is malformed")
    active = ActiveGeneration(
        generation_id=_digest(values["generation_id"], "active generation ID"),
        revision=_positive_int(values["revision"], "active generation revision"),
        evaluation_report_id=cast(
            str, _report_id(values["evaluation_report_id"], "active generation report ID")
        ),
        activated_at=_string(values["activated_at"], "active generation timestamp"),
    )
    _active_values(active)
    return active


def _approval_from_item(
    values: Mapping[str, object], expected_approval_id: str
) -> ProtectedApproval:
    fields = {
        "pk",
        "record_type",
        "approval_id",
        "action",
        "generation_id",
        "expected_active_generation",
        "evaluation_report_id",
        "approver",
        "approved_at",
    }
    _exact_keys(values, fields, "protected approval")
    if (
        values["pk"] != f"approval#{expected_approval_id}"
        or values["record_type"] != "protected_approval"
        or values["approval_id"] != expected_approval_id
    ):
        raise ValueError("protected approval identity is malformed")
    action_value = values["action"]
    action: Literal["activate", "rollback"]
    if action_value == "activate":
        action = "activate"
    elif action_value == "rollback":
        action = "rollback"
    else:
        raise ValueError("protected approval action is malformed")
    expected = values["expected_active_generation"]
    if expected is not None:
        expected = _digest(expected, "protected approval expected generation")
    approver = _string(values["approver"], "protected approval approver")
    if len(approver) > 128:
        raise ValueError("protected approval approver is malformed")
    approved_at = _string(values["approved_at"], "protected approval timestamp")
    if not _TIMESTAMP.fullmatch(approved_at):
        raise ValueError("protected approval timestamp is malformed")
    return ProtectedApproval(
        approval_id=expected_approval_id,
        action=action,
        generation_id=_digest(values["generation_id"], "protected approval generation ID"),
        expected_active_generation=expected,
        evaluation_report_id=cast(
            str, _report_id(values["evaluation_report_id"], "protected approval report ID")
        ),
        approver=approver,
        approved_at=approved_at,
    )


def _exact_condition(
    values: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, dict[str, object]], str]:
    names: dict[str, str] = {}
    expression_values: dict[str, dict[str, object]] = {}
    clauses: list[str] = []
    for index, (name, value) in enumerate(sorted(values.items())):
        name_key = f"#n{index}"
        value_key = f":v{index}"
        names[name_key] = name
        expression_values[value_key] = _attribute(value)
        clauses.append(f"{name_key} = {value_key}")
    return names, expression_values, " AND ".join(clauses)
