import json
from collections.abc import Mapping
from typing import cast

import pytest

from valkeyrie.aws_adapters import AwsLambdaInvoker
from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedInvocation,
)


class Payload:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.closed = False
        self.read_limits: list[int] = []

    def read(self, amount: int) -> bytes:
        self.read_limits.append(amount)
        return self.content[:amount]

    def close(self) -> None:
        self.closed = True


class LambdaClient:
    def __init__(self, response: Mapping[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def invoke(self, **kwargs: object) -> Mapping[str, object]:
        self.calls.append(dict(kwargs))
        return self.response


def _invocation() -> DeployedInvocation:
    return DeployedInvocation(
        function_name="valkeyrie-development-application",
        function_qualifier="7",
        application_revision="sha256:" + "a" * 64,
        evaluation_suite_revision="sha256:" + "b" * 64,
        generation_id="sha256:" + "c" * 64,
        request_id="req_deployed-eval-" + "d" * 64,
        payload=("{" + '"request_id":"req_deployed-eval-' + "d" * 64 + '"}').encode(),
    )


def _result() -> dict[str, object]:
    return {
        "outcome": "answer",
        "request_id": _invocation().request_id,
        "message": "Verified answer.",
        "claims": [],
        "citations": [],
        "generation_id": _invocation().generation_id,
        "request_revision": "sha256:" + "e" * 64,
        "request_fence": 1,
    }


def test_invokes_exact_immutable_version_once_and_returns_strict_json() -> None:
    payload = Payload(json.dumps(_result(), separators=(",", ":")).encode())
    client = LambdaClient({"StatusCode": 200, "ExecutedVersion": "7", "Payload": payload})

    result = AwsLambdaInvoker(client).invoke(_invocation())

    assert result == _result()
    assert client.calls == [
        {
            "FunctionName": "valkeyrie-development-application",
            "Qualifier": "7",
            "InvocationType": "RequestResponse",
            "Payload": _invocation().payload,
        }
    ]
    assert payload.read_limits == [1024 * 1024 + 1]
    assert payload.closed is True


@pytest.mark.parametrize(
    "response",
    [
        {"StatusCode": 202, "ExecutedVersion": "7", "Payload": Payload(b"{}")},
        {
            "StatusCode": 200,
            "ExecutedVersion": "7",
            "FunctionError": "Unhandled",
            "Payload": Payload(b"{}"),
        },
        {"StatusCode": 200, "ExecutedVersion": "6", "Payload": Payload(b"{}")},
    ],
)
def test_rejects_non_successful_or_wrong_version_envelopes(
    response: Mapping[str, object],
) -> None:
    client = LambdaClient(response)
    with pytest.raises(DeployedEvaluationError):
        AwsLambdaInvoker(client).invoke(_invocation())
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b'{"outcome":"answer","outcome":"error"}',
        b'{"value":NaN}',
        b"[]",
        b"x" * (1024 * 1024 + 1),
    ],
)
def test_rejects_empty_duplicate_nonfinite_nonobject_or_oversized_payloads(
    content: bytes,
) -> None:
    payload = Payload(content)
    client = LambdaClient({"StatusCode": 200, "ExecutedVersion": "7", "Payload": payload})
    with pytest.raises(DeployedEvaluationError):
        AwsLambdaInvoker(client).invoke(_invocation())
    assert payload.closed is True


def test_rejects_non_mapping_sdk_response() -> None:
    class MalformedClient:
        def invoke(self, **kwargs: object) -> Mapping[str, object]:
            del kwargs
            return cast(Mapping[str, object], object())

    with pytest.raises(DeployedEvaluationError, match="malformed response"):
        AwsLambdaInvoker(MalformedClient()).invoke(_invocation())
