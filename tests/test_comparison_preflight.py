import base64
import json
from collections.abc import Mapping
from typing import cast

import pytest

from infra.run_comparison_case import (
    AwsComparisonPreflight,
    ComparisonEvaluationIdentity,
)
from valkeyrie.deployed_evaluation import DeployedEvaluationError

ARTIFACT = "sha256:d0cfac9167b343eb75f3875e688d6ef3bf596f21c2f0a1588f5de49f2d6ae303"
APPLICATION = "sha256:fca8f94a37959e127240239526448855ca9a93fc16a910b0294ebfc800725a54"
PROFILE = "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-opus-5"


def _identity() -> ComparisonEvaluationIdentity:
    return ComparisonEvaluationIdentity(
        "valkeyrie-development-application",
        "8",
        APPLICATION,
        ARTIFACT,
        "sha256:394c725de332cb66cff5808d50c39f19a66979d27e377ac89c0eb4e62295ef89",
        "sha256:6760b22e3e09ac0e2ac64c9c23b3a77733f5bd1f8f8c9704c66d2c69d1362b60",
        "ONVASJDDNX",
        "us.anthropic.claude-opus-5",
        PROFILE,
        "sha256:5bfca020eed7a24b06c6e92c10cb2ae99b99b1a82a257e679eba40bd8faba8d5",
        "owner_directed_comparison",
        "not_run_not_qualified",
    )


class Payload:
    def __init__(self, value: Mapping[str, object]) -> None:
        self.content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def read(self, amount: int) -> bytes:
        return self.content[:amount]

    def close(self) -> None:
        pass


class Client:
    def __init__(self) -> None:
        self.configuration: dict[str, object] = {
            "FunctionName": _identity().function_name,
            "Version": _identity().function_qualifier,
            "CodeSha256": base64.b64encode(bytes.fromhex(ARTIFACT[7:])).decode(),
            "Timeout": 240,
            "Environment": {
                "Variables": {
                    "APPLICATION_REVISION": APPLICATION,
                    "SELECTED_INFERENCE_PROFILE_ARN": PROFILE,
                }
            },
        }
        self.health: dict[str, object] = {
            "status": "healthy",
            "application_revision": APPLICATION,
            "selection_id": _identity().selection_id,
            "selected_model_revision": _identity().selected_model_revision,
            "selected_inference_profile_arn": PROFILE,
            "execution_authorization": "owner_directed_comparison",
            "qualification_status": "not_run_not_qualified",
            "synthetic_path": {},
        }
        self.get_calls: list[dict[str, object]] = []
        self.invoke_calls: list[dict[str, object]] = []

    def get_function(self, **kwargs: object) -> Mapping[str, object]:
        self.get_calls.append(dict(kwargs))
        return {"Configuration": self.configuration}

    def invoke(self, **kwargs: object) -> Mapping[str, object]:
        self.invoke_calls.append(dict(kwargs))
        return {
            "StatusCode": 200,
            "ExecutedVersion": _identity().function_qualifier,
            "Payload": Payload(self.health),
        }


def test_preflight_binds_code_configuration_and_runtime_manifest_identity() -> None:
    client = Client()
    proof = AwsComparisonPreflight(client).verify(_identity())

    configuration = cast(Mapping[str, object], proof["configuration"])
    assert configuration["CodeSha256"] == client.configuration["CodeSha256"]
    assert proof["health"] == client.health
    assert client.get_calls == [
        {
            "FunctionName": _identity().function_name,
            "Qualifier": _identity().function_qualifier,
        }
    ]
    assert len(client.invoke_calls) == 1
    health_request = json.loads(cast(bytes, client.invoke_calls[0]["Payload"]))
    assert health_request == {"action": "health", "application_revision": APPLICATION}


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("configuration", "CodeSha256", "wrong", "configuration differs"),
        ("configuration", "Timeout", 30, "configuration differs"),
        ("health", "selected_model_revision", "us.anthropic.claude-fable-5", "health identity"),
        ("health", "qualification_status", "qualified", "health identity"),
    ],
)
def test_preflight_rejects_mismatched_deployment_identity(
    target: str, field: str, value: object, message: str
) -> None:
    client = Client()
    getattr(client, target)[field] = value

    with pytest.raises(DeployedEvaluationError, match=message):
        AwsComparisonPreflight(client).verify(_identity())
