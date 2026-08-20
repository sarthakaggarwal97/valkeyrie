from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Final, NoReturn, cast

NETWORK_POLICY_LOGICAL_ID: Final = "VectorNetworkPolicy"
APPROVED_FINAL_TEMPLATE_SHA256: Final = (
    "sha256:85021302ee2b6afe61fac0c77617e7db606042dc58696b2e0d9c1ebae43d477c"
)
COLLECTION_RESOURCE: Final = "collection/valkeyrie-development-vectors"
PRIVATE_NETWORK_POLICY: Final = [
    {
        "AllowFromPublic": False,
        "Description": "Bedrock private collection access only",
        "Rules": [
            {
                "Resource": [COLLECTION_RESOURCE],
                "ResourceType": "collection",
            }
        ],
        "SourceServices": ["bedrock.amazonaws.com"],
    }
]
BOOTSTRAP_NETWORK_POLICY: Final = [
    {
        "AllowFromPublic": True,
        "Description": "Temporary CloudFormation index provisioning only",
        "Rules": [
            {
                "Resource": [COLLECTION_RESOURCE],
                "ResourceType": "collection",
            }
        ],
    }
]
_EXPECTED_INDEX_DEPENDENCIES: Final = {
    "VectorCollection",
    "VectorDataAccessPolicy",
    "VectorNetworkPolicy",
}


class DeploymentTemplateError(ValueError):
    """The final template cannot safely produce the reviewed bootstrap template."""


def template_sha256(template_bytes: bytes) -> str:
    """Return the exact SHA-256 identity of deployment template bytes."""
    if not isinstance(template_bytes, bytes):
        raise DeploymentTemplateError("template must be bytes")
    return f"sha256:{hashlib.sha256(template_bytes).hexdigest()}"


def build_index_bootstrap_template(final_template_bytes: bytes) -> bytes:
    """Derive the one-property public bootstrap needed by the native index provider."""
    template = _load_exact_json(final_template_bytes)
    if not isinstance(template, dict):
        raise DeploymentTemplateError("template root must be an object")
    if template.get("Parameters") not in (None, {}):
        raise DeploymentTemplateError("deployment template must not accept parameters")

    resources_value = template.get("Resources")
    if not isinstance(resources_value, dict):
        raise DeploymentTemplateError("template Resources must be an object")
    resources = cast(dict[str, object], resources_value)
    if len(resources) != 52:
        raise DeploymentTemplateError("deployment template must contain exactly 52 resources")
    if any(
        isinstance(resource, Mapping) and resource.get("Type") == "AWS::Budgets::Budget"
        for resource in resources.values()
    ):
        raise DeploymentTemplateError("deployment template must not contain an AWS Budget")

    network_resources = [
        (logical_id, resource)
        for logical_id, resource in resources.items()
        if isinstance(resource, Mapping)
        and resource.get("Type") == "AWS::OpenSearchServerless::SecurityPolicy"
        and isinstance(resource.get("Properties"), Mapping)
        and resource["Properties"].get("Type") == "network"
    ]
    if len(network_resources) != 1 or network_resources[0][0] != NETWORK_POLICY_LOGICAL_ID:
        raise DeploymentTemplateError("template must contain the exact single network policy")
    network_resource = cast(dict[str, object], network_resources[0][1])
    properties_value = network_resource.get("Properties")
    if not isinstance(properties_value, dict):
        raise DeploymentTemplateError("network policy Properties must be an object")
    properties = cast(dict[str, object], properties_value)
    policy_value = properties.get("Policy")
    if not isinstance(policy_value, str):
        raise DeploymentTemplateError("network policy must be canonical JSON text")
    try:
        network_policy = _load_exact_json(policy_value.encode("utf-8"))
    except DeploymentTemplateError as error:
        raise DeploymentTemplateError("network policy must be valid duplicate-free JSON") from error
    if network_policy != PRIVATE_NETWORK_POLICY:
        raise DeploymentTemplateError(
            "final network policy must be exact Bedrock-only private access"
        )

    index_value = resources.get("VectorIndex")
    if not isinstance(index_value, Mapping):
        raise DeploymentTemplateError("template must contain VectorIndex")
    dependencies = index_value.get("DependsOn")
    if not isinstance(dependencies, list) or set(dependencies) != _EXPECTED_INDEX_DEPENDENCIES:
        raise DeploymentTemplateError("VectorIndex dependencies do not match the reviewed contract")
    if template_sha256(final_template_bytes) != APPROVED_FINAL_TEMPLATE_SHA256:
        raise DeploymentTemplateError(
            "final template digest does not match the approved deployment"
        )

    bootstrap = copy.deepcopy(template)
    bootstrap_resources = cast(dict[str, dict[str, object]], bootstrap["Resources"])
    bootstrap_properties = cast(
        dict[str, object], bootstrap_resources[NETWORK_POLICY_LOGICAL_ID]["Properties"]
    )
    bootstrap_properties["Policy"] = json.dumps(
        BOOTSTRAP_NETWORK_POLICY,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (json.dumps(bootstrap, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _load_exact_json(document: bytes) -> object:
    if not isinstance(document, bytes):
        raise DeploymentTemplateError("template must be bytes")
    try:
        text = document.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DeploymentTemplateError("template must be valid UTF-8") from error
    if text.startswith("\ufeff"):
        raise DeploymentTemplateError("template must not contain a UTF-8 BOM")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _raise_duplicate(key)
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, TypeError) as error:
        raise DeploymentTemplateError("template must be valid duplicate-free JSON") from error


def _raise_duplicate(key: str) -> NoReturn:
    raise DeploymentTemplateError(f"duplicate JSON key: {key}")
