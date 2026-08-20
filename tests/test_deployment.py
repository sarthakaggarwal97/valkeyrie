from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from infra.app import synthesize
from infra.deployment import (
    APPROVED_FINAL_TEMPLATE_SHA256,
    BOOTSTRAP_NETWORK_POLICY,
    NETWORK_POLICY_LOGICAL_ID,
    PRIVATE_NETWORK_POLICY,
    DeploymentTemplateError,
    build_index_bootstrap_template,
    template_sha256,
)


def _final_template(tmp_path: Path) -> bytes:
    output_dir = tmp_path / "cdk.out"
    synthesize(output_dir)
    return (output_dir / "KnowledgePlane.template.json").read_bytes()


def _policy(template: dict[str, object]) -> object:
    resources = template["Resources"]
    assert isinstance(resources, dict)
    network = resources[NETWORK_POLICY_LOGICAL_ID]
    assert isinstance(network, dict)
    properties = network["Properties"]
    assert isinstance(properties, dict)
    value = properties["Policy"]
    assert isinstance(value, str)
    return json.loads(value)


def test_bootstrap_changes_only_the_network_policy(tmp_path: Path) -> None:
    final_bytes = _final_template(tmp_path)
    final = json.loads(final_bytes)
    bootstrap_bytes = build_index_bootstrap_template(final_bytes)
    bootstrap = json.loads(bootstrap_bytes)

    assert len(final["Resources"]) == 52
    assert _policy(final) == PRIVATE_NETWORK_POLICY
    assert _policy(bootstrap) == BOOTSTRAP_NETWORK_POLICY

    expected = copy.deepcopy(final)
    expected["Resources"][NETWORK_POLICY_LOGICAL_ID]["Properties"]["Policy"] = json.dumps(
        BOOTSTRAP_NETWORK_POLICY,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert bootstrap == expected


def test_bootstrap_has_no_dashboard_rule_or_source_service(tmp_path: Path) -> None:
    bootstrap = json.loads(build_index_bootstrap_template(_final_template(tmp_path)))
    policy = _policy(bootstrap)

    assert policy == BOOTSTRAP_NETWORK_POLICY
    assert policy[0]["Rules"] == [
        {
            "Resource": ["collection/valkeyrie-development-vectors"],
            "ResourceType": "collection",
        }
    ]
    assert "SourceServices" not in policy[0]
    assert all(rule["ResourceType"] != "dashboard" for rule in policy[0]["Rules"])


def test_bootstrap_and_hashes_are_deterministic(tmp_path: Path) -> None:
    final_bytes = _final_template(tmp_path)
    assert template_sha256(final_bytes) == APPROVED_FINAL_TEMPLATE_SHA256

    first = build_index_bootstrap_template(final_bytes)
    second = build_index_bootstrap_template(final_bytes)

    assert first == second
    assert template_sha256(first) == template_sha256(second)
    assert template_sha256(first) != template_sha256(final_bytes)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda template: template.__setitem__("Parameters", {"Unsafe": {"Type": "String"}}),
            "must not accept parameters",
        ),
        (
            lambda template: template["Resources"].__setitem__(
                "CorpusBucket", {"Type": "AWS::Budgets::Budget"}
            ),
            "must not contain an AWS Budget",
        ),
        (
            lambda template: template["Resources"].__setitem__(
                NETWORK_POLICY_LOGICAL_ID,
                {"Type": "AWS::SSM::Parameter"},
            ),
            "exact single network policy",
        ),
        (
            lambda template: template["Resources"][NETWORK_POLICY_LOGICAL_ID][
                "Properties"
            ].__setitem__("Policy", json.dumps(BOOTSTRAP_NETWORK_POLICY)),
            "exact Bedrock-only private access",
        ),
        (
            lambda template: template["Resources"]["VectorIndex"].__setitem__(
                "DependsOn", ["VectorCollection"]
            ),
            "dependencies do not match",
        ),
    ],
)
def test_bootstrap_rejects_unreviewed_template_shapes(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    template = json.loads(_final_template(tmp_path))
    assert callable(mutation)
    mutation(template)

    with pytest.raises(DeploymentTemplateError, match=message):
        build_index_bootstrap_template(json.dumps(template).encode())


def test_bootstrap_rejects_unapproved_unrelated_resource_change(tmp_path: Path) -> None:
    template = json.loads(_final_template(tmp_path))
    template["Resources"]["CorpusBucket"]["Properties"]["PublicAccessBlockConfiguration"][
        "BlockPublicPolicy"
    ] = False

    with pytest.raises(DeploymentTemplateError, match="digest does not match"):
        build_index_bootstrap_template(json.dumps(template).encode())


def test_bootstrap_rejects_unapproved_top_level_change(tmp_path: Path) -> None:
    template = json.loads(_final_template(tmp_path))
    template["Metadata"] = {"Unreviewed": True}

    with pytest.raises(DeploymentTemplateError, match="digest does not match"):
        build_index_bootstrap_template(json.dumps(template).encode())


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (b"[]", "root must be an object"),
        (b"{", "valid duplicate-free JSON"),
        (b'{"Resources":{},"Resources":{}}', "duplicate JSON key: Resources"),
        (b"\xef\xbb\xbf{}", "must not contain a UTF-8 BOM"),
        (b"\xff", "valid UTF-8"),
    ],
)
def test_bootstrap_rejects_invalid_json_documents(document: bytes, message: str) -> None:
    with pytest.raises(DeploymentTemplateError, match=message):
        build_index_bootstrap_template(document)


def test_template_hash_rejects_text() -> None:
    with pytest.raises(DeploymentTemplateError, match="template must be bytes"):
        template_sha256("not bytes")  # type: ignore[arg-type]
