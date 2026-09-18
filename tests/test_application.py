from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest

from infra import application_handler
from infra.app import synthesize
from infra.application import (
    APPLICATION_ARTIFACT_BUCKET,
    APPLICATION_FUNCTION_NAME,
    APPLICATION_RUNTIME_ROLE_NAME,
    APPLICATION_SERVICE_ROLE_ARN,
    APPLICATION_SERVICE_ROLE_NAME,
    D01_TEMPLATE_SHA256,
    KNOWLEDGE_BASE_ID,
    ApplicationSynthesis,
    build_application_app,
    synthesize_application,
)
from infra.application_artifact import build_application_artifact

ROOT = Path(__file__).resolve().parents[1]


def _synth(
    tmp_path: Path, name: str = "a07a"
) -> tuple[ApplicationSynthesis, dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = synthesize_application(ROOT, tmp_path / name)
    bootstrap = json.loads(result.bootstrap_template_path.read_text())
    application = json.loads(result.template_path.read_text())
    evidence = json.loads(result.evidence_path.read_text())
    return result, bootstrap, application, evidence


def _resources(template: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return cast(dict[str, dict[str, Any]], template["Resources"])


def _statements(resource: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        statement["Sid"]: statement
        for statement in resource["Properties"]["PolicyDocument"]["Statement"]
    }


def test_two_stacks_and_artifact_are_byte_deterministic(tmp_path: Path) -> None:
    first, _, _, _ = _synth(tmp_path, "first")
    second, _, _, _ = _synth(tmp_path, "second")
    assert first.artifact.content == second.artifact.content
    assert first.bootstrap_template_path.read_bytes() == second.bootstrap_template_path.read_bytes()
    assert first.template_path.read_bytes() == second.template_path.read_bytes()
    assert first.evidence_path.read_bytes() == second.evidence_path.read_bytes()
    manifest = json.loads((tmp_path / "first/manifest.json").read_text())
    stacks = {
        key
        for key, value in manifest["artifacts"].items()
        if value.get("type") == "aws:cloudformation:stack"
    }
    assert stacks == {"ApplicationBootstrap", "ApplicationPlane"}
    for stack in stacks:
        assert manifest["artifacts"][stack].get("dependencies") in (None, [])
    assert manifest.get("missing") in (None, [])
    assert not any(
        value.get("type") == "cdk:asset-manifest" for value in manifest["artifacts"].values()
    )


def test_knowledge_plane_remains_exact_deployed_d01(tmp_path: Path) -> None:
    output = tmp_path / "knowledge"
    synthesize(output)
    content = (output / "KnowledgePlane.template.json").read_bytes()
    template = json.loads(content)
    assert len(template["Resources"]) == 52
    assert "sha256:" + hashlib.sha256(content).hexdigest() == D01_TEMPLATE_SHA256
    assert (
        D01_TEMPLATE_SHA256
        == "sha256:75bff0ceecf4324255d997b88c6a5efde4ad77c7b1e724ea8af6fde9614754b0"
    )
    for role_name in ("valkeyrie-development-application-deployer", APPLICATION_RUNTIME_ROLE_NAME):
        role_ids = [
            logical
            for logical, resource in template["Resources"].items()
            if resource["Type"] == "AWS::IAM::Role"
            and resource["Properties"]["RoleName"] == role_name
        ]
        assert len(role_ids) == 1
        assert not any(
            resource["Type"] == "AWS::IAM::Policy"
            and {"Ref": role_ids[0]} in resource["Properties"].get("Roles", [])
            for resource in template["Resources"].values()
        )


def test_bootstrap_owns_only_bucket_and_exact_iam(tmp_path: Path) -> None:
    _, bootstrap, _, _ = _synth(tmp_path)
    resources = _resources(bootstrap)
    assert (
        bootstrap["Description"]
        == "Valkeyrie D02 application bootstrap (human-created, synthesis only)"
    )
    assert "Parameters" not in bootstrap and "Outputs" not in bootstrap
    assert Counter(item["Type"] for item in resources.values()) == {
        "AWS::S3::Bucket": 1,
        "AWS::S3::BucketPolicy": 1,
        "AWS::IAM::Role": 1,
        "AWS::IAM::Policy": 2,
    }
    bucket = resources["ApplicationArtifactBucket"]
    assert bucket["DeletionPolicy"] == bucket["UpdateReplacePolicy"] == "Retain"
    properties = bucket["Properties"]
    assert properties["BucketName"] == APPLICATION_ARTIFACT_BUCKET
    assert properties["VersioningConfiguration"] == {"Status": "Enabled"}
    assert properties["BucketEncryption"] == {
        "ServerSideEncryptionConfiguration": [
            {"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
        ]
    }
    assert properties["PublicAccessBlockConfiguration"] == {
        "BlockPublicAcls": True,
        "BlockPublicPolicy": True,
        "IgnorePublicAcls": True,
        "RestrictPublicBuckets": True,
    }
    policy = resources["ApplicationArtifactBucketPolicy"]["Properties"]["PolicyDocument"]
    statements = {item["Sid"]: item for item in policy["Statement"]}
    assert set(statements) == {
        "DenyInsecureTransport",
        "DenyArtifactBucketAndObjectDeletion",
        "DenyDeployerArtifactOverwrite",
    }
    assert statements["DenyInsecureTransport"]["Condition"] == {
        "Bool": {"aws:SecureTransport": "false"}
    }
    assert set(statements["DenyArtifactBucketAndObjectDeletion"]["Action"]) == {
        "s3:DeleteBucket",
        "s3:DeleteObject",
        "s3:DeleteObjectVersion",
    }
    assert statements["DenyDeployerArtifactOverwrite"]["Condition"] == {
        "StringNotEquals": {"s3:if-none-match": "*"}
    }


def test_service_role_and_existing_role_policies_are_exact(tmp_path: Path) -> None:
    result, bootstrap, _, evidence = _synth(tmp_path)
    resources = _resources(bootstrap)
    role = resources["ApplicationCloudFormationServiceRole"]
    assert role["DeletionPolicy"] == role["UpdateReplacePolicy"] == "Retain"
    assert role["Properties"]["RoleName"] == APPLICATION_SERVICE_ROLE_NAME
    trust = role["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    assert trust == [
        {
            "Action": "sts:AssumeRole",
            "Condition": {"StringEquals": {"aws:SourceAccount": "968533178160"}},
            "Effect": "Allow",
            "Principal": {"Service": "cloudformation.amazonaws.com"},
        }
    ]
    service = {
        item["Sid"]: item
        for item in role["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    }
    assert set(service) == {
        "ReadExactApplicationArtifact",
        "ManageExactApplicationFunction",
        "ManageExactApplicationLogGroup",
        "PassExactRuntimeRoleToLambda",
    }
    assert service["ReadExactApplicationArtifact"] == {
        "Action": "s3:GetObject",
        "Effect": "Allow",
        "Resource": {
            "Fn::Join": [
                "",
                [
                    "arn:",
                    {"Ref": "AWS::Partition"},
                    (
                        f":s3:::{APPLICATION_ARTIFACT_BUCKET}/artifacts/"
                        f"{result.artifact.artifact_sha256[7:]}.zip"
                    ),
                ],
            ]
        },
        "Sid": "ReadExactApplicationArtifact",
    }
    assert set(service["ManageExactApplicationFunction"]["Action"]) == {
        "lambda:AddPermission",
        "lambda:CreateFunction",
        "lambda:CreateFunctionUrlConfig",
        "lambda:DeleteFunction",
        "lambda:DeleteFunctionConcurrency",
        "lambda:DeleteFunctionUrlConfig",
        "lambda:GetFunction",
        "lambda:GetFunctionConcurrency",
        "lambda:GetFunctionConfiguration",
        "lambda:GetFunctionUrlConfig",
        "lambda:GetPolicy",
        "lambda:ListVersionsByFunction",
        "lambda:PublishVersion",
        "lambda:PutFunctionConcurrency",
        "lambda:RemovePermission",
        "lambda:TagResource",
        "lambda:UntagResource",
        "lambda:UpdateFunctionCode",
        "lambda:UpdateFunctionConfiguration",
        "lambda:UpdateFunctionUrlConfig",
    }
    assert service["PassExactRuntimeRoleToLambda"]["Condition"] == {
        "StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}
    }
    assert (
        service["PassExactRuntimeRoleToLambda"]["Resource"]["Fn::Join"][1][-1]
        == f":iam::968533178160:role/{APPLICATION_RUNTIME_ROLE_NAME}"
    )

    deployer = _statements(resources["ApplicationDeployerPolicy"])
    assert set(deployer) == {
        "DeployExactApplicationStack",
        "ReadWriteExactArtifactPrefix",
        "ListExactArtifactPrefix",
        "InvokeExactImmutableHealthVersion",
        "PassExactApplicationServiceRole",
    }
    assert deployer["PassExactApplicationServiceRole"] == {
        "Action": "iam:PassRole",
        "Condition": {"StringEquals": {"iam:PassedToService": "cloudformation.amazonaws.com"}},
        "Effect": "Allow",
        "Resource": APPLICATION_SERVICE_ROLE_ARN,
        "Sid": "PassExactApplicationServiceRole",
    }
    all_actions = {
        action
        for statement in deployer.values()
        for action in (
            [statement["Action"]] if isinstance(statement["Action"], str) else statement["Action"]
        )
    }
    assert not (
        {
            "iam:CreateRole",
            "iam:PutRolePolicy",
            "iam:AttachRolePolicy",
            "cloudformation:DeleteStack",
            "s3:DeleteObject",
        }
        & all_actions
    )

    runtime = _statements(resources["ApplicationRuntimePolicy"])
    assert set(runtime) == {
        "InvokeExactQualifiedFableRoute",
        "InvokeExactOwnerDirectedOpusComparisonRoute",
        "ReadExactGitHubTokenSecret",
        "RetrieveExactKnowledgeBase",
        "ReadCorpusPointersAndOwnRequests",
        "ConditionallyAuditOwnRequests",
        "ReadFailClosedRuntimeControls",
        "WriteExactApplicationTelemetry",
    }
    # One exact secret, never a prefix or wildcard: the account also holds the ops webhook
    # signing secret, which the answer runtime has no business reading.
    secret_grant = runtime["ReadExactGitHubTokenSecret"]
    assert secret_grant["Action"] == "secretsmanager:GetSecretValue"
    assert secret_grant["Resource"] == (
        "arn:aws:secretsmanager:us-east-1:968533178160:secret:"
        "valkeyrie/development/github-read-token-0mNbc2"
    )
    assert "*" not in secret_grant["Resource"]
    model_resources = runtime["InvokeExactQualifiedFableRoute"]["Resource"]
    assert model_resources == [
        "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-fable-5",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-fable-5",
        "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-fable-5",
        "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-fable-5",
    ]
    # Immutable Lambda version 8 runs Claude Opus 5 under owner-directed authorization, so the
    # runtime keeps an exact Opus grant. Synthesis must reproduce the deployed policy exactly,
    # otherwise a future bootstrap deploy would revoke that version's ability to invoke Opus.
    assert runtime["InvokeExactOwnerDirectedOpusComparisonRoute"]["Resource"] == [
        "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-opus-5",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-5",
        "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-5",
        "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-opus-5",
    ]
    assert runtime["RetrieveExactKnowledgeBase"] == {
        "Action": "bedrock:Retrieve",
        "Effect": "Allow",
        "Resource": {
            "Fn::Join": [
                "",
                [
                    "arn:",
                    {"Ref": "AWS::Partition"},
                    (f":bedrock:us-east-1:968533178160:knowledge-base/{KNOWLEDGE_BASE_ID}"),
                ],
            ]
        },
        "Sid": "RetrieveExactKnowledgeBase",
    }
    # Reads and writes are separate statements so the runtime can never move the active pointer
    # or record an approval: writes are confined to its own request audit rows.
    reads = runtime["ReadCorpusPointersAndOwnRequests"]
    assert reads["Action"] == ["dynamodb:GetItem"]
    assert reads["Condition"]["ForAllValues:StringLike"]["dynamodb:LeadingKeys"] == [
        "active_generation",
        "version#*",
        "generation#*",
        "structured#*",
        "request#*",
    ]
    writes = runtime["ConditionallyAuditOwnRequests"]
    assert set(writes["Action"]) == {"dynamodb:PutItem", "dynamodb:UpdateItem"}
    assert writes["Condition"] == {
        "ForAllValues:StringLike": {"dynamodb:LeadingKeys": ["request#*"]}
    }
    serialized = json.dumps(runtime)
    for prohibited in (
        "Scan",
        "DeleteItem",
        "s3:",
        "StartIngestionJob",
        "aoss:",
        "ssm:PutParameter",
        "iam:PassRole",
    ):
        assert prohibited not in serialized
    after = evidence["iam_action_resource_diff"]["application_bootstrap_after"]
    assert set(after) == {
        "valkeyrie-development-application-deployer",
        "valkeyrie-development-runtime",
        APPLICATION_SERVICE_ROLE_NAME,
    }
    service_evidence = {
        item["Sid"]: item for item in after[APPLICATION_SERVICE_ROLE_NAME]["statements"]
    }
    assert (
        service_evidence["ReadExactApplicationArtifact"] == service["ReadExactApplicationArtifact"]
    )
    assert evidence["iam_action_resource_diff"]["prohibited_actions"] == []


def test_application_stack_is_iam_free_and_retains_exact_version(tmp_path: Path) -> None:
    result, _, application, _ = _synth(tmp_path)
    resources = _resources(application)
    assert application["Description"] == "Valkeyrie private qualified application (synthesis only)"
    assert Counter(item["Type"] for item in resources.values()) == {
        "AWS::Lambda::Function": 1,
        "AWS::Lambda::Version": 1,
        "AWS::Lambda::Url": 1,
        "AWS::Logs::LogGroup": 1,
    }
    assert "Parameters" not in application and "Outputs" not in application
    assert not any(item["Type"].startswith("AWS::IAM::") for item in resources.values())
    function = resources["ApplicationFunction"]["Properties"]
    assert function["FunctionName"] == APPLICATION_FUNCTION_NAME
    assert function["Architectures"] == ["x86_64"]
    assert function["Code"] == {
        "S3Bucket": APPLICATION_ARTIFACT_BUCKET,
        "S3Key": f"artifacts/{result.artifact.artifact_sha256[7:]}.zip",
    }
    assert function["Handler"] == "infra.application_handler.handler"
    assert (
        function["Environment"]["Variables"]["APPLICATION_REVISION"]
        == result.artifact.application_revision
    )
    version = resources["ApplicationVersion"]
    assert version["DeletionPolicy"] == version["UpdateReplacePolicy"] == "Retain"
    assert (
        version["Properties"]["CodeSha256"]
        == base64.b64encode(bytes.fromhex(result.artifact.artifact_sha256[7:])).decode()
    )
    assert resources["ApplicationLogGroup"]["DeletionPolicy"] == "Retain"

    # The endpoint is account-scoped, never world accessible. A public function url is
    # prohibited in this account and actively policed: AuthType NONE returned 403 for
    # anonymous callers and Palisade raised epoxy-engage_mitigations against it. AWS_IAM
    # needs no resource policy, so any Lambda::Permission here would only widen the
    # function, and a public CloudFront front door would restore the world reach the
    # mitigation removed. Both are asserted absent so a regression fails loudly.
    assert resources["ApplicationUrl"]["Properties"]["AuthType"] == "AWS_IAM"
    assert not any(item["Type"] == "AWS::Lambda::Permission" for item in resources.values())
    assert not any(item["Type"].startswith("AWS::CloudFront::") for item in resources.values())
    # A public caller must not be serialised behind one execution, and needs headroom
    # beyond the 30s that cut off the densest answers.
    assert function["ReservedConcurrentExecutions"] == 5
    assert function["Timeout"] == 120


def test_evidence_is_derived_executable_and_cost_truthful(tmp_path: Path) -> None:
    result, _, _, evidence = _synth(tmp_path)
    identity = evidence["identity"]
    assert identity["knowledge_template_sha256"] == D01_TEMPLATE_SHA256
    assert identity["bootstrap_template_sha256"] == result.bootstrap_template_sha256
    assert identity["application_template_sha256"] == result.template_sha256
    assert identity["artifact_sha256"] == result.artifact.artifact_sha256
    assert identity["selection_id"] == result.artifact.selection_id
    before = evidence["iam_action_resource_diff"]["deployed_d01_before"]
    assert all(
        item["attached_policy_resources"] == [] and item["statements"] == []
        for item in before.values()
    )
    health = evidence["health"]
    commands = "\n".join(health["immutable_version_commands"])
    assert "describe-stack-resource" in commands and "ApplicationVersion" in commands
    assert "$VERSION" in commands and '{"Ref"' not in commands
    cloud = health["cloud_answer_probe"]
    assert "KnowledgeBase" in cloud["knowledge_base_resolver"]
    assert cloud["payload"]["request_id"] == "req_postdeploy-a07a"
    rollback = evidence["rollback"]
    assert all(
        value.startswith("<OPERATOR-SUPPLIED")
        for key, value in rollback["operator_supplied_prior_identity"].items()
        if key not in {"PRIOR_ARTIFACT_BUCKET"}
    )
    assert APPLICATION_SERVICE_ROLE_ARN in "\n".join(rollback["commands"])
    assert rollback["corpus_action"] == "none"
    cost = evidence["cost"]
    assert cost["qualification_measurement"] == {
        "requests": 249,
        "input_tokens": 522810,
        "output_tokens": 36639,
        "formula_usd": "522810/1e6*3 + 36639/1e6*15",
        "calculated_usd": 2.118015,
    }
    assert cost["assumptions"]["bedrock_retrieve_per_1000_usd"] == 1.0
    assert set(cost) == {
        "currency",
        "assumptions",
        "qualification_measurement",
        "one_time_bootstrap",
        "per_deploy",
        "per_cloud_answer",
    }


def test_health_handler_runs_offline_a03_a06_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = build_application_artifact(ROOT)
    manifest = tmp_path / "application-artifact.json"
    manifest.write_text(json.dumps(artifact.manifest))
    monkeypatch.setattr(application_handler, "_MANIFEST", manifest)
    monkeypatch.delenv("STATE_TABLE_NAME", raising=False)
    monkeypatch.delenv("SELECTED_INFERENCE_PROFILE_ARN", raising=False)
    result = application_handler.handler(
        {"action": "health", "application_revision": artifact.application_revision},
        None,
    )
    assert result["status"] == "healthy"
    assert result["synthetic_path"] == {
        "a03_route": "static_semantic",
        "a04_normalization": "unchanged",
        "a04_output": "answer",
        "a05_conditional_claim": True,
        "a05_completion": True,
        "a06_outcome": "answer",
    }


def test_application_cdk_configuration_and_independence(tmp_path: Path) -> None:
    assert json.loads((ROOT / "cdk.application.json").read_text()) == {
        "app": "uv run python infra/application.py",
        "output": "cdk.application.out",
        "stacks": ["ApplicationBootstrap", "ApplicationPlane"],
    }
    artifact = build_application_artifact(ROOT)
    out = tmp_path / "assembly"
    build_application_app(out, artifact).synth()
    assert not (out / "KnowledgePlane.template.json").exists()
