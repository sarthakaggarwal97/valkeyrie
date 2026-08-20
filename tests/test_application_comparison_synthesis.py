import base64
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from infra.application import (
    APPLICATION_ARTIFACT_BUCKET,
    APPLICATION_FUNCTION_NAME,
    ApplicationSynthesis,
    synthesize_application_comparison,
    synthesize_application_comparison_rollback_window,
)

ROOT = Path(__file__).resolve().parents[1]
APPLICATION_REVISION = "sha256:fca8f94a37959e127240239526448855ca9a93fc16a910b0294ebfc800725a54"
ARTIFACT_SHA256 = "sha256:d0cfac9167b343eb75f3875e688d6ef3bf596f21c2f0a1588f5de49f2d6ae303"
OPUS_PROFILE = "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-opus-5"
OPUS_MODELS = [
    "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-5",
    "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-5",
    "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-opus-5",
]
FABLE_RESOURCES = [
    "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-fable-5",
    "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-fable-5",
    "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-fable-5",
    "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-fable-5",
]


def _synth(
    tmp_path: Path, name: str
) -> tuple[ApplicationSynthesis, dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = synthesize_application_comparison(ROOT, tmp_path / name)
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


def test_comparison_synthesis_is_deterministic_and_explicit(tmp_path: Path) -> None:
    first, first_bootstrap, first_application, first_evidence = _synth(tmp_path, "first")
    second, second_bootstrap, second_application, second_evidence = _synth(tmp_path, "second")

    assert first.artifact.content == second.artifact.content
    assert first.artifact.application_revision == APPLICATION_REVISION
    assert first.artifact.artifact_sha256 == ARTIFACT_SHA256
    assert first_bootstrap == second_bootstrap
    assert first_application == second_application
    assert first_evidence == second_evidence
    assert first_evidence["identity"]["execution_authorization"] == ("owner_directed_comparison")
    assert first_evidence["identity"]["qualification_status"] == "not_run_not_qualified"


def test_comparison_application_is_iam_free_retained_and_opus_pinned(tmp_path: Path) -> None:
    result, _, application, _ = _synth(tmp_path, "application")
    resources = _resources(application)

    assert application["Description"] == (
        "Valkeyrie private owner-directed model comparison (synthesis only)"
    )
    assert Counter(item["Type"] for item in resources.values()) == {
        "AWS::Lambda::Function": 1,
        "AWS::Lambda::Version": 1,
        "AWS::Logs::LogGroup": 1,
    }
    assert not any(item["Type"].startswith("AWS::IAM::") for item in resources.values())
    function = resources["ApplicationFunction"]["Properties"]
    assert function["FunctionName"] == APPLICATION_FUNCTION_NAME
    assert function["Timeout"] == 240
    assert function["Code"] == {
        "S3Bucket": APPLICATION_ARTIFACT_BUCKET,
        "S3Key": f"artifacts/{ARTIFACT_SHA256[7:]}.zip",
    }
    environment = function["Environment"]["Variables"]
    assert environment["APPLICATION_REVISION"] == APPLICATION_REVISION
    assert environment["SELECTED_INFERENCE_PROFILE_ARN"] == OPUS_PROFILE

    version = resources["ApplicationVersion"]
    assert version["DeletionPolicy"] == version["UpdateReplacePolicy"] == "Retain"
    assert (
        version["Properties"]["CodeSha256"]
        == base64.b64encode(bytes.fromhex(ARTIFACT_SHA256[7:])).decode()
    )
    assert "authorization=owner_directed_comparison" in version["Properties"]["Description"]
    assert result.artifact.manifest["qualification_status"] == "not_run_not_qualified"


def test_comparison_bootstrap_preserves_fable_and_adds_only_exact_opus(
    tmp_path: Path,
) -> None:
    _, bootstrap, _, _ = _synth(tmp_path, "bootstrap")
    resources = _resources(bootstrap)
    runtime = _statements(resources["ApplicationRuntimePolicy"])

    assert set(runtime) == {
        "InvokeExactQualifiedFableRoute",
        "InvokeExactOwnerDirectedOpusComparisonRoute",
        "RetrieveExactKnowledgeBase",
        "ReadAndConditionallyAuditRequests",
        "ReadFailClosedRuntimeControls",
        "WriteExactApplicationTelemetry",
    }
    assert runtime["InvokeExactQualifiedFableRoute"]["Resource"] == FABLE_RESOURCES
    assert runtime["InvokeExactOwnerDirectedOpusComparisonRoute"]["Resource"] == [
        OPUS_PROFILE,
        *OPUS_MODELS,
    ]
    assert "*" not in json.dumps(
        {
            "fable": runtime["InvokeExactQualifiedFableRoute"],
            "opus": runtime["InvokeExactOwnerDirectedOpusComparisonRoute"],
        }
    )

    service_role = resources["ApplicationCloudFormationServiceRole"]
    service = {
        item["Sid"]: item
        for item in service_role["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    }
    artifact_resource = service["ReadExactApplicationArtifact"]["Resource"]
    assert artifact_resource["Fn::Join"][1][-1] == (
        f":s3:::{APPLICATION_ARTIFACT_BUCKET}/artifacts/{ARTIFACT_SHA256[7:]}.zip"
    )
    assert "98bbbd6d9808f12518ba67654d36343984dab9122f5c5445b3a504261aa93cb9" not in (
        json.dumps(artifact_resource)
    )


def test_rollback_window_changes_only_exact_artifact_read_set(tmp_path: Path) -> None:
    final, final_bootstrap, final_application, _ = _synth(tmp_path, "final")
    rollback = synthesize_application_comparison_rollback_window(ROOT, tmp_path / "rollback")
    rollback_bootstrap = json.loads(rollback.bootstrap_template_path.read_text())
    rollback_application = json.loads(rollback.template_path.read_text())

    assert rollback.artifact == final.artifact
    assert rollback_application == final_application
    final_resources = _resources(final_bootstrap)
    rollback_resources = _resources(rollback_bootstrap)
    final_service = {
        item["Sid"]: item
        for item in final_resources["ApplicationCloudFormationServiceRole"]["Properties"][
            "Policies"
        ][0]["PolicyDocument"]["Statement"]
    }
    rollback_service = {
        item["Sid"]: item
        for item in rollback_resources["ApplicationCloudFormationServiceRole"]["Properties"][
            "Policies"
        ][0]["PolicyDocument"]["Statement"]
    }
    final_resource = final_service["ReadExactApplicationArtifact"]["Resource"]
    rollback_resource = rollback_service["ReadExactApplicationArtifact"]["Resource"]
    assert isinstance(rollback_resource, list)
    assert [item["Fn::Join"][1][-1] for item in rollback_resource] == [
        (
            f":s3:::{APPLICATION_ARTIFACT_BUCKET}/artifacts/"
            "98bbbd6d9808f12518ba67654d36343984dab9122f5c5445b3a504261aa93cb9.zip"
        ),
        f":s3:::{APPLICATION_ARTIFACT_BUCKET}/artifacts/{ARTIFACT_SHA256[7:]}.zip",
    ]

    normalized = json.loads(json.dumps(rollback_bootstrap))
    normalized_service = {
        item["Sid"]: item
        for item in _resources(normalized)["ApplicationCloudFormationServiceRole"]["Properties"][
            "Policies"
        ][0]["PolicyDocument"]["Statement"]
    }
    normalized_service["ReadExactApplicationArtifact"]["Resource"] = final_resource
    assert normalized == final_bootstrap
