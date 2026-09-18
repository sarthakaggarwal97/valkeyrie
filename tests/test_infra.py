from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest

from infra.app import (
    APPLICATION_ENVIRONMENT,
    CONTROL_DEFAULTS,
    CORPUS_DATA_SOURCE_PREFIX,
    CORPUS_ENVIRONMENT,
    CUSTOM_METRIC_NAMESPACE,
    DEFAULT_CONFIG,
    DEPLOYMENT_PRINCIPAL_NAME,
    GENERATION_PREFIXES,
    GITHUB_REPOSITORY,
    INFRASTRUCTURE_OWNER,
    SAFEGUARD_THRESHOLDS,
    FoundationConfig,
    build_app,
    synthesize,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TAGS = {
    "Project": "valkeyrie",
    "Environment": "development",
    "ManagedBy": "aws-cdk",
    "Boundary": "knowledge-plane",
}
EXPECTED_RESOURCE_COUNTS = {
    "AWS::Bedrock::DataSource": 1,
    "AWS::Bedrock::KnowledgeBase": 1,
    "AWS::CloudWatch::Alarm": 14,
    "AWS::CloudWatch::Dashboard": 1,
    "AWS::DynamoDB::Table": 1,
    "AWS::IAM::Policy": 2,
    "AWS::IAM::Role": 4,
    "AWS::KMS::Alias": 1,
    "AWS::KMS::Key": 1,
    "AWS::Logs::LogGroup": 1,
    "AWS::Logs::MetricFilter": 13,
    "AWS::OpenSearchServerless::AccessPolicy": 1,
    "AWS::OpenSearchServerless::Collection": 1,
    "AWS::OpenSearchServerless::Index": 1,
    "AWS::OpenSearchServerless::SecurityPolicy": 2,
    "AWS::S3::Bucket": 1,
    "AWS::S3::BucketPolicy": 1,
    "AWS::SNS::Topic": 1,
    "AWS::SNS::TopicPolicy": 1,
    "AWS::SSM::Parameter": 3,
}
CUSTOM_METRIC_UNITS = {
    "CorpusAgeHours": "None",
    "AossStorageBytes": "Bytes",
    "IngestionFailures": "Count",
    "InvalidCitations": "Count",
    "DependencyThrottles": "Count",
    "RequestLatencySeconds": "Seconds",
    "AcceptedRequests": "Count",
    "ExternalApiCallsPerRequest": "Count",
    "ModelCallsPerRequest": "Count",
    "InputTokensPerRequest": "Count",
    "OutputTokensPerRequest": "Count",
    "ConcurrentRequests": "Count",
    "EstimatedMonthlySpendUsd": "None",
}


def _assembly_manifest(
    outdir: Path, config: FoundationConfig = DEFAULT_CONFIG
) -> dict[str, object]:
    build_app(outdir, config).synth()
    return cast(
        dict[str, object],
        json.loads((outdir / "manifest.json").read_text(encoding="utf-8")),
    )


def _template(outdir: Path, config: FoundationConfig = DEFAULT_CONFIG) -> dict[str, object]:
    _assembly_manifest(outdir, config)
    return cast(
        dict[str, object],
        json.loads((outdir / "KnowledgePlane.template.json").read_text(encoding="utf-8")),
    )


def _resources(template: dict[str, object]) -> dict[str, dict[str, Any]]:
    return cast(dict[str, dict[str, Any]], template["Resources"])


def _one(template: dict[str, object], resource_type: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (logical_id, resource)
        for logical_id, resource in _resources(template).items()
        if resource["Type"] == resource_type
    ]
    assert len(matches) == 1
    return matches[0]


def _properties(template: dict[str, object], logical_id: str) -> dict[str, Any]:
    return cast(dict[str, Any], _resources(template)[logical_id]["Properties"])


def test_foundation_configuration_defines_exact_target_namespace_and_tags() -> None:
    assert DEFAULT_CONFIG == FoundationConfig(account="968533178160", region="us-east-1")
    assert DEFAULT_CONFIG.namespace == "valkeyrie-development"
    assert DEFAULT_CONFIG.stack_name == "valkeyrie-development-knowledge-plane"
    assert DEFAULT_CONFIG.tags == EXPECTED_TAGS
    assert GITHUB_REPOSITORY == "sarthakaggarwal97/valkeyrie"
    assert APPLICATION_ENVIRONMENT == "application"
    assert CORPUS_ENVIRONMENT == "corpus"
    assert INFRASTRUCTURE_OWNER == "sarthakaggarwal97"
    assert CORPUS_DATA_SOURCE_PREFIX == "kb-documents/"
    assert GENERATION_PREFIXES == (
        "kb-documents/generations/*",
        "control/generations/*",
    )
    assert set(FoundationConfig.__dataclass_fields__) == {
        "account",
        "region",
        "project",
        "environment",
        "boundary",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("github_repository", "attacker/other-repo"),
        ("application_environment", "other-application"),
        ("corpus_environment", "other-corpus"),
    ],
)
def test_oidc_trust_boundary_is_not_configurable(field: str, value: str) -> None:
    values: dict[str, Any] = {
        "account": "968533178160",
        "region": "us-east-1",
        field: value,
    }
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        FoundationConfig(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("account", "123", "12-digit AWS account ID"),
        ("region", "us_east_1", "explicit AWS region"),
        ("project", "Valkeyrie", "lowercase namespacing component"),
        ("environment", "dev!", "lowercase namespacing component"),
        ("boundary", "knowledge_plane", "lowercase namespacing component"),
    ],
)
def test_foundation_configuration_rejects_unsafe_inputs(
    field: str, value: str, message: str
) -> None:
    values = {
        "account": "968533178160",
        "region": "us-east-1",
        "project": "valkeyrie",
        "environment": "development",
        "boundary": "knowledge-plane",
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        FoundationConfig(**values)


def test_synthesis_has_one_environment_bound_tagged_stack_and_no_assets(
    tmp_path: Path,
) -> None:
    manifest = _assembly_manifest(tmp_path / "assembly")
    artifacts = cast(dict[str, dict[str, object]], manifest["artifacts"])
    stacks = {
        artifact_id: artifact
        for artifact_id, artifact in artifacts.items()
        if artifact.get("type") == "aws:cloudformation:stack"
    }
    assert set(stacks) == {"KnowledgePlane"}
    stack = stacks["KnowledgePlane"]
    assert stack["environment"] == "aws://968533178160/us-east-1"
    assert stack.get("dependencies") in (None, [])
    assert stack["properties"] == {
        "templateFile": "KnowledgePlane.template.json",
        "terminationProtection": False,
        "tags": EXPECTED_TAGS,
        "validateOnSynth": False,
        "stackName": "valkeyrie-development-knowledge-plane",
    }
    assert "cdk:asset-manifest" not in {artifact.get("type") for artifact in artifacts.values()}
    assert manifest.get("missing") in (None, [])


def test_template_has_exact_dedicated_resource_inventory(tmp_path: Path) -> None:
    template = _template(tmp_path / "assembly")
    assert template["Description"] == (
        "Valkeyrie isolated development knowledge plane (synthesis only)"
    )
    assert (
        Counter(resource["Type"] for resource in _resources(template).values())
        == EXPECTED_RESOURCE_COUNTS
    )
    assert "Outputs" not in template


def test_template_binds_exact_approved_deployment_principal_without_parameters(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    assert DEPLOYMENT_PRINCIPAL_NAME == "sarthagg"
    assert "Parameters" not in template
    serialized = json.dumps(template, sort_keys=True)
    assert '"Ref": "AWS::Partition"' in serialized
    assert ":iam::968533178160:user/sarthagg" in serialized
    for prohibited in (
        "KnowledgePlaneIndexProvisionerRoleName",
        "VectorIndexProvisionerPolicy",
        "index-provisioner",
    ):
        assert prohibited not in serialized


def test_corpus_bucket_is_predictable_encrypted_versioned_private_and_retained(
    tmp_path: Path,
) -> None:
    logical_id, bucket = _one(_template(tmp_path / "assembly"), "AWS::S3::Bucket")
    assert logical_id == "CorpusBucket"
    assert bucket["DeletionPolicy"] == "Retain"
    assert bucket["UpdateReplacePolicy"] == "Retain"
    assert bucket["Properties"] == {
        "BucketEncryption": {
            "ServerSideEncryptionConfiguration": [
                {"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
            ]
        },
        "BucketName": "valkeyrie-development-corpus-968533178160-us-east-1",
        "LifecycleConfiguration": {
            "Rules": [
                {
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
                    "Id": "AbortIncompleteMultipartUploads",
                    "Status": "Enabled",
                }
            ]
        },
        "OwnershipControls": {"Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]},
        "PublicAccessBlockConfiguration": {
            "BlockPublicAcls": True,
            "BlockPublicPolicy": True,
            "IgnorePublicAcls": True,
            "RestrictPublicBuckets": True,
        },
        "Tags": [{"Key": "Owner", "Value": INFRASTRUCTURE_OWNER}],
        "VersioningConfiguration": {"Status": "Enabled"},
    }


def test_state_table_is_encrypted_recoverable_retained_and_billing_bounded(
    tmp_path: Path,
) -> None:
    logical_id, table = _one(_template(tmp_path / "assembly"), "AWS::DynamoDB::Table")
    assert logical_id == "StateTable"
    assert table["DeletionPolicy"] == table["UpdateReplacePolicy"] == "Retain"
    properties = table["Properties"]
    assert properties["TableName"] == "valkeyrie-development-state"
    assert properties["BillingMode"] == "PAY_PER_REQUEST"
    assert properties["OnDemandThroughput"] == {
        "MaxReadRequestUnits": 25,
        "MaxWriteRequestUnits": 25,
    }
    assert properties["PointInTimeRecoverySpecification"] == {
        "PointInTimeRecoveryEnabled": True,
        "RecoveryPeriodInDays": 35,
    }
    assert properties["SSESpecification"] == {"SSEEnabled": True, "SSEType": "KMS"}
    assert properties["DeletionProtectionEnabled"] is True
    assert properties["TimeToLiveSpecification"] == {
        "AttributeName": "expires_at",
        "Enabled": True,
    }
    assert properties["KeySchema"] == [{"AttributeName": "pk", "KeyType": "HASH"}]


def test_vector_collection_policies_are_explicit_encrypted_and_non_public(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    security_policies = {
        resource["Properties"]["Type"]: resource["Properties"]
        for resource in _resources(template).values()
        if resource["Type"] == "AWS::OpenSearchServerless::SecurityPolicy"
    }
    assert set(security_policies) == {"encryption", "network"}
    assert json.loads(security_policies["encryption"]["Policy"]) == {
        "AWSOwnedKey": True,
        "Rules": [
            {
                "Resource": ["collection/valkeyrie-development-vectors"],
                "ResourceType": "collection",
            }
        ],
    }
    network = json.loads(security_policies["network"]["Policy"])
    assert network == [
        {
            "AllowFromPublic": False,
            "Description": "Bedrock private collection access only",
            "Rules": [
                {
                    "Resource": ["collection/valkeyrie-development-vectors"],
                    "ResourceType": "collection",
                }
            ],
            "SourceServices": ["bedrock.amazonaws.com"],
        }
    ]
    assert "dashboard" not in json.dumps(network)
    _, collection = _one(template, "AWS::OpenSearchServerless::Collection")
    assert collection["Properties"] == {
        "DeletionProtection": "ENABLED",
        "Description": "Derived Valkeyrie Bedrock Knowledge Base vector index",
        "Name": "valkeyrie-development-vectors",
        "StandbyReplicas": "DISABLED",
        "Tags": [{"Key": "Owner", "Value": INFRASTRUCTURE_OWNER}],
        "Type": "VECTORSEARCH",
    }
    assert set(collection["DependsOn"]) == {"VectorEncryptionPolicy", "VectorNetworkPolicy"}


def test_vector_index_exactly_matches_frozen_1024_faiss_hnsw_l2_mapping(
    tmp_path: Path,
) -> None:
    logical_id, index = _one(_template(tmp_path / "assembly"), "AWS::OpenSearchServerless::Index")
    assert logical_id == "VectorIndex"
    assert index["Properties"]["IndexName"] == "valkeyrie-development-kb"
    assert index["Properties"]["Settings"] == {"Index": {"Knn": True}}
    assert index["Properties"]["Mappings"] == {
        "Properties": {
            "AMAZON_BEDROCK_METADATA": {"Index": False, "Type": "text"},
            "AMAZON_BEDROCK_TEXT_CHUNK": {"Index": True, "Type": "text"},
            "bedrock-knowledge-base-default-vector": {
                "Dimension": 1024,
                "Method": {
                    "Engine": "faiss",
                    "Name": "hnsw",
                    "SpaceType": "l2",
                },
                "Type": "knn_vector",
            },
        }
    }
    assert set(index["DependsOn"]) == {
        "VectorCollection",
        "VectorDataAccessPolicy",
        "VectorNetworkPolicy",
    }


def test_single_bedrock_knowledge_base_and_data_source_use_frozen_configuration(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    _, knowledge_base = _one(template, "AWS::Bedrock::KnowledgeBase")
    kb_properties = knowledge_base["Properties"]
    assert kb_properties["Name"] == "valkeyrie-development-kb"
    vector = kb_properties["KnowledgeBaseConfiguration"]["VectorKnowledgeBaseConfiguration"]
    assert "amazon.titan-embed-text-v2:0" in json.dumps(vector["EmbeddingModelArn"])
    assert vector["EmbeddingModelConfiguration"] == {
        "BedrockEmbeddingModelConfiguration": {
            "Dimensions": 1024,
            "EmbeddingDataType": "FLOAT32",
        }
    }
    assert "Normalization" not in json.dumps(vector)
    assert kb_properties["StorageConfiguration"]["OpensearchServerlessConfiguration"][
        "FieldMapping"
    ] == {
        "MetadataField": "AMAZON_BEDROCK_METADATA",
        "TextField": "AMAZON_BEDROCK_TEXT_CHUNK",
        "VectorField": "bedrock-knowledge-base-default-vector",
    }

    _, data_source = _one(template, "AWS::Bedrock::DataSource")
    ds_properties = data_source["Properties"]
    assert ds_properties["DataDeletionPolicy"] == "RETAIN"
    assert ds_properties["DataSourceConfiguration"]["S3Configuration"]["InclusionPrefixes"] == [
        "kb-documents/"
    ]
    assert "control/" not in json.dumps(ds_properties["DataSourceConfiguration"])
    assert ds_properties["VectorIngestionConfiguration"] == {
        "ChunkingConfiguration": {
            "ChunkingStrategy": "HIERARCHICAL",
            "HierarchicalChunkingConfiguration": {
                # Parent first: it is what the answer stage receives; the child is what matches.
                "LevelConfigurations": [{"MaxTokens": 1500}, {"MaxTokens": 300}],
                "OverlapTokens": 60,
            },
        }
    }
    # The retiring list is empty once a migration completes, so exactly one data source exists
    # and it retains its vectors on deletion, as production data must.
    assert ds_properties["DataDeletionPolicy"] == "RETAIN"


def test_observability_topic_uses_exact_cmk_and_constrained_service_publishers(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    _, key = _one(template, "AWS::KMS::Key")
    assert key["DeletionPolicy"] == key["UpdateReplacePolicy"] == "Retain"
    key_properties = key["Properties"]
    assert key_properties["Description"] == (
        "Symmetric CMK for safeguard topic publishing; owner subscription is intentionally "
        "deferred to deployed I-10B"
    )
    assert key_properties["EnableKeyRotation"] is True
    assert key_properties["KeySpec"] == "SYMMETRIC_DEFAULT"
    assert key_properties["KeyUsage"] == "ENCRYPT_DECRYPT"
    key_statements = {
        statement["Sid"]: statement for statement in key_properties["KeyPolicy"]["Statement"]
    }
    assert set(key_statements) == {
        "AccountAdministration",
        "CloudWatchEncryptedTopicPublishing",
    }
    assert key_statements["AccountAdministration"] == {
        "Action": "kms:*",
        "Effect": "Allow",
        "Principal": {
            "AWS": {
                "Fn::Join": [
                    "",
                    [
                        "arn:",
                        {"Ref": "AWS::Partition"},
                        ":iam::968533178160:root",
                    ],
                ]
            }
        },
        "Resource": "*",
        "Sid": "AccountAdministration",
    }
    for sid, service in (("CloudWatchEncryptedTopicPublishing", "cloudwatch.amazonaws.com"),):
        assert key_statements[sid] == {
            "Action": ["kms:GenerateDataKey*", "kms:Decrypt"],
            "Condition": {
                "StringEquals": {
                    "aws:SourceAccount": "968533178160",
                    "kms:ViaService": "sns.us-east-1.amazonaws.com",
                }
            },
            "Effect": "Allow",
            "Principal": {"Service": service},
            "Resource": "*",
            "Sid": sid,
        }

    _, alias = _one(template, "AWS::KMS::Alias")
    assert alias["Properties"] == {
        "AliasName": "alias/valkeyrie-development-alerts",
        "TargetKeyId": {"Ref": "AlertKey"},
    }
    _, topic = _one(template, "AWS::SNS::Topic")
    assert topic["Properties"] == {
        "DisplayName": (
            "Valkeyrie alerts; owner subscription intentionally deferred to deployed I-10B"
        ),
        "KmsMasterKeyId": {"Fn::GetAtt": ["AlertKey", "Arn"]},
        "Tags": [{"Key": "Owner", "Value": INFRASTRUCTURE_OWNER}],
        "TopicName": "valkeyrie-development-alerts",
    }
    assert "Subscription" not in topic["Properties"]
    assert not any(
        resource["Type"] == "AWS::SNS::Subscription" for resource in _resources(template).values()
    )

    _, topic_policy = _one(template, "AWS::SNS::TopicPolicy")
    statements = {
        statement["Sid"]: statement
        for statement in topic_policy["Properties"]["PolicyDocument"]["Statement"]
    }
    assert topic_policy["Properties"]["Topics"] == [{"Ref": "AlertTopic"}]
    assert set(statements) == {"CloudWatchPublishOnly"}
    for sid, service in (("CloudWatchPublishOnly", "cloudwatch.amazonaws.com"),):
        assert statements[sid] == {
            "Action": "sns:Publish",
            "Condition": {"StringEquals": {"aws:SourceAccount": "968533178160"}},
            "Effect": "Allow",
            "Principal": {"Service": service},
            "Resource": {"Fn::GetAtt": ["AlertTopic", "TopicArn"]},
            "Sid": sid,
        }


def test_log_group_metric_filters_define_exact_json_producer_contract(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    _, log_group = _one(template, "AWS::Logs::LogGroup")
    assert log_group["DeletionPolicy"] == log_group["UpdateReplacePolicy"] == "Retain"
    assert log_group["Properties"] == {
        "DeletionProtectionEnabled": True,
        "LogGroupClass": "STANDARD",
        "LogGroupName": "/valkeyrie-development/knowledge-plane",
        "RetentionInDays": 30,
        "Tags": [{"Key": "Owner", "Value": INFRASTRUCTURE_OWNER}],
    }

    filters = {
        logical_id: resource
        for logical_id, resource in _resources(template).items()
        if resource["Type"] == "AWS::Logs::MetricFilter"
    }
    assert set(filters) == {f"{name}MetricFilter" for name in CUSTOM_METRIC_UNITS}
    for metric_name, unit in CUSTOM_METRIC_UNITS.items():
        resource = filters[f"{metric_name}MetricFilter"]
        assert resource["DependsOn"] == ["KnowledgePlaneLogGroup"]
        assert resource["Properties"] == {
            "FilterPattern": (f'{{ $.metric_name = "{metric_name}" && $.value = * }}'),
            "LogGroupName": {"Ref": "KnowledgePlaneLogGroup"},
            "MetricTransformations": [
                {
                    "MetricName": metric_name,
                    "MetricNamespace": CUSTOM_METRIC_NAMESPACE,
                    "MetricValue": "$.value",
                    "Unit": unit,
                }
            ],
        }
    assert "PutMetricData" not in json.dumps(template)


def test_template_has_no_budget_or_cost_allocation_dependency(tmp_path: Path) -> None:
    template = _template(tmp_path / "assembly")
    serialized = json.dumps(template, sort_keys=True)
    for prohibited in (
        "AWS::Budgets::Budget",
        "budgets.amazonaws.com",
        "TagKeyValue",
        "cost-allocation",
        "D01EvidencePrerequisite",
    ):
        assert prohibited not in serialized


def test_safeguard_alarms_have_exact_thresholds_and_missing_signal_routing(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    alarms = {
        logical_id: resource["Properties"]
        for logical_id, resource in _resources(template).items()
        if resource["Type"] == "AWS::CloudWatch::Alarm"
    }
    expected = {
        "CorpusAge": ("CorpusAgeHours", 168, 3600),
        "S3Storage": ("BucketSizeBytes", 10_737_418_240, 86400),
        "AossStorage": ("AossStorageBytes", 10_737_418_240, 3600),
        "IngestionFailures": ("IngestionFailures", 1, 300),
        "InvalidCitations": ("InvalidCitations", 1, 300),
        "DependencyThrottles": ("DependencyThrottles", 1, 300),
        "RequestP95Latency": ("RequestLatencySeconds", 15, 300),
        "DailyAcceptedRequests": ("AcceptedRequests", 100, 86400),
        "ExternalApiCallsPerRequest": ("ExternalApiCallsPerRequest", 20, 300),
        "ModelCallsPerRequest": ("ModelCallsPerRequest", 2, 300),
        "InputTokensPerRequest": ("InputTokensPerRequest", 20_000, 300),
        "OutputTokensPerRequest": ("OutputTokensPerRequest", 2_000, 300),
        "ConcurrentRequests": ("ConcurrentRequests", 5, 60),
    }
    topic_arn = {"Fn::GetAtt": ["AlertTopic", "TopicArn"]}
    assert set(alarms) == {*expected, "AnomalousSpend"}
    for logical_id, (metric, threshold, period) in expected.items():
        properties = alarms[logical_id]
        assert (properties["MetricName"], properties["Threshold"], properties["Period"]) == (
            metric,
            threshold,
            period,
        )
        assert properties["ActionsEnabled"] is True
        assert properties["TreatMissingData"] == "missing"
        assert properties["Tags"] == [{"Key": "Owner", "Value": INFRASTRUCTURE_OWNER}]
        assert f"Owner={INFRASTRUCTURE_OWNER}" in properties["AlarmDescription"]
        assert (
            "owner subscription intentionally deferred to deployed I-10B"
            in properties["AlarmDescription"]
        )
        assert properties["AlarmActions"] == [topic_arn]
        if properties["Namespace"] == CUSTOM_METRIC_NAMESPACE:
            assert properties["InsufficientDataActions"] == [topic_arn]
            assert metric in CUSTOM_METRIC_UNITS
        else:
            assert logical_id == "S3Storage"
            assert properties["Namespace"] == "AWS/S3"
            assert "InsufficientDataActions" not in properties

    anomaly = alarms["AnomalousSpend"]
    assert anomaly["ComparisonOperator"] == "GreaterThanUpperThreshold"
    assert anomaly["ThresholdMetricId"] == "ad1"
    assert anomaly["Metrics"][1]["Expression"] == "ANOMALY_DETECTION_BAND(m1, 2)"
    assert anomaly["Metrics"][0]["MetricStat"]["Metric"] == {
        "MetricName": "EstimatedMonthlySpendUsd",
        "Namespace": CUSTOM_METRIC_NAMESPACE,
    }
    assert anomaly["InsufficientDataActions"] == [topic_arn]
    assert anomaly["TreatMissingData"] == "missing"
    serialized = json.dumps(alarms)
    for prohibited in (
        "SpendAlert",
        "SpendStop",
        "ssm:PutParameter",
        "lambda:InvokeFunction",
        "cloudformation:",
    ):
        assert prohibited not in serialized


def test_safeguard_contract_matches_approved_limits_and_defaults() -> None:
    assert CUSTOM_METRIC_NAMESPACE == "Valkeyrie/KnowledgePlane"
    assert SAFEGUARD_THRESHOLDS == {
        "corpus_age_hours": 168,
        "s3_storage_bytes": 10_737_418_240,
        "aoss_storage_bytes": 10_737_418_240,
        "ingestion_failures": 1,
        "invalid_citations": 1,
        "dependency_throttles": 1,
        "p95_latency_seconds": 15,
        "accepted_requests_per_day": 100,
        "external_api_calls_per_request": 20,
        "model_calls_per_request": 2,
        "input_tokens_per_request": 20_000,
        "output_tokens_per_request": 2_000,
        "maximum_concurrency": 5,
    }
    assert CONTROL_DEFAULTS == {
        "promotion-enabled": "false",
        "runtime-enabled": "false",
        "model-processing-enabled": "false",
    }


def test_controls_default_disabled_and_d01_runtime_remains_permission_empty(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    parameters = {
        resource["Properties"]["Name"]: resource["Properties"]
        for resource in _resources(template).values()
        if resource["Type"] == "AWS::SSM::Parameter"
    }
    assert set(parameters) == {
        f"/valkeyrie-development/controls/{control}" for control in CONTROL_DEFAULTS
    }
    for properties in parameters.values():
        assert properties["Value"] == "false"
        assert properties["AllowedPattern"] == "^(false|true)$"
        assert properties["Tier"] == "Standard"
        assert properties["Tags"] == {
            "FailClosed": "true",
            "Owner": INFRASTRUCTURE_OWNER,
        }
        assert "missing, unreadable, malformed" in properties["Description"]
    policies = [
        resource
        for resource in _resources(template).values()
        if resource["Type"] == "AWS::IAM::Policy"
    ]
    serialized_policies = json.dumps(policies)
    assert "ssm:GetParameter" not in serialized_policies
    for prohibited in ("ssm:PutParameter", "ssm:DeleteParameter", "ssm:AddTagsToResource"):
        assert prohibited not in serialized_policies


def test_no_runtime_endpoint_secret_credential_provider_or_deploy_path(tmp_path: Path) -> None:
    template = _template(tmp_path / "assembly")
    serialized = json.dumps(template, sort_keys=True)
    for prohibited in (
        "AWS::Lambda::",
        "AWS::SQS::",
        "AWS::ApiGateway::",
        "AWS::ApiGatewayV2::",
        "AWS::SecretsManager::",
        "AWS::IAM::OIDCProvider",
        "AWS::IAM::AccessKey",
        "AWS::CloudFormation::",
        "slack",
        "github_token",
        "private_key",
        "alias/aws/sns",
        "AWS::SNS::Subscription",
        "cloudwatch:PutMetricData",
    ):
        assert prohibited not in serialized
    assert "Outputs" not in template


def test_configuration_is_explicit_not_context_or_lookup_driven(tmp_path: Path) -> None:
    config = FoundationConfig(account="111122223333", region="eu-west-1", environment="test")
    outdir = tmp_path / "explicit"
    manifest = _assembly_manifest(outdir, config)
    artifacts = cast(dict[str, dict[str, object]], manifest["artifacts"])
    stack = artifacts["KnowledgePlane"]
    assert stack["environment"] == "aws://111122223333/eu-west-1"
    properties = cast(dict[str, object], stack["properties"])
    assert properties["stackName"] == "valkeyrie-test-knowledge-plane"
    assert cast(dict[str, str], properties["tags"])["Environment"] == "test"
    template = json.loads((outdir / "KnowledgePlane.template.json").read_text())
    assert _properties(template, "CorpusBucket")["BucketName"] == (
        "valkeyrie-test-corpus-111122223333-eu-west-1"
    )
    assert manifest.get("missing") in (None, [])
    assert not (outdir / "cdk.context.json").exists()
    assert build_app(tmp_path / "context-check").node.try_get_context("unsafe") is None


def test_cdk_inputs_define_only_local_synthesis(tmp_path: Path) -> None:
    config = json.loads((ROOT / "cdk.json").read_text(encoding="utf-8"))
    assert config == {"app": "uv run python infra/app.py", "output": "cdk.out"}
    assert not (ROOT / "cdk.context.json").exists()

    outdir = tmp_path / "cdk.out"
    synthesize(outdir)
    assert {path.name for path in outdir.iterdir()} == {
        "KnowledgePlane.metadata.json",
        "KnowledgePlane.template.json",
        "cdk.out",
        "manifest.json",
        "validation-report.json",
    }


def test_synthesis_is_deterministic(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = _assembly_manifest(first_dir)
    second = _assembly_manifest(second_dir)
    assert first == second
    assert (first_dir / "KnowledgePlane.template.json").read_bytes() == (
        second_dir / "KnowledgePlane.template.json"
    ).read_bytes()
