from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from infra.app import build_app

ROLE_NAMES = {
    "valkeyrie-development-application-deployer",
    "valkeyrie-development-corpus-publisher",
    "valkeyrie-development-runtime",
    "valkeyrie-development-bedrock-kb",
}
OIDC_PROVIDER = {
    "Fn::Join": [
        "",
        [
            "arn:",
            {"Ref": "AWS::Partition"},
            ":iam::968533178160:oidc-provider/token.actions.githubusercontent.com",
        ],
    ]
}
GENERATION_SUFFIXES = {
    "/kb-documents/generations/*",
    "/control/generations/*",
}


def _template(outdir: Path) -> dict[str, object]:
    build_app(outdir).synth()
    return cast(
        dict[str, object],
        json.loads((outdir / "KnowledgePlane.template.json").read_text(encoding="utf-8")),
    )


def _resources(template: dict[str, object]) -> dict[str, dict[str, Any]]:
    return cast(dict[str, dict[str, Any]], template["Resources"])


def _roles(template: dict[str, object]) -> dict[str, tuple[str, dict[str, Any]]]:
    roles = {}
    for logical_id, resource in _resources(template).items():
        if resource["Type"] != "AWS::IAM::Role":
            continue
        properties = cast(dict[str, Any], resource["Properties"])
        roles[cast(str, properties["RoleName"])] = (logical_id, properties)
    assert len(roles) == 4
    return roles


def _policies(template: dict[str, object]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (logical_id, cast(dict[str, Any], resource["Properties"]))
        for logical_id, resource in _resources(template).items()
        if resource["Type"] == "AWS::IAM::Policy"
    ]


def _policy_for_role(
    template: dict[str, object], role_logical_id: str
) -> tuple[str, dict[str, Any]] | None:
    matches = [
        (logical_id, policy)
        for logical_id, policy in _policies(template)
        if policy["Roles"] == [{"Ref": role_logical_id}]
    ]
    assert len(matches) <= 1
    return matches[0] if matches else None


def _trust(role: dict[str, Any]) -> dict[str, Any]:
    policy = role["AssumeRolePolicyDocument"]
    assert policy["Version"] == "2012-10-17"
    statements = policy["Statement"]
    assert len(statements) == 1
    return cast(dict[str, Any], statements[0])


def _statements(policy: dict[str, Any]) -> list[dict[str, Any]]:
    document = policy["PolicyDocument"]
    assert document["Version"] == "2012-10-17"
    return cast(list[dict[str, Any]], document["Statement"])


def _resource_suffixes(statement: dict[str, Any]) -> set[str]:
    resources = statement["Resource"]
    if not isinstance(resources, list):
        resources = [resources]
    suffixes = set()
    for resource in resources:
        if isinstance(resource, dict) and "Fn::Join" in resource:
            suffixes.add(cast(list[Any], resource["Fn::Join"])[1][-1])
    return suffixes


def test_identity_registry_is_exact_separate_and_bounded(tmp_path: Path) -> None:
    template = _template(tmp_path / "assembly")
    roles = _roles(template)
    assert set(roles) == ROLE_NAMES
    expected_descriptions = {
        "valkeyrie-development-application-deployer": (
            "Protected application environment deployment identity"
        ),
        "valkeyrie-development-corpus-publisher": (
            "Protected conditional write-once corpus publication identity"
        ),
        "valkeyrie-development-runtime": "Valkeyrie Lambda runtime identity",
        "valkeyrie-development-bedrock-kb": (
            "Least-privilege Bedrock Knowledge Base ingestion and retrieval role"
        ),
    }
    for role_name, (_, role) in roles.items():
        assert set(role) == {
            "AssumeRolePolicyDocument",
            "Description",
            "MaxSessionDuration",
            "RoleName",
        }
        assert role["Description"] == expected_descriptions[role_name]
        assert role["MaxSessionDuration"] == 3600
        assert "ManagedPolicyArns" not in role
        assert "PermissionsBoundary" not in role


def test_github_roles_preserve_exact_repository_audience_and_environment_trust(
    tmp_path: Path,
) -> None:
    roles = _roles(_template(tmp_path / "assembly"))
    expected_subjects = {
        "valkeyrie-development-application-deployer": (
            "repo:sarthakaggarwal97/valkeyrie:environment:application"
        ),
        "valkeyrie-development-corpus-publisher": (
            "repo:sarthakaggarwal97/valkeyrie:environment:corpus"
        ),
    }
    for role_name, subject in expected_subjects.items():
        statement = _trust(roles[role_name][1])
        assert statement == {
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                    "token.actions.githubusercontent.com:sub": subject,
                }
            },
            "Effect": "Allow",
            "Principal": {"Federated": OIDC_PROVIDER},
        }
        assert "*" not in subject
        assert ":ref:" not in subject
        assert ":pull_request" not in subject


def test_runtime_trust_is_lambda_only_and_operator_role_is_deferred(tmp_path: Path) -> None:
    roles = _roles(_template(tmp_path / "assembly"))
    assert _trust(roles["valkeyrie-development-runtime"][1]) == {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
    }
    assert "valkeyrie-development-operator" not in roles
    serialized = json.dumps(roles, sort_keys=True)
    assert ":root" not in serialized
    assert "aws:MultiFactorAuthPresent" not in serialized


def test_application_deployer_and_runtime_remain_permission_empty_in_d01(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    roles = _roles(template)
    for role_name in (
        "valkeyrie-development-application-deployer",
        "valkeyrie-development-runtime",
    ):
        logical_id, _ = roles[role_name]
        assert _policy_for_role(template, logical_id) is None
    serialized = json.dumps(template, sort_keys=True)
    for prohibited in (
        "cloudformation:CreateStack",
        "cloudformation:UpdateStack",
        "lambda:UpdateFunctionCode",
        "bedrock:Retrieve",
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "iam:PassRole",
    ):
        assert prohibited not in serialized


def test_publisher_may_only_conditionally_create_read_and_list_generation_objects(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    publisher_logical_id, _ = _roles(template)["valkeyrie-development-corpus-publisher"]
    match = _policy_for_role(template, publisher_logical_id)
    assert match is not None
    _, policy = match
    statements = {statement["Sid"]: statement for statement in _statements(policy)}
    assert set(statements) == {
        "ConditionalGenerationCreate",
        "ReadGenerationObjects",
        "ListGenerationObjects",
    }
    create = statements["ConditionalGenerationCreate"]
    assert create["Effect"] == "Allow"
    assert create["Action"] == "s3:PutObject"
    assert create["Condition"] == {"StringEquals": {"s3:if-none-match": "*"}}
    assert _resource_suffixes(create) == GENERATION_SUFFIXES

    read = statements["ReadGenerationObjects"]
    assert read["Action"] == "s3:GetObject"
    assert _resource_suffixes(read) == GENERATION_SUFFIXES

    listing = statements["ListGenerationObjects"]
    assert listing["Action"] == "s3:ListBucket"
    assert listing["Condition"] == {
        "ForAnyValue:StringLike": {
            "s3:prefix": [
                "kb-documents/generations/*",
                "control/generations/*",
            ]
        }
    }
    actions = {statement["Action"] for statement in statements.values()}
    assert actions == {"s3:PutObject", "s3:GetObject", "s3:ListBucket"}
    serialized = json.dumps(policy, sort_keys=True)
    for prohibited in (
        "DeleteObject",
        "DeleteObjectVersion",
        "AbortMultipartUpload",
        "ListMultipartUploadParts",
        "ListBucketMultipartUploads",
        "PutObjectAcl",
        "PutObjectTagging",
        "bedrock:",
        "aoss:",
        "dynamodb:",
        "ssm:",
        "iam:PassRole",
    ):
        assert prohibited not in serialized


def test_bucket_policy_denies_nonconditional_overwrite_and_all_generation_deletes(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    bucket_policies = [
        resource["Properties"]
        for resource in _resources(template).values()
        if resource["Type"] == "AWS::S3::BucketPolicy"
    ]
    assert len(bucket_policies) == 1
    statements = {
        statement["Sid"]: statement
        for statement in bucket_policies[0]["PolicyDocument"]["Statement"]
    }
    overwrite = statements["DenyPublisherNonConditionalWrites"]
    assert overwrite["Effect"] == "Deny"
    assert overwrite["Action"] == "s3:PutObject"
    assert overwrite["Condition"] == {"StringNotEquals": {"s3:if-none-match": "*"}}
    assert _resource_suffixes(overwrite) == GENERATION_SUFFIXES
    deletion = statements["DenyPublisherGenerationDeletion"]
    assert deletion["Effect"] == "Deny"
    assert deletion["Action"] == ["s3:DeleteObject", "s3:DeleteObjectVersion"]
    assert _resource_suffixes(deletion) == GENERATION_SUFFIXES
    assert statements["DenyInsecureTransport"]["Condition"] == {
        "Bool": {"aws:SecureTransport": "false"}
    }
    publisher_arn = {"Fn::GetAtt": ["CorpusPublisherRole36D344EA", "Arn"]}
    assert overwrite["Principal"] == deletion["Principal"] == {"AWS": publisher_arn}


def test_no_stack_owned_policy_mutates_the_approved_deployment_principal(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    policies = dict(_policies(template))
    assert len(policies) == 2
    assert all("Roles" in policy and "Users" not in policy for policy in policies.values())
    serialized = json.dumps(template, sort_keys=True)
    for prohibited in (
        "VectorIndexProvisionerPolicy",
        "KnowledgePlaneIndexProvisionerRoleName",
        "valkeyrie-development-index-provisioner",
        "AWS::IAM::UserPolicy",
    ):
        assert prohibited not in serialized


def test_bedrock_role_has_exact_service_trust_and_least_resource_permissions(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    role_logical_id, role = _roles(template)["valkeyrie-development-bedrock-kb"]
    assert _trust(role) == {
        "Action": "sts:AssumeRole",
        "Condition": {
            "ArnLike": {
                "aws:SourceArn": {
                    "Fn::Join": [
                        "",
                        [
                            "arn:",
                            {"Ref": "AWS::Partition"},
                            ":bedrock:us-east-1:968533178160:knowledge-base/*",
                        ],
                    ]
                }
            },
            "StringEquals": {"aws:SourceAccount": "968533178160"},
        },
        "Effect": "Allow",
        "Principal": {"Service": "bedrock.amazonaws.com"},
    }
    match = _policy_for_role(template, role_logical_id)
    assert match is not None
    _, policy = match
    statements = {statement["Sid"]: statement for statement in _statements(policy)}
    assert set(statements) == {
        "InvokeFrozenEmbeddingModel",
        "ListIngestibleCorpusPrefix",
        "ReadIngestibleCorpusPrefix",
        "UseDedicatedVectorCollection",
    }
    assert statements["InvokeFrozenEmbeddingModel"]["Action"] == "bedrock:InvokeModel"
    assert "amazon.titan-embed-text-v2:0" in json.dumps(
        statements["InvokeFrozenEmbeddingModel"]["Resource"]
    )
    assert statements["ListIngestibleCorpusPrefix"]["Condition"] == {
        "StringLike": {"s3:prefix": ["kb-documents/*"]}
    }
    assert _resource_suffixes(statements["ReadIngestibleCorpusPrefix"]) == {"/kb-documents/*"}
    assert statements["UseDedicatedVectorCollection"]["Action"] == "aoss:APIAccessAll"
    assert statements["UseDedicatedVectorCollection"]["Resource"] == {
        "Fn::GetAtt": ["VectorCollection", "Arn"]
    }
    serialized = json.dumps(policy, sort_keys=True)
    for prohibited in (
        "control/",
        "s3:PutObject",
        "s3:DeleteObject",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:StartIngestionJob",
        "aoss:DashboardsAccessAll",
        "iam:PassRole",
        'Resource": "*',
    ):
        assert prohibited not in serialized


def test_aoss_data_policy_separates_bedrock_operations_from_exact_index_control_plane(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    policies = [
        resource["Properties"]
        for resource in _resources(template).values()
        if resource["Type"] == "AWS::OpenSearchServerless::AccessPolicy"
    ]
    assert len(policies) == 1
    policy = policies[0]
    assert policy["Name"] == "valkeyrie-development-data"
    assert policy["Type"] == "data"
    assert policy["Description"] == (
        "Bedrock document operations and D-01-bound CloudFormation index control-plane "
        "operations on the exact dedicated index"
    )
    join = policy["Policy"]["Fn::Join"]
    assert join[0] == ""
    tokens = [part for part in join[1] if isinstance(part, dict)]
    assert tokens == [
        {"Fn::GetAtt": ["BedrockKnowledgeBaseRole24C5E17B", "Arn"]},
        {"Ref": "AWS::Partition"},
    ]

    rendered_parts = []
    for part in join[1]:
        if not isinstance(part, dict):
            rendered_parts.append(part)
        elif part == {"Fn::GetAtt": ["BedrockKnowledgeBaseRole24C5E17B", "Arn"]}:
            rendered_parts.append("arn:aws:iam::968533178160:role/bedrock-kb")
        elif part == {"Ref": "AWS::Partition"}:
            rendered_parts.append("aws")
        else:  # pragma: no cover - tokens are asserted above
            raise AssertionError(part)
    assert json.loads("".join(rendered_parts)) == [
        {
            "Description": "Bedrock Knowledge Base document operations",
            "Principal": ["arn:aws:iam::968533178160:role/bedrock-kb"],
            "Rules": [
                {
                    "Resource": ["collection/valkeyrie-development-vectors"],
                    "ResourceType": "collection",
                    "Permission": [
                        "aoss:CreateCollectionItems",
                        "aoss:DescribeCollectionItems",
                        "aoss:UpdateCollectionItems",
                    ],
                },
                {
                    "Resource": ["index/valkeyrie-development-vectors/*"],
                    "ResourceType": "index",
                    "Permission": [
                        "aoss:CreateIndex",
                        "aoss:DescribeIndex",
                        "aoss:ReadDocument",
                        "aoss:UpdateIndex",
                        "aoss:WriteDocument",
                    ],
                },
            ],
        },
        {
            "Description": "CloudFormation exact-index control-plane operations",
            "Principal": ["arn:aws:iam::968533178160:user/sarthagg"],
            "Rules": [
                {
                    "Resource": ["collection/valkeyrie-development-vectors"],
                    "ResourceType": "collection",
                    "Permission": ["aoss:DescribeCollectionItems"],
                },
                {
                    "Resource": ["index/valkeyrie-development-vectors/valkeyrie-development-kb"],
                    "ResourceType": "index",
                    "Permission": [
                        "aoss:CreateIndex",
                        "aoss:DescribeIndex",
                        "aoss:UpdateIndex",
                        "aoss:DeleteIndex",
                    ],
                },
            ],
        },
    ]
    serialized = json.dumps(policy, sort_keys=True)
    for prohibited in (
        "valkeyrie-development-application-deployer",
        "valkeyrie-development-corpus-publisher",
        "valkeyrie-development-runtime",
        "aoss:DashboardsAccessAll",
        'Principal":["*',
    ):
        assert prohibited not in serialized


def test_alert_key_and_topic_policies_only_authorize_bounded_service_delivery(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    keys = [
        resource["Properties"]
        for resource in _resources(template).values()
        if resource["Type"] == "AWS::KMS::Key"
    ]
    assert len(keys) == 1
    key_statements = keys[0]["KeyPolicy"]["Statement"]
    service_statements = [
        statement for statement in key_statements if "Service" in statement["Principal"]
    ]
    assert {statement["Principal"]["Service"] for statement in service_statements} == {
        "cloudwatch.amazonaws.com",
    }
    for statement in service_statements:
        assert statement["Action"] == ["kms:GenerateDataKey*", "kms:Decrypt"]
        assert statement["Resource"] == "*"
        assert statement["Condition"] == {
            "StringEquals": {
                "aws:SourceAccount": "968533178160",
                "kms:ViaService": "sns.us-east-1.amazonaws.com",
            }
        }

    topic_policies = [
        resource["Properties"]
        for resource in _resources(template).values()
        if resource["Type"] == "AWS::SNS::TopicPolicy"
    ]
    assert len(topic_policies) == 1
    topic_statements = topic_policies[0]["PolicyDocument"]["Statement"]
    assert {statement["Principal"]["Service"] for statement in topic_statements} == {
        "cloudwatch.amazonaws.com",
    }
    for statement in topic_statements:
        assert statement["Action"] == "sns:Publish"
        assert statement["Resource"] == {"Fn::GetAtt": ["AlertTopic", "TopicArn"]}
        assert statement["Condition"] == {"StringEquals": {"aws:SourceAccount": "968533178160"}}
    serialized = json.dumps([service_statements, topic_statements], sort_keys=True)
    for prohibited in (
        '"Service": "sns.amazonaws.com"',
        '"Principal": "*"',
        '"Action": "sns:*"',
        '"Action": "kms:*"',
        "kms:Encrypt",
        "valkeyrie-development-application-deployer",
        "valkeyrie-development-corpus-publisher",
        "valkeyrie-development-runtime",
    ):
        assert prohibited not in serialized


def test_template_creates_no_provider_credential_project_write_or_unbounded_policy(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path / "assembly")
    serialized = json.dumps(template, sort_keys=True)
    for prohibited in (
        "AWS::IAM::OIDCProvider",
        "AWS::IAM::AccessKey",
        "AWS::IAM::ManagedPolicy",
        "AdministratorAccess",
        "PowerUserAccess",
        "iam:PassRole",
        "github_token",
        "private_key",
        "workflow_dispatch",
        "pull_request_target",
        "ssm:PutParameter",
        "dynamodb:UpdateItem",
        "dynamodb:PutItem",
    ):
        assert prohibited not in serialized
    assert "Outputs" not in template


def test_all_trust_actions_are_bounded_and_no_role_can_assume_another(tmp_path: Path) -> None:
    roles = _roles(_template(tmp_path / "assembly"))
    statements = [_trust(role) for _, role in roles.values()]
    assert {statement["Action"] for statement in statements} == {
        "sts:AssumeRole",
        "sts:AssumeRoleWithWebIdentity",
    }
    assert all(statement["Effect"] == "Allow" for statement in statements)
    assert all(statement["Action"] != "sts:*" for statement in statements)
    serialized = json.dumps(statements, sort_keys=True)
    for role_name in ROLE_NAMES:
        assert f":role/{role_name}" not in serialized
    aws_principals = [
        statement["Principal"].get("AWS")
        for statement in statements
        if isinstance(statement["Principal"], dict)
    ]
    assert all(principal is None for principal in aws_principals)
