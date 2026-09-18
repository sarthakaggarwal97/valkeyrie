"""Synthesize the isolated Valkeyrie development knowledge plane."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from aws_cdk import (
    App,
    ArnFormat,
    CfnDeletionPolicy,
    CfnTag,
    Duration,
    Environment,
    LegacyStackSynthesizer,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_opensearchserverless as aoss
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sns as sns
from aws_cdk import aws_ssm as ssm

from valkeyrie.retrieval_config import ChunkingConfiguration, load_retrieval_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = Path("cdk.out")
_NAME = re.compile(r"^[a-z][a-z0-9-]*$")
_ACCOUNT = re.compile(r"^[0-9]{12}$")
_REGION = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-[0-9]+$")
GITHUB_REPOSITORY: Final = "sarthakaggarwal97/valkeyrie"
# GitHub's immutable numeric ids for the owner and repository. Its OIDC tokens now carry the
# subject as repo:<owner>@<owner_id>/<name>@<repo_id>:environment:<env>, which survives a rename;
# a trust on the bare name never matched it, and every scheduled refresh since August failed at
# the credential step with "Not authorized to perform sts:AssumeRoleWithWebIdentity", seen in
# CloudTrail. Pinning the ids is STRICTER than the name: a repository recreated under the same
# name has a different id and does not match. Confirmed against the GitHub API on 2026-09-18.
GITHUB_OWNER_ID: Final = 25262500
GITHUB_REPOSITORY_ID: Final = 1338597592
APPLICATION_ENVIRONMENT: Final = "application"
CORPUS_ENVIRONMENT: Final = "corpus"
INFRASTRUCTURE_OWNER: Final = "sarthakaggarwal97"
DEPLOYMENT_PRINCIPAL_NAME: Final = "sarthagg"
CORPUS_DATA_SOURCE_PREFIX: Final = "kb-documents/"


@dataclass(frozen=True)
class RetiringDataSource:
    """A data source kept alive during a chunking migration, with vector deletion armed."""

    logical_id: str
    name: str
    chunking: ChunkingConfiguration


def _data_source_logical_id(candidate_id: str) -> str:
    # "CorpusDataSource" was the pre-migration logical ID for titan-v2-1024-fixed-300-20 and is
    # kept for it so the deployed resource is recognised rather than replaced.
    if candidate_id == "titan-v2-1024-fixed-300-20":
        return "CorpusDataSource"
    return "CorpusDataSource" + "".join(part.capitalize() for part in candidate_id.split("-"))


# Empty since the hierarchical migration completed on 2026-09-18: the fixed-300 source ingested
# nothing new, was flipped to DELETE in Phase A, and was removed in Phase B so its vectors went
# with it. During the overlap the runtime rejected some answers because two chunkings of one
# document share an evidence identity, so a future migration should keep the drain short.
RETIRING_DATA_SOURCES: Final[tuple[RetiringDataSource, ...]] = ()
GENERATION_PREFIXES: Final = (
    "kb-documents/generations/*",
    "control/generations/*",
)
CUSTOM_METRIC_NAMESPACE: Final = "Valkeyrie/KnowledgePlane"
CONTROL_DEFAULTS: Final = {
    "promotion-enabled": "false",
    "runtime-enabled": "false",
    "model-processing-enabled": "false",
}
SAFEGUARD_THRESHOLDS: Final = {
    "corpus_age_hours": 168,
    "s3_storage_bytes": 10 * 1024 * 1024 * 1024,
    "aoss_storage_bytes": 10 * 1024 * 1024 * 1024,
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


@dataclass(frozen=True)
class FoundationConfig:
    """Reviewed inputs for the isolated development knowledge plane."""

    account: str
    region: str
    project: str = "valkeyrie"
    environment: str = "development"
    boundary: str = "knowledge-plane"

    def __post_init__(self) -> None:
        if not _ACCOUNT.fullmatch(self.account):
            raise ValueError("account must be a 12-digit AWS account ID")
        if not _REGION.fullmatch(self.region):
            raise ValueError("region must be an explicit AWS region")
        for field, value in (
            ("project", self.project),
            ("environment", self.environment),
            ("boundary", self.boundary),
        ):
            if not _NAME.fullmatch(value):
                raise ValueError(f"{field} must be a lowercase namespacing component")

    @property
    def namespace(self) -> str:
        """Return the mandatory prefix for every dedicated resource."""
        return f"{self.project}-{self.environment}"

    @property
    def stack_name(self) -> str:
        """Return the isolated knowledge-plane stack name."""
        return f"{self.namespace}-{self.boundary}"

    @property
    def tags(self) -> dict[str, str]:
        """Return the tags inherited by taggable resources in this stack."""
        return {
            "Project": self.project,
            "Environment": self.environment,
            "ManagedBy": "aws-cdk",
            "Boundary": self.boundary,
        }


DEFAULT_CONFIG = FoundationConfig(account="968533178160", region="us-east-1")


class KnowledgePlaneStack(Stack):
    """Dedicated corpus, retrieval, identity, and safeguard definitions."""

    def __init__(self, scope: App, construct_id: str, *, config: FoundationConfig) -> None:
        super().__init__(
            scope,
            construct_id,
            stack_name=config.stack_name,
            env=Environment(account=config.account, region=config.region),
            description="Valkeyrie isolated development knowledge plane (synthesis only)",
            tags=config.tags,
            synthesizer=LegacyStackSynthesizer(),
        )
        self.namespace = config.namespace
        self._config = config
        self._owner_tags = [CfnTag(key="Owner", value=INFRASTRUCTURE_OWNER)]

        oidc_provider = (
            f"arn:{self.partition}:iam::{config.account}:"
            "oidc-provider/token.actions.githubusercontent.com"
        )

        def github_environment(environment: str) -> iam.FederatedPrincipal:
            claims = "token.actions.githubusercontent.com"
            owner, name = GITHUB_REPOSITORY.split("/")
            return iam.FederatedPrincipal(
                oidc_provider,
                {
                    "StringEquals": {
                        f"{claims}:aud": "sts.amazonaws.com",
                        # Both subject forms GitHub has issued. StringEquals with a list is an OR
                        # of exact matches; there is still no wildcard anywhere in the trust.
                        f"{claims}:sub": [
                            f"repo:{owner}@{GITHUB_OWNER_ID}/{name}@{GITHUB_REPOSITORY_ID}"
                            f":environment:{environment}",
                            f"repo:{GITHUB_REPOSITORY}:environment:{environment}",
                        ],
                    }
                },
                "sts:AssumeRoleWithWebIdentity",
            )

        iam.Role(
            self,
            "ApplicationDeployerRole",
            role_name=f"{self.namespace}-application-deployer",
            description="Protected application environment deployment identity",
            assumed_by=github_environment(APPLICATION_ENVIRONMENT),
            max_session_duration=Duration.hours(1),
        )
        publisher = iam.Role(
            self,
            "CorpusPublisherRole",
            role_name=f"{self.namespace}-corpus-publisher",
            description="Protected conditional write-once corpus publication identity",
            assumed_by=github_environment(CORPUS_ENVIRONMENT),
            # A refresh runs 60 to 90 minutes warm and longer cold, in one assumed session that
            # configure-aws-credentials does not renew. One hour expired mid-ingestion.
            max_session_duration=Duration.hours(3),
        )
        iam.Role(
            self,
            "RuntimeRole",
            role_name=f"{self.namespace}-runtime",
            description="Valkeyrie Lambda runtime identity",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            max_session_duration=Duration.hours(1),
        )
        # An operator role remains unsafe until a distinct approved human or SSO principal exists.

        corpus_bucket = self._create_corpus_bucket(config)
        self._grant_write_once_generation_publication(corpus_bucket, publisher)
        state_table = self._create_state_table()
        vector_collection, vector_index = self._create_vector_store()
        knowledge_base = self._create_knowledge_base(
            config,
            corpus_bucket,
            vector_collection,
            vector_index,
        )
        data_source = self._create_data_source(config, corpus_bucket, knowledge_base)
        self._grant_corpus_refresh_execution(publisher, state_table, knowledge_base, data_source)
        self._create_observability(config, corpus_bucket, state_table, vector_collection)
        self._create_disabled_controls()

    def _create_corpus_bucket(self, config: FoundationConfig) -> s3.CfnBucket:
        bucket = s3.CfnBucket(
            self,
            "CorpusBucket",
            bucket_name=(f"{self.namespace}-corpus-{config.account}-{config.region}"),
            bucket_encryption=s3.CfnBucket.BucketEncryptionProperty(
                server_side_encryption_configuration=[
                    s3.CfnBucket.ServerSideEncryptionRuleProperty(
                        server_side_encryption_by_default=(
                            s3.CfnBucket.ServerSideEncryptionByDefaultProperty(
                                sse_algorithm="AES256"
                            )
                        )
                    )
                ]
            ),
            lifecycle_configuration=s3.CfnBucket.LifecycleConfigurationProperty(
                rules=[
                    s3.CfnBucket.RuleProperty(
                        id="AbortIncompleteMultipartUploads",
                        status="Enabled",
                        abort_incomplete_multipart_upload=(
                            s3.CfnBucket.AbortIncompleteMultipartUploadProperty(
                                days_after_initiation=7
                            )
                        ),
                    )
                ]
            ),
            ownership_controls=s3.CfnBucket.OwnershipControlsProperty(
                rules=[
                    s3.CfnBucket.OwnershipControlsRuleProperty(
                        object_ownership="BucketOwnerEnforced"
                    )
                ]
            ),
            public_access_block_configuration=s3.CfnBucket.PublicAccessBlockConfigurationProperty(
                block_public_acls=True,
                block_public_policy=True,
                ignore_public_acls=True,
                restrict_public_buckets=True,
            ),
            versioning_configuration=s3.CfnBucket.VersioningConfigurationProperty(status="Enabled"),
            tags=self._owner_tags,
        )
        bucket.cfn_options.deletion_policy = CfnDeletionPolicy.RETAIN
        bucket.cfn_options.update_replace_policy = CfnDeletionPolicy.RETAIN
        return bucket

    def _grant_corpus_refresh_execution(
        self,
        publisher: iam.Role,
        state_table: dynamodb.CfnTable,
        knowledge_base: bedrock.CfnKnowledgeBase,
        data_source: bedrock.CfnDataSource,
    ) -> None:
        """Grant the publisher what a refresh needs beyond writing objects.

        Publication was the only granted step, so a scheduled refresh could write generation
        objects and then fail on its first lifecycle write. The remaining steps are the ones the
        release performs: conditional lifecycle transitions on the state table, starting and
        polling one ingestion job, and the generation-filtered retrieval smoke that activation
        runs before it swaps the active pointer.

        No delete anywhere: a generation is write-once, and activation moves a pointer rather than
        removing what it replaces, so the previous corpus stays retrievable for rollback.
        """
        publisher.add_to_policy(
            iam.PolicyStatement(
                sid="ConditionalGenerationLifecycle",
                # No DeleteItem: lifecycle rows are the audit trail of what was published.
                # TransactWriteItems is the activation compare-and-swap: a condition check on the
                # candidate and a put of the active pointer in one transaction. Without it every
                # refresh would ingest for an hour and then fail at the final write.
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:TransactWriteItems",
                ],
                resources=[state_table.attr_arn],
                # Scoped to the corpus-owned key families. The same table also holds request
                # audits and protected approvals, which the publisher has no reason to touch.
                conditions={
                    "ForAllValues:StringLike": {
                        "dynamodb:LeadingKeys": [
                            "generation#*",
                            "candidate_generation",
                            "active_generation",
                        ]
                    }
                },
            )
        )
        publisher.add_to_policy(
            iam.PolicyStatement(
                sid="IngestExactKnowledgeBase",
                actions=["bedrock:StartIngestionJob", "bedrock:GetIngestionJob"],
                # Scoped by resource only. A bedrock:DataSourceId condition was tried first and
                # simulate-principal-policy returned implicitDeny for both actions: the key does not
                # apply to them, so the condition could never be satisfied and the grant would have
                # denied every ingestion while looking correct in review. This knowledge base has
                # exactly one data source, which the stack also owns, so the ARN is the boundary.
                resources=[knowledge_base.attr_knowledge_base_arn],
            )
        )
        publisher.add_to_policy(
            iam.PolicyStatement(
                sid="SmokeTestCandidateRetrieval",
                # Activation refuses to swap the pointer until a live retrieval against the
                # candidate generation returns evidence, so this read is part of the gate.
                actions=["bedrock:Retrieve"],
                resources=[knowledge_base.attr_knowledge_base_arn],
            )
        )

    def _grant_write_once_generation_publication(
        self,
        bucket: s3.CfnBucket,
        publisher: iam.Role,
    ) -> None:
        object_arns = [f"{bucket.attr_arn}/{prefix}" for prefix in GENERATION_PREFIXES]
        publisher.add_to_policy(
            iam.PolicyStatement(
                sid="ConditionalGenerationCreate",
                actions=["s3:PutObject"],
                resources=object_arns,
                conditions={"StringEquals": {"s3:if-none-match": "*"}},
            )
        )
        publisher.add_to_policy(
            iam.PolicyStatement(
                sid="ReadGenerationObjects",
                actions=["s3:GetObject"],
                resources=object_arns,
            )
        )
        publisher.add_to_policy(
            iam.PolicyStatement(
                sid="ListGenerationObjects",
                actions=["s3:ListBucket"],
                resources=[bucket.attr_arn],
                conditions={"ForAnyValue:StringLike": {"s3:prefix": list(GENERATION_PREFIXES)}},
            )
        )

        bucket_policy = s3.CfnBucketPolicy(
            self,
            "CorpusBucketPolicy",
            bucket=bucket.ref,
            policy_document={
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "DenyInsecureTransport",
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:*",
                        "Resource": [bucket.attr_arn, f"{bucket.attr_arn}/*"],
                        "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                    },
                    {
                        "Sid": "DenyPublisherNonConditionalWrites",
                        "Effect": "Deny",
                        "Principal": {"AWS": publisher.role_arn},
                        "Action": "s3:PutObject",
                        "Resource": object_arns,
                        "Condition": {"StringNotEquals": {"s3:if-none-match": "*"}},
                    },
                    {
                        "Sid": "DenyPublisherGenerationDeletion",
                        "Effect": "Deny",
                        "Principal": {"AWS": publisher.role_arn},
                        "Action": ["s3:DeleteObject", "s3:DeleteObjectVersion"],
                        "Resource": object_arns,
                    },
                ],
            },
        )
        bucket_policy.add_resource_dependency(bucket)

    def _create_state_table(self) -> dynamodb.CfnTable:
        table = dynamodb.CfnTable(
            self,
            "StateTable",
            table_name=f"{self.namespace}-state",
            attribute_definitions=[
                dynamodb.CfnTable.AttributeDefinitionProperty(
                    attribute_name="pk", attribute_type="S"
                )
            ],
            key_schema=[dynamodb.CfnTable.KeySchemaProperty(attribute_name="pk", key_type="HASH")],
            billing_mode="PAY_PER_REQUEST",
            on_demand_throughput=dynamodb.CfnTable.OnDemandThroughputProperty(
                max_read_request_units=25,
                max_write_request_units=25,
            ),
            point_in_time_recovery_specification=(
                dynamodb.CfnTable.PointInTimeRecoverySpecificationProperty(
                    point_in_time_recovery_enabled=True,
                    recovery_period_in_days=35,
                )
            ),
            sse_specification=dynamodb.CfnTable.SSESpecificationProperty(
                sse_enabled=True,
                sse_type="KMS",
            ),
            table_class="STANDARD",
            deletion_protection_enabled=True,
            time_to_live_specification=dynamodb.CfnTable.TimeToLiveSpecificationProperty(
                attribute_name="expires_at",
                enabled=True,
            ),
            tags=self._owner_tags,
        )
        table.cfn_options.deletion_policy = CfnDeletionPolicy.RETAIN
        table.cfn_options.update_replace_policy = CfnDeletionPolicy.RETAIN
        return table

    def _create_vector_store(self) -> tuple[aoss.CfnCollection, aoss.CfnIndex]:
        collection_name = f"{self.namespace}-vectors"
        index_name = f"{self.namespace}-kb"
        provisioner_principal_arn = self.format_arn(
            service="iam",
            region="",
            account=self._config.account,
            resource="user",
            resource_name=DEPLOYMENT_PRINCIPAL_NAME,
            arn_format=ArnFormat.SLASH_RESOURCE_NAME,
        )
        encryption_policy = aoss.CfnSecurityPolicy(
            self,
            "VectorEncryptionPolicy",
            name=f"{self.namespace}-enc",
            type="encryption",
            description="AWS-owned-key encryption for the dedicated Valkeyrie vector collection",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "Resource": [f"collection/{collection_name}"],
                            "ResourceType": "collection",
                        }
                    ],
                    "AWSOwnedKey": True,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        network_policy = aoss.CfnSecurityPolicy(
            self,
            "VectorNetworkPolicy",
            name=f"{self.namespace}-network",
            type="network",
            description="Private Bedrock-only network access for the Valkeyrie vector collection",
            policy=json.dumps(
                [
                    {
                        "Description": "Bedrock private collection access only",
                        "Rules": [
                            {
                                "Resource": [f"collection/{collection_name}"],
                                "ResourceType": "collection",
                            }
                        ],
                        "AllowFromPublic": False,
                        "SourceServices": ["bedrock.amazonaws.com"],
                    },
                ],
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        collection = aoss.CfnCollection(
            self,
            "VectorCollection",
            name=collection_name,
            description="Derived Valkeyrie Bedrock Knowledge Base vector index",
            type="VECTORSEARCH",
            standby_replicas="DISABLED",
            deletion_protection="ENABLED",
            tags=self._owner_tags,
        )
        collection.add_resource_dependency(encryption_policy)
        collection.add_resource_dependency(network_policy)

        service_role = iam.Role(
            self,
            "BedrockKnowledgeBaseRole",
            role_name=f"{self.namespace}-bedrock-kb",
            description="Least-privilege Bedrock Knowledge Base ingestion and retrieval role",
            assumed_by=iam.ServicePrincipal(
                "bedrock.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": self._config.account},
                    "ArnLike": {
                        "aws:SourceArn": self.format_arn(
                            service="bedrock",
                            resource="knowledge-base",
                            resource_name="*",
                            arn_format=ArnFormat.SLASH_RESOURCE_NAME,
                        )
                    },
                },
            ),
            max_session_duration=Duration.hours(1),
        )
        self._bedrock_role = service_role
        data_access_policy = aoss.CfnAccessPolicy(
            self,
            "VectorDataAccessPolicy",
            name=f"{self.namespace}-data",
            type="data",
            description=(
                "Bedrock document operations and D-01-bound CloudFormation index control-plane "
                "operations on the exact dedicated index"
            ),
            policy=Stack.of(self).to_json_string(
                [
                    {
                        "Description": "Bedrock Knowledge Base document operations",
                        "Principal": [service_role.role_arn],
                        "Rules": [
                            {
                                "Resource": [f"collection/{collection_name}"],
                                "ResourceType": "collection",
                                "Permission": [
                                    "aoss:CreateCollectionItems",
                                    "aoss:DescribeCollectionItems",
                                    "aoss:UpdateCollectionItems",
                                ],
                            },
                            {
                                "Resource": [f"index/{collection_name}/*"],
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
                        "Principal": [provisioner_principal_arn],
                        "Rules": [
                            {
                                "Resource": [f"collection/{collection_name}"],
                                "ResourceType": "collection",
                                "Permission": ["aoss:DescribeCollectionItems"],
                            },
                            {
                                "Resource": [f"index/{collection_name}/{index_name}"],
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
            ),
        )
        data_access_policy.node.add_dependency(service_role)

        selected = load_retrieval_config(ROOT / "retrieval-config.yaml", project_root=ROOT).selected
        vector_index = aoss.CfnIndex(
            self,
            "VectorIndex",
            collection_endpoint=collection.attr_collection_endpoint,
            index_name=index_name,
            settings=aoss.CfnIndex.IndexSettingsProperty(
                index=aoss.CfnIndex.IndexProperty(knn=True)
            ),
            mappings=aoss.CfnIndex.MappingsProperty(
                properties={
                    selected.index.vector_field: aoss.CfnIndex.PropertyMappingProperty(
                        type="knn_vector",
                        dimension=selected.index.dimensions,
                        method=aoss.CfnIndex.MethodProperty(
                            name=selected.index.algorithm,
                            engine=selected.index.engine,
                            space_type=selected.index.distance_metric,
                        ),
                    ),
                    selected.index.text_field: aoss.CfnIndex.PropertyMappingProperty(
                        type="text",
                        index=True,
                    ),
                    selected.index.metadata_field: aoss.CfnIndex.PropertyMappingProperty(
                        type="text",
                        index=False,
                    ),
                }
            ),
        )
        vector_index.add_resource_dependency(data_access_policy)
        vector_index.add_resource_dependency(collection)
        vector_index.add_resource_dependency(network_policy)
        return collection, vector_index

    def _create_knowledge_base(
        self,
        config: FoundationConfig,
        bucket: s3.CfnBucket,
        collection: aoss.CfnCollection,
        vector_index: aoss.CfnIndex,
    ) -> bedrock.CfnKnowledgeBase:
        selected = load_retrieval_config(ROOT / "retrieval-config.yaml", project_root=ROOT).selected
        model_arn = self.format_arn(
            service="bedrock",
            account="",
            resource="foundation-model",
            resource_name=selected.embedding.model_id,
            arn_format=ArnFormat.SLASH_RESOURCE_NAME,
        )
        self._bedrock_role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeFrozenEmbeddingModel",
                actions=["bedrock:InvokeModel"],
                resources=[model_arn],
            )
        )
        self._bedrock_role.add_to_policy(
            iam.PolicyStatement(
                sid="ListIngestibleCorpusPrefix",
                actions=["s3:ListBucket"],
                resources=[bucket.attr_arn],
                conditions={"StringLike": {"s3:prefix": [f"{CORPUS_DATA_SOURCE_PREFIX}*"]}},
            )
        )
        self._bedrock_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadIngestibleCorpusPrefix",
                actions=["s3:GetObject"],
                resources=[f"{bucket.attr_arn}/{CORPUS_DATA_SOURCE_PREFIX}*"],
            )
        )
        self._bedrock_role.add_to_policy(
            iam.PolicyStatement(
                sid="UseDedicatedVectorCollection",
                actions=["aoss:APIAccessAll"],
                resources=[collection.attr_arn],
            )
        )

        knowledge_base = bedrock.CfnKnowledgeBase(
            self,
            "KnowledgeBase",
            name=f"{self.namespace}-kb",
            description="Single frozen Valkeyrie beta Knowledge Base",
            role_arn=self._bedrock_role.role_arn,
            knowledge_base_configuration=(
                bedrock.CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
                    type="VECTOR",
                    vector_knowledge_base_configuration=(
                        bedrock.CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
                            embedding_model_arn=model_arn,
                            embedding_model_configuration=(
                                bedrock.CfnKnowledgeBase.EmbeddingModelConfigurationProperty(
                                    bedrock_embedding_model_configuration=(
                                        bedrock.CfnKnowledgeBase.BedrockEmbeddingModelConfigurationProperty(
                                            dimensions=selected.embedding.dimensions,
                                            embedding_data_type="FLOAT32",
                                        )
                                    )
                                )
                            ),
                        )
                    ),
                )
            ),
            storage_configuration=bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
                type="OPENSEARCH_SERVERLESS",
                opensearch_serverless_configuration=(
                    bedrock.CfnKnowledgeBase.OpenSearchServerlessConfigurationProperty(
                        collection_arn=collection.attr_arn,
                        vector_index_name=f"{self.namespace}-kb",
                        field_mapping=(
                            bedrock.CfnKnowledgeBase.OpenSearchServerlessFieldMappingProperty(
                                vector_field=selected.index.vector_field,
                                text_field=selected.index.text_field,
                                metadata_field=selected.index.metadata_field,
                            )
                        ),
                    )
                ),
            ),
            tags={"Owner": INFRASTRUCTURE_OWNER},
        )
        knowledge_base.add_resource_dependency(vector_index)
        return knowledge_base

    def _create_data_source(
        self,
        config: FoundationConfig,
        bucket: s3.CfnBucket,
        knowledge_base: bedrock.CfnKnowledgeBase,
    ) -> bedrock.CfnDataSource:
        """Create the selected data source, and keep any retiring one alive until it is drained.

        Bedrock cannot change chunking on an existing data source, so a chunking migration is
        additive: the new source is created beside the old, ingests the active generation, and
        only then is the old one retired. The logical ID is derived from the candidate so a new
        candidate is a new resource rather than an in-place update CloudFormation would have to
        replace, which would delete the old source before the new one held any vectors.

        RETIRING_DATA_SOURCES lists old sources still present during the drain. Each is flipped
        to a DELETE data policy so that removing it from this list, on the next deploy, removes
        its vectors too; a RETAIN policy would leave the old chunking retrievable for the active
        generation forever, silently mixing two chunkings in every answer.
        """
        frozen = load_retrieval_config(ROOT / "retrieval-config.yaml", project_root=ROOT)
        selected = frozen.selected
        data_source = self._data_source(
            config,
            bucket,
            knowledge_base,
            logical_id=_data_source_logical_id(frozen.selected_candidate),
            name=f"{self.namespace}-corpus-{frozen.selected_candidate}",
            chunking=selected.chunking,
            deletion_policy="RETAIN",
        )
        for retiring in RETIRING_DATA_SOURCES:
            self._data_source(
                config,
                bucket,
                knowledge_base,
                logical_id=retiring.logical_id,
                name=retiring.name,
                chunking=retiring.chunking,
                deletion_policy="DELETE",
            )
        return data_source

    def _data_source(
        self,
        config: FoundationConfig,
        bucket: s3.CfnBucket,
        knowledge_base: bedrock.CfnKnowledgeBase,
        *,
        logical_id: str,
        name: str,
        chunking: ChunkingConfiguration,
        deletion_policy: str,
    ) -> bedrock.CfnDataSource:
        if chunking.strategy == "HIERARCHICAL":
            chunking_property = bedrock.CfnDataSource.ChunkingConfigurationProperty(
                chunking_strategy="HIERARCHICAL",
                hierarchical_chunking_configuration=(
                    bedrock.CfnDataSource.HierarchicalChunkingConfigurationProperty(
                        level_configurations=[
                            bedrock.CfnDataSource.HierarchicalChunkingLevelConfigurationProperty(
                                max_tokens=cast(int, chunking.parent_max_tokens)
                            ),
                            bedrock.CfnDataSource.HierarchicalChunkingLevelConfigurationProperty(
                                max_tokens=chunking.max_tokens
                            ),
                        ],
                        overlap_tokens=cast(int, chunking.overlap_tokens),
                    )
                ),
            )
        else:
            chunking_property = bedrock.CfnDataSource.ChunkingConfigurationProperty(
                chunking_strategy="FIXED_SIZE",
                fixed_size_chunking_configuration=(
                    bedrock.CfnDataSource.FixedSizeChunkingConfigurationProperty(
                        max_tokens=chunking.max_tokens,
                        overlap_percentage=cast(int, chunking.overlap_percentage),
                    )
                ),
            )
        data_source = bedrock.CfnDataSource(
            self,
            logical_id,
            name=name,
            description="Fixed S3 data source; control/ is intentionally not ingestible",
            knowledge_base_id=knowledge_base.attr_knowledge_base_id,
            data_deletion_policy=deletion_policy,
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                type="S3",
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=bucket.attr_arn,
                    bucket_owner_account_id=config.account,
                    inclusion_prefixes=[CORPUS_DATA_SOURCE_PREFIX],
                ),
            ),
            vector_ingestion_configuration=(
                bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                    chunking_configuration=chunking_property
                )
            ),
        )
        data_source.add_resource_dependency(knowledge_base)
        return data_source

    def _create_observability(
        self,
        config: FoundationConfig,
        bucket: s3.CfnBucket,
        state_table: dynamodb.CfnTable,
        collection: aoss.CfnCollection,
    ) -> None:
        log_group = logs.CfnLogGroup(
            self,
            "KnowledgePlaneLogGroup",
            log_group_name=f"/{self.namespace}/knowledge-plane",
            retention_in_days=30,
            log_group_class="STANDARD",
            deletion_protection_enabled=True,
            tags=self._owner_tags,
        )
        log_group.apply_removal_policy(RemovalPolicy.RETAIN)

        account_root = self.format_arn(
            service="iam",
            region="",
            account=config.account,
            resource="root",
        )
        sns_via_service = f"sns.{config.region}.amazonaws.com"
        alert_key = kms.CfnKey(
            self,
            "AlertKey",
            description=(
                "Symmetric CMK for safeguard topic publishing; owner subscription is intentionally "
                "deferred to deployed I-10B"
            ),
            enable_key_rotation=True,
            key_spec="SYMMETRIC_DEFAULT",
            key_usage="ENCRYPT_DECRYPT",
            key_policy={
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "AccountAdministration",
                        "Effect": "Allow",
                        "Principal": {"AWS": account_root},
                        "Action": "kms:*",
                        "Resource": "*",
                    },
                    {
                        "Sid": "CloudWatchEncryptedTopicPublishing",
                        "Effect": "Allow",
                        "Principal": {"Service": "cloudwatch.amazonaws.com"},
                        "Action": ["kms:GenerateDataKey*", "kms:Decrypt"],
                        "Resource": "*",
                        "Condition": {
                            "StringEquals": {
                                "aws:SourceAccount": config.account,
                                "kms:ViaService": sns_via_service,
                            }
                        },
                    },
                ],
            },
            tags=self._owner_tags,
        )
        alert_key.cfn_options.deletion_policy = CfnDeletionPolicy.RETAIN
        alert_key.cfn_options.update_replace_policy = CfnDeletionPolicy.RETAIN
        kms.CfnAlias(
            self,
            "AlertKeyAlias",
            alias_name=f"alias/{self.namespace}-alerts",
            target_key_id=alert_key.ref,
        )

        topic = sns.CfnTopic(
            self,
            "AlertTopic",
            topic_name=f"{self.namespace}-alerts",
            display_name=(
                "Valkeyrie alerts; owner subscription intentionally deferred to deployed I-10B"
            ),
            kms_master_key_id=alert_key.attr_arn,
            tags=self._owner_tags,
        )
        alarm_arn = topic.attr_topic_arn
        sns.CfnTopicPolicy(
            self,
            "AlertTopicPolicy",
            topics=[topic.ref],
            policy_document={
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "CloudWatchPublishOnly",
                        "Effect": "Allow",
                        "Principal": {"Service": "cloudwatch.amazonaws.com"},
                        "Action": "sns:Publish",
                        "Resource": alarm_arn,
                        "Condition": {"StringEquals": {"aws:SourceAccount": config.account}},
                    },
                ],
            },
        )

        custom_metric_units = {
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
        for metric_name, unit in custom_metric_units.items():
            metric_filter = logs.CfnMetricFilter(
                self,
                f"{metric_name}MetricFilter",
                log_group_name=log_group.ref,
                filter_pattern=(f'{{ $.metric_name = "{metric_name}" && $.value = * }}'),
                metric_transformations=[
                    logs.CfnMetricFilter.MetricTransformationProperty(
                        metric_name=metric_name,
                        metric_namespace=CUSTOM_METRIC_NAMESPACE,
                        metric_value="$.value",
                        unit=unit,
                    )
                ],
            )
            metric_filter.add_resource_dependency(log_group)

        def alarm(
            construct_id: str,
            *,
            metric_name: str,
            threshold: int,
            period: int,
            statistic: str = "Maximum",
            comparison: str = "GreaterThanThreshold",
            namespace: str = CUSTOM_METRIC_NAMESPACE,
            dimensions: list[cloudwatch.CfnAlarm.DimensionProperty] | None = None,
            extended_statistic: str | None = None,
            unit: str | None = None,
        ) -> cloudwatch.CfnAlarm:
            custom_metric = namespace == CUSTOM_METRIC_NAMESPACE
            return cloudwatch.CfnAlarm(
                self,
                construct_id,
                alarm_name=f"{self.namespace}-{construct_id}",
                alarm_description=(
                    f"Owner={INFRASTRUCTURE_OWNER}; safeguard={metric_name}; alarm route only; "
                    "owner subscription intentionally deferred to deployed I-10B; never enables "
                    "runtime or promotion"
                ),
                actions_enabled=True,
                alarm_actions=[alarm_arn],
                insufficient_data_actions=[alarm_arn] if custom_metric else None,
                comparison_operator=comparison,
                dimensions=dimensions,
                evaluation_periods=1,
                datapoints_to_alarm=1,
                metric_name=metric_name,
                namespace=namespace,
                period=period,
                statistic=None if extended_statistic else statistic,
                extended_statistic=extended_statistic,
                threshold=threshold,
                treat_missing_data="missing",
                unit=unit,
                tags=self._owner_tags,
            )

        alarms = [
            alarm(
                "CorpusAge",
                metric_name="CorpusAgeHours",
                threshold=SAFEGUARD_THRESHOLDS["corpus_age_hours"],
                period=3600,
                unit="None",
            ),
            alarm(
                "S3Storage",
                metric_name="BucketSizeBytes",
                threshold=SAFEGUARD_THRESHOLDS["s3_storage_bytes"],
                period=86400,
                statistic="Average",
                namespace="AWS/S3",
                dimensions=[
                    cloudwatch.CfnAlarm.DimensionProperty(name="BucketName", value=bucket.ref),
                    cloudwatch.CfnAlarm.DimensionProperty(
                        name="StorageType", value="StandardStorage"
                    ),
                ],
                unit="Bytes",
            ),
            alarm(
                "AossStorage",
                metric_name="AossStorageBytes",
                threshold=SAFEGUARD_THRESHOLDS["aoss_storage_bytes"],
                period=3600,
                dimensions=[
                    cloudwatch.CfnAlarm.DimensionProperty(
                        name="CollectionName", value=f"{self.namespace}-vectors"
                    )
                ],
                unit="Bytes",
            ),
            alarm(
                "IngestionFailures",
                metric_name="IngestionFailures",
                threshold=SAFEGUARD_THRESHOLDS["ingestion_failures"],
                period=300,
                statistic="Sum",
                comparison="GreaterThanOrEqualToThreshold",
                unit="Count",
            ),
            alarm(
                "InvalidCitations",
                metric_name="InvalidCitations",
                threshold=SAFEGUARD_THRESHOLDS["invalid_citations"],
                period=300,
                statistic="Sum",
                comparison="GreaterThanOrEqualToThreshold",
                unit="Count",
            ),
            alarm(
                "DependencyThrottles",
                metric_name="DependencyThrottles",
                threshold=SAFEGUARD_THRESHOLDS["dependency_throttles"],
                period=300,
                statistic="Sum",
                comparison="GreaterThanOrEqualToThreshold",
                unit="Count",
            ),
            alarm(
                "RequestP95Latency",
                metric_name="RequestLatencySeconds",
                threshold=SAFEGUARD_THRESHOLDS["p95_latency_seconds"],
                period=300,
                extended_statistic="p95",
                unit="Seconds",
            ),
            alarm(
                "DailyAcceptedRequests",
                metric_name="AcceptedRequests",
                threshold=SAFEGUARD_THRESHOLDS["accepted_requests_per_day"],
                period=86400,
                statistic="Sum",
                unit="Count",
            ),
            alarm(
                "ExternalApiCallsPerRequest",
                metric_name="ExternalApiCallsPerRequest",
                threshold=SAFEGUARD_THRESHOLDS["external_api_calls_per_request"],
                period=300,
                unit="Count",
            ),
            alarm(
                "ModelCallsPerRequest",
                metric_name="ModelCallsPerRequest",
                threshold=SAFEGUARD_THRESHOLDS["model_calls_per_request"],
                period=300,
                unit="Count",
            ),
            alarm(
                "InputTokensPerRequest",
                metric_name="InputTokensPerRequest",
                threshold=SAFEGUARD_THRESHOLDS["input_tokens_per_request"],
                period=300,
                unit="Count",
            ),
            alarm(
                "OutputTokensPerRequest",
                metric_name="OutputTokensPerRequest",
                threshold=SAFEGUARD_THRESHOLDS["output_tokens_per_request"],
                period=300,
                unit="Count",
            ),
            alarm(
                "ConcurrentRequests",
                metric_name="ConcurrentRequests",
                threshold=SAFEGUARD_THRESHOLDS["maximum_concurrency"],
                period=60,
                unit="Count",
            ),
        ]

        spend_anomaly = cloudwatch.CfnAlarm(
            self,
            "AnomalousSpend",
            alarm_name=f"{self.namespace}-AnomalousSpend",
            alarm_description=(
                f"Owner={INFRASTRUCTURE_OWNER}; two-standard-deviation project spend anomaly "
                "from the JSON log metric contract; alarm route only; owner subscription "
                "intentionally deferred to deployed I-10B; never enables runtime or promotion"
            ),
            actions_enabled=True,
            alarm_actions=[alarm_arn],
            insufficient_data_actions=[alarm_arn],
            comparison_operator="GreaterThanUpperThreshold",
            evaluation_periods=1,
            datapoints_to_alarm=1,
            threshold_metric_id="ad1",
            treat_missing_data="missing",
            metrics=[
                cloudwatch.CfnAlarm.MetricDataQueryProperty(
                    id="m1",
                    return_data=True,
                    metric_stat=cloudwatch.CfnAlarm.MetricStatProperty(
                        metric=cloudwatch.CfnAlarm.MetricProperty(
                            namespace=CUSTOM_METRIC_NAMESPACE,
                            metric_name="EstimatedMonthlySpendUsd",
                        ),
                        period=21600,
                        stat="Maximum",
                    ),
                ),
                cloudwatch.CfnAlarm.MetricDataQueryProperty(
                    id="ad1",
                    expression="ANOMALY_DETECTION_BAND(m1, 2)",
                    label="Expected Valkeyrie spend band",
                    return_data=True,
                ),
            ],
            tags=self._owner_tags,
        )
        alarms.append(spend_anomaly)

        dashboard_metrics = list(custom_metric_units)
        cloudwatch.CfnDashboard(
            self,
            "SafeguardDashboard",
            dashboard_name=f"{self.namespace}-safeguards",
            dashboard_body=json.dumps(
                {
                    "start": "-PT24H",
                    "periodOverride": "inherit",
                    "widgets": [
                        {
                            "type": "metric",
                            "width": 24,
                            "height": 12,
                            "properties": {
                                "title": "Valkeyrie knowledge-plane safeguards",
                                "region": config.region,
                                "view": "timeSeries",
                                "stat": "Maximum",
                                "period": 300,
                                "metrics": [
                                    [CUSTOM_METRIC_NAMESPACE, metric]
                                    for metric in dashboard_metrics
                                ],
                            },
                        }
                    ],
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            tags=self._owner_tags,
        )

        for resource in [log_group, alert_key, topic, state_table, collection, *alarms]:
            Tags.of(resource).add("Owner", INFRASTRUCTURE_OWNER)

    def _create_disabled_controls(self) -> None:
        for control, default in CONTROL_DEFAULTS.items():
            ssm.CfnParameter(
                self,
                f"{control.title().replace('-', '')}Control",
                name=f"/{self.namespace}/controls/{control}",
                type="String",
                value=default,
                allowed_pattern="^(false|true)$",
                description=(
                    "Default disabled. Consumers must treat missing, unreadable, malformed, or any "
                    "value other than exact 'true' as disabled. No synthesized role may change it."
                ),
                tier="Standard",
                tags={"Owner": INFRASTRUCTURE_OWNER, "FailClosed": "true"},
            )


def build_app(outdir: Path, config: FoundationConfig = DEFAULT_CONFIG) -> App:
    """Build one explicit stack without lookups, assets, or a deploy action."""
    app = App(
        analytics_reporting=False,
        outdir=str(outdir),
        stack_traces=False,
        tree_metadata=False,
    )
    KnowledgePlaneStack(app, "KnowledgePlane", config=config)
    return app


def synthesize(outdir: Path = DEFAULT_OUTDIR, config: FoundationConfig = DEFAULT_CONFIG) -> None:
    """Write only the local cloud assembly."""
    build_app(outdir, config).synth()


if __name__ == "__main__":
    synthesize()
