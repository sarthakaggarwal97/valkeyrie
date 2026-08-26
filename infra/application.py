"""Independent synthesis-only bootstrap and qualified application stacks."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from aws_cdk import (
    App,
    ArnFormat,
    CfnDeletionPolicy,
    CfnTag,
    Environment,
    LegacyStackSynthesizer,
    Stack,
)
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3

from infra.app import DEFAULT_CONFIG, build_app
from infra.application_artifact import ApplicationArtifact, build_application_artifact

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPLICATION_OUTDIR = Path("cdk.application.out")
BOOTSTRAP_STACK_ID: Final = "ApplicationBootstrap"
BOOTSTRAP_STACK_NAME: Final = "valkeyrie-development-application-bootstrap"
APPLICATION_STACK_ID: Final = "ApplicationPlane"
APPLICATION_STACK_NAME: Final = "valkeyrie-development-application"
APPLICATION_FUNCTION_NAME: Final = "valkeyrie-development-application"
APPLICATION_LOG_GROUP_NAME: Final = "/valkeyrie-development/application"
APPLICATION_ARTIFACT_BUCKET: Final = "valkeyrie-dev-app-artifacts-968533178160-us-east-1"
APPLICATION_ARTIFACT_PREFIX: Final = "artifacts"
APPLICATION_SERVICE_ROLE_NAME: Final = "valkeyrie-development-application-cloudformation"
# Owner-directed Claude Opus 5 route, retained as plain exact configuration so the runtime
# grant does not depend on any comparison-artifact derivation. Immutable Lambda version 8
# runs this model; the Sid matches the deployed policy so re-synthesis is a no-op.
OWNER_DIRECTED_OPUS_SID: Final = "InvokeExactOwnerDirectedOpusComparisonRoute"
OPUS_INFERENCE_PROFILE_ARN: Final = (
    "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-opus-5"
)
OPUS_FOUNDATION_MODEL_ARNS: Final = (
    "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-5",
    "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-5",
    "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-opus-5",
)
APPLICATION_SERVICE_ROLE_ARN: Final = (
    "arn:aws:iam::968533178160:role/valkeyrie-development-application-cloudformation"
)
APPLICATION_DEPLOYER_ROLE_NAME: Final = "valkeyrie-development-application-deployer"
APPLICATION_RUNTIME_ROLE_NAME: Final = "valkeyrie-development-runtime"
KNOWLEDGE_STACK_NAME: Final = "valkeyrie-development-knowledge-plane"
KNOWLEDGE_BASE_ID: Final = "ONVASJDDNX"
D01_TEMPLATE_SHA256: Final = (
    "sha256:85021302ee2b6afe61fac0c77617e7db606042dc58696b2e0d9c1ebae43d477c"
)


@dataclass(frozen=True)
class ApplicationSynthesis:
    artifact: ApplicationArtifact
    bootstrap_template_path: Path
    template_path: Path
    artifact_path: Path
    evidence_path: Path
    bootstrap_template_sha256: str
    template_sha256: str
    evidence_sha256: str


class ApplicationBootstrapStack(Stack):
    """Human-created D02 bootstrap: artifact store and exact role policies."""

    def __init__(
        self,
        scope: App,
        construct_id: str,
        *,
        artifact: ApplicationArtifact,
    ) -> None:
        super().__init__(
            scope,
            construct_id,
            stack_name=BOOTSTRAP_STACK_NAME,
            env=Environment(account=DEFAULT_CONFIG.account, region=DEFAULT_CONFIG.region),
            description="Valkeyrie D02 application bootstrap (human-created, synthesis only)",
            tags=_tags("application-bootstrap"),
            synthesizer=LegacyStackSynthesizer(),
        )
        owner = [CfnTag(key="Owner", value="sarthakaggarwal97")]
        bucket = s3.CfnBucket(
            self,
            "ApplicationArtifactBucket",
            bucket_name=APPLICATION_ARTIFACT_BUCKET,
            bucket_encryption=s3.CfnBucket.BucketEncryptionProperty(
                server_side_encryption_configuration=[
                    s3.CfnBucket.ServerSideEncryptionRuleProperty(
                        server_side_encryption_by_default=s3.CfnBucket.ServerSideEncryptionByDefaultProperty(  # noqa: E501
                            sse_algorithm="AES256"
                        )
                    )
                ]
            ),
            lifecycle_configuration=s3.CfnBucket.LifecycleConfigurationProperty(
                rules=[
                    s3.CfnBucket.RuleProperty(
                        id="AbortIncompleteMultipartUploadsOnly",
                        status="Enabled",
                        abort_incomplete_multipart_upload=s3.CfnBucket.AbortIncompleteMultipartUploadProperty(  # noqa: E501
                            days_after_initiation=7
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
            tags=owner,
        )
        bucket.cfn_options.deletion_policy = CfnDeletionPolicy.RETAIN
        bucket.cfn_options.update_replace_policy = CfnDeletionPolicy.RETAIN
        deployer_arn = self.format_arn(
            service="iam", region="", resource="role", resource_name=APPLICATION_DEPLOYER_ROLE_NAME
        )
        s3.CfnBucketPolicy(
            self,
            "ApplicationArtifactBucketPolicy",
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
                        "Sid": "DenyArtifactBucketAndObjectDeletion",
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": ["s3:DeleteBucket", "s3:DeleteObject", "s3:DeleteObjectVersion"],
                        "Resource": [bucket.attr_arn, f"{bucket.attr_arn}/*"],
                    },
                    {
                        "Sid": "DenyDeployerArtifactOverwrite",
                        "Effect": "Deny",
                        "Principal": {"AWS": deployer_arn},
                        "Action": "s3:PutObject",
                        "Resource": f"{bucket.attr_arn}/{APPLICATION_ARTIFACT_PREFIX}/*",
                        "Condition": {"StringNotEquals": {"s3:if-none-match": "*"}},
                    },
                ],
            },
        )

        service_role = iam.CfnRole(
            self,
            "ApplicationCloudFormationServiceRole",
            role_name=APPLICATION_SERVICE_ROLE_NAME,
            description="Exact CloudFormation execution role for the IAM-free application stack",
            max_session_duration=3600,
            assume_role_policy_document={
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "cloudformation.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                        "Condition": {
                            "StringEquals": {"aws:SourceAccount": DEFAULT_CONFIG.account}
                        },
                    }
                ],
            },
            policies=[
                iam.CfnRole.PolicyProperty(
                    policy_name="ApplicationResourceLifecycle",
                    policy_document={
                        "Version": "2012-10-17",
                        "Statement": _service_role_statements(self, artifact),
                    },
                )
            ],
            tags=owner,
        )
        service_role.cfn_options.deletion_policy = CfnDeletionPolicy.RETAIN
        service_role.cfn_options.update_replace_policy = CfnDeletionPolicy.RETAIN

        iam.CfnPolicy(
            self,
            "ApplicationDeployerPolicy",
            policy_name="valkeyrie-development-application-deployer",
            roles=[APPLICATION_DEPLOYER_ROLE_NAME],
            policy_document={"Version": "2012-10-17", "Statement": _deployer_statements(self)},
        )
        iam.CfnPolicy(
            self,
            "ApplicationRuntimePolicy",
            policy_name="valkeyrie-development-application-runtime",
            roles=[APPLICATION_RUNTIME_ROLE_NAME],
            policy_document={
                "Version": "2012-10-17",
                "Statement": _runtime_statements(self, artifact),
            },
        )


class ApplicationStack(Stack):
    """IAM-free application resources deployable only through the exact service role."""

    def __init__(self, scope: App, construct_id: str, *, artifact: ApplicationArtifact) -> None:
        super().__init__(
            scope,
            construct_id,
            stack_name=APPLICATION_STACK_NAME,
            env=Environment(account=DEFAULT_CONFIG.account, region=DEFAULT_CONFIG.region),
            description="Valkeyrie private qualified application (synthesis only)",
            tags=_tags("application"),
            synthesizer=LegacyStackSynthesizer(),
        )
        owner = [CfnTag(key="Owner", value="sarthakaggarwal97")]
        log_group = logs.CfnLogGroup(
            self,
            "ApplicationLogGroup",
            log_group_name=APPLICATION_LOG_GROUP_NAME,
            retention_in_days=30,
            log_group_class="STANDARD",
            deletion_protection_enabled=True,
            tags=owner,
        )
        log_group.cfn_options.deletion_policy = CfnDeletionPolicy.RETAIN
        log_group.cfn_options.update_replace_policy = CfnDeletionPolicy.RETAIN
        role_arn = self.format_arn(
            service="iam", region="", resource="role", resource_name=APPLICATION_RUNTIME_ROLE_NAME
        )
        key = f"{APPLICATION_ARTIFACT_PREFIX}/{artifact.artifact_sha256[7:]}.zip"
        function = lambda_.CfnFunction(
            self,
            "ApplicationFunction",
            function_name=APPLICATION_FUNCTION_NAME,
            description=f"Qualified Valkeyrie application {artifact.application_revision}",
            architectures=["x86_64"],
            code=lambda_.CfnFunction.CodeProperty(
                s3_bucket=APPLICATION_ARTIFACT_BUCKET, s3_key=key
            ),
            environment=lambda_.CfnFunction.EnvironmentProperty(
                variables={
                    "APPLICATION_REVISION": artifact.application_revision,
                    "STATE_TABLE_NAME": "valkeyrie-development-state",
                    "SELECTED_INFERENCE_PROFILE_ARN": cast(
                        str, artifact.manifest["selected_inference_profile_arn"]
                    ),
                }
            ),
            handler="infra.application_handler.handler",
            logging_config=lambda_.CfnFunction.LoggingConfigProperty(
                application_log_level="INFO",
                log_format="JSON",
                log_group=APPLICATION_LOG_GROUP_NAME,
                system_log_level="WARN",
            ),
            memory_size=512,
            # Bounds both parallel spend and blast radius. A public URL with 1 would
            # serialise every caller behind an 8-23 second answer.
            reserved_concurrent_executions=5,
            role=role_arn,
            runtime="python3.11",
            # Fable exceeded 30s on the densest questions (multi-part replication and
            # governance-process answers), which surfaced as a truncated request.
            timeout=120,
            tracing_config=lambda_.CfnFunction.TracingConfigProperty(mode="PassThrough"),
            tags=owner,
        )
        function.add_resource_dependency(log_group)
        version = lambda_.CfnVersion(
            self,
            "ApplicationVersion",
            code_sha256=base64.b64encode(bytes.fromhex(artifact.artifact_sha256[7:])).decode(),
            description=(
                f"application={artifact.application_revision};selection={artifact.selection_id}"
            ),
            function_name=function.ref,
        )
        version.add_resource_dependency(function)
        version.cfn_options.deletion_policy = CfnDeletionPolicy.RETAIN
        version.cfn_options.update_replace_policy = CfnDeletionPolicy.RETAIN

        # Account-scoped endpoint. A world-accessible function url is prohibited in
        # this account and actively policed: an AuthType NONE url with a correct
        # public permission returned 403 for anonymous callers, and Palisade then
        # raised epoxy-engage_mitigations and stripped the public statement. Fronting
        # the same function with a public CloudFront distribution would restore world
        # reach and circumvent that mitigation, so it is deliberately not done.
        #
        # AWS_IAM needs no resource policy: a caller in this account authorizes with
        # its own IAM identity and signs the request with SigV4, which is verified
        # working. That keeps the grant in the caller's identity policy rather than
        # widening the function itself.
        url = lambda_.CfnUrl(
            self,
            "ApplicationUrl",
            auth_type="AWS_IAM",
            target_function_arn=function.ref,
        )
        url.add_resource_dependency(function)


def build_application_app(
    outdir: Path,
    artifact: ApplicationArtifact,
) -> App:
    app = App(
        analytics_reporting=False, outdir=str(outdir), stack_traces=False, tree_metadata=False
    )
    ApplicationBootstrapStack(
        app,
        BOOTSTRAP_STACK_ID,
        artifact=artifact,
    )
    ApplicationStack(app, APPLICATION_STACK_ID, artifact=artifact)
    return app


def synthesize_application(
    root: Path = ROOT, outdir: Path = DEFAULT_APPLICATION_OUTDIR
) -> ApplicationSynthesis:
    return _synthesize_application(root, outdir, build_application_artifact(root))


def _synthesize_application(
    root: Path,
    outdir: Path,
    artifact: ApplicationArtifact,
) -> ApplicationSynthesis:
    build_application_app(outdir, artifact).synth()
    bootstrap_path = outdir / f"{BOOTSTRAP_STACK_ID}.template.json"
    template_path = outdir / f"{APPLICATION_STACK_ID}.template.json"
    artifact_dir = outdir / "application-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{artifact.artifact_sha256[7:]}.zip"
    artifact_path.write_bytes(artifact.content)
    with tempfile.TemporaryDirectory(prefix="valkeyrie-a07a-") as temporary:
        knowledge_out = Path(temporary) / "knowledge"
        build_app(knowledge_out).synth()
        knowledge_bytes = (knowledge_out / "KnowledgePlane.template.json").read_bytes()
    evidence = derive_application_evidence(
        artifact,
        bootstrap_path.read_bytes(),
        template_path.read_bytes(),
        knowledge_bytes,
    )
    evidence_path = outdir / "a07a-evidence.json"
    evidence_path.write_bytes(_pretty(evidence))
    return ApplicationSynthesis(
        artifact,
        bootstrap_path,
        template_path,
        artifact_path,
        evidence_path,
        _sha256(bootstrap_path.read_bytes()),
        _sha256(template_path.read_bytes()),
        _sha256(evidence_path.read_bytes()),
    )


def derive_application_evidence(
    artifact: ApplicationArtifact,
    bootstrap_bytes: bytes,
    application_bytes: bytes,
    knowledge_bytes: bytes,
) -> dict[str, object]:
    bootstrap = _json(bootstrap_bytes)
    application = _json(application_bytes)
    knowledge = _json(knowledge_bytes)
    knowledge_resources = _resources(knowledge)
    if len(knowledge_resources) != 52 or _sha256(knowledge_bytes) != D01_TEMPLATE_SHA256:
        raise ValueError("KnowledgePlane differs from deployed D01 identity")
    baseline = _baseline_role_policies(knowledge)
    after = _bootstrap_attached_policies(bootstrap)
    app_resources = _resources(application)
    if any(resource["Type"].startswith("AWS::IAM::") for resource in app_resources.values()):
        raise ValueError("application stack contains IAM")
    function = cast(dict[str, object], app_resources["ApplicationFunction"]["Properties"])
    version = app_resources["ApplicationVersion"]
    if (
        cast(dict[str, object], version["Properties"])["CodeSha256"]
        != base64.b64encode(bytes.fromhex(artifact.artifact_sha256[7:])).decode()
    ):
        raise ValueError("Lambda version is not bound to exact artifact bytes")
    if version.get("DeletionPolicy") != "Retain" or version.get("UpdateReplacePolicy") != "Retain":
        raise ValueError("Lambda version is not retained")
    artifact_key = cast(dict[str, object], function["Code"])["S3Key"]
    health_payload = {"action": "health", "application_revision": artifact.application_revision}
    answer_payload = {
        "action": "answer",
        "request_id": "req_postdeploy-a07a",
        "question": "Where should documentation for a newly added Valkey command be written?",
        "version_requirement": "none",
        "requested_version": None,
        "knowledge_base_id": "${KNOWLEDGE_BASE_ID}",
        "owner": "postdeploy-a07a",
        "now": "${POSTDEPLOY_STARTED_AT}",
        "completed_at": "${POSTDEPLOY_COMPLETED_AT}",
        "lease_duration_seconds": 300,
    }
    return {
        "api_version": "valkeyrie.io/a07a-evidence/2",
        "kind": "ApplicationReleaseEvidence",
        "target": {"account": DEFAULT_CONFIG.account, "region": DEFAULT_CONFIG.region},
        "no_deploy_status": {"status": "synthesized_only", "aws_calls_performed": 0},
        "identity": {
            "knowledge_template_sha256": D01_TEMPLATE_SHA256,
            "bootstrap_template_sha256": _sha256(bootstrap_bytes),
            "application_template_sha256": _sha256(application_bytes),
            "application_revision": artifact.application_revision,
            "artifact_sha256": artifact.artifact_sha256,
            "prompt_revision": artifact.prompt_revision,
            "selected_model_revision": artifact.selected_model_revision,
            "selected_profile_revision": artifact.selected_profile_revision,
            "selected_inference_config_revision": artifact.selected_inference_config_revision,
            "selected_report_id": artifact.selected_report_id,
            "selection_id": artifact.selection_id,
            "execution_authorization": "qualified_model_selection",
            "qualification_status": "qualified",
            "response_normalization_policy_revision": artifact.response_normalization_policy_revision,  # noqa: E501
            "raw_evidence_sha256": artifact.manifest["raw_evidence_sha256"],
        },
        "resource_inventory": {
            "knowledge": dict(
                sorted(Counter(item["Type"] for item in knowledge_resources.values()).items())
            ),
            "bootstrap": dict(
                sorted(Counter(item["Type"] for item in _resources(bootstrap).values()).items())
            ),
            "application": dict(
                sorted(Counter(item["Type"] for item in app_resources.values()).items())
            ),
            "public_ingress": [],
            "parameters": [],
            "lookups": [],
        },
        "iam_action_resource_diff": {
            "deployed_d01_before": baseline,
            "application_bootstrap_after": after,
            "application_stack_iam_resources": [],
            "service_role_arn": APPLICATION_SERVICE_ROLE_ARN,
            "prohibited_actions": _prohibited_actions(after),
        },
        "artifact_staging": {
            "a07a_local_path": f"application-artifacts/{artifact.artifact_sha256[7:]}.zip",
            "bucket": APPLICATION_ARTIFACT_BUCKET,
            "key": artifact_key,
            "upload_phase": "A07B only",
            "commands": [
                f"aws s3api head-object --bucket {APPLICATION_ARTIFACT_BUCKET} --key {artifact_key}",  # noqa: E501
                f"aws s3api put-object --bucket {APPLICATION_ARTIFACT_BUCKET} --key {artifact_key} --body <LOCAL_ZIP> --if-none-match '*' --checksum-algorithm SHA256 --checksum-sha256 {base64.b64encode(bytes.fromhex(artifact.artifact_sha256[7:])).decode()}",  # noqa: E501
                f"aws s3api head-object --bucket {APPLICATION_ARTIFACT_BUCKET} --key {artifact_key} --checksum-mode ENABLED",  # noqa: E501
            ],
            "verification": "accept pre-existing object only when ContentLength and ChecksumSHA256 match exact local bytes; never overwrite",  # noqa: E501
            "retention": "content-hash keys, bucket versions, and Lambda versions are retained",
        },
        "health": {
            "offline_expected": {
                "status": "healthy",
                "synthetic_path": {
                    "a03_route": "static_semantic",
                    "a04_normalization": "unchanged",
                    "a04_output": "answer",
                    "a05_conditional_claim": True,
                    "a05_completion": True,
                    "a06_outcome": "answer",
                },
            },
            "immutable_version_commands": [
                f"VERSION=$(aws cloudformation describe-stack-resource --stack-name {APPLICATION_STACK_NAME} --logical-resource-id ApplicationVersion --query 'StackResourceDetail.PhysicalResourceId' --output text)",  # noqa: E501
                f"aws lambda invoke --function-name {APPLICATION_FUNCTION_NAME} --qualifier \"$VERSION\" --cli-binary-format raw-in-base64-out --payload '{json.dumps(health_payload, separators=(',', ':'))}' /tmp/a07a-health.json",  # noqa: E501
            ],
            "cloud_answer_probe": {
                "knowledge_base_resolver": f"KNOWLEDGE_BASE_ID=$(aws cloudformation describe-stack-resource --stack-name {KNOWLEDGE_STACK_NAME} --logical-resource-id KnowledgeBase --query 'StackResourceDetail.PhysicalResourceId' --output text)",  # noqa: E501
                "payload": answer_payload,
                "predicates": [
                    "outcome == answer",
                    "generation_id is sha256",
                    "claims is non-empty",
                    "citations is non-empty and every URL is immutable github.com/blob/<40-hex>/",
                    "request_id == req_postdeploy-a07a writes only request#req_postdeploy-a07a audit state",  # noqa: E501
                ],
                "failure_stop": "do not continue deployment or remove any retained version",
            },
        },
        "rollback": {
            "operator_supplied_prior_identity": {
                "PRIOR_TEMPLATE_FILE": "<OPERATOR-SUPPLIED retained exact prior template>",
                "PRIOR_TEMPLATE_SHA256": "<OPERATOR-SUPPLIED prior template sha256>",
                "PRIOR_ARTIFACT_BUCKET": APPLICATION_ARTIFACT_BUCKET,
                "PRIOR_ARTIFACT_KEY": "<OPERATOR-SUPPLIED retained prior content-hash key>",
                "PRIOR_ARTIFACT_VERSION_ID": "<OPERATOR-SUPPLIED retained S3 version ID>",
                "PRIOR_LAMBDA_VERSION": "<OPERATOR-SUPPLIED retained physical Lambda version>",
            },
            "commands": [
                'sha256sum -c <(printf \'%s  %s\\n\' "${PRIOR_TEMPLATE_SHA256#sha256:}" "$PRIOR_TEMPLATE_FILE")',  # noqa: E501
                f"aws cloudformation create-change-set --stack-name {APPLICATION_STACK_NAME} --change-set-name a07a-rollback-<UNIQUE> --change-set-type UPDATE --template-body file://$PRIOR_TEMPLATE_FILE --role-arn {APPLICATION_SERVICE_ROLE_ARN} --capabilities CAPABILITY_NAMED_IAM",  # noqa: E501
                f"aws cloudformation execute-change-set --stack-name {APPLICATION_STACK_NAME} --change-set-name a07a-rollback-<UNIQUE>",  # noqa: E501
                f"aws lambda invoke --function-name {APPLICATION_FUNCTION_NAME} --qualifier \"$PRIOR_LAMBDA_VERSION\" --cli-binary-format raw-in-base64-out --payload '<PRIOR-EXACT-HEALTH-PAYLOAD>' /tmp/a07a-prior-health.json",  # noqa: E501
            ],
            "corpus_action": "none",
        },
        "cost": _cost_evidence(),
        "state_contract": {
            "reads": [
                "pk=active_generation or pk=version#<scope>",
                "pk=generation#<64-hex>",
                "pk=request#<request_id>",
            ],
            "first_claim": "PutItem pk=request#<request_id> with condition attribute_not_exists(pk)",  # noqa: E501
            "completion": "UpdateItem same key with revision/fence equality and attribute_not_exists(outcome)",  # noqa: E501
            "retry": "strongly consistent GetItem resolves exact stored model/prompt/generation/evidence plan",  # noqa: E501
            "forbidden": [
                "Scan",
                "DeleteItem",
                "active-generation writes",
                "generation lifecycle writes",
            ],
        },
    }


def _service_role_statements(
    stack: Stack,
    artifact: ApplicationArtifact,
) -> list[dict[str, object]]:
    function = stack.format_arn(
        service="lambda",
        resource="function",
        resource_name=APPLICATION_FUNCTION_NAME,
        arn_format=ArnFormat.COLON_RESOURCE_NAME,
    )
    log_group = stack.format_arn(
        service="logs",
        resource="log-group",
        resource_name=f"{APPLICATION_LOG_GROUP_NAME}:*",
        arn_format=ArnFormat.COLON_RESOURCE_NAME,
    )
    runtime = stack.format_arn(
        service="iam", region="", resource="role", resource_name=APPLICATION_RUNTIME_ROLE_NAME
    )

    def artifact_arn(digest: str) -> str:
        if (
            not digest.startswith("sha256:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
        ):
            raise ValueError("application artifact digest is malformed")
        return (
            f"arn:{stack.partition}:s3:::{APPLICATION_ARTIFACT_BUCKET}/"
            f"{APPLICATION_ARTIFACT_PREFIX}/{digest[7:]}.zip"
        )

    current_artifact_arn = artifact_arn(artifact.artifact_sha256)
    artifact_resources: str | list[str] = current_artifact_arn
    return [
        {
            "Sid": "ReadExactApplicationArtifact",
            "Effect": "Allow",
            "Action": "s3:GetObject",
            "Resource": artifact_resources,
        },
        {
            "Sid": "ManageExactApplicationFunction",
            "Effect": "Allow",
            "Action": [
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
            ],
            "Resource": [function, f"{function}:*"],
        },
        {
            "Sid": "ManageExactApplicationLogGroup",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:DeleteLogGroup",
                "logs:DeleteRetentionPolicy",
                "logs:PutRetentionPolicy",
                "logs:TagResource",
                "logs:UntagResource",
            ],
            "Resource": log_group,
        },
        {
            "Sid": "PassExactRuntimeRoleToLambda",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": runtime,
            "Condition": {"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}},
        },
    ]


def _deployer_statements(stack: Stack) -> list[dict[str, object]]:
    stack_arn = stack.format_arn(
        service="cloudformation",
        resource="stack",
        resource_name=f"{APPLICATION_STACK_NAME}/*",
        arn_format=ArnFormat.SLASH_RESOURCE_NAME,
    )
    bucket_arn = f"arn:{stack.partition}:s3:::{APPLICATION_ARTIFACT_BUCKET}"
    function = stack.format_arn(
        service="lambda",
        resource="function",
        resource_name=f"{APPLICATION_FUNCTION_NAME}:*",
        arn_format=ArnFormat.COLON_RESOURCE_NAME,
    )
    return [
        {
            "Sid": "DeployExactApplicationStack",
            "Effect": "Allow",
            "Action": [
                "cloudformation:ContinueUpdateRollback",
                "cloudformation:CreateChangeSet",
                "cloudformation:CreateStack",
                "cloudformation:DeleteChangeSet",
                "cloudformation:DescribeChangeSet",
                "cloudformation:DescribeStackEvents",
                "cloudformation:DescribeStackResource",
                "cloudformation:DescribeStacks",
                "cloudformation:ExecuteChangeSet",
                "cloudformation:GetTemplate",
                "cloudformation:RollbackStack",
                "cloudformation:UpdateStack",
            ],
            "Resource": stack_arn,
        },
        {
            "Sid": "ReadWriteExactArtifactPrefix",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"],
            "Resource": f"{bucket_arn}/{APPLICATION_ARTIFACT_PREFIX}/*",
        },
        {
            "Sid": "ListExactArtifactPrefix",
            "Effect": "Allow",
            "Action": "s3:ListBucket",
            "Resource": bucket_arn,
            "Condition": {"StringLike": {"s3:prefix": [f"{APPLICATION_ARTIFACT_PREFIX}/*"]}},
        },
        {
            "Sid": "InvokeExactImmutableHealthVersion",
            "Effect": "Allow",
            "Action": "lambda:InvokeFunction",
            "Resource": function,
        },
        {
            "Sid": "PassExactApplicationServiceRole",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": APPLICATION_SERVICE_ROLE_ARN,
            "Condition": {"StringEquals": {"iam:PassedToService": "cloudformation.amazonaws.com"}},
        },
    ]


def _runtime_statements(stack: Stack, artifact: ApplicationArtifact) -> list[dict[str, object]]:
    profile = cast(str, artifact.manifest["selected_inference_profile_arn"])
    models = cast(list[str], artifact.manifest["selected_foundation_model_arns"])
    model_statements: list[dict[str, object]] = [
        {
            "Sid": "InvokeExactQualifiedFableRoute",
            "Effect": "Allow",
            "Action": "bedrock:InvokeModel",
            "Resource": [profile, *models],
        },
        {
            "Sid": OWNER_DIRECTED_OPUS_SID,
            "Effect": "Allow",
            "Action": "bedrock:InvokeModel",
            "Resource": [OPUS_INFERENCE_PROFILE_ARN, *OPUS_FOUNDATION_MODEL_ARNS],
        },
    ]
    table = stack.format_arn(
        service="dynamodb",
        resource="table",
        resource_name="valkeyrie-development-state",
        arn_format=ArnFormat.SLASH_RESOURCE_NAME,
    )
    kb = stack.format_arn(
        service="bedrock",
        resource="knowledge-base",
        resource_name=KNOWLEDGE_BASE_ID,
        arn_format=ArnFormat.SLASH_RESOURCE_NAME,
    )
    controls = [
        stack.format_arn(
            service="ssm",
            resource="parameter",
            resource_name=f"valkeyrie-development/controls/{name}",
            arn_format=ArnFormat.SLASH_RESOURCE_NAME,
        )
        for name in ("model-processing-enabled", "runtime-enabled")
    ]
    log = stack.format_arn(
        service="logs",
        resource="log-group",
        resource_name=f"{APPLICATION_LOG_GROUP_NAME}:*",
        arn_format=ArnFormat.COLON_RESOURCE_NAME,
    )
    return [
        *model_statements,
        {
            "Sid": "RetrieveExactKnowledgeBase",
            "Effect": "Allow",
            "Action": "bedrock:Retrieve",
            "Resource": kb,
        },
        {
            "Sid": "ReadAndConditionallyAuditRequests",
            "Effect": "Allow",
            "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"],
            "Resource": table,
        },
        {
            "Sid": "ReadFailClosedRuntimeControls",
            "Effect": "Allow",
            "Action": ["ssm:GetParameter", "ssm:GetParameters"],
            "Resource": controls,
        },
        {
            "Sid": "WriteExactApplicationTelemetry",
            "Effect": "Allow",
            "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
            "Resource": log,
        },
    ]


def _baseline_role_policies(template: dict[str, object]) -> dict[str, object]:
    resources = _resources(template)
    result: dict[str, object] = {}
    for name in (APPLICATION_DEPLOYER_ROLE_NAME, APPLICATION_RUNTIME_ROLE_NAME):
        role_ids = [
            logical
            for logical, resource in resources.items()
            if resource["Type"] == "AWS::IAM::Role"
            and cast(dict[str, object], resource["Properties"]).get("RoleName") == name
        ]
        if len(role_ids) != 1:
            raise ValueError("D01 baseline role identity is missing")
        attached = [
            logical
            for logical, resource in resources.items()
            if resource["Type"] == "AWS::IAM::Policy"
            and {"Ref": role_ids[0]}
            in cast(list[object], cast(dict[str, object], resource["Properties"]).get("Roles", []))
        ]
        result[name] = {
            "role_logical_id": role_ids[0],
            "attached_policy_resources": sorted(attached),
            "statements": [] if not attached else "unexpected",
        }
        if attached:
            raise ValueError("D01 application role is not permission-empty")
    return result


def _bootstrap_attached_policies(template: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    resources = _resources(template)
    for logical, resource in resources.items():
        if resource["Type"] != "AWS::IAM::Policy":
            continue
        properties = cast(dict[str, object], resource["Properties"])
        roles = properties.get("Roles")
        if roles in ([APPLICATION_DEPLOYER_ROLE_NAME], [APPLICATION_RUNTIME_ROLE_NAME]):
            result[cast(str, roles[0])] = {
                "policy_logical_id": logical,
                "statements": cast(dict[str, object], properties["PolicyDocument"])["Statement"],
            }
    service_roles = [
        (logical, cast(dict[str, object], resource["Properties"]))
        for logical, resource in resources.items()
        if resource["Type"] == "AWS::IAM::Role"
        and cast(dict[str, object], resource["Properties"]).get("RoleName")
        == APPLICATION_SERVICE_ROLE_NAME
    ]
    if len(service_roles) != 1:
        raise ValueError("bootstrap service role identity is missing")
    service_logical, service_properties = service_roles[0]
    inline = cast(list[dict[str, object]], service_properties.get("Policies", []))
    if len(inline) != 1 or inline[0].get("PolicyName") != "ApplicationResourceLifecycle":
        raise ValueError("bootstrap service role policy is not exact")
    result[APPLICATION_SERVICE_ROLE_NAME] = {
        "role_logical_id": service_logical,
        "inline_policy_name": inline[0]["PolicyName"],
        "statements": cast(dict[str, object], inline[0]["PolicyDocument"])["Statement"],
    }
    if set(result) != {
        APPLICATION_DEPLOYER_ROLE_NAME,
        APPLICATION_RUNTIME_ROLE_NAME,
        APPLICATION_SERVICE_ROLE_NAME,
    }:
        raise ValueError("bootstrap does not attach exact application policies")
    return result


def _prohibited_actions(after: dict[str, object]) -> list[str]:
    actions: set[str] = set()
    for value in after.values():
        for statement in cast(
            list[dict[str, object]], cast(dict[str, object], value)["statements"]
        ):
            action = statement["Action"]
            actions.update([action] if isinstance(action, str) else cast(list[str], action))
    prohibited = {
        "s3:DeleteObject",
        "s3:DeleteObjectVersion",
        "bedrock:StartIngestionJob",
        "ssm:PutParameter",
        "dynamodb:DeleteItem",
        "aoss:APIAccessAll",
        "iam:CreateRole",
        "iam:PutRolePolicy",
    }
    return sorted(actions & prohibited)


def _cost_evidence() -> dict[str, object]:
    input_tokens, output_tokens = 522810, 36639
    return {
        "currency": "USD",
        "assumptions": {
            "lambda_request_usd_per_million": 0.20,
            "lambda_gb_second_usd": 0.0000166667,
            "s3_standard_gb_month_usd": 0.023,
            "s3_put_per_1000_usd": 0.005,
            "s3_get_per_1000_usd": 0.0004,
            "logs_ingest_gb_usd": 0.50,
            "bedrock_retrieve_per_1000_usd": 1.00,
            "fable_input_per_million_tokens_usd": 3.00,
            "fable_output_per_million_tokens_usd": 15.00,
            "pricing_note": "Bedrock Retrieve uses the official $1/1,000 requests page; Fable token prices are the recorded 2026-08-19 qualification assumption.",  # noqa: E501
        },
        "qualification_measurement": {
            "requests": 249,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "formula_usd": "522810/1e6*3 + 36639/1e6*15",
            "calculated_usd": 2.118015,
        },
        "one_time_bootstrap": {
            "lambda_requests": 0,
            "state_writes": 0,
            "artifact_puts": 0,
            "estimated_usd": 0.0,
            "ongoing_s3_storage": "artifact_size_bytes/1e9*0.023 per month; retained versions accumulate",  # noqa: E501
        },
        "per_deploy": {
            "artifact_puts": 1,
            "artifact_gets": 2,
            "health_lambda_requests": 1,
            "health_lambda_gb_seconds_upper": 15.0,
            "health_bedrock_retrieves": 0,
            "health_model_tokens": 0,
            "estimated_upper_usd_excluding_storage": 0.00026,
        },
        "per_cloud_answer": {
            "lambda_request": 1,
            "lambda_gb_seconds_upper": 15.0,
            "bedrock_retrieve_requests": 1,
            "fable_formula": "input_tokens/1e6*3 + output_tokens/1e6*15",
            "qualification_average_tokens": {
                "input": input_tokens / 249,
                "output": output_tokens / 249,
            },
            "state": "2-5 strongly consistent reads, one conditional PutItem, one conditional UpdateItem; bounded on-demand estimate < $0.00001/request",  # noqa: E501
            "logs": "bounded response metadata only; estimate bytes/1e9*0.50",
            "estimated_using_qualification_average_usd": 2.118015 / 249 + 0.001 + 0.00026,
        },
    }


def _tags(boundary: str) -> dict[str, str]:
    return {
        "Project": "valkeyrie",
        "Environment": "development",
        "ManagedBy": "aws-cdk",
        "Boundary": boundary,
    }


def _resources(template: dict[str, object]) -> dict[str, dict[str, Any]]:
    return cast(dict[str, dict[str, Any]], template["Resources"])


def _json(value: bytes) -> dict[str, object]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("template must be an object")
    return cast(dict[str, object], parsed)


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _pretty(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


if __name__ == "__main__":
    synthesize_application()
