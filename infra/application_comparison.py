from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

from infra.application_artifact import (
    _MANIFEST_FIELDS as _BASE_MANIFEST_FIELDS,
)
from infra.application_artifact import (
    _MAX_ARTIFACT,
    _MAX_ENTRY,
    _TIMESTAMP,
    _canonical,
    _load_manifest,
    _read,
    _revision,
    _sha256,
    _zip,
    build_application_artifact,
    verify_application_artifact,
)


class ApplicationComparisonError(ValueError):
    """An owner-directed comparison declaration or artifact is invalid."""


@dataclass(frozen=True)
class ApplicationComparisonArtifact:
    application_revision: str
    prompt_revision: str
    candidate_inventory_digest: str
    evaluation_suite_revision: str
    evaluation_inputs_digest: str
    evaluation_reports_digest: str
    qualification_artifacts_digest: str
    response_normalization_policy_revision: str
    selected_model_revision: str
    selected_profile_revision: str
    selected_inference_config_revision: str
    selected_report_id: str
    selection_id: str
    artifact_sha256: str
    content: bytes
    manifest: Mapping[str, object]
    authorization: str
    qualification_status: str
    base_application_revision: str
    base_artifact_sha256: str
    base_selection_id: str
    corpus_generation: str
    knowledge_base_id: str
    lambda_timeout_seconds: int


COMPARISON_DECLARATION_PATH: Final = "evals/comparisons/claude-opus-5-ten-real.json"
_AUTHORIZATION: Final = "owner_directed_comparison"
_QUALIFICATION_STATUS: Final = "not_run_not_qualified"
_BASE_APPLICATION_REVISION: Final = (
    "sha256:08079d56dcbde9f2c44f421ae0b3dd7e1b157d3ba457d84d971acf3a5efb7f7e"
)
_BASE_ARTIFACT_SHA256: Final = (
    "sha256:98bbbd6d9808f12518ba67654d36343984dab9122f5c5445b3a504261aa93cb9"
)
_BASE_SELECTION_ID: Final = (
    "sha256:3cae47af3b2b2d8a2dbe60a167e1664d57f95efe8bb8314eb39d7bfa10d79dc8"
)
_PROMPT_REVISION: Final = "sha256:d06c1262c98feda8661e494b92fb5dd92985e23ec896dbd596947b5635107768"
_EVALUATION_SUITE_REVISION: Final = (
    "sha256:64f9a99faefc432eca78ed0bbf68d3fb6da3a0867e558fc07111ced476e66193"
)
_CORPUS_GENERATION: Final = (
    "sha256:6760b22e3e09ac0e2ac64c9c23b3a77733f5bd1f8f8c9704c66d2c69d1362b60"
)
_KNOWLEDGE_BASE_ID: Final = "ONVASJDDNX"
_MODEL_REVISION: Final = "us.anthropic.claude-opus-5"
_PROFILE_ARN: Final = (
    "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-opus-5"
)
_PROFILE_STATUS: Final = "ACTIVE"
_FOUNDATION_MODEL_ARNS: Final = (
    "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-5",
    "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-5",
    "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-opus-5",
)
_INFERENCE: Final[dict[str, object]] = {
    "maximum_output_tokens": 2048,
    "reasoning_effort": "low",
    "temperature": None,
    "top_p": None,
}
_LAMBDA_TIMEOUT_SECONDS: Final = 240
_RUNTIME_PATH: Final = "valkeyrie/application_runtime.py"
_BASE_RUNTIME_SHA256: Final = (
    "sha256:2a4e86ea23085b3bcf66bb5d14864d97f394bae4ec6fd1634078413797cbb241"
)
_AUTHORIZATION_FUNCTION: Final = (
    b"\n".join(
        (
            b"def _execution_authorization(manifest: Mapping[str, object]) -> str:",
            b'    authorization = manifest.get("authorization", "qualified_model_selection")',
            b'    if authorization == "qualified_model_selection":',
            b"        return authorization",
            b'    if authorization != "owner_directed_comparison":',
            b'        raise ApplicationRuntimeError("execution authorization is unsupported")',
            b'    if manifest.get("qualification_status") != "not_run_not_qualified":',
            (
                b'        raise ApplicationRuntimeError("comparison qualification status '
                b'is incompatible")'
            ),
            b'    if manifest.get("selected_model_revision") != "us.anthropic.claude-opus-5":',
            b'        raise ApplicationRuntimeError("comparison model identity is incompatible")',
            b"    return authorization",
            b"",
        )
    )
    + b"\n\n"
)
_AUTHORIZATION_PLAN_CHECK: Final = b"\n".join(
    (
        b'    if plan.get("execution_authorization") != _execution_authorization(manifest):',
        (
            b'        raise ApplicationRuntimeError("request is pinned to different execution '
            b'authorization")'
        ),
        b"",
    )
)
_HEALTH_IDENTITY_OVERLAY: Final = b"\n".join(
    (
        b'        "selected_model_revision": manifest["selected_model_revision"],',
        (b'        "selected_inference_profile_arn": manifest["selected_inference_profile_arn"],'),
        b'        "execution_authorization": _execution_authorization(manifest),',
        (b'        "qualification_status": manifest.get("qualification_status", "qualified"),'),
        b"",
    )
)
_AUTHORIZATION_OVERLAY: Final = (
    _AUTHORIZATION_FUNCTION,
    _HEALTH_IDENTITY_OVERLAY,
    b'        "execution_authorization": _execution_authorization(manifest),\n',
    _AUTHORIZATION_PLAN_CHECK,
    b'        "execution_authorization",\n',
    b"    _execution_authorization(value)\n",
)
_DECLARATION_SCHEMA: Final = "valkeyrie.io/application-comparison/1"
_ARTIFACT_SCHEMA: Final = "valkeyrie.io/application-comparison-artifact/1"
_REVISION_DOMAIN: Final = b"valkeyrie-application-comparison-revision/1"
_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_DECLARATION_FIELDS: Final = {
    "api_version",
    "kind",
    "authorization",
    "qualification_status",
    "base_application_revision",
    "base_artifact_sha256",
    "base_selection_id",
    "prompt_revision",
    "evaluation_suite_revision",
    "corpus_generation",
    "knowledge_base_id",
    "selected_model_revision",
    "selected_profile_revision",
    "selected_inference_config_revision",
    "selected_inference_profile_arn",
    "selected_inference_profile_status",
    "selected_foundation_model_arns",
    "selected_inference",
    "lambda_timeout_seconds",
    "selection_id",
}
_COMPARISON_FIELDS: Final = _BASE_MANIFEST_FIELDS | {
    "authorization",
    "qualification_status",
    "comparison_declaration_path",
    "comparison_declaration_sha256",
    "base_application_revision",
    "base_artifact_sha256",
    "base_selection_id",
    "base_manifest",
    "corpus_generation",
    "knowledge_base_id",
    "selected_inference_profile_status",
    "lambda_timeout_seconds",
}


def build_application_comparison_artifact(root: Path) -> ApplicationComparisonArtifact:
    """Derive the pinned owner-directed comparison from the exact qualified base artifact."""
    qualified = build_application_artifact(root)
    base_manifest, _ = _pinned_base(qualified.content, qualified.manifest)
    declaration_content = _read(root / COMPARISON_DECLARATION_PATH)
    declaration = _load_declaration(declaration_content)
    expected_declaration = _expected_declaration()
    if declaration != expected_declaration or declaration_content != (
        _canonical(expected_declaration) + b"\n"
    ):
        raise ApplicationComparisonError("comparison declaration differs from its immutable pins")

    files = _archive_files(qualified.content)
    del files["application-artifact.json"]
    files[COMPARISON_DECLARATION_PATH] = declaration_content
    manifest = _comparison_manifest(base_manifest, declaration_content, declaration, files)
    files["application-artifact.json"] = _canonical(manifest)
    content = _zip(files)
    if len(content) > _MAX_ARTIFACT:
        raise ApplicationComparisonError("comparison artifact exceeds its byte bound")
    verify_application_comparison_artifact(
        content,
        expected_application_revision=cast(str, manifest["application_revision"]),
        expected_artifact_sha256=_sha256(content),
    )
    return ApplicationComparisonArtifact(
        application_revision=cast(str, manifest["application_revision"]),
        prompt_revision=cast(str, manifest["prompt_revision"]),
        candidate_inventory_digest=cast(str, manifest["candidate_inventory_digest"]),
        evaluation_suite_revision=cast(str, manifest["evaluation_suite_revision"]),
        evaluation_inputs_digest=cast(str, manifest["evaluation_inputs_digest"]),
        evaluation_reports_digest=cast(str, manifest["evaluation_reports_digest"]),
        qualification_artifacts_digest=cast(str, manifest["qualification_artifacts_digest"]),
        response_normalization_policy_revision=cast(
            str, manifest["response_normalization_policy_revision"]
        ),
        selected_model_revision=cast(str, manifest["selected_model_revision"]),
        selected_profile_revision=cast(str, manifest["selected_profile_revision"]),
        selected_inference_config_revision=cast(
            str, manifest["selected_inference_config_revision"]
        ),
        selected_report_id=cast(str, manifest["selected_report_id"]),
        selection_id=cast(str, manifest["selection_id"]),
        artifact_sha256=_sha256(content),
        content=content,
        manifest=manifest,
        authorization=_AUTHORIZATION,
        qualification_status=_QUALIFICATION_STATUS,
        base_application_revision=_BASE_APPLICATION_REVISION,
        base_artifact_sha256=_BASE_ARTIFACT_SHA256,
        base_selection_id=_BASE_SELECTION_ID,
        corpus_generation=_CORPUS_GENERATION,
        knowledge_base_id=_KNOWLEDGE_BASE_ID,
        lambda_timeout_seconds=_LAMBDA_TIMEOUT_SECONDS,
    )


def verify_application_comparison_artifact(
    content: bytes,
    *,
    expected_application_revision: str | None = None,
    expected_artifact_sha256: str | None = None,
) -> Mapping[str, object]:
    """Verify exact comparison fields, paths, metadata, base bytes, and identities."""
    if not isinstance(content, bytes) or not 1 <= len(content) <= _MAX_ARTIFACT:
        raise ApplicationComparisonError("comparison artifact is outside its byte bound")
    if expected_artifact_sha256 is not None and _sha256(content) != expected_artifact_sha256:
        raise ApplicationComparisonError("comparison artifact hash differs from expected identity")

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != sorted(names) or len(names) != len(set(names)):
                raise ApplicationComparisonError("comparison artifact paths are not canonical")
            for info in infos:
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or info.is_dir():
                    raise ApplicationComparisonError("comparison artifact has an unsafe path")
                if (
                    info.date_time != _TIMESTAMP
                    or info.create_system != 3
                    or info.external_attr >> 16 != (stat.S_IFREG | 0o644)
                    or info.compress_type != zipfile.ZIP_DEFLATED
                ):
                    raise ApplicationComparisonError(
                        "comparison artifact entry metadata is not exact"
                    )
                if info.file_size > _MAX_ENTRY:
                    raise ApplicationComparisonError(
                        "comparison artifact entry exceeds its byte bound"
                    )
            if "application-artifact.json" not in names or COMPARISON_DECLARATION_PATH not in names:
                raise ApplicationComparisonError("comparison artifact lacks its immutable identity")
            files = {name: archive.read(name) for name in names}
    except ApplicationComparisonError:
        raise
    except (KeyError, OSError, zipfile.BadZipFile) as error:
        raise ApplicationComparisonError("cannot read comparison artifact") from error

    if _zip(files) != content:
        raise ApplicationComparisonError("comparison artifact ZIP encoding is not canonical")
    document = _load_comparison_manifest(files["application-artifact.json"])
    declaration = _load_declaration(files[COMPARISON_DECLARATION_PATH])
    expected_declaration = _expected_declaration()
    if declaration != expected_declaration or files[COMPARISON_DECLARATION_PATH] != (
        _canonical(expected_declaration) + b"\n"
    ):
        raise ApplicationComparisonError("comparison declaration differs from its immutable pins")

    base_manifest = document.get("base_manifest")
    if not isinstance(base_manifest, Mapping):
        raise ApplicationComparisonError("comparison base manifest is malformed")
    base_document = _load_manifest(_canonical(base_manifest))
    _verify_base_pins(base_document, _BASE_ARTIFACT_SHA256)
    expected_document = _comparison_manifest(
        base_document,
        files[COMPARISON_DECLARATION_PATH],
        declaration,
        files,
    )
    if document != expected_document or files["application-artifact.json"] != _canonical(document):
        raise ApplicationComparisonError("comparison manifest differs from its exact derivation")

    inventory = document.get("files")
    if not isinstance(inventory, list):
        raise ApplicationComparisonError("comparison file inventory is malformed")
    expected_paths = set(files) - {"application-artifact.json"}
    observed_paths: set[str] = set()
    for item in inventory:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "size", "mode"}:
            raise ApplicationComparisonError("comparison file entry is malformed")
        manifest_path = item.get("path")
        if (
            not isinstance(manifest_path, str)
            or manifest_path not in expected_paths
            or manifest_path in observed_paths
        ):
            raise ApplicationComparisonError("comparison file inventory is inconsistent")
        value = files[manifest_path]
        if (
            item.get("mode") != "0644"
            or item.get("size") != len(value)
            or item.get("sha256") != _sha256(value)
        ):
            raise ApplicationComparisonError("comparison file content was tampered")
        observed_paths.add(manifest_path)
    if observed_paths != expected_paths:
        raise ApplicationComparisonError("comparison file inventory is incomplete")

    base_files = dict(files)
    del base_files[COMPARISON_DECLARATION_PATH]
    del base_files["application-artifact.json"]
    base_files[_RUNTIME_PATH] = _remove_authorization_overlay(base_files[_RUNTIME_PATH])
    base_files["application-artifact.json"] = _canonical(base_document)
    base_content = _zip(base_files)
    if _sha256(base_content) != _BASE_ARTIFACT_SHA256:
        raise ApplicationComparisonError("comparison does not preserve the pinned base artifact")
    verify_application_artifact(
        base_content, expected_application_revision=_BASE_APPLICATION_REVISION
    )

    revision = document["application_revision"]
    if not isinstance(revision, str) or revision != _comparison_revision(document):
        raise ApplicationComparisonError("comparison application revision is invalid")
    if expected_application_revision is not None and revision != expected_application_revision:
        raise ApplicationComparisonError(
            "comparison application revision differs from expected content identity"
        )
    return document


def _expected_declaration() -> dict[str, object]:
    inference_revision = _content_id(b"valkeyrie-owner-directed-inference-config/1", _INFERENCE)
    profile_preimage = {
        "model_revision": _MODEL_REVISION,
        "inference_profile_arn": _PROFILE_ARN,
        "inference_profile_status": _PROFILE_STATUS,
        "foundation_model_arns": list(_FOUNDATION_MODEL_ARNS),
        "inference_config_revision": inference_revision,
        "prompt_revision": _PROMPT_REVISION,
        "evaluation_suite_revision": _EVALUATION_SUITE_REVISION,
        "corpus_generation": _CORPUS_GENERATION,
        "knowledge_base_id": _KNOWLEDGE_BASE_ID,
    }
    profile_revision = _content_id(
        b"valkeyrie-owner-directed-comparison-profile/1", profile_preimage
    )
    preimage: dict[str, object] = {
        "api_version": _DECLARATION_SCHEMA,
        "kind": "OwnerDirectedApplicationComparison",
        "authorization": _AUTHORIZATION,
        "qualification_status": _QUALIFICATION_STATUS,
        "base_application_revision": _BASE_APPLICATION_REVISION,
        "base_artifact_sha256": _BASE_ARTIFACT_SHA256,
        "base_selection_id": _BASE_SELECTION_ID,
        "prompt_revision": _PROMPT_REVISION,
        "evaluation_suite_revision": _EVALUATION_SUITE_REVISION,
        "corpus_generation": _CORPUS_GENERATION,
        "knowledge_base_id": _KNOWLEDGE_BASE_ID,
        "selected_model_revision": _MODEL_REVISION,
        "selected_profile_revision": profile_revision,
        "selected_inference_config_revision": inference_revision,
        "selected_inference_profile_arn": _PROFILE_ARN,
        "selected_inference_profile_status": _PROFILE_STATUS,
        "selected_foundation_model_arns": list(_FOUNDATION_MODEL_ARNS),
        "selected_inference": dict(_INFERENCE),
        "lambda_timeout_seconds": _LAMBDA_TIMEOUT_SECONDS,
    }
    return {
        **preimage,
        "selection_id": _content_id(b"valkeyrie-owner-directed-comparison-selection/1", preimage),
    }


def _comparison_manifest(
    base_manifest: Mapping[str, object],
    declaration_content: bytes,
    declaration: Mapping[str, object],
    artifact_files: Mapping[str, bytes],
) -> dict[str, object]:
    if not isinstance(base_manifest.get("files"), list):
        raise ApplicationComparisonError("base application file inventory is malformed")
    entries = [
        {"path": path, "sha256": _sha256(content), "size": len(content), "mode": "0644"}
        for path, content in sorted(artifact_files.items())
        if path != "application-artifact.json"
    ]
    preimage = dict(base_manifest)
    preimage.pop("application_revision", None)
    preimage.update(
        {
            "api_version": _ARTIFACT_SCHEMA,
            "kind": "ApplicationComparisonArtifact",
            "authorization": _AUTHORIZATION,
            "qualification_status": _QUALIFICATION_STATUS,
            "comparison_declaration_path": COMPARISON_DECLARATION_PATH,
            "comparison_declaration_sha256": _sha256(declaration_content),
            "base_application_revision": _BASE_APPLICATION_REVISION,
            "base_artifact_sha256": _BASE_ARTIFACT_SHA256,
            "base_selection_id": _BASE_SELECTION_ID,
            "base_manifest": dict(base_manifest),
            "corpus_generation": _CORPUS_GENERATION,
            "knowledge_base_id": _KNOWLEDGE_BASE_ID,
            "selected_model_revision": _MODEL_REVISION,
            "selected_profile_revision": declaration["selected_profile_revision"],
            "selected_inference_config_revision": declaration["selected_inference_config_revision"],
            "selected_report_id": _QUALIFICATION_STATUS,
            "selection_id": declaration["selection_id"],
            "selected_inference": dict(_INFERENCE),
            "selected_inference_profile_arn": _PROFILE_ARN,
            "selected_inference_profile_status": _PROFILE_STATUS,
            "selected_foundation_model_arns": list(_FOUNDATION_MODEL_ARNS),
            "lambda_timeout_seconds": _LAMBDA_TIMEOUT_SECONDS,
            "files": entries,
        }
    )
    return {**preimage, "application_revision": _comparison_revision(preimage)}


def _load_declaration(content: bytes) -> dict[str, object]:
    value = _load_json_object(content, "comparison declaration")
    if set(value) != _DECLARATION_FIELDS:
        raise ApplicationComparisonError("comparison declaration has an unknown or missing field")
    return value


def _load_comparison_manifest(content: bytes) -> dict[str, object]:
    value = _load_json_object(content, "comparison manifest")
    if set(value) != _COMPARISON_FIELDS:
        raise ApplicationComparisonError("comparison manifest has an unknown or missing field")
    if (value.get("api_version"), value.get("kind")) != (
        _ARTIFACT_SCHEMA,
        "ApplicationComparisonArtifact",
    ):
        raise ApplicationComparisonError("comparison manifest identity is incompatible")
    return value


def _load_json_object(content: bytes, label: str) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ApplicationComparisonError(f"duplicate JSON key in {label}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(content.decode(), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ApplicationComparisonError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ApplicationComparisonError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _verify_base_pins(base_manifest: Mapping[str, object], artifact_sha256: str) -> None:
    pins = {
        "application_revision": _BASE_APPLICATION_REVISION,
        "selection_id": _BASE_SELECTION_ID,
        "prompt_revision": _PROMPT_REVISION,
        "evaluation_suite_revision": _EVALUATION_SUITE_REVISION,
    }
    for field, expected in pins.items():
        if base_manifest.get(field) != expected:
            raise ApplicationComparisonError(f"base application {field} differs from its pin")
    if artifact_sha256 != _BASE_ARTIFACT_SHA256:
        raise ApplicationComparisonError("base application artifact hash differs from its pin")


def _pinned_base(
    qualified_content: bytes, qualified_manifest: Mapping[str, object]
) -> tuple[dict[str, object], bytes]:
    files = _archive_files(qualified_content)
    del files["application-artifact.json"]
    files[_RUNTIME_PATH] = _remove_authorization_overlay(files[_RUNTIME_PATH])
    entries = [
        {"path": path, "sha256": _sha256(content), "size": len(content), "mode": "0644"}
        for path, content in sorted(files.items())
    ]
    preimage = dict(qualified_manifest)
    preimage.pop("application_revision", None)
    preimage["files"] = entries
    base_manifest = {**preimage, "application_revision": _revision(preimage)}
    files["application-artifact.json"] = _canonical(base_manifest)
    base_content = _zip(files)
    _verify_base_pins(base_manifest, _sha256(base_content))
    verify_application_artifact(
        base_content, expected_application_revision=_BASE_APPLICATION_REVISION
    )
    return base_manifest, base_content


def _remove_authorization_overlay(runtime: bytes) -> bytes:
    base = runtime
    for addition in _AUTHORIZATION_OVERLAY:
        if base.count(addition) != 1:
            raise ApplicationComparisonError(
                "comparison runtime authorization overlay differs from its exact patch"
            )
        base = base.replace(addition, b"", 1)
    if _sha256(base) != _BASE_RUNTIME_SHA256:
        raise ApplicationComparisonError("reconstructed base runtime differs from its pin")
    return base


def _archive_files(content: bytes) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return {name: archive.read(name) for name in archive.namelist()}
    except (OSError, zipfile.BadZipFile) as error:
        raise ApplicationComparisonError("cannot read base application artifact") from error


def _comparison_revision(value: Mapping[str, object]) -> str:
    preimage = dict(value)
    preimage.pop("application_revision", None)
    return _framed(_REVISION_DOMAIN, _canonical(preimage))


def _content_id(domain: bytes, value: object) -> str:
    return _framed(domain, _canonical(value))


def _framed(domain: bytes, value: bytes) -> str:
    digest = hashlib.sha256()
    for part in (domain, value):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return "sha256:" + digest.hexdigest()


if any(
    _DIGEST.fullmatch(value) is None
    for value in (_BASE_APPLICATION_REVISION, _BASE_ARTIFACT_SHA256, _BASE_SELECTION_ID)
):
    raise RuntimeError("comparison immutable digest pin is malformed")
