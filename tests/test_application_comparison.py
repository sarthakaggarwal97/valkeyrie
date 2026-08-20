from __future__ import annotations

import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path
from typing import cast

import pytest

from infra.application_artifact import _canonical, _zip, build_application_artifact
from infra.application_comparison import (
    COMPARISON_DECLARATION_PATH,
    ApplicationComparisonError,
    build_application_comparison_artifact,
    verify_application_comparison_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_APPLICATION_REVISION = (
    "sha256:08079d56dcbde9f2c44f421ae0b3dd7e1b157d3ba457d84d971acf3a5efb7f7e"
)
BASE_ARTIFACT_SHA256 = "sha256:98bbbd6d9808f12518ba67654d36343984dab9122f5c5445b3a504261aa93cb9"
BASE_SELECTION_ID = "sha256:3cae47af3b2b2d8a2dbe60a167e1664d57f95efe8bb8314eb39d7bfa10d79dc8"
PROMPT_REVISION = "sha256:d06c1262c98feda8661e494b92fb5dd92985e23ec896dbd596947b5635107768"
SUITE_REVISION = "sha256:64f9a99faefc432eca78ed0bbf68d3fb6da3a0867e558fc07111ced476e66193"
GENERATION = "sha256:6760b22e3e09ac0e2ac64c9c23b3a77733f5bd1f8f8c9704c66d2c69d1362b60"
MODEL = "us.anthropic.claude-opus-5"
PROFILE = "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-opus-5"
FOUNDATION_MODELS = [
    "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-5",
    "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-5",
    "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-opus-5",
]
INFERENCE = {
    "maximum_output_tokens": 2048,
    "reasoning_effort": "low",
    "temperature": None,
    "top_p": None,
}


def _archive(content: bytes) -> tuple[dict[str, bytes], dict[str, zipfile.ZipInfo]]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return (
            {name: archive.read(name) for name in archive.namelist()},
            {info.filename: info for info in archive.infolist()},
        )


def _rewrite(content: bytes, values: dict[str, bytes]) -> bytes:
    files, _ = _archive(content)
    files.update(values)
    return _zip(files)


def _qualification_hashes() -> dict[str, str]:
    paths = [ROOT / "evals/model-selection.json"]
    paths.extend(
        path
        for directory in (ROOT / "evals/reports", ROOT / "evals/qualification-attempts")
        for path in directory.rglob("*")
        if path.is_file()
    )
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def test_comparison_artifact_is_deterministic_canonical_and_content_addressed() -> None:
    first = build_application_comparison_artifact(ROOT)
    second = build_application_comparison_artifact(ROOT)

    assert first == second
    assert first.artifact_sha256 == "sha256:" + hashlib.sha256(first.content).hexdigest()
    assert (
        verify_application_comparison_artifact(
            first.content,
            expected_application_revision=first.application_revision,
            expected_artifact_sha256=first.artifact_sha256,
        )
        == first.manifest
    )

    files, infos = _archive(first.content)
    assert list(files) == sorted(files)
    assert files["application-artifact.json"] == _canonical(first.manifest)
    assert files[COMPARISON_DECLARATION_PATH] == (ROOT / COMPARISON_DECLARATION_PATH).read_bytes()
    for info in infos.values():
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.create_system == 3
        assert info.external_attr >> 16 == stat.S_IFREG | 0o644
        assert info.compress_type == zipfile.ZIP_DEFLATED


def test_declaration_and_manifest_pin_owner_authorization_and_exact_opus_target() -> None:
    artifact = build_application_comparison_artifact(ROOT)
    declaration_content = (ROOT / COMPARISON_DECLARATION_PATH).read_bytes()
    declaration = json.loads(declaration_content)
    assert declaration_content == _canonical(declaration) + b"\n"

    exact = {
        "authorization": "owner_directed_comparison",
        "qualification_status": "not_run_not_qualified",
        "base_application_revision": BASE_APPLICATION_REVISION,
        "base_artifact_sha256": BASE_ARTIFACT_SHA256,
        "base_selection_id": BASE_SELECTION_ID,
        "prompt_revision": PROMPT_REVISION,
        "evaluation_suite_revision": SUITE_REVISION,
        "corpus_generation": GENERATION,
        "knowledge_base_id": "ONVASJDDNX",
        "selected_model_revision": MODEL,
        "selected_inference_profile_arn": PROFILE,
        "selected_inference_profile_status": "ACTIVE",
        "selected_foundation_model_arns": FOUNDATION_MODELS,
        "selected_inference": INFERENCE,
        "lambda_timeout_seconds": 240,
    }
    for field, expected in exact.items():
        assert declaration[field] == expected
        assert artifact.manifest[field] == expected

    assert artifact.authorization == exact["authorization"]
    assert artifact.qualification_status == exact["qualification_status"]
    assert artifact.base_application_revision == BASE_APPLICATION_REVISION
    assert artifact.base_artifact_sha256 == BASE_ARTIFACT_SHA256
    assert artifact.base_selection_id == BASE_SELECTION_ID
    assert artifact.prompt_revision == PROMPT_REVISION
    assert artifact.evaluation_suite_revision == SUITE_REVISION
    assert artifact.corpus_generation == GENERATION
    assert artifact.knowledge_base_id == "ONVASJDDNX"
    assert artifact.lambda_timeout_seconds == 240
    assert artifact.selected_model_revision == MODEL
    assert artifact.selected_report_id == "not_run_not_qualified"
    assert artifact.selection_id == declaration["selection_id"]
    assert artifact.selected_profile_revision == declaration["selected_profile_revision"]
    assert (
        artifact.selected_inference_config_revision
        == declaration["selected_inference_config_revision"]
    )


def test_comparison_preserves_qualified_payload_with_exact_runtime_overlay() -> None:
    qualified = build_application_artifact(ROOT)
    comparison = build_application_comparison_artifact(ROOT)
    qualified_files, qualified_infos = _archive(qualified.content)
    comparison_files, comparison_infos = _archive(comparison.content)

    assert qualified.application_revision != BASE_APPLICATION_REVISION
    assert qualified.artifact_sha256 != BASE_ARTIFACT_SHA256
    assert qualified.selection_id == BASE_SELECTION_ID
    assert set(comparison_files) == set(qualified_files) | {COMPARISON_DECLARATION_PATH}
    assert (
        comparison_files["application-artifact.json"]
        != qualified_files["application-artifact.json"]
    )
    for path, expected in qualified_files.items():
        if path == "application-artifact.json":
            continue
        assert comparison_files[path] == expected
        qualified_info = qualified_infos[path]
        comparison_info = comparison_infos[path]
        assert (
            comparison_info.date_time,
            comparison_info.create_system,
            comparison_info.external_attr,
            comparison_info.compress_type,
        ) == (
            qualified_info.date_time,
            qualified_info.create_system,
            qualified_info.external_attr,
            qualified_info.compress_type,
        )
    base_manifest = cast(dict[str, object], comparison.manifest["base_manifest"])
    assert base_manifest["application_revision"] == BASE_APPLICATION_REVISION
    assert base_manifest["selection_id"] == BASE_SELECTION_ID
    base_inventory = {
        cast(str, item["path"]): item
        for item in cast(list[dict[str, object]], base_manifest["files"])
    }
    comparison_inventory = {
        cast(str, item["path"]): item
        for item in cast(list[dict[str, object]], comparison.manifest["files"])
    }
    assert base_inventory["valkeyrie/application_runtime.py"]["sha256"] == (
        "sha256:2a4e86ea23085b3bcf66bb5d14864d97f394bae4ec6fd1634078413797cbb241"
    )
    assert comparison_inventory["valkeyrie/application_runtime.py"]["sha256"] == (
        "sha256:" + hashlib.sha256(qualified_files["valkeyrie/application_runtime.py"]).hexdigest()
    )


@pytest.mark.parametrize("tamper", ["payload", "declaration", "manifest", "extra_path"])
def test_comparison_verifier_rejects_tampering(tamper: str) -> None:
    artifact = build_application_comparison_artifact(ROOT)
    files, _ = _archive(artifact.content)
    replacement: dict[str, bytes]
    if tamper == "payload":
        replacement = {"valkeyrie/routing.py": b"tampered\n"}
    elif tamper == "declaration":
        declaration = json.loads(files[COMPARISON_DECLARATION_PATH])
        declaration["qualification_status"] = "qualified"
        replacement = {COMPARISON_DECLARATION_PATH: _canonical(declaration)}
    elif tamper == "manifest":
        manifest = json.loads(files["application-artifact.json"])
        manifest["selected_model_revision"] = "us.anthropic.claude-fable-5"
        replacement = {"application-artifact.json": _canonical(manifest)}
    else:
        replacement = {"evil.py": b"raise SystemExit('evil')\n"}

    with pytest.raises(ApplicationComparisonError, match="tamper|differs|preserve|inventory"):
        verify_application_comparison_artifact(_rewrite(artifact.content, replacement))

    with pytest.raises(ApplicationComparisonError, match="artifact hash"):
        verify_application_comparison_artifact(
            artifact.content, expected_artifact_sha256="sha256:" + "0" * 64
        )


def test_build_leaves_all_qualification_file_hashes_unchanged() -> None:
    before = _qualification_hashes()
    artifact = build_application_comparison_artifact(ROOT)
    after = _qualification_hashes()

    assert artifact.qualification_status == "not_run_not_qualified"
    assert before
    assert after == before
    assert (
        cast(dict[str, object], artifact.manifest["base_manifest"])["selection_id"]
        == BASE_SELECTION_ID
    )
