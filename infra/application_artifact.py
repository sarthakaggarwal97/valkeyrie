"""Deterministic deployable ZIP for one fully qualified Valkeyrie application."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import distribution
from pathlib import Path, PurePosixPath
from typing import Final, cast

from valkeyrie.answer_models import load_answer_model_inventory
from valkeyrie.bedrock_response import BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION
from valkeyrie.evaluations import load_evaluation_suite
from valkeyrie.live_qualification import LiveQualificationError, verify_live_qualification_artifacts
from valkeyrie.prompts import load_prompt_package


class ApplicationArtifactError(ValueError):
    """An application package input or immutable artifact is invalid."""


@dataclass(frozen=True)
class ApplicationArtifact:
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


_SCHEMA: Final = "valkeyrie.io/application-artifact/2"
_DOMAIN: Final = b"valkeyrie-application-revision/2"
_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_ENTRY = 4 * 1024 * 1024
_MAX_ARTIFACT = 32 * 1024 * 1024
_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MODEL_SELECTION_PATH: Final = "evals/model-selection.json"
_RUNTIME_MODULES: Final = (
    "__init__.py",
    "acquisition.py",
    "answer_models.py",
    "application_runtime.py",
    "bedrock_response.py",
    "drafting.py",
    "evaluations.py",
    "evidence.py",
    "generation.py",
    "github.py",
    "live_github.py",
    "normalization.py",
    "prompts.py",
    "request_audit.py",
    "retrieval.py",
    "retrieval_config.py",
    "revisions.py",
    "routing.py",
    "sources.py",
    "structured.py",
)
_NON_RUNTIME_MODULES: Final = frozenset(
    {
        "aws_adapters.py",
        "corpus.py",
        "deployed_evaluation.py",
        "git_acquisition.py",
        "ingestion.py",
        "live_qualification.py",
        "promotion.py",
        "publication.py",
    }
)
_STATIC_INPUTS: Final = (
    "answer-models.yaml",
    "retrieval-config.yaml",
    "prompts/answer.md",
    "prompts/citations.md",
    "prompts/clarification.md",
    "prompts/evidence-use.md",
    "prompts/manifest.yaml",
    "prompts/system.md",
    "evals/criteria/model.yaml",
    "evals/criteria/retrieval.yaml",
    "evals/criteria/usage-safeguards.yaml",
    "evals/holdout.yaml",
    "evals/manifest.yaml",
    "evals/public.yaml",
    "evals/retrieval.yaml",
    "infra/__init__.py",
    "infra/application_handler.py",
)
_DEPENDENCIES: Final = (
    ("PyYAML", "6.0.3", ("yaml",)),
    ("jsonschema", "4.26.0", ("jsonschema",)),
    ("pathspec", "1.1.1", ("pathspec",)),
    ("attrs", "26.1.0", ("attr", "attrs")),
    ("jsonschema-specifications", "2025.9.1", ("jsonschema_specifications",)),
    ("referencing", "0.37.0", ("referencing",)),
    ("rpds-py", "2026.6.3", ("rpds",)),
    ("typing-extensions", "4.16.0", ("typing_extensions.py",)),
)
_MANIFEST_FIELDS: Final = {
    "api_version",
    "kind",
    "application_revision",
    "architecture",
    "handler",
    "prompt_revision",
    "candidate_inventory_digest",
    "approved_candidate_models",
    "evaluation_suite_revision",
    "evaluation_inputs_digest",
    "evaluation_reports_digest",
    "evaluation_report_paths",
    "qualification_evidence_paths",
    "model_selection_path",
    "qualification_artifacts_digest",
    "response_normalization_policy_revision",
    "selected_model_revision",
    "selected_profile_revision",
    "selected_inference_config_revision",
    "selected_report_id",
    "selection_id",
    "selected_inference",
    "selected_inference_profile_arn",
    "selected_foundation_model_arns",
    "raw_evidence_sha256",
    "dependencies",
    "files",
    "corpus_state_included",
}


def build_application_artifact(root: Path) -> ApplicationArtifact:
    """Build only when the exact authoritative qualification set fully verifies."""
    _validate_root(root)
    _validate_source_inventory(root)
    prompt = load_prompt_package(root)
    candidates = load_answer_model_inventory(root / "answer-models.yaml")
    suite = load_evaluation_suite(root)
    try:
        verified = verify_live_qualification_artifacts(root)
    except (LiveQualificationError, ValueError) as error:
        raise ApplicationArtifactError(
            f"qualification artifacts are not approved: {error}"
        ) from error
    selection = dict(verified.selection_record)
    _validate_selection(
        selection,
        prompt.prompt_revision,
        suite.revision,
        tuple(candidate.model_revision for candidate in candidates),
    )
    qualification_paths = _qualification_paths(selection)

    files: dict[str, bytes] = {}
    for module in _RUNTIME_MODULES:
        files[f"valkeyrie/{module}"] = _read(root / f"src/valkeyrie/{module}")
    files["valkeyrie/schemas/contracts.schema.json"] = _read(
        root / "src/valkeyrie/schemas/contracts.schema.json"
    )
    for relative in _STATIC_INPUTS:
        files[relative] = _read(root / relative)
    for relative in qualification_paths:
        files[relative] = _read(root / relative)

    dependency_entries: list[dict[str, object]] = []
    dependency_files, dependency_entries = _vendored_dependencies()
    overlap = set(files) & set(dependency_files)
    if overlap:
        raise ApplicationArtifactError(f"dependency path collision: {sorted(overlap)!r}")
    files.update(dependency_files)

    report_paths = [
        path for path in qualification_paths if "/reports/" in path and "/evidence/" not in path
    ]
    evidence_paths = [path for path in qualification_paths if "/reports/evidence/" in path]
    raw_hashes = {path: _sha256(files[path]) for path in evidence_paths}
    evaluation_inputs_digest = _framed(
        b"valkeyrie-evaluation-inputs/1",
        tuple(
            part
            for path in _STATIC_INPUTS
            if path.startswith("evals/")
            for part in (path.encode(), files[path])
        ),
    )
    reports_digest = _framed(
        b"valkeyrie-evaluation-reports/1",
        tuple(part for path in report_paths for part in (path.encode(), files[path])),
    )
    qualification_digest = _framed(
        b"valkeyrie-qualification-artifacts/1",
        tuple(part for path in qualification_paths for part in (path.encode(), files[path])),
    )
    inventory_digest = _sha256(files["answer-models.yaml"])
    entries = [
        {"path": path, "sha256": _sha256(content), "size": len(content), "mode": "0644"}
        for path, content in sorted(files.items())
    ]
    caller_identity = selection.get("caller_and_model_identity")
    if not isinstance(caller_identity, Mapping):
        raise ApplicationArtifactError("model selection caller identity is malformed")
    preimage: dict[str, object] = {
        "api_version": _SCHEMA,
        "kind": "ApplicationArtifact",
        "architecture": "x86_64",
        "handler": "infra.application_handler.handler",
        "prompt_revision": prompt.prompt_revision,
        "candidate_inventory_digest": inventory_digest,
        "approved_candidate_models": [candidate.model_revision for candidate in candidates],
        "evaluation_suite_revision": suite.revision,
        "evaluation_inputs_digest": evaluation_inputs_digest,
        "evaluation_reports_digest": reports_digest,
        "evaluation_report_paths": report_paths,
        "qualification_evidence_paths": evidence_paths,
        "model_selection_path": _MODEL_SELECTION_PATH,
        "qualification_artifacts_digest": qualification_digest,
        "response_normalization_policy_revision": BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION,
        "selected_model_revision": selection["selected_model_revision"],
        "selected_profile_revision": selection["selected_profile_revision"],
        "selected_inference_config_revision": selection["selected_inference_config_revision"],
        "selected_report_id": selection["selected_report_id"],
        "selection_id": selection["selection_id"],
        "selected_inference": selection["selected_inference"],
        "selected_inference_profile_arn": caller_identity["fable_profile_arn"],
        "selected_foundation_model_arns": caller_identity["fable_model_arns"],
        "raw_evidence_sha256": raw_hashes,
        "dependencies": dependency_entries,
        "files": entries,
        "corpus_state_included": False,
    }
    revision = _revision(preimage)
    manifest = {**preimage, "application_revision": revision}
    archive = dict(files)
    archive["application-artifact.json"] = _canonical(manifest)
    content = _zip(archive)
    if len(content) > _MAX_ARTIFACT:
        raise ApplicationArtifactError("application artifact exceeds its byte bound")
    verify_application_artifact(content, expected_application_revision=revision)
    return ApplicationArtifact(
        revision,
        prompt.prompt_revision,
        inventory_digest,
        suite.revision,
        evaluation_inputs_digest,
        reports_digest,
        qualification_digest,
        BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION,
        cast(str, selection["selected_model_revision"]),
        cast(str, selection["selected_profile_revision"]),
        cast(str, selection["selected_inference_config_revision"]),
        cast(str, selection["selected_report_id"]),
        cast(str, selection["selection_id"]),
        _sha256(content),
        content,
        manifest,
    )


def verify_application_artifact(
    content: bytes,
    *,
    expected_application_revision: str | None = None,
) -> Mapping[str, object]:
    """Verify exact fields, paths, dependencies, metadata, and content identity."""
    if not isinstance(content, bytes) or not 1 <= len(content) <= _MAX_ARTIFACT:
        raise ApplicationArtifactError("application artifact is outside its byte bound")
    expected_dependency_files, expected_dependencies = _vendored_dependencies()
    base_required_paths = {
        *(f"valkeyrie/{module}" for module in _RUNTIME_MODULES),
        "valkeyrie/schemas/contracts.schema.json",
        *_STATIC_INPUTS,
        *expected_dependency_files,
    }
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != sorted(names) or len(names) != len(set(names)):
                raise ApplicationArtifactError("application artifact paths are not canonical")
            if "application-artifact.json" not in names or _MODEL_SELECTION_PATH not in names:
                raise ApplicationArtifactError("application artifact lacks qualification identity")
            document = _load_manifest(archive.read("application-artifact.json"))
            selection = _load_manifest(archive.read(_MODEL_SELECTION_PATH), exact_fields=False)
            approved = document.get("approved_candidate_models")
            if not isinstance(approved, list) or not all(
                isinstance(item, str) for item in approved
            ):
                raise ApplicationArtifactError("approved candidate inventory is malformed")
            _validate_selection(
                selection,
                cast(str, document["prompt_revision"]),
                cast(str, document["evaluation_suite_revision"]),
                tuple(cast(list[str], approved)),
            )
            qualification_paths = _qualification_paths(selection)
            required_paths = base_required_paths | set(qualification_paths)
            if set(names) != required_paths | {"application-artifact.json"}:
                raise ApplicationArtifactError(
                    "application artifact has an unknown or missing path"
                )
            for info in infos:
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or info.is_dir():
                    raise ApplicationArtifactError("application artifact has an unsafe path")
                if info.date_time != _TIMESTAMP:
                    raise ApplicationArtifactError(
                        "application artifact timestamp is not deterministic"
                    )
                if info.create_system != 3 or info.external_attr >> 16 != (stat.S_IFREG | 0o644):
                    raise ApplicationArtifactError("application artifact permissions are not exact")
                if info.file_size > _MAX_ENTRY:
                    raise ApplicationArtifactError("application artifact entry exceeds its bound")
            if document["dependencies"] != expected_dependencies:
                raise ApplicationArtifactError("application dependency inventory is incompatible")
            values = document["files"]
            if not isinstance(values, list):
                raise ApplicationArtifactError("application file inventory is malformed")
            observed_paths: set[str] = set()
            file_bytes: dict[str, bytes] = {}
            for item in values:
                if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "size", "mode"}:
                    raise ApplicationArtifactError("application file entry is malformed")
                manifest_path = item.get("path")
                if (
                    not isinstance(manifest_path, str)
                    or manifest_path not in required_paths
                    or manifest_path in observed_paths
                ):
                    raise ApplicationArtifactError("application file inventory is inconsistent")
                value = archive.read(manifest_path)
                if (
                    item.get("mode") != "0644"
                    or item.get("size") != len(value)
                    or item.get("sha256") != _sha256(value)
                ):
                    raise ApplicationArtifactError("application file content was tampered")
                observed_paths.add(manifest_path)
                file_bytes[manifest_path] = value
            if observed_paths != required_paths:
                raise ApplicationArtifactError("application file inventory is incomplete")
            _verify_bundled_qualification(document, selection, qualification_paths, file_bytes)
            _verify_artifact_input_digests(document, qualification_paths, file_bytes)
    except ApplicationArtifactError:
        raise
    except (KeyError, OSError, zipfile.BadZipFile) as error:
        raise ApplicationArtifactError("cannot read application artifact") from error
    revision = document["application_revision"]
    preimage = dict(document)
    preimage.pop("application_revision")
    if _revision(preimage) != revision:
        raise ApplicationArtifactError("application revision does not match artifact identity")
    if expected_application_revision is not None and revision != expected_application_revision:
        raise ApplicationArtifactError(
            "application revision differs from expected content identity"
        )
    if document["corpus_state_included"] is not False:
        raise ApplicationArtifactError("application artifact must not contain corpus state")
    return document


def _qualification_paths(selection: Mapping[str, object]) -> tuple[str, ...]:
    reports = selection.get("candidate_reports")
    if not isinstance(reports, list) or len(reports) != 2:
        raise ApplicationArtifactError("model selection does not cover exactly two candidates")
    revisions: list[str] = []
    for item in reports:
        if not isinstance(item, Mapping):
            raise ApplicationArtifactError("model selection candidate report is malformed")
        revision = item.get("profile_revision")
        if not isinstance(revision, str) or _DIGEST.fullmatch(revision) is None:
            raise ApplicationArtifactError("model selection candidate profile is malformed")
        revisions.append(revision.removeprefix("sha256:"))
    if len(set(revisions)) != 2:
        raise ApplicationArtifactError("model selection candidate profiles are not unique")
    return (
        _MODEL_SELECTION_PATH,
        *(f"evals/reports/{revision}.json" for revision in revisions),
        *(f"evals/reports/evidence/{revision}.json" for revision in revisions),
    )


def _validate_selection(
    value: Mapping[str, object],
    prompt: str,
    suite: str,
    candidate_models: tuple[str, ...],
) -> None:
    if value.get("selected_model_revision") != "us.anthropic.claude-fable-5":
        raise ApplicationArtifactError("model selection must select the required Fable model")
    if value.get("response_normalization_policy_revision") != (
        BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION
    ):
        raise ApplicationArtifactError("model selection normalization identity is incompatible")
    for field in (
        "selected_profile_revision",
        "selected_inference_config_revision",
        "selection_id",
    ):
        item = value.get(field)
        if not isinstance(item, str) or _DIGEST.fullmatch(item) is None:
            raise ApplicationArtifactError(f"model selection {field} is malformed")
    report_id = value.get("selected_report_id")
    if not isinstance(report_id, str) or re.fullmatch(r"eval_[0-9a-f]{64}", report_id) is None:
        raise ApplicationArtifactError("model selection selected_report_id is malformed")
    if value.get("evaluation_suite_revision") != suite:
        raise ApplicationArtifactError("model selection suite identity is stale")
    identities = value.get("input_identities")
    if (
        not isinstance(identities, Mapping)
        or identities.get("prompt_revision") != prompt
        or identities.get("evaluation_suite_revision") != suite
        or identities.get("response_normalization_policy_revision")
        != BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION
    ):
        raise ApplicationArtifactError("model selection input identity is stale")
    reports = value.get("candidate_reports")
    if not isinstance(reports, list) or len(reports) != 2:
        raise ApplicationArtifactError("model selection does not cover exactly two candidates")
    observed_models: list[str] = []
    selected_candidate: Mapping[str, object] | None = None
    for item in reports:
        if not isinstance(item, Mapping):
            raise ApplicationArtifactError("model selection candidate report is malformed")
        model = item.get("model_revision")
        profile = item.get("profile_revision")
        candidate_report = item.get("report_id")
        if (
            not isinstance(model, str)
            or not isinstance(profile, str)
            or _DIGEST.fullmatch(profile) is None
            or not isinstance(candidate_report, str)
            or re.fullmatch(r"eval_[0-9a-f]{64}", candidate_report) is None
        ):
            raise ApplicationArtifactError("model selection candidate identity is malformed")
        observed_models.append(model)
        if model == value["selected_model_revision"]:
            selected_candidate = cast(Mapping[str, object], item)
    if tuple(observed_models) != candidate_models or len(set(observed_models)) != 2:
        raise ApplicationArtifactError("model selection candidate inventory is incompatible")
    if (
        selected_candidate is None
        or selected_candidate.get("profile_revision") != value["selected_profile_revision"]
        or selected_candidate.get("report_id") != value["selected_report_id"]
    ):
        raise ApplicationArtifactError("model selection selected identity is inconsistent")
    selection_preimage = dict(value)
    selection_id = selection_preimage.pop("selection_id")
    if selection_id != _content_id("answer-model-selection-record/1", selection_preimage):
        raise ApplicationArtifactError("model selection content identity is invalid")


def _verify_bundled_qualification(
    document: Mapping[str, object],
    selection: Mapping[str, object],
    qualification_paths: tuple[str, ...],
    files: Mapping[str, bytes],
) -> None:
    report_paths = [
        path for path in qualification_paths if "/reports/" in path and "/evidence/" not in path
    ]
    evidence_paths = [path for path in qualification_paths if "/reports/evidence/" in path]
    if document.get("evaluation_report_paths") != report_paths:
        raise ApplicationArtifactError("qualification reports are not the exact authoritative set")
    if document.get("qualification_evidence_paths") != evidence_paths:
        raise ApplicationArtifactError(
            "raw qualification evidence is not the exact authoritative set"
        )
    if document.get("model_selection_path") != _MODEL_SELECTION_PATH:
        raise ApplicationArtifactError("model selection is absent")
    for field in (
        "selected_model_revision",
        "selected_profile_revision",
        "selected_inference_config_revision",
        "selected_report_id",
        "selection_id",
        "response_normalization_policy_revision",
    ):
        if document.get(field) != selection.get(field):
            raise ApplicationArtifactError(f"artifact {field} differs from model selection")
    candidates = cast(list[Mapping[str, object]], selection["candidate_reports"])
    for candidate, report_path, evidence_path in zip(
        candidates, report_paths, evidence_paths, strict=True
    ):
        report = _load_manifest(files[report_path], exact_fields=False)
        report_preimage = dict(report)
        report_id = report_preimage.pop("report_id", None)
        expected_report_id = "eval_" + hashlib.sha256(_canonical(report_preimage)).hexdigest()
        if (
            report_id != expected_report_id
            or report_id != candidate.get("report_id")
            or report.get("candidate_revision") != candidate.get("profile_revision")
            or report.get("suite_revision") != document.get("evaluation_suite_revision")
            or report.get("result") != candidate.get("result")
        ):
            raise ApplicationArtifactError("bundled evaluation report identity is inconsistent")
        evidence = _load_manifest(files[evidence_path], exact_fields=False)
        profile = evidence.get("profile")
        inputs = evidence.get("input_identities")
        if (
            evidence.get("complete") is not True
            or evidence.get("report_id") != report_id
            or not isinstance(profile, Mapping)
            or profile.get("model_revision") != candidate.get("model_revision")
            or profile.get("profile_revision") != candidate.get("profile_revision")
            or profile.get("prompt_revision") != document.get("prompt_revision")
            or profile.get("evaluation_suite_revision") != document.get("evaluation_suite_revision")
            or not isinstance(inputs, Mapping)
            or inputs.get("response_normalization_policy_revision")
            != BEDROCK_RESPONSE_NORMALIZATION_POLICY_REVISION
        ):
            raise ApplicationArtifactError(
                "bundled qualification evidence identity is inconsistent"
            )


def _verify_artifact_input_digests(
    document: Mapping[str, object],
    qualification_paths: tuple[str, ...],
    files: Mapping[str, bytes],
) -> None:
    report_paths = [
        path for path in qualification_paths if "/reports/" in path and "/evidence/" not in path
    ]
    evidence_paths = [path for path in qualification_paths if "/reports/evidence/" in path]
    expected = {
        "candidate_inventory_digest": _sha256(files["answer-models.yaml"]),
        "evaluation_inputs_digest": _framed(
            b"valkeyrie-evaluation-inputs/1",
            tuple(
                part
                for path in _STATIC_INPUTS
                if path.startswith("evals/")
                for part in (path.encode(), files[path])
            ),
        ),
        "evaluation_reports_digest": _framed(
            b"valkeyrie-evaluation-reports/1",
            tuple(part for path in report_paths for part in (path.encode(), files[path])),
        ),
        "qualification_artifacts_digest": _framed(
            b"valkeyrie-qualification-artifacts/1",
            tuple(part for path in qualification_paths for part in (path.encode(), files[path])),
        ),
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise ApplicationArtifactError(f"artifact {field} is inconsistent")
    raw_hashes = document.get("raw_evidence_sha256")
    if raw_hashes != {path: _sha256(files[path]) for path in evidence_paths}:
        raise ApplicationArtifactError("raw qualification evidence hashes are inconsistent")


def _content_id(domain: str, value: object) -> str:
    return _framed(domain.encode(), (_canonical(value),))


def _assert_target_abi(relative: str, content: bytes) -> None:
    """Refuse a native extension that cannot load on the Lambda this artifact declares.

    Dependencies are vendored from the host environment while the artifact and the function both
    declare x86_64 Linux. A build on another platform therefore produces extensions the runtime
    cannot import, and every other check still passes: the digest is computed over whatever bytes
    were copied, so the artifact verifies and the failure only appears as an ImportError after
    deployment. The ELF header is what distinguishes them, so it is read here rather than trusted
    from the filename.
    """
    if not relative.endswith((".so", ".pyd", ".dylib")):
        return
    # ELF magic, then EI_CLASS 2 (64-bit) and EI_DATA 1 (little endian), then e_machine 0x3E.
    header = content[:20]
    machine = int.from_bytes(header[18:20], "little") if len(header) >= 20 else 0
    if header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1 or machine != 0x3E:
        raise ApplicationArtifactError(
            f"vendored native extension is not Linux x86_64 and cannot load on the "
            f"declared architecture: {relative}. Build in a Linux x86_64 environment."
        )


def _vendored_dependencies() -> tuple[dict[str, bytes], list[dict[str, object]]]:
    files: dict[str, bytes] = {}
    entries: list[dict[str, object]] = []
    site = Path(str(distribution("jsonschema").locate_file("")))
    for name, version, roots in _DEPENDENCIES:
        dist = distribution(name)
        if dist.version != version:
            raise ApplicationArtifactError(f"locked dependency version differs for {name}")
        resolved_roots: list[str] = []
        for root in roots:
            candidates = [site / root]
            if root in {"_yaml", "rpds"}:
                candidates = sorted(site.glob(f"{root}*.so")) if root == "_yaml" else [site / root]
            if not candidates:
                raise ApplicationArtifactError(f"locked dependency root is absent: {root}")
            for candidate in candidates:
                if candidate.is_file():
                    relative = candidate.relative_to(site).as_posix()
                    content = _read(candidate)
                    _assert_target_abi(relative, content)
                    files[relative] = content
                    resolved_roots.append(relative)
                elif candidate.is_dir():
                    resolved_roots.append(candidate.relative_to(site).as_posix())
                    for path in sorted(candidate.rglob("*")):
                        if (
                            path.is_file()
                            and "__pycache__" not in path.parts
                            and "tests" not in path.relative_to(candidate).parts
                            and path.stat().st_size > 0
                            and path.suffix != ".pyc"
                        ):
                            relative = path.relative_to(site).as_posix()
                            content = _read(path)
                            _assert_target_abi(relative, content)
                            files[relative] = content
                else:
                    raise ApplicationArtifactError(f"locked dependency root is absent: {root}")
        entries.append({"name": name, "version": version, "roots": sorted(resolved_roots)})
    return files, entries


def _validate_source_inventory(root: Path) -> None:
    actual = {path.name for path in (root / "src/valkeyrie").glob("*.py")}
    if actual != set(_RUNTIME_MODULES) | set(_NON_RUNTIME_MODULES):
        raise ApplicationArtifactError("application sources have an unknown or missing Python file")


def _validate_root(root: Path) -> None:
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise ApplicationArtifactError("application root must be a regular local directory")


def _read(path: Path) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ApplicationArtifactError(f"application input must be a regular file: {path}")
        if not 1 <= before.st_size <= _MAX_ENTRY:
            raise ApplicationArtifactError(f"application input is outside its bound: {path}")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            content = os.read(descriptor, _MAX_ENTRY + 1)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except ApplicationArtifactError:
        raise
    except OSError as error:
        raise ApplicationArtifactError(f"cannot read application input: {path}") from error
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(content) != before.st_size:
        raise ApplicationArtifactError(f"application input changed while loading: {path}")
    return content


def _zip(files: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True
    ) as archive:
        for path, content in sorted(files.items()):
            info = zipfile.ZipInfo(path, _TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def _load_manifest(content: bytes, *, exact_fields: bool = True) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ApplicationArtifactError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(content.decode(), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ApplicationArtifactError("application JSON is invalid") from error
    if not isinstance(value, dict):
        raise ApplicationArtifactError("application JSON must be an object")
    if exact_fields and set(value) != _MANIFEST_FIELDS:
        raise ApplicationArtifactError(
            "application artifact manifest has an unknown or missing field"
        )
    if exact_fields and (value.get("api_version"), value.get("kind")) != (
        _SCHEMA,
        "ApplicationArtifact",
    ):
        raise ApplicationArtifactError("application artifact manifest identity is incompatible")
    return cast(dict[str, object], value)


def _revision(preimage: Mapping[str, object]) -> str:
    return _framed(_DOMAIN, (_canonical(preimage),))


def _framed(domain: bytes, values: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    for value in (domain, *values):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return "sha256:" + digest.hexdigest()


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
