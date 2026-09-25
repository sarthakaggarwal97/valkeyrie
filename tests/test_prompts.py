from __future__ import annotations

import os
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from tests.helpers import compute_prompt_revision
from valkeyrie.prompts import PromptPackageError, load_prompt_package

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_FILES = (
    "answer.md",
    "citations.md",
    "clarification.md",
    "evidence-use.md",
    "system.md",
)


def _copy_package(tmp_path: Path, name: str = "package") -> Path:
    package = tmp_path / name
    shutil.copytree(ROOT / "prompts", package / "prompts")
    return package


def _manifest(package: Path) -> dict[str, Any]:
    value = yaml.safe_load((package / "prompts" / "manifest.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _write_manifest(package: Path, document: dict[str, Any]) -> None:
    (package / "prompts" / "manifest.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def test_reviewed_prompt_package_loads_as_exact_immutable_snapshot() -> None:
    package = load_prompt_package(ROOT)

    assert package.api_version == "valkeyrie.io/prompts/1"
    assert package.kind == "PromptPackage"
    assert package.output_contract == "evidence_ids_only_for_citations"
    assert package.prompt_revision == (
        "sha256:59bcaae84982a94f02b2d631eba07569d066c6545e2184f045345c2853d3dc9d"
    )
    assert len(package.prompt_revision) == 71
    assert tuple(template.name for template in package.templates) == (
        "answer",
        "citations",
        "clarification",
        "evidence-use",
        "system",
    )
    assert tuple(template.path for template in package.templates) == tuple(
        f"prompts/{name}" for name in TEMPLATE_FILES
    )
    for template in package.templates:
        assert template.content == (ROOT / template.path).read_text(encoding="utf-8")

    package_field = "prompt_revision"
    template_field = "content"
    with pytest.raises(FrozenInstanceError):
        setattr(package, package_field, "sha256:" + "0" * 64)
    with pytest.raises(FrozenInstanceError):
        setattr(package.templates[0], template_field, "changed")


def test_answer_prompt_states_the_exact_json_only_model_output_contract() -> None:
    package = load_prompt_package(ROOT)
    answer = next(template.content for template in package.templates if template.name == "answer")

    for exact_shape in (
        '"outcome":"answer","claims"',
        '"outcome":"clarification","question"',
        '"outcome":"abstention","reason"',
    ):
        assert exact_shape in answer
    normalized = " ".join(answer.split())
    for requirement in (
        "Return exactly one JSON object and nothing else",
        "with no additional fields",
        "one claim object per independently supported factual claim",
        "40 words or fewer",
        "one concise question of 20 words or fewer",
        "reason of 20 words or fewer",
        "Insufficient validated evidence.",
        "Never explain source authority, policy,",
        "or capability in output",
        "evidence_ids` separate",
        "no Markdown",
        "evidence IDs",
        "URLs",
        "release-readiness",
        "external write completed",
        "qualified partial result",
    ):
        assert requirement in normalized


def test_prompts_default_unqualified_valkey_questions_to_canonical_core() -> None:
    package = load_prompt_package(ROOT)
    prompts = {template.name: " ".join(template.content.split()) for template in package.templates}

    for name in ("system", "evidence-use", "clarification", "answer"):
        assert "unqualified Valkey feature or command question" in prompts[name]
        assert "Valkey core" in prompts[name]
        assert "module or client evidence" in prompts[name]
    assert "explicitly missing component or version scope" in prompts["system"]
    assert "materially changes the supported answer" in prompts["clarification"]


def test_identical_package_bytes_produce_identical_revision(tmp_path: Path) -> None:
    first = _copy_package(tmp_path, "first")
    second = _copy_package(tmp_path, "second")

    assert load_prompt_package(first) == load_prompt_package(second)
    assert load_prompt_package(first).prompt_revision == load_prompt_package(second).prompt_revision


@pytest.mark.parametrize("filename", TEMPLATE_FILES)
def test_every_template_content_is_revision_bound(tmp_path: Path, filename: str) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    original = document["revision"]["prompt_revision"]
    template = package / "prompts" / filename
    template.write_bytes(template.read_bytes() + b"\nReviewed semantic change.\n")

    assert compute_prompt_revision(document, package) != original
    with pytest.raises(PromptPackageError, match="revision does not match"):
        load_prompt_package(package)


def test_manifest_is_canonical_and_derived_digest_is_not_self_referential(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    original = load_prompt_package(package).prompt_revision
    document = _manifest(package)
    cast(list[object], document["prompts"]).reverse()
    _write_manifest(package, document)

    assert load_prompt_package(package).prompt_revision == original

    manifest_path = package / "prompts" / "manifest.yaml"
    manifest_path.write_text(
        "# Semantically irrelevant YAML comment.\n" + manifest_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert load_prompt_package(package).prompt_revision == original

    document = _manifest(package)
    document["revision"]["prompt_revision"] = "sha256:" + "f" * 64
    _write_manifest(package, document)
    with pytest.raises(PromptPackageError, match="revision does not match"):
        load_prompt_package(package)


def test_loaded_snapshot_does_not_retain_mutable_file_inputs(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    loaded = load_prompt_package(package)
    answer = package / "prompts" / "answer.md"
    original = loaded.templates[0].content
    answer.write_text("replacement", encoding="utf-8")

    assert loaded.templates[0].content == original
    with pytest.raises(PromptPackageError, match="revision does not match"):
        load_prompt_package(package)


@pytest.mark.parametrize(
    "field",
    ["credentials", "capabilities", "policy", "model_provider", "provider_configuration"],
)
def test_manifest_rejects_external_authority_or_provider_inputs(tmp_path: Path, field: str) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document[field] = "external-input"
    _write_manifest(package, document)

    with pytest.raises(PromptPackageError, match="unknown or missing top-level field"):
        load_prompt_package(package)


def test_manifest_rejects_missing_top_level_field(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    del document["output_contract"]
    _write_manifest(package, document)

    with pytest.raises(PromptPackageError, match="unknown or missing top-level field"):
        load_prompt_package(package)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("api_version", "valkeyrie.io/prompts/2", "incompatible identity"),
        ("kind", "ExternalPromptPackage", "incompatible identity"),
        ("output_contract", "free_form_links", "unsupported output contract"),
    ],
)
def test_manifest_rejects_incompatible_identity_or_contract(
    tmp_path: Path, field: str, value: str, error: str
) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document[field] = value
    _write_manifest(package, document)

    with pytest.raises(PromptPackageError, match=error):
        load_prompt_package(package)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("algorithm", "sha256_content_only"),
        ("canonical_order", "manifest_order"),
        ("newline_normalization", "platform"),
        ("derived_field", "revision"),
        ("prompt_revision_is_not_part_of_preimage", False),
        ("prompt_revision_is_not_part_of_preimage", 1),
    ],
)
def test_manifest_rejects_changed_revision_semantics(
    tmp_path: Path, field: str, value: object
) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document["revision"][field] = value
    _write_manifest(package, document)

    with pytest.raises(PromptPackageError, match=f"invalid {field}"):
        load_prompt_package(package)


def test_manifest_rejects_unknown_or_missing_revision_fields(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document["revision"]["external"] = True
    _write_manifest(package, document)
    with pytest.raises(PromptPackageError, match="revision has an unknown or missing field"):
        load_prompt_package(package)

    document = _manifest(ROOT)
    del document["revision"]["canonical_order"]
    _write_manifest(package, document)
    with pytest.raises(PromptPackageError, match="revision has an unknown or missing field"):
        load_prompt_package(package)


@pytest.mark.parametrize("value", [None, 7, "sha256:ABC", "sha256:abcd", "not-a-digest"])
def test_manifest_rejects_invalid_declared_revision(tmp_path: Path, value: object) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document["revision"]["prompt_revision"] = value
    _write_manifest(package, document)

    with pytest.raises(PromptPackageError, match="must be a sha256 digest"):
        load_prompt_package(package)


def test_manifest_rejects_non_mapping_revision_and_non_list_prompts(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document["revision"] = []
    _write_manifest(package, document)
    with pytest.raises(PromptPackageError, match="revision must be a mapping"):
        load_prompt_package(package)

    document = _manifest(ROOT)
    document["prompts"] = {}
    _write_manifest(package, document)
    with pytest.raises(PromptPackageError, match="prompts must be a list"):
        load_prompt_package(package)


def test_manifest_rejects_unknown_or_missing_prompt_entry_fields(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document["prompts"][0]["external"] = True
    _write_manifest(package, document)
    with pytest.raises(PromptPackageError, match="exactly name and file"):
        load_prompt_package(package)

    document = _manifest(ROOT)
    del document["prompts"][0]["file"]
    _write_manifest(package, document)
    with pytest.raises(PromptPackageError, match="exactly name and file"):
        load_prompt_package(package)


@pytest.mark.parametrize("field", ["name", "file"])
def test_manifest_rejects_non_string_prompt_identity(tmp_path: Path, field: str) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document["prompts"][0][field] = 1
    _write_manifest(package, document)

    with pytest.raises(PromptPackageError, match="names and paths must be strings"):
        load_prompt_package(package)


@pytest.mark.parametrize(
    ("duplicate", "error"), [("name", "prompt names"), ("file", "prompt paths")]
)
def test_manifest_rejects_duplicate_prompt_identities(
    tmp_path: Path, duplicate: str, error: str
) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document["prompts"][1][duplicate] = document["prompts"][0][duplicate]
    _write_manifest(package, document)

    with pytest.raises(PromptPackageError, match=f"{error} must be unique"):
        load_prompt_package(package)


def test_manifest_rejects_unknown_and_missing_required_prompts(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document["prompts"].pop()
    _write_manifest(package, document)
    with pytest.raises(PromptPackageError, match="unknown or missing required prompt"):
        load_prompt_package(package)

    document = _manifest(ROOT)
    document["prompts"][0]["name"] = "routing"
    _write_manifest(package, document)
    with pytest.raises(PromptPackageError, match="unknown or missing required prompt"):
        load_prompt_package(package)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "../answer.md",
        "prompts/../answer.md",
        "prompts/./answer.md",
        "/prompts/answer.md",
        "prompts\\answer.md",
        "prompts/subdir/answer.md",
        "https://example.invalid/answer.md",
        "${PROMPT_PATH}",
        "prompts/answer\x00.md",
    ],
)
def test_manifest_rejects_unsafe_or_external_paths(tmp_path: Path, path: str) -> None:
    package = _copy_package(tmp_path)
    document = _manifest(package)
    document["prompts"][0]["file"] = path
    _write_manifest(package, document)

    with pytest.raises(PromptPackageError, match="unsafe or external prompt path"):
        load_prompt_package(package)


def test_manifest_rejects_duplicate_keys_and_yaml_merges(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    manifest = package / "prompts" / "manifest.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\nkind: UnsafeDuplicate\n", encoding="utf-8"
    )
    with pytest.raises(PromptPackageError, match="duplicate key"):
        load_prompt_package(package)

    shutil.copyfile(ROOT / "prompts" / "manifest.yaml", manifest)
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text("base: &base {}\n<<: *base\n" + text, encoding="utf-8")
    with pytest.raises(PromptPackageError, match="merge keys are not allowed"):
        load_prompt_package(package)


def test_package_rejects_unknown_or_missing_directory_paths(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    (package / "prompts" / "unreviewed.md").write_text("unreviewed", encoding="utf-8")
    with pytest.raises(PromptPackageError, match="unknown or missing package path"):
        load_prompt_package(package)

    (package / "prompts" / "unreviewed.md").unlink()
    (package / "prompts" / "answer.md").unlink()
    with pytest.raises(PromptPackageError, match="unknown or missing package path"):
        load_prompt_package(package)


@pytest.mark.parametrize("filename", ["manifest.yaml", *TEMPLATE_FILES])
def test_package_rejects_symlinked_files(tmp_path: Path, filename: str) -> None:
    package = _copy_package(tmp_path)
    path = package / "prompts" / filename
    target = tmp_path / f"external-{filename}"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(PromptPackageError, match="must not be a symlink"):
        load_prompt_package(package)


def test_package_rejects_symlinked_root_prompts_directory_and_ancestor(tmp_path: Path) -> None:
    package = _copy_package(tmp_path, "real")
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(package, target_is_directory=True)
    with pytest.raises(PromptPackageError, match="root must not use a symlink"):
        load_prompt_package(linked_root)

    ancestor = tmp_path / "ancestor"
    nested_package = _copy_package(ancestor, "nested")
    linked_ancestor = tmp_path / "linked-ancestor"
    linked_ancestor.symlink_to(ancestor, target_is_directory=True)
    with pytest.raises(PromptPackageError, match="root must not use a symlink"):
        load_prompt_package(linked_ancestor / nested_package.name)

    prompts_target = tmp_path / "external-prompts"
    shutil.copytree(package / "prompts", prompts_target)
    shutil.rmtree(package / "prompts")
    (package / "prompts").symlink_to(prompts_target, target_is_directory=True)
    with pytest.raises(PromptPackageError, match="directory must not use a symlink"):
        load_prompt_package(package)


def test_package_rejects_unsupported_file_types_without_reading_them(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    answer = package / "prompts" / "answer.md"
    answer.unlink()
    os.mkfifo(answer)

    with pytest.raises(PromptPackageError, match="regular file"):
        load_prompt_package(package)


@pytest.mark.parametrize("filename", ["manifest.yaml", "answer.md"])
@pytest.mark.parametrize(
    ("content", "error"),
    [
        (b"\xff", "not valid UTF-8"),
        (b"text\x00binary", "binary control data"),
        (b" \t\n", "must not be empty"),
    ],
)
def test_package_rejects_invalid_binary_or_empty_text(
    tmp_path: Path, filename: str, content: bytes, error: str
) -> None:
    package = _copy_package(tmp_path)
    (package / "prompts" / filename).write_bytes(content)

    with pytest.raises(PromptPackageError, match=error):
        load_prompt_package(package)


@pytest.mark.parametrize("filename", ["manifest.yaml", "answer.md"])
def test_package_rejects_non_lf_newlines(tmp_path: Path, filename: str) -> None:
    package = _copy_package(tmp_path)
    path = package / "prompts" / filename
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(PromptPackageError, match="must use LF newlines"):
        load_prompt_package(package)


def test_package_rejects_oversized_manifest_and_template(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    (package / "prompts" / "manifest.yaml").write_bytes(b"x" * (16 * 1024 + 1))
    with pytest.raises(PromptPackageError, match="exceeds the 16384-byte limit"):
        load_prompt_package(package)

    shutil.copyfile(ROOT / "prompts" / "manifest.yaml", package / "prompts" / "manifest.yaml")
    (package / "prompts" / "answer.md").write_bytes(b"x" * (64 * 1024 + 1))
    with pytest.raises(PromptPackageError, match="exceeds the 65536-byte limit"):
        load_prompt_package(package)


def test_package_rejects_missing_or_non_directory_roots(tmp_path: Path) -> None:
    with pytest.raises(PromptPackageError, match="root must be a local directory"):
        load_prompt_package(tmp_path / "missing")

    file_root = tmp_path / "file"
    file_root.write_text("not a package", encoding="utf-8")
    with pytest.raises(PromptPackageError, match="root must be a local directory"):
        load_prompt_package(file_root)
