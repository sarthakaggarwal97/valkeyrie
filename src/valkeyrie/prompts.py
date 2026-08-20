"""Fail-closed loading of reviewed, immutable prompt packages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import yaml

from valkeyrie.sources import _load_yaml_mapping


class PromptPackageError(ValueError):
    """A prompt package is malformed, unsafe, or changed while loading."""


@dataclass(frozen=True)
class PromptTemplate:
    """One immutable reviewed prompt template."""

    name: str
    path: str
    content: str


@dataclass(frozen=True)
class PromptPackage:
    """An immutable prompt package snapshot for one application revision."""

    api_version: str
    kind: str
    output_contract: str
    prompt_revision: str
    templates: tuple[PromptTemplate, ...]


_MANIFEST_BYTES = 16 * 1024
_TEMPLATE_BYTES = 64 * 1024
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUIRED_PROMPTS = {
    "answer": "prompts/answer.md",
    "citations": "prompts/citations.md",
    "clarification": "prompts/clarification.md",
    "evidence-use": "prompts/evidence-use.md",
    "system": "prompts/system.md",
}
_MANIFEST_FILES = {
    "manifest.yaml",
    *(PurePosixPath(path).name for path in _REQUIRED_PROMPTS.values()),
}
_TOP_LEVEL_FIELDS = {"api_version", "kind", "revision", "output_contract", "prompts"}
_REVISION_FIELDS = {
    "algorithm",
    "canonical_order",
    "newline_normalization",
    "derived_field",
    "prompt_revision",
    "prompt_revision_is_not_part_of_preimage",
}
_REVISION_SEMANTICS = {
    "algorithm": "sha256_canonical_roles_paths_and_bytes",
    "canonical_order": "lexical_path",
    "newline_normalization": "lf",
    "derived_field": "prompt_revision",
    "prompt_revision_is_not_part_of_preimage": True,
}


def load_prompt_package(root: Path) -> PromptPackage:
    """Load one fixed local prompt package into deterministic immutable data."""
    prompts_dir = root / "prompts"
    _require_real_directory(root, "prompt package root")
    _require_real_directory(prompts_dir, "prompts directory")
    _validate_directory_entries(prompts_dir)

    manifest_bytes = _read_regular_file(
        prompts_dir / "manifest.yaml", _MANIFEST_BYTES, "prompt manifest"
    )
    manifest_text = _decode_text(manifest_bytes, "prompt manifest")
    try:
        document = _load_yaml_mapping(
            manifest_text, str(prompts_dir / "manifest.yaml"), reject_merge_keys=True
        )
    except (UnicodeError, yaml.YAMLError, ValueError) as error:
        raise PromptPackageError(f"cannot load prompt manifest: {error}") from error

    prompts = _validate_manifest(document)
    loaded: list[tuple[dict[str, str], bytes, str]] = []
    for entry in sorted(prompts, key=lambda item: item["file"]):
        path = root / entry["file"]
        content_bytes = _read_regular_file(path, _TEMPLATE_BYTES, f"prompt {entry['name']}")
        content = _decode_text(content_bytes, f"prompt {entry['name']}")
        loaded.append((entry, content_bytes, content))

    revision = _prompt_revision(document, loaded)
    declared_revision = cast(Mapping[str, object], document["revision"])["prompt_revision"]
    if declared_revision != revision:
        raise PromptPackageError("prompt manifest revision does not match package contents")
    templates = tuple(
        PromptTemplate(entry["name"], entry["file"], content) for entry, _, content in loaded
    )
    return PromptPackage(
        api_version="valkeyrie.io/prompts/1",
        kind="PromptPackage",
        output_contract="evidence_ids_only_for_citations",
        prompt_revision=revision,
        templates=templates,
    )


def _validate_manifest(document: Mapping[str, object]) -> list[dict[str, str]]:
    if set(document) != _TOP_LEVEL_FIELDS:
        raise PromptPackageError("prompt manifest has an unknown or missing top-level field")
    if (document.get("api_version"), document.get("kind")) != (
        "valkeyrie.io/prompts/1",
        "PromptPackage",
    ):
        raise PromptPackageError("prompt manifest has an incompatible identity")
    if document.get("output_contract") != "evidence_ids_only_for_citations":
        raise PromptPackageError("prompt manifest has an unsupported output contract")

    revision_value = document.get("revision")
    if not isinstance(revision_value, Mapping) or not all(
        isinstance(key, str) for key in revision_value
    ):
        raise PromptPackageError("prompt manifest revision must be a mapping with string keys")
    revision = cast(Mapping[str, object], revision_value)
    if set(revision) != _REVISION_FIELDS:
        raise PromptPackageError("prompt manifest revision has an unknown or missing field")
    for field, expected in _REVISION_SEMANTICS.items():
        if revision.get(field) != expected or type(revision.get(field)) is not type(expected):
            raise PromptPackageError(f"prompt manifest revision has invalid {field}")
    declared = revision.get("prompt_revision")
    if not isinstance(declared, str) or _DIGEST.fullmatch(declared) is None:
        raise PromptPackageError("prompt manifest prompt_revision must be a sha256 digest")

    prompt_value = document.get("prompts")
    if not isinstance(prompt_value, list):
        raise PromptPackageError("prompt manifest prompts must be a list")
    prompts: list[dict[str, str]] = []
    for value in prompt_value:
        if not isinstance(value, Mapping) or set(value) != {"name", "file"}:
            raise PromptPackageError("each prompt must contain exactly name and file")
        name = value.get("name")
        path = value.get("file")
        if not isinstance(name, str) or not isinstance(path, str):
            raise PromptPackageError("prompt names and paths must be strings")
        _validate_prompt_path(path)
        prompts.append({"name": name, "file": path})

    names = [entry["name"] for entry in prompts]
    paths = [entry["file"] for entry in prompts]
    if len(names) != len(set(names)):
        raise PromptPackageError("prompt names must be unique")
    if len(paths) != len(set(paths)):
        raise PromptPackageError("prompt paths must be unique")
    if {entry["name"]: entry["file"] for entry in prompts} != _REQUIRED_PROMPTS:
        raise PromptPackageError("prompt manifest has an unknown or missing required prompt")
    return prompts


def _validate_prompt_path(value: str) -> None:
    path = PurePosixPath(value)
    parts = value.split("/")
    if (
        not value
        or path.is_absolute()
        or len(parts) != 2
        or parts[0] != "prompts"
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise PromptPackageError(f"unsafe or external prompt path: {value!r}")


def _require_real_directory(path: Path, label: str) -> None:
    try:
        absolute = path.absolute()
        if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
            raise PromptPackageError(f"{label} must not use a symlink")
        if not path.is_dir():
            raise PromptPackageError(f"{label} must be a local directory")
    except OSError as error:
        raise PromptPackageError(f"cannot inspect {label}: {error}") from error


def _validate_directory_entries(prompts_dir: Path) -> None:
    try:
        with os.scandir(prompts_dir) as entries:
            observed = {entry.name: entry for entry in entries}
    except OSError as error:
        raise PromptPackageError(f"cannot inspect prompts directory: {error}") from error
    if set(observed) != _MANIFEST_FILES:
        raise PromptPackageError("prompts directory has an unknown or missing package path")
    for name, entry in observed.items():
        if entry.is_symlink():
            raise PromptPackageError(f"prompt package path must not be a symlink: {name}")
        if not entry.is_file(follow_symlinks=False):
            raise PromptPackageError(f"prompt package path must be a regular file: {name}")


def _read_regular_file(path: Path, maximum: int, label: str) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise PromptPackageError(f"{label} must not be a symlink")
        if not stat.S_ISREG(before.st_mode):
            raise PromptPackageError(f"{label} must be a regular file")
        if before.st_size == 0:
            raise PromptPackageError(f"{label} must not be empty")
        if before.st_size > maximum:
            raise PromptPackageError(f"{label} exceeds the {maximum}-byte limit")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise PromptPackageError(f"{label} changed while loading")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                content = stream.read(maximum + 1)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except PromptPackageError:
        raise
    except OSError as error:
        raise PromptPackageError(f"cannot read {label}: {error}") from error
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise PromptPackageError(f"{label} changed while loading")
    if not content:
        raise PromptPackageError(f"{label} must not be empty")
    if len(content) > maximum:
        raise PromptPackageError(f"{label} exceeds the {maximum}-byte limit")
    return content


def _decode_text(content: bytes, label: str) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PromptPackageError(f"{label} is not valid UTF-8") from error
    if "\r" in text:
        raise PromptPackageError(f"{label} must use LF newlines")
    if any(
        (ord(character) < 32 and character not in "\t\n") or 127 <= ord(character) <= 159
        for character in text
    ):
        raise PromptPackageError(f"{label} contains binary control data")
    if not text.strip():
        raise PromptPackageError(f"{label} must not be empty")
    return text


def _prompt_revision(
    document: Mapping[str, object], loaded: list[tuple[dict[str, str], bytes, str]]
) -> str:
    canonical = dict(document)
    revision = dict(cast(Mapping[str, object], canonical["revision"]))
    revision.pop("prompt_revision")
    canonical["revision"] = revision
    canonical["prompts"] = [entry for entry, _, _ in loaded]
    manifest_bytes = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256()
    _hash_part(digest, b"valkeyrie-prompt-package/1")
    _hash_part(digest, manifest_bytes)
    for entry, content, _ in loaded:
        _hash_part(digest, entry["name"].encode("utf-8"))
        _hash_part(digest, entry["file"].encode("utf-8"))
        _hash_part(digest, content)
    return f"sha256:{digest.hexdigest()}"


def _hash_part(digest: hashlib._Hash, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)
