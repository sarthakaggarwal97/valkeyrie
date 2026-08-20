"""Deterministic normalization of acquired reviewed source documents."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, Literal, TypeAlias, cast

from valkeyrie.acquisition import AcquiredFile, AcquiredRepository
from valkeyrie.revisions import ResolvedRevision
from valkeyrie.sources import (
    SourceInventoryError,
    classify_path,
    validate_source_inventory,
)


class NormalizationError(ValueError):
    """Acquired source cannot be represented as a canonical reviewed document."""


ContentType: TypeAlias = Literal[
    "text/plain",
    "text/markdown",
    "text/code",
    "application/json",
    "application/yaml",
]


@dataclass(frozen=True)
class NormalizationLimits:
    """Hard local bounds for normalizing one acquired repository."""

    max_files: int = 10_000
    max_path_bytes: int = 4_096
    max_file_bytes: int = 1024 * 1024
    max_total_bytes: int = 100 * 1024 * 1024


@dataclass(frozen=True)
class NormalizedDocument:
    """One source-bound normalized document before generation identity exists."""

    source: ResolvedRevision
    path: str
    content_type: ContentType
    content: str
    content_digest: str
    document_id: str


_DEFAULT_LIMITS = NormalizationLimits()
_MAX_REPOSITORY_BYTES: Final = 100
_MAX_REQUESTED_REF_BYTES: Final = 255
_MAX_VERSION_SCOPE_BYTES: Final = 256
_DOCUMENT_API_VERSION: Final = "valkeyrie.io/normalized-document/1"
_DOCUMENT_ID_API_VERSION: Final = "valkeyrie.io/normalized-document-identity/1"
_METADATA_API_VERSION: Final = "valkeyrie.io/metadata-sidecar/1"
_REPOSITORY = re.compile(r"^[a-z0-9.][a-z0-9._-]*$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MARKDOWN_SUFFIXES = frozenset({".md", ".mdx"})
_JSON_SUFFIXES = frozenset({".json"})
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_PLAIN_SUFFIXES = frozenset({".adoc", ".csv", ".docs", ".install", ".rst", ".txt"})
_CODE_SUFFIXES = frozenset(
    {
        ".bash",
        ".c",
        ".cc",
        ".cfg",
        ".conf",
        ".cpp",
        ".cs",
        ".csproj",
        ".css",
        ".cxx",
        ".fish",
        ".go",
        ".gradle",
        ".h",
        ".hh",
        ".hpp",
        ".html",
        ".ini",
        ".in",
        ".java",
        ".jade",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".lua",
        ".php",
        ".pl",
        ".properties",
        ".proto",
        ".py",
        ".pyi",
        ".rb",
        ".rs",
        ".sbt",
        ".scala",
        ".scss",
        ".sh",
        ".sql",
        ".swift",
        ".tcl",
        ".toml",
        ".ts",
        ".tsx",
        ".xml",
    }
)
_EXTENSIONLESS_NAMES = frozenset(
    {
        "00-releasenotes",
        "authors",
        "changes",
        "contributors",
        "copying",
        "dockerfile",
        "makefile",
        "notice",
    }
)
_EXTENSIONLESS_PREFIXES = (
    "changelog",
    "code_of_conduct",
    "contributing",
    "governance",
    "license",
    "migrat",
    "readme",
    "release",
    "runtest",
    "security",
)
_FORBIDDEN_UNICODE_CONTROLS = frozenset(
    {
        "\u0085",
        "\u2028",
        "\u2029",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
        "\ufeff",
    }
)


def normalize_repository(
    source_inventory: Mapping[str, object],
    resolved: ResolvedRevision,
    acquired: AcquiredRepository,
    *,
    limits: NormalizationLimits = _DEFAULT_LIMITS,
) -> tuple[NormalizedDocument, ...]:
    """Normalize one complete exact-commit acquisition in lexical path order."""
    _validate_limits(limits)
    entry = _validate_source_binding(source_inventory, resolved, acquired)
    if not isinstance(acquired.files, tuple):
        raise NormalizationError("acquired repository files must be an immutable tuple")
    if len(acquired.files) > limits.max_files:
        raise NormalizationError("normalization exceeded its file-count bound")

    policy_file_bytes, policy_total_bytes = _policy_limits(source_inventory)
    max_file_bytes = min(limits.max_file_bytes, policy_file_bytes)
    max_total_bytes = min(limits.max_total_bytes, policy_total_bytes)
    seen_paths: set[str] = set()
    normalized: list[NormalizedDocument] = []
    acquired_total_bytes = 0
    for acquired_file in acquired.files:
        if not isinstance(acquired_file, AcquiredFile):
            raise NormalizationError("acquired repository contains a malformed file")
        path = _validate_path(acquired_file.path, limits.max_path_bytes)
        if path in seen_paths:
            raise NormalizationError(f"acquired repository repeats path {path!r}")
        seen_paths.add(path)
        try:
            included = classify_path(source_inventory, resolved.repository, path) == "include"
        except SourceInventoryError as error:  # pragma: no cover - inventory/path checked above
            raise NormalizationError(f"cannot classify acquired path {path!r}: {error}") from error
        if not included:
            raise NormalizationError(f"acquired path {path!r} is outside reviewed source policy")
        content_type = _content_type(path)
        content, content_bytes = _validate_content(acquired_file.content, path, max_file_bytes)
        acquired_total_bytes += len(acquired_file.content)
        if acquired_total_bytes > max_total_bytes:
            raise NormalizationError("normalization exceeded its total-byte bound")
        content_digest = _sha256(content_bytes)
        document_id = _sha256(
            _document_identity_bytes(resolved, path, content_type, content_digest)
        )
        normalized.append(
            NormalizedDocument(
                source=resolved,
                path=path,
                content_type=content_type,
                content=content,
                content_digest=content_digest,
                document_id=document_id,
            )
        )

    if (
        not isinstance(acquired.total_bytes, int)
        or isinstance(acquired.total_bytes, bool)
        or acquired.total_bytes != acquired_total_bytes
    ):
        raise NormalizationError("acquired repository total_bytes is inconsistent with its files")
    if entry["authority"] != resolved.authority:  # explicit trust-boundary assertion
        raise NormalizationError("resolved authority conflicts with reviewed source policy")
    return tuple(sorted(normalized, key=lambda document: document.path))


def canonical_document_identity_bytes(document: NormalizedDocument) -> bytes:
    """Return the source-bound preimage used to derive ``document_id``.

    The preimage has neither ``document_id`` nor any checksum of itself.
    """
    _validate_document(document)
    return _document_identity_bytes(
        document.source,
        document.path,
        document.content_type,
        document.content_digest,
    )


def canonical_document_bytes(document: NormalizedDocument) -> bytes:
    """Return strict canonical JSON for the normalized-document contract."""
    _validate_document(document)
    return _canonical_json(
        {
            "api_version": _DOCUMENT_API_VERSION,
            "content": document.content,
            "content_type": document.content_type,
            "document_id": document.document_id,
            "kind": "NormalizedDocument",
        }
    )


def canonical_metadata_identity_bytes(document: NormalizedDocument) -> bytes:
    """Return a generation-free metadata-sidecar identity template.

    ``generation_id`` and a metadata checksum are deliberately absent. The
    source content checksum is retained because it is not derived from this
    metadata template.
    """
    _validate_document(document)
    source = document.source
    return _canonical_json(
        {
            "api_version": _METADATA_API_VERSION,
            "authority": source.authority,
            "commit": source.commit,
            "content_digest": document.content_digest,
            "content_type": document.content_type,
            "document_id": document.document_id,
            "kind": "MetadataSidecar",
            "path": document.path,
            "repository": source.repository,
            "version_scope": source.version_scope,
        }
    )


def _validate_limits(limits: NormalizationLimits) -> None:
    if not isinstance(limits, NormalizationLimits):
        raise NormalizationError("normalization limits are malformed")
    values = (
        limits.max_files,
        limits.max_path_bytes,
        limits.max_file_bytes,
        limits.max_total_bytes,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
        raise NormalizationError("normalization bounds must be positive integers")


def _validate_source_binding(
    source_inventory: Mapping[str, object],
    resolved: ResolvedRevision,
    acquired: AcquiredRepository,
) -> Mapping[str, object]:
    if not isinstance(source_inventory, Mapping):
        raise NormalizationError("source inventory must be a mapping")
    try:
        validate_source_inventory(source_inventory)
    except (SourceInventoryError, KeyError, TypeError, ValueError) as error:
        raise NormalizationError(f"source inventory is invalid: {error}") from error
    if not isinstance(resolved, ResolvedRevision):
        raise NormalizationError("resolved revision metadata is malformed")
    if not isinstance(acquired, AcquiredRepository):
        raise NormalizationError("acquired repository metadata is malformed")
    _validate_resolved_revision(resolved)

    repositories = cast(list[dict[str, object]], source_inventory["repositories"])
    entry = next(
        (repository for repository in repositories if repository["name"] == resolved.repository),
        None,
    )
    if entry is None:
        raise NormalizationError(f"unknown reviewed source repository: {resolved.repository}")
    expected = (
        (resolved.repository_url, entry["url"]),
        (resolved.requested_ref, entry["requested_ref"]),
        (resolved.authority, entry["authority"]),
        (resolved.version_scope, entry["version_scope"]),
        (resolved.source_policy_digest, _source_policy_digest(source_inventory)),
    )
    if (
        entry["classification"] != "curated"
        or entry["ingestion_mode"] != "documents"
        or entry["authority"] not in {"canonical", "secondary"}
        or any(actual != wanted for actual, wanted in expected)
    ):
        raise NormalizationError("resolved revision metadata conflicts with reviewed source policy")
    if acquired.repository != resolved.repository or acquired.commit != resolved.commit:
        raise NormalizationError("acquisition identity does not match the resolved revision")
    return entry


def _validate_resolved_revision(source: ResolvedRevision) -> None:
    _validate_bounded_text(source.repository, "repository", _MAX_REPOSITORY_BYTES)
    if _REPOSITORY.fullmatch(source.repository) is None:
        raise NormalizationError("resolved repository name is malformed")
    if source.repository_url != f"https://github.com/valkey-io/{source.repository}":
        raise NormalizationError("resolved repository URL is not canonical")
    _validate_bounded_text(source.requested_ref, "requested ref", _MAX_REQUESTED_REF_BYTES)
    if not isinstance(source.ref_kind, str) or source.ref_kind not in {"branch", "tag", "commit"}:
        raise NormalizationError("resolved ref kind is invalid")
    if not isinstance(source.commit, str) or _SHA.fullmatch(source.commit) is None:
        raise NormalizationError("resolved commit must be a full lowercase 40-hex SHA")
    if not isinstance(source.authority, str) or source.authority not in {"canonical", "secondary"}:
        raise NormalizationError("resolved authority is not valid for document normalization")
    _validate_bounded_text(source.version_scope, "version scope", _MAX_VERSION_SCOPE_BYTES)
    if (
        not isinstance(source.source_policy_digest, str)
        or _DIGEST.fullmatch(source.source_policy_digest) is None
    ):
        raise NormalizationError("resolved source policy digest is malformed")


def _validate_document(document: NormalizedDocument) -> None:
    if not isinstance(document, NormalizedDocument):
        raise NormalizationError("normalized document is malformed")
    if not isinstance(document.source, ResolvedRevision):
        raise NormalizationError("normalized document source is malformed")
    _validate_resolved_revision(document.source)
    path = _validate_path(document.path, _DEFAULT_LIMITS.max_path_bytes)
    expected_type = _content_type(path)
    if document.content_type != expected_type:
        raise NormalizationError("normalized document content type is inconsistent with its path")
    if not isinstance(document.content, str):
        raise NormalizationError("normalized document content must be text")
    try:
        content_bytes = document.content.encode("utf-8")
    except UnicodeEncodeError as error:
        raise NormalizationError("normalized document content is not valid UTF-8") from error
    _validate_content(content_bytes, path, _DEFAULT_LIMITS.max_file_bytes)
    content_digest = _sha256(content_bytes)
    if document.content_digest != content_digest:
        raise NormalizationError("normalized document content checksum is inconsistent")
    document_id = _sha256(
        _document_identity_bytes(
            document.source,
            path,
            document.content_type,
            document.content_digest,
        )
    )
    if document.document_id != document_id:
        raise NormalizationError("normalized document ID is inconsistent")


def _validate_path(value: object, max_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise NormalizationError("document path must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise NormalizationError("document path is not valid UTF-8") from error
    candidate = PurePosixPath(value)
    parts = value.split("/")
    if (
        len(encoded) > max_bytes
        or candidate.is_absolute()
        or candidate.as_posix() != value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise NormalizationError(f"unsafe document path: {value!r}")
    return value


def _validate_content(value: object, path: str, max_bytes: int) -> tuple[str, bytes]:
    if not isinstance(value, bytes):
        raise NormalizationError(f"document {path!r} content must be bytes")
    if not value:
        raise NormalizationError(f"document {path!r} content must not be empty")
    if len(value) > max_bytes:
        raise NormalizationError(f"document {path!r} exceeded its file-byte bound")
    normalized = value.removeprefix(b"\xef\xbb\xbf")
    if not normalized:
        raise NormalizationError(f"document {path!r} content must not be empty")
    try:
        text = normalized.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NormalizationError(f"document {path!r} is not canonical UTF-8 text") from error
    if text.encode("utf-8") != normalized:
        raise NormalizationError(f"document {path!r} is not canonical UTF-8 text")
    text = text.replace("\r\n", "\n")
    if "\r" in text:
        raise NormalizationError(f"document {path!r} contains a bare carriage return")
    normalized = text.encode("utf-8")
    if any(
        (ord(character) < 32 and character not in {"\t", "\n"})
        or 127 <= ord(character) <= 159
        or character in _FORBIDDEN_UNICODE_CONTROLS
        for character in text
    ):
        raise NormalizationError(f"document {path!r} contains binary or unsafe control content")
    return text, normalized


def _content_type(path: str) -> ContentType:
    candidate = PurePosixPath(path)
    suffix = candidate.suffix.casefold()
    name = candidate.name.casefold()
    if suffix in _MARKDOWN_SUFFIXES:
        return "text/markdown"
    if suffix in _JSON_SUFFIXES:
        return "application/json"
    if suffix in _YAML_SUFFIXES:
        return "application/yaml"
    if suffix in _CODE_SUFFIXES:
        return "text/code"
    if (
        suffix in _PLAIN_SUFFIXES
        or name in _EXTENSIONLESS_NAMES
        or (not suffix and name.startswith(_EXTENSIONLESS_PREFIXES))
    ):
        return "text/plain"
    if not suffix:
        return "text/code"
    raise NormalizationError(f"document path {path!r} has an unsupported content type")


def _validate_bounded_text(value: object, field: str, maximum_bytes: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise NormalizationError(f"{field} must be a non-blank string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise NormalizationError(f"{field} is not valid UTF-8") from error
    if len(encoded) > maximum_bytes:
        raise NormalizationError(f"{field} exceeds its {maximum_bytes}-byte bound")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise NormalizationError(f"{field} contains a control character")


def _policy_limits(source_inventory: Mapping[str, object]) -> tuple[int, int]:
    defaults = cast(Mapping[str, object], source_inventory["defaults"])
    return cast(int, defaults["max_file_bytes"]), cast(int, defaults["max_repository_bytes"])


def _source_policy_digest(source_inventory: Mapping[str, object]) -> str:
    try:
        canonical = _canonical_json(source_inventory)
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise NormalizationError("source inventory cannot be canonically encoded") from error
    return _sha256(canonical)


def _document_identity_bytes(
    source: ResolvedRevision,
    path: str,
    content_type: ContentType,
    content_digest: str,
) -> bytes:
    return _canonical_json(
        {
            "api_version": _DOCUMENT_ID_API_VERSION,
            "content_digest": content_digest,
            "content_type": content_type,
            "kind": "NormalizedDocumentIdentity",
            "path": path,
            "source": {
                "authority": source.authority,
                "commit": source.commit,
                "ref_kind": source.ref_kind,
                "repository": source.repository,
                "repository_url": source.repository_url,
                "requested_ref": source.requested_ref,
                "source_policy_digest": source.source_policy_digest,
                "version_scope": source.version_scope,
            },
        }
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"
