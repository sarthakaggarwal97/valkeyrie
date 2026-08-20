"""Deterministic structured-record templates and fail-closed exact lookup."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, Literal, TypeAlias
from urllib.parse import quote

from valkeyrie.acquisition import AcquiredFile, AcquiredRepository
from valkeyrie.revisions import ResolvedRevision


class StructuredRecordError(ValueError):
    """A structured record or exact identifier is invalid."""


class DuplicateIdentifierError(StructuredRecordError):
    """An identical exact record was supplied more than once."""


class AmbiguousIdentifierError(StructuredRecordError):
    """One exact identifier resolves to conflicting records."""


class MissingIdentifierError(StructuredRecordError):
    """An exact identifier has no record in the index."""


MAX_REPOSITORY_BYTES: Final = 100
MAX_PATH_BYTES: Final = 4096
MAX_SYMBOL_BYTES: Final = 512
MAX_COMMAND_BYTES: Final = 256
MAX_RELEASE_BYTES: Final = 128
MAX_GITHUB_OBJECT_ID: Final = (1 << 63) - 1

GitHubObjectType: TypeAlias = Literal[
    "repository",
    "issue",
    "pull_request",
    "review",
    "check",
    "workflow_run",
    "release",
]


@dataclass(frozen=True)
class RepositoryIdentifier:
    """A canonical valkey-io repository name."""

    repository: str


@dataclass(frozen=True)
class CommitIdentifier:
    """A full lowercase Git commit SHA."""

    commit: str


@dataclass(frozen=True)
class PathIdentifier:
    """A bounded repository-relative path."""

    repository: str
    path: str


@dataclass(frozen=True)
class SymbolIdentifier:
    """A bounded symbol at an exact repository path."""

    repository: str
    path: str
    symbol: str


@dataclass(frozen=True)
class GitHubObjectIdentifier:
    """A numeric GitHub object scoped by repository and object type."""

    repository: str
    object_type: GitHubObjectType
    object_id: int


@dataclass(frozen=True)
class CommandIdentifier:
    """A canonical uppercase Valkey command identifier."""

    command: str


@dataclass(frozen=True)
class ReleaseArtifactIdentifier:
    """A Valkey release artifact scoped by release and artifact path."""

    release: str
    artifact: str


@dataclass(frozen=True)
class StructuredParsingLimits:
    """Hard bounds for parsing one acquired structured source."""

    max_files: int = 256
    max_file_bytes: int = 1024 * 1024
    max_total_bytes: int = 8 * 1024 * 1024
    max_lines: int = 100_000
    max_line_bytes: int = 8_192
    max_records: int = 10_000


ExactIdentifier: TypeAlias = (
    RepositoryIdentifier
    | CommitIdentifier
    | PathIdentifier
    | SymbolIdentifier
    | GitHubObjectIdentifier
    | CommandIdentifier
    | ReleaseArtifactIdentifier
)


@dataclass(frozen=True)
class RepositoryRecord:
    source: ResolvedRevision
    identifier: RepositoryIdentifier


@dataclass(frozen=True)
class CommitRecord:
    source: ResolvedRevision
    identifier: CommitIdentifier


@dataclass(frozen=True)
class PathRecord:
    source: ResolvedRevision
    identifier: PathIdentifier

    @property
    def provenance_path(self) -> str:
        """Return the exact upstream path represented by this record."""
        return self.identifier.path


@dataclass(frozen=True)
class SymbolRecord:
    source: ResolvedRevision
    identifier: SymbolIdentifier

    @property
    def provenance_path(self) -> str:
        """Return the exact upstream path containing this symbol."""
        return self.identifier.path


@dataclass(frozen=True)
class GitHubObjectRecord:
    source: ResolvedRevision
    identifier: GitHubObjectIdentifier


@dataclass(frozen=True)
class CommandRecord:
    source: ResolvedRevision
    identifier: CommandIdentifier


@dataclass(frozen=True)
class ReleaseArtifactDigestRecord:
    source: ResolvedRevision
    identifier: ReleaseArtifactIdentifier
    digest: str
    provenance_path: str = "README"


StructuredRecord: TypeAlias = (
    RepositoryRecord
    | CommitRecord
    | PathRecord
    | SymbolRecord
    | GitHubObjectRecord
    | CommandRecord
    | ReleaseArtifactDigestRecord
)

_RECORD_API_VERSION = "valkeyrie.io/structured-record/1"
_REPOSITORY = re.compile(r"^[a-z0-9.][a-z0-9._-]*$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMAND = re.compile(r"^[A-Z][A-Z0-9_.-]*(?: [A-Z][A-Z0-9_.-]*)*$")
_RELEASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_VALKEY_RELEASE = r"(?:[0-9]+\.[0-9]+\.[0-9]+(?:-rc[0-9]+)?|unstable)"
_VALKEY_ARTIFACT = re.compile(rf"^valkey-(?P<release>{_VALKEY_RELEASE})\.tar\.gz$")
_VALKEY_HASH_LINE = re.compile(
    rf"^hash (?P<artifact>valkey-(?P<release>{_VALKEY_RELEASE})\.tar\.gz) "
    r"sha256 (?P<digest>[0-9a-f]{64}) (?P<url>https://[^ ]+)$"
)
_SHA256SUM_LINE = re.compile(
    rf"^(?P<digest>[0-9a-f]{{64}})  "
    rf"(?P<artifact>valkey-(?P<release>{_VALKEY_RELEASE})\.tar\.gz)$"
)
_SHA256_PROVENANCE_PATH = re.compile(r"^releases/[A-Za-z0-9][A-Za-z0-9._+-]*\.sha256$")
_AUXILIARY_HASH_PATHS = frozenset({"LICENSE", "LICENSE.md", "LICENSE.txt", "README.md"})
_GITHUB_OBJECT_TYPES = {
    "repository",
    "issue",
    "pull_request",
    "review",
    "check",
    "workflow_run",
    "release",
}
_DEFAULT_PARSING_LIMITS = StructuredParsingLimits()


class ExactLookup:
    """An immutable typed index with no semantic-retrieval fallback."""

    def __init__(self, records: Sequence[StructuredRecord]) -> None:
        indexed: dict[ExactIdentifier, StructuredRecord] = {}
        for record in records:
            _, _, identifier, _ = _record_parts(record)
            existing = indexed.get(identifier)
            if existing is not None:
                if canonical_record_bytes(existing) == canonical_record_bytes(record):
                    raise DuplicateIdentifierError(
                        f"duplicate exact identifier: {_identifier_description(identifier)}"
                    )
                raise AmbiguousIdentifierError(
                    f"ambiguous exact identifier: {_identifier_description(identifier)}"
                )
            indexed[identifier] = record
        self._indexed = indexed
        self._records = tuple(sorted(indexed.values(), key=_record_sort_key))

    @property
    def records(self) -> tuple[StructuredRecord, ...]:
        """Return records in their canonical deterministic order."""
        return self._records

    def lookup(self, identifier: ExactIdentifier) -> StructuredRecord:
        """Resolve one validated typed identifier exactly or fail closed."""
        _identifier_parts(identifier)
        record = self._indexed.get(identifier)
        if record is None:
            raise MissingIdentifierError(
                f"missing exact identifier: {_identifier_description(identifier)}"
            )
        return record


def canonical_record_bytes(record: StructuredRecord) -> bytes:
    """Return the canonical identity preimage for one record.

    The template intentionally has no generation ID, record checksum, or other
    self-derived field.
    """
    record_type, record_id, identifier, value = _record_parts(record)
    template: dict[str, object] = {
        "api_version": _RECORD_API_VERSION,
        "kind": "StructuredRecord",
        "record_id": record_id,
        "record_type": record_type,
        "source": _source_template(record.source),
        "identifier": _identifier_template(identifier),
    }
    provenance = _provenance_template(record)
    if provenance is not None:
        template["provenance"] = provenance
    if value is not None:
        template["value"] = value
    return _canonical_json(template)


def canonical_records_bytes(records: Sequence[StructuredRecord]) -> bytes:
    """Return a canonical, input-order-independent record collection preimage."""
    index = ExactLookup(records)
    return b"[" + b",".join(canonical_record_bytes(record) for record in index.records) + b"]"


def record_checksum(record: StructuredRecord) -> str:
    """Hash the canonical record preimage without inserting the hash into it."""
    return f"sha256:{hashlib.sha256(canonical_record_bytes(record)).hexdigest()}"


def build_release_artifact_digest_records(
    acquired: AcquiredRepository,
    source: ResolvedRevision,
    *,
    limits: StructuredParsingLimits = _DEFAULT_PARSING_LIMITS,
) -> tuple[ReleaseArtifactDigestRecord, ...]:
    """Build exact release digests from bounded acquired ``valkey-hashes`` bytes.

    Only the repository's actual canonical ``README`` hash lines and canonical
    two-space sha256sum lines under ``releases/*.sha256`` are accepted. The
    exact source commit and provenance path in every record bind the parsed
    value to immutable upstream content without a network read or fallback.
    """
    _validate_parsing_limits(limits)
    _validate_source(source)
    _validate_valkey_hashes_source(source)
    if not isinstance(acquired, AcquiredRepository):
        raise StructuredRecordError("structured acquisition has the wrong runtime type")
    if acquired.repository != source.repository or acquired.commit != source.commit:
        raise StructuredRecordError("structured acquisition does not match its exact revision")
    if not isinstance(acquired.files, tuple):
        raise StructuredRecordError("structured acquisition files must be an immutable tuple")
    if not 1 <= len(acquired.files) <= limits.max_files:
        raise StructuredRecordError("structured acquisition file count is outside its bound")

    records: dict[ReleaseArtifactIdentifier, ReleaseArtifactDigestRecord] = {}
    seen: dict[ReleaseArtifactIdentifier, str] = {}
    ambiguous: set[ReleaseArtifactIdentifier] = set()
    seen_paths: set[str] = set()
    paths: list[str] = []
    total_bytes = 0
    total_lines = 0
    for acquired_file in acquired.files:
        if not isinstance(acquired_file, AcquiredFile):
            raise StructuredRecordError("structured acquisition contains a malformed file")
        path = acquired_file.path
        _validate_path(path, "structured provenance path")
        if path in seen_paths:
            raise StructuredRecordError(f"duplicate structured source path: {path}")
        seen_paths.add(path)
        paths.append(path)
        content = acquired_file.content
        if not isinstance(content, bytes):
            raise StructuredRecordError("structured source content must be bytes")
        if len(content) > limits.max_file_bytes:
            raise StructuredRecordError("structured source file exceeds its byte bound")
        total_bytes += len(content)
        if total_bytes > limits.max_total_bytes:
            raise StructuredRecordError("structured source exceeds its total-byte bound")

        format_name = _structured_provenance_format(path)
        if format_name is None:
            if path not in _AUXILIARY_HASH_PATHS:
                raise StructuredRecordError(f"unreviewed structured source path: {path}")
            continue
        lines = _decode_structured_lines(content, path, limits)
        total_lines += len(lines)
        if total_lines > limits.max_lines:
            raise StructuredRecordError("structured source exceeds its line-count bound")
        for line_number, line in enumerate(lines, 1):
            parsed = _parse_digest_line(format_name, line, path, line_number)
            if parsed is None:
                continue
            release, artifact, digest = parsed
            identifier = ReleaseArtifactIdentifier(release, artifact)
            previous = seen.get(identifier)
            if previous is not None:
                if previous == digest:
                    raise DuplicateIdentifierError(
                        f"duplicate exact identifier: {_identifier_description(identifier)}"
                    )
                ambiguous.add(identifier)
                records.pop(identifier, None)
                continue
            seen[identifier] = digest
            if identifier in ambiguous:
                continue
            record = ReleaseArtifactDigestRecord(source, identifier, digest, path)
            _record_parts(record)
            records[identifier] = record
            if len(records) > limits.max_records:
                raise StructuredRecordError("structured record count exceeds its bound")

    if paths != sorted(paths):
        raise StructuredRecordError("structured acquisition files are not in lexical order")
    if (
        not isinstance(acquired.total_bytes, int)
        or isinstance(acquired.total_bytes, bool)
        or acquired.total_bytes != total_bytes
    ):
        raise StructuredRecordError("structured acquisition total_bytes is inconsistent")
    if not records:
        raise StructuredRecordError("structured source contains no unambiguous sha256 records")
    return tuple(sorted(records.values(), key=_record_sort_key))


def _record_parts(
    record: StructuredRecord,
) -> tuple[str, str, ExactIdentifier, dict[str, object] | None]:
    if not isinstance(
        record,
        (
            RepositoryRecord,
            CommitRecord,
            PathRecord,
            SymbolRecord,
            GitHubObjectRecord,
            CommandRecord,
            ReleaseArtifactDigestRecord,
        ),
    ):
        raise StructuredRecordError("unsupported structured record type")
    _validate_record_pair(record)
    _validate_source(record.source)
    identifier_type, identifier_value = _identifier_parts(record.identifier)
    _validate_record_source(record)
    record_id = _record_id(identifier_type, identifier_value)
    value: dict[str, object] | None = None
    if isinstance(record, ReleaseArtifactDigestRecord):
        _validate_digest(record.digest, "release artifact digest")
        _validate_reviewed_provenance_path(record.provenance_path)
        if record.provenance_path == record.identifier.artifact:
            raise StructuredRecordError(
                "release provenance path must be distinct from the artifact name"
            )
        value = {"digest": record.digest}
    return identifier_type, record_id, record.identifier, value


def _identifier_parts(identifier: ExactIdentifier) -> tuple[str, tuple[str | int, ...]]:
    if isinstance(identifier, RepositoryIdentifier):
        _validate_repository(identifier.repository)
        return "repository", (identifier.repository,)
    if isinstance(identifier, CommitIdentifier):
        _validate_sha(identifier.commit, "commit identifier")
        return "commit", (identifier.commit,)
    if isinstance(identifier, PathIdentifier):
        _validate_repository(identifier.repository)
        _validate_path(identifier.path, "path")
        return "path", (identifier.repository, identifier.path)
    if isinstance(identifier, SymbolIdentifier):
        _validate_repository(identifier.repository)
        _validate_path(identifier.path, "symbol path")
        _validate_bounded_text(identifier.symbol, "symbol", MAX_SYMBOL_BYTES)
        return "symbol", (identifier.repository, identifier.path, identifier.symbol)
    if isinstance(identifier, GitHubObjectIdentifier):
        _validate_repository(identifier.repository)
        if (
            not isinstance(identifier.object_type, str)
            or identifier.object_type not in _GITHUB_OBJECT_TYPES
        ):
            raise StructuredRecordError("unsupported GitHub object type")
        if (
            not isinstance(identifier.object_id, int)
            or isinstance(identifier.object_id, bool)
            or not 1 <= identifier.object_id <= MAX_GITHUB_OBJECT_ID
        ):
            raise StructuredRecordError("GitHub object ID must be a bounded positive integer")
        return "github_object", (
            identifier.repository,
            identifier.object_type,
            identifier.object_id,
        )
    if isinstance(identifier, CommandIdentifier):
        _validate_bounded_text(identifier.command, "command", MAX_COMMAND_BYTES)
        if _COMMAND.fullmatch(identifier.command) is None:
            raise StructuredRecordError("command identifier must use canonical uppercase tokens")
        return "command", (identifier.command,)
    if isinstance(identifier, ReleaseArtifactIdentifier):
        _validate_bounded_text(identifier.release, "release", MAX_RELEASE_BYTES)
        if _RELEASE.fullmatch(identifier.release) is None:
            raise StructuredRecordError("release identifier is malformed")
        _validate_path(identifier.artifact, "release artifact")
        return "release_artifact_digest", (identifier.release, identifier.artifact)
    raise StructuredRecordError("unsupported exact identifier type")


def _validate_record_source(record: StructuredRecord) -> None:
    source = record.source
    identifier = record.identifier
    if isinstance(identifier, (RepositoryIdentifier, PathIdentifier, SymbolIdentifier)):
        if identifier.repository != source.repository:
            raise StructuredRecordError("record identifier repository does not match its source")
    elif isinstance(identifier, CommitIdentifier):
        if identifier.commit != source.commit:
            raise StructuredRecordError("commit identifier does not match its source revision")
    elif isinstance(identifier, GitHubObjectIdentifier):
        if identifier.repository != source.repository:
            raise StructuredRecordError("GitHub object repository does not match its source")
    elif isinstance(identifier, CommandIdentifier):
        if source.repository != "valkey" or source.authority != "canonical":
            raise StructuredRecordError(
                "Valkey command records require the canonical valkey source"
            )
    elif isinstance(identifier, ReleaseArtifactIdentifier):
        _validate_valkey_hashes_source(source)


def _validate_record_pair(record: StructuredRecord) -> None:
    valid_pair = (
        isinstance(record, RepositoryRecord)
        and isinstance(record.identifier, RepositoryIdentifier)
        or isinstance(record, CommitRecord)
        and isinstance(record.identifier, CommitIdentifier)
        or isinstance(record, PathRecord)
        and isinstance(record.identifier, PathIdentifier)
        or isinstance(record, SymbolRecord)
        and isinstance(record.identifier, SymbolIdentifier)
        or isinstance(record, GitHubObjectRecord)
        and isinstance(record.identifier, GitHubObjectIdentifier)
        or isinstance(record, CommandRecord)
        and isinstance(record.identifier, CommandIdentifier)
        or isinstance(record, ReleaseArtifactDigestRecord)
        and isinstance(record.identifier, ReleaseArtifactIdentifier)
    )
    if not valid_pair:
        raise StructuredRecordError("structured record and identifier types do not match")


def _validate_source(source: ResolvedRevision) -> None:
    if not isinstance(source, ResolvedRevision):
        raise StructuredRecordError("record source must be a resolved revision")
    _validate_repository(source.repository)
    expected_url = f"https://github.com/valkey-io/{source.repository}"
    if source.repository_url != expected_url:
        raise StructuredRecordError("record source repository URL is not canonical")
    _validate_bounded_text(source.requested_ref, "requested ref", 255)
    if not isinstance(source.ref_kind, str) or source.ref_kind not in {
        "branch",
        "tag",
        "commit",
    }:
        raise StructuredRecordError("record source ref kind is invalid")
    _validate_sha(source.commit, "record source revision")
    if not isinstance(source.authority, str) or source.authority not in {
        "canonical",
        "secondary",
        "structured",
    }:
        raise StructuredRecordError("record source authority is invalid")
    _validate_bounded_text(source.version_scope, "version scope", 256)
    _validate_digest(source.source_policy_digest, "source policy digest")


def _validate_repository(repository: str) -> None:
    _validate_bounded_text(repository, "repository", MAX_REPOSITORY_BYTES)
    if _REPOSITORY.fullmatch(repository) is None:
        raise StructuredRecordError("repository identifier is malformed")


def _validate_path(path: str, field: str) -> None:
    _validate_bounded_text(path, field, MAX_PATH_BYTES)
    candidate = PurePosixPath(path)
    parts = path.split("/")
    if candidate.is_absolute() or "\\" in path or any(part in {"", ".", ".."} for part in parts):
        raise StructuredRecordError(f"{field} must be a safe repository-relative path")


def _validate_reviewed_provenance_path(path: str) -> None:
    _validate_path(path, "release provenance path")
    if path != "README" and _SHA256_PROVENANCE_PATH.fullmatch(path) is None:
        raise StructuredRecordError("release provenance path is not a reviewed digest source")


def _validate_bounded_text(value: str, field: str, maximum_bytes: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StructuredRecordError(f"{field} must be a non-blank string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise StructuredRecordError(f"{field} is not valid UTF-8") from error
    if len(encoded) > maximum_bytes:
        raise StructuredRecordError(f"{field} exceeds its {maximum_bytes}-byte bound")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise StructuredRecordError(f"{field} contains a control character")


def _validate_sha(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise StructuredRecordError(f"{field} must be a full lowercase 40-hex SHA")


def _validate_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise StructuredRecordError(f"{field} must be a lowercase sha256 digest")


def _record_id(record_type: str, values: tuple[str | int, ...]) -> str:
    encoded = [
        quote(str(value).lower() if record_type == "command" else str(value), safe="")
        for value in values
    ]
    return ":".join((record_type, *encoded))


def _source_template(source: ResolvedRevision) -> dict[str, object]:
    return {
        "authority": source.authority,
        "commit": source.commit,
        "ref_kind": source.ref_kind,
        "repository": source.repository,
        "repository_url": source.repository_url,
        "requested_ref": source.requested_ref,
        "source_policy_digest": source.source_policy_digest,
        "version_scope": source.version_scope,
    }


def _identifier_template(identifier: ExactIdentifier) -> dict[str, object]:
    if isinstance(identifier, RepositoryIdentifier):
        return {"repository": identifier.repository}
    if isinstance(identifier, CommitIdentifier):
        return {"commit": identifier.commit}
    if isinstance(identifier, PathIdentifier):
        return {"path": identifier.path, "repository": identifier.repository}
    if isinstance(identifier, SymbolIdentifier):
        return {
            "path": identifier.path,
            "repository": identifier.repository,
            "symbol": identifier.symbol,
        }
    if isinstance(identifier, GitHubObjectIdentifier):
        return {
            "object_id": identifier.object_id,
            "object_type": identifier.object_type,
            "repository": identifier.repository,
        }
    if isinstance(identifier, CommandIdentifier):
        return {"command": identifier.command}
    if isinstance(identifier, ReleaseArtifactIdentifier):
        return {"artifact": identifier.artifact, "release": identifier.release}
    raise StructuredRecordError("unsupported exact identifier type")


def _provenance_template(record: StructuredRecord) -> dict[str, object] | None:
    path: str | None = None
    if isinstance(record, PathRecord | SymbolRecord):
        path = record.provenance_path
    elif isinstance(record, ReleaseArtifactDigestRecord):
        path = record.provenance_path
    if path is None:
        return None
    return {
        "commit": record.source.commit,
        "path": path,
        "repository": record.source.repository,
    }


def _validate_valkey_hashes_source(source: ResolvedRevision) -> None:
    if (
        source.repository != "valkey-hashes"
        or source.authority != "structured"
        or source.version_scope != "release_artifacts"
    ):
        raise StructuredRecordError(
            "release artifact digests require the structured valkey-hashes source"
        )


def _validate_parsing_limits(limits: object) -> None:
    if not isinstance(limits, StructuredParsingLimits):
        raise StructuredRecordError("structured parsing limits have the wrong runtime type")
    values = (
        limits.max_files,
        limits.max_file_bytes,
        limits.max_total_bytes,
        limits.max_lines,
        limits.max_line_bytes,
        limits.max_records,
    )
    if any(type(value) is not int or value < 1 for value in values):
        raise StructuredRecordError("structured parsing bounds must be positive integers")


def _structured_provenance_format(path: str) -> Literal["hash", "sha256sum"] | None:
    if path == "README":
        return "hash"
    if _SHA256_PROVENANCE_PATH.fullmatch(path) is not None:
        return "sha256sum"
    return None


def _decode_structured_lines(
    content: bytes,
    path: str,
    limits: StructuredParsingLimits,
) -> tuple[str, ...]:
    if content.startswith(b"\xef\xbb\xbf") or b"\r" in content or b"\x00" in content:
        raise StructuredRecordError(f"structured source {path} is not canonical LF-only text")
    try:
        lines = tuple(line.decode("utf-8") for line in content.split(b"\n"))
    except UnicodeDecodeError as error:
        raise StructuredRecordError(f"structured source {path} is not valid UTF-8") from error
    if any(len(line.encode("utf-8")) > limits.max_line_bytes for line in lines):
        raise StructuredRecordError(
            f"structured source {path} contains a line outside its byte bound"
        )
    return lines


def _parse_digest_line(
    format_name: Literal["hash", "sha256sum"],
    line: str,
    path: str,
    line_number: int,
) -> tuple[str, str, str] | None:
    if not line or line.startswith("#"):
        return None
    match = (
        _VALKEY_HASH_LINE.fullmatch(line)
        if format_name == "hash"
        else _SHA256SUM_LINE.fullmatch(line)
    )
    if match is None:
        raise StructuredRecordError(f"malformed structured digest line at {path}:{line_number}")
    release = match.group("release")
    artifact = match.group("artifact")
    if _VALKEY_ARTIFACT.fullmatch(artifact) is None:  # pragma: no cover - regex invariant
        raise StructuredRecordError(f"malformed release artifact at {path}:{line_number}")
    if format_name == "hash":
        expected_url = (
            "https://github.com/valkey-io/valkey/archive/unstable.tar.gz"
            if release == "unstable"
            else f"https://github.com/valkey-io/valkey/archive/refs/tags/{release}.tar.gz"
        )
        if match.group("url") != expected_url:
            raise StructuredRecordError(
                f"non-canonical release artifact URL at {path}:{line_number}"
            )
    return release, artifact, f"sha256:{match.group('digest')}"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _record_sort_key(record: StructuredRecord) -> tuple[str, str, bytes]:
    record_type, record_id, _, _ = _record_parts(record)
    return record_type, record_id, canonical_record_bytes(record)


def _identifier_description(identifier: ExactIdentifier) -> str:
    record_type, values = _identifier_parts(identifier)
    return _record_id(record_type, values)
