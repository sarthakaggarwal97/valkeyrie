"""Bounded acquisition of reviewed text files from one exact GitHub commit."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from time import monotonic
from typing import cast
from urllib.parse import quote

from valkeyrie.github import GitHubFetcher, GitHubReadError, HttpResponse, fetch_public_github
from valkeyrie.revisions import (
    ResolvedRevision,
    RevisionError,
    RevisionLimits,
    verify_source_revision,
)
from valkeyrie.sources import (
    SourceInventoryError,
    classify_path,
    validate_source_inventory,
)


class AcquisitionError(ValueError):
    """A reviewed source could not be acquired completely and safely."""


@dataclass(frozen=True)
class AcquisitionLimits:
    """Hard bounds for acquiring one repository at one exact commit."""

    max_paths: int = 100_000
    max_path_bytes: int = 4_096
    max_files: int = 10_000
    max_file_bytes: int = 1024 * 1024
    max_total_bytes: int = 100 * 1024 * 1024
    max_api_requests: int = 10_010
    max_response_bytes: int = 160 * 1024 * 1024
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class AcquiredFile:
    """One policy-selected UTF-8 source file; content remains untrusted data."""

    path: str
    content: bytes


@dataclass(frozen=True)
class AcquiredRepository:
    """A complete, deterministically ordered acquisition from one exact commit."""

    repository: str
    commit: str
    files: tuple[AcquiredFile, ...]
    total_bytes: int


@dataclass(frozen=True)
class _TreeFile:
    path: str
    sha: str
    size: int


_DEFAULT_LIMITS = AcquisitionLimits()
_API_ROOT = "https://api.github.com"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_POLICY_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REGULAR_MODES = frozenset({"100644", "100755"})
_SUPPORTED_SUFFIXES = frozenset(
    {
        ".adoc",
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
        ".csv",
        ".docs",
        ".fish",
        ".go",
        ".gradle",
        ".h",
        ".hh",
        ".hpp",
        ".html",
        ".ini",
        ".in",
        ".install",
        ".java",
        ".jade",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".kts",
        ".lua",
        ".md",
        ".mdx",
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
        ".rst",
        ".scss",
        ".sh",
        ".sql",
        ".swift",
        ".tcl",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_SUPPORTED_EXTENSIONLESS = frozenset(
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
_SUPPORTED_EXTENSIONLESS_PREFIXES = (
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


def acquire_source(
    source_inventory: Mapping[str, object],
    resolved: ResolvedRevision,
    *,
    fetch: GitHubFetcher | None = None,
    limits: AcquisitionLimits = _DEFAULT_LIMITS,
    elapsed_clock: Callable[[], float] = monotonic,
) -> AcquiredRepository:
    """Acquire every reviewed text file at ``resolved.commit`` or fail without a result."""
    _validate_limits(limits)
    entry = _validate_source(source_inventory, resolved)
    policy_file_limit, policy_total_limit = _policy_limits(source_inventory)
    max_file_bytes = min(limits.max_file_bytes, policy_file_limit)
    max_total_bytes = min(limits.max_total_bytes, policy_total_limit)
    budget = _RequestBudget(fetch or fetch_public_github, limits, elapsed_clock)

    repository = cast(str, entry["name"])
    encoded_repository = quote(repository, safe="")
    commit_path = f"repos/valkey-io/{encoded_repository}/git/commits/{resolved.commit}"
    commit = budget.get_json(commit_path, "repository commit")
    tree_sha = _parse_commit_tree(commit, repository, resolved.commit)
    tree_path = f"repos/valkey-io/{encoded_repository}/git/trees/{tree_sha}?recursive=1"
    tree = budget.get_json(tree_path, "repository tree")
    candidates = _parse_tree(
        tree,
        source_inventory,
        repository,
        tree_sha,
        limits,
        max_file_bytes,
        max_total_bytes,
    )

    blobs: dict[str, bytes] = {}
    acquired: list[AcquiredFile] = []
    actual_total = 0
    for candidate in candidates:
        content = blobs.get(candidate.sha)
        if content is None:
            blob_path = f"repos/valkey-io/{encoded_repository}/git/blobs/{candidate.sha}"
            blob = budget.get_json(blob_path, f"blob for {candidate.path}")
            content = _parse_blob(blob, repository, candidate, max_file_bytes)
            blobs[candidate.sha] = content
        elif len(content) != candidate.size:
            raise AcquisitionError(f"conflicting sizes for Git blob {candidate.sha}")
        actual_total += len(content)
        if actual_total > max_total_bytes:
            raise AcquisitionError("source acquisition exceeded its total-byte bound")
        acquired.append(AcquiredFile(candidate.path, content))

    try:
        verify_source_revision(
            source_inventory,
            resolved,
            fetch=budget.fetch,
            limits=RevisionLimits(
                max_requests=limits.max_api_requests,
                max_response_bytes=limits.max_response_bytes,
                timeout_seconds=limits.timeout_seconds,
            ),
            elapsed_clock=elapsed_clock,
        )
    except RevisionError as error:
        raise AcquisitionError(f"source revision verification failed: {error}") from error

    return AcquiredRepository(repository, resolved.commit, tuple(acquired), actual_total)


class _RequestBudget:
    def __init__(
        self,
        fetch: GitHubFetcher,
        limits: AcquisitionLimits,
        elapsed_clock: Callable[[], float],
    ) -> None:
        self._transport = fetch
        self._limits = limits
        self._clock = elapsed_clock
        self._started_at = elapsed_clock()
        self._requests = 0
        self._remaining_bytes = limits.max_response_bytes

    def fetch(self, url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        """Apply one shared budget to acquisition and final revision verification."""
        if self._requests >= self._limits.max_api_requests:
            raise GitHubReadError("source acquisition exceeded its API-request bound")
        if self._remaining_bytes < 1:
            raise GitHubReadError("source acquisition exceeded its response-byte bound")
        remaining_time = self._remaining_time()
        self._requests += 1
        response = self._transport(
            url,
            min(float(timeout_seconds), remaining_time),
            min(max_bytes, self._remaining_bytes),
        )
        if self._clock() - self._started_at > self._limits.timeout_seconds:
            raise GitHubReadError("source acquisition exceeded its time bound")
        if len(response.body) > self._remaining_bytes:
            raise GitHubReadError("source acquisition exceeded its response-byte bound")
        self._remaining_bytes -= len(response.body)
        return response

    def get_json(self, path: str, description: str) -> Mapping[str, object]:
        try:
            response = self.fetch(
                f"{_API_ROOT}/{path}",
                self._remaining_time(),
                self._remaining_bytes,
            )
        except GitHubReadError as error:
            raise AcquisitionError(f"cannot read public GitHub {description}: {error}") from error
        if response.status != 200:
            raise AcquisitionError(f"public GitHub {description} returned HTTP {response.status}")
        return _decode_json_object(response.body, description)

    def _remaining_time(self) -> float:
        remaining = self._limits.timeout_seconds - (self._clock() - self._started_at)
        if remaining <= 0:
            raise GitHubReadError("source acquisition exceeded its time bound")
        return remaining


def _validate_limits(limits: AcquisitionLimits) -> None:
    integers = (
        limits.max_paths,
        limits.max_path_bytes,
        limits.max_files,
        limits.max_file_bytes,
        limits.max_total_bytes,
        limits.max_api_requests,
        limits.max_response_bytes,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in integers
    ):
        raise AcquisitionError("source acquisition bounds must be positive integers")
    timeout = limits.timeout_seconds
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
        or not math.isfinite(timeout)
    ):
        raise AcquisitionError("source acquisition timeout must be positive and finite")


def _validate_source(
    source_inventory: Mapping[str, object], resolved: ResolvedRevision
) -> Mapping[str, object]:
    try:
        validate_source_inventory(source_inventory)
    except SourceInventoryError as error:
        raise AcquisitionError(f"source inventory is invalid: {error}") from error
    if not isinstance(resolved, ResolvedRevision):
        raise AcquisitionError("resolved revision metadata is malformed")

    repositories = cast(list[dict[str, object]], source_inventory["repositories"])
    entry = next(
        (repository for repository in repositories if repository["name"] == resolved.repository),
        None,
    )
    if entry is None:
        raise AcquisitionError(f"unknown reviewed source repository: {resolved.repository}")
    expected = (
        (resolved.repository_url, entry["url"]),
        (resolved.requested_ref, entry["requested_ref"]),
        (resolved.authority, entry["authority"]),
        (resolved.version_scope, entry["version_scope"]),
    )
    if (
        entry["classification"] not in {"curated", "structured_exact"}
        or entry["ingestion_mode"] not in {"documents", "structured_records"}
        or any(actual != wanted for actual, wanted in expected)
        or resolved.ref_kind not in {"branch", "tag", "commit"}
        or not isinstance(resolved.commit, str)
        or _SHA.fullmatch(resolved.commit) is None
        or not isinstance(resolved.source_policy_digest, str)
        or _POLICY_DIGEST.fullmatch(resolved.source_policy_digest) is None
        or resolved.source_policy_digest != _source_policy_digest(source_inventory)
    ):
        raise AcquisitionError("resolved revision metadata conflicts with reviewed source policy")
    return entry


def _policy_limits(source_inventory: Mapping[str, object]) -> tuple[int, int]:
    defaults = cast(Mapping[str, object], source_inventory["defaults"])
    return cast(int, defaults["max_file_bytes"]), cast(int, defaults["max_repository_bytes"])


def _parse_commit_tree(value: Mapping[str, object], repository: str, commit: str) -> str:
    expected_url = f"{_API_ROOT}/repos/valkey-io/{repository}/git/commits/{commit}"
    tree = value.get("tree")
    if value.get("sha") != commit or value.get("url") != expected_url or not isinstance(tree, dict):
        raise AcquisitionError(f"repository {repository} returned conflicting commit identity")
    tree_sha = tree.get("sha")
    if not isinstance(tree_sha, str) or _SHA.fullmatch(tree_sha) is None:
        raise AcquisitionError(f"repository {repository} commit returned an invalid root tree")
    expected_tree_url = f"{_API_ROOT}/repos/valkey-io/{repository}/git/trees/{tree_sha}"
    if tree.get("url") != expected_tree_url:
        raise AcquisitionError(f"repository {repository} commit returned an unsafe root tree URL")
    return tree_sha


def _source_policy_digest(source_inventory: Mapping[str, object]) -> str:
    canonical = json.dumps(
        source_inventory,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _parse_tree(
    value: Mapping[str, object],
    source_inventory: Mapping[str, object],
    repository: str,
    tree_sha: str,
    limits: AcquisitionLimits,
    max_file_bytes: int,
    max_total_bytes: int,
) -> tuple[_TreeFile, ...]:
    expected_url = f"{_API_ROOT}/repos/valkey-io/{repository}/git/trees/{tree_sha}"
    if value.get("sha") != tree_sha or value.get("url") != expected_url:
        raise AcquisitionError(f"repository {repository} returned conflicting tree identity")
    if value.get("truncated") is not False:
        raise AcquisitionError(f"repository {repository} returned a truncated or malformed tree")
    raw_entries = value.get("tree")
    if not isinstance(raw_entries, list):
        raise AcquisitionError(f"repository {repository} tree entries must be an array")
    if len(raw_entries) > limits.max_paths:
        raise AcquisitionError("source acquisition exceeded its path-count bound")

    seen_paths: set[str] = set()
    selected: list[_TreeFile] = []
    selected_bytes = 0
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise AcquisitionError(f"repository {repository} tree contains a non-object entry")
        entry = cast(Mapping[str, object], raw_entry)
        path = _tree_path(entry.get("path"), limits.max_path_bytes)
        if path in seen_paths:
            raise AcquisitionError(f"repository {repository} tree repeats path {path!r}")
        seen_paths.add(path)
        kind = entry.get("type")
        mode = entry.get("mode")
        sha = entry.get("sha")
        if not isinstance(sha, str) or _SHA.fullmatch(sha) is None:
            raise AcquisitionError(f"repository {repository} path {path!r} has an invalid Git SHA")
        collection = "blobs" if kind == "blob" else "trees" if kind == "tree" else "commits"
        expected_entry_url = f"{_API_ROOT}/repos/valkey-io/{repository}/git/{collection}/{sha}"
        if entry.get("url") != expected_entry_url:
            raise AcquisitionError(
                f"repository {repository} path {path!r} has an unsafe object URL"
            )

        try:
            included = classify_path(source_inventory, repository, path) == "include"
        except SourceInventoryError as error:  # pragma: no cover - inventory and path checked above
            raise AcquisitionError(f"cannot classify repository path {path!r}: {error}") from error

        if kind == "tree" and mode == "040000" and "size" not in entry:
            continue
        if kind == "commit" and mode == "160000" and "size" not in entry:
            if included:
                raise AcquisitionError(f"reviewed path {path!r} is a submodule or gitlink")
            continue
        if kind == "blob" and mode == "120000":
            _entry_size(entry, repository, path)
            if included:
                raise AcquisitionError(f"reviewed path {path!r} is a symlink or link target")
            continue
        if kind != "blob" or mode not in _REGULAR_MODES:
            raise AcquisitionError(
                f"repository {repository} path {path!r} has an unsupported Git type or mode"
            )
        size = _entry_size(entry, repository, path)
        if not included:
            continue
        if not _supported_file_type(path):
            raise AcquisitionError(f"reviewed path {path!r} has an unsupported file type")
        if size > max_file_bytes:
            raise AcquisitionError(f"reviewed path {path!r} exceeded its file-byte bound")
        selected_bytes += size
        if selected_bytes > max_total_bytes:
            raise AcquisitionError("source acquisition exceeded its total-byte bound")
        selected.append(_TreeFile(path, sha, size))
        if len(selected) > limits.max_files:
            raise AcquisitionError("source acquisition exceeded its file-count bound")

    return tuple(sorted(selected, key=lambda item: item.path))


def _tree_path(value: object, max_path_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise AcquisitionError("repository tree path must be a non-empty string")
    path = PurePosixPath(value)
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise AcquisitionError(f"repository tree path is not valid UTF-8: {value!r}") from error
    if (
        len(encoded) > max_path_bytes
        or path.is_absolute()
        or path.as_posix() != value
        or value in {".", ".."}
        or ".." in path.parts
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AcquisitionError(f"unsafe repository tree path: {value!r}")
    return value


def _entry_size(entry: Mapping[str, object], repository: str, path: str) -> int:
    size = entry.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise AcquisitionError(f"repository {repository} path {path!r} has an invalid size")
    return size


def _supported_file_type(path: str) -> bool:
    name = PurePosixPath(path).name.casefold()
    suffix = PurePosixPath(path).suffix.casefold()
    return (
        suffix in _SUPPORTED_SUFFIXES
        or not suffix
        or name in _SUPPORTED_EXTENSIONLESS
        or (not suffix and name.startswith(_SUPPORTED_EXTENSIONLESS_PREFIXES))
    )


def _parse_blob(
    value: Mapping[str, object],
    repository: str,
    expected: _TreeFile,
    max_file_bytes: int,
) -> bytes:
    expected_url = f"{_API_ROOT}/repos/valkey-io/{repository}/git/blobs/{expected.sha}"
    size = value.get("size")
    content = value.get("content")
    if (
        value.get("sha") != expected.sha
        or value.get("url") != expected_url
        or value.get("encoding") != "base64"
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(content, str)
    ):
        raise AcquisitionError(f"reviewed path {expected.path!r} returned malformed blob metadata")
    if size != expected.size or size > max_file_bytes:
        raise AcquisitionError(f"reviewed path {expected.path!r} returned a conflicting blob size")
    try:
        encoded = content.replace("\n", "").replace("\r", "").encode("ascii")
        decoded = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error) as error:
        raise AcquisitionError(
            f"reviewed path {expected.path!r} returned invalid base64"
        ) from error
    if len(decoded) != size:
        raise AcquisitionError(f"reviewed path {expected.path!r} returned a conflicting blob size")
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(decoded)}\0".encode())
    digest.update(decoded)
    if digest.hexdigest() != expected.sha:
        raise AcquisitionError(f"reviewed path {expected.path!r} returned conflicting blob content")
    if b"\0" in decoded:
        raise AcquisitionError(f"reviewed path {expected.path!r} contains NUL or binary content")
    try:
        decoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AcquisitionError(f"reviewed path {expected.path!r} is not UTF-8 text") from error
    return decoded


class _StrictJsonError(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            raise _StrictJsonError(f"duplicate JSON key: {key}")
        result[key] = item
    return result


def _reject_json_constant(value: str) -> object:
    raise _StrictJsonError(f"non-finite JSON constant: {value}")


def _decode_json_object(body: bytes, description: str) -> Mapping[str, object]:
    try:
        value = cast(
            object,
            json.loads(
                body.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _StrictJsonError) as error:
        raise AcquisitionError(f"public GitHub {description} is not strict JSON") from error
    if not isinstance(value, dict):
        raise AcquisitionError(f"public GitHub {description} must be an object")
    return cast(Mapping[str, object], value)
