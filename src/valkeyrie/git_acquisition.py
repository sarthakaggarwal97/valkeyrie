from __future__ import annotations

import hashlib
import json
import os
import re

# Git is invoked with a fixed executable, validated argv, and shell=False.
import subprocess  # nosec B404
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

from valkeyrie.acquisition import AcquiredFile, AcquiredRepository, AcquisitionLimits
from valkeyrie.corpus import CorpusFunctions, build_corpus
from valkeyrie.generation import GenerationBundle
from valkeyrie.retrieval_config import FrozenRetrievalConfiguration
from valkeyrie.revisions import ResolvedRevision, resolve_source_revision
from valkeyrie.sources import SourceInventoryError, classify_path, validate_source_inventory


class GitAcquisitionError(ValueError):
    """A locked source set or exact local Git acquisition is invalid."""


class RevisionResolver(Protocol):
    def __call__(
        self, source_inventory: Mapping[str, object], repository_name: str, /
    ) -> ResolvedRevision: ...


class GitRunner(Protocol):
    def __call__(
        self,
        arguments: tuple[str, ...],
        *,
        input_bytes: bytes | None,
        maximum_output_bytes: int,
    ) -> bytes: ...


@dataclass(frozen=True)
class LockedSourceSet:
    """Every reviewed static source pinned under one exact source policy."""

    lock_id: str
    source_policy_digest: str
    revisions: tuple[ResolvedRevision, ...]

    def resolve(
        self, source_inventory: Mapping[str, object], repository_name: str, /
    ) -> ResolvedRevision:
        if _source_policy_digest(source_inventory) != self.source_policy_digest:
            raise GitAcquisitionError("source policy does not match the immutable source lock")
        matches = tuple(item for item in self.revisions if item.repository == repository_name)
        if len(matches) != 1:
            raise GitAcquisitionError(
                f"source lock does not contain exactly one revision for {repository_name!r}"
            )
        return matches[0]


_LOCK_API_VERSION: Final = "valkeyrie.io/corpus-source-lock/1"
_LOCK_KIND: Final = "CorpusSourceLock"
_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA: Final = re.compile(r"^[0-9a-f]{40}$")
_REGULAR_MODES: Final = frozenset({"100644", "100755"})
_MAX_TREE_BYTES: Final = 64 * 1024 * 1024
_DEFAULT_LIMITS = AcquisitionLimits()


def create_source_lock(
    source_inventory: Mapping[str, object],
    *,
    resolve: RevisionResolver = resolve_source_revision,
) -> bytes:
    """Resolve every reviewed static source once and return canonical lock bytes."""
    entries = _reviewed_entries(source_inventory)
    if not callable(resolve):
        raise GitAcquisitionError("source lock resolver is not callable")
    revisions: list[ResolvedRevision] = []
    for entry in entries:
        repository = cast(str, entry["name"])
        try:
            revision = resolve(source_inventory, repository)
        except Exception as error:
            raise GitAcquisitionError(
                f"cannot resolve reviewed source {repository!r}: {error}"
            ) from error
        _validate_locked_revision(source_inventory, entry, revision)
        revisions.append(revision)
    preimage = _lock_preimage(tuple(revisions))
    lock_id = _digest(_canonical_json(preimage))
    return _canonical_json({**preimage, "lock_id": lock_id})


def load_source_lock(source_inventory: Mapping[str, object], document: bytes) -> LockedSourceSet:
    """Load one exact lock and bind it to the complete current reviewed inventory."""
    _reviewed_entries(source_inventory)
    if not isinstance(document, bytes) or not document:
        raise GitAcquisitionError("source lock must be non-empty bytes")
    try:
        value = json.loads(
            document.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, _StrictJsonError) as error:
        raise GitAcquisitionError(f"source lock is not strict canonical JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {
        "api_version",
        "kind",
        "source_policy_digest",
        "repositories",
        "lock_id",
    }:
        raise GitAcquisitionError("source lock has an unknown or missing field")
    if (value.get("api_version"), value.get("kind")) != (_LOCK_API_VERSION, _LOCK_KIND):
        raise GitAcquisitionError("source lock has an incompatible identity")
    if _canonical_json(value) != document:
        raise GitAcquisitionError("source lock is not canonical JSON")
    lock_id = value.get("lock_id")
    preimage = {key: item for key, item in value.items() if key != "lock_id"}
    if not isinstance(lock_id, str) or lock_id != _digest(_canonical_json(preimage)):
        raise GitAcquisitionError("source lock content identity is invalid")
    policy_digest = value.get("source_policy_digest")
    if policy_digest != _source_policy_digest(source_inventory):
        raise GitAcquisitionError("source lock policy identity does not match sources.yaml")
    raw_revisions = value.get("repositories")
    if not isinstance(raw_revisions, list):
        raise GitAcquisitionError("source lock repository list is malformed")
    entries = {cast(str, item["name"]): item for item in _reviewed_entries(source_inventory)}
    revisions: list[ResolvedRevision] = []
    for raw in raw_revisions:
        revision = _revision_from_value(raw)
        entry = entries.get(revision.repository)
        if entry is None:
            raise GitAcquisitionError(
                f"source lock contains unknown reviewed source {revision.repository!r}"
            )
        _validate_locked_revision(source_inventory, entry, revision)
        revisions.append(revision)
    names = tuple(item.repository for item in revisions)
    if names != tuple(sorted(entries)):
        raise GitAcquisitionError("source lock does not contain the exact reviewed source set")
    return LockedSourceSet(lock_id, cast(str, policy_digest), tuple(revisions))


def build_locked_corpus(
    sources_yaml: bytes,
    source_inventory: Mapping[str, object],
    retrieval_config: FrozenRetrievalConfiguration,
    lock_document: bytes,
    cache_root: Path,
    *,
    created_at: str,
    run_git: GitRunner | None = None,
) -> GenerationBundle:
    """Build one complete generation from an immutable lock and exact Git objects."""
    locked = load_source_lock(source_inventory, lock_document)
    acquirer = GitObjectAcquirer(cache_root, run_git=run_git)
    return build_corpus(
        sources_yaml,
        retrieval_config,
        created_at=created_at,
        functions=CorpusFunctions(resolve=locked.resolve, acquire=acquirer.acquire),
    )


class GitObjectAcquirer:
    """Acquire every policy-selected file from a commit-pinned bare Git cache."""

    def __init__(
        self,
        cache_root: Path,
        *,
        run_git: GitRunner | None = None,
        limits: AcquisitionLimits = _DEFAULT_LIMITS,
    ) -> None:
        if not isinstance(cache_root, Path) or not cache_root.is_absolute():
            raise GitAcquisitionError("Git cache root must be an absolute path")
        _validate_limits(limits)
        self._cache_root = cache_root
        self._run_git = run_git or _run_git
        self._limits = limits

    def acquire(
        self, source_inventory: Mapping[str, object], resolved: ResolvedRevision, /
    ) -> AcquiredRepository:
        _reviewed_entries(source_inventory)
        if resolved.source_policy_digest != _source_policy_digest(source_inventory):
            raise GitAcquisitionError("locked revision source policy is stale")
        repository_path = self._repository_path(resolved.repository)
        self._prepare_repository(repository_path, resolved)
        tree = self._run(
            repository_path,
            ("ls-tree", "-r", "-z", "-l", resolved.commit),
            maximum_output_bytes=_MAX_TREE_BYTES,
        )
        selected = _selected_tree(source_inventory, resolved.repository, tree, self._limits)
        batch_input = b"".join(f"{sha}\n".encode("ascii") for _, sha, _ in selected)
        batch_output = self._run(
            repository_path,
            ("cat-file", "--batch"),
            input_bytes=batch_input,
            maximum_output_bytes=self._limits.max_total_bytes + len(selected) * 96,
        )
        contents = _parse_batch(batch_output, selected, self._limits)
        files = tuple(
            AcquiredFile(path, content)
            for (path, _, _), content in zip(selected, contents, strict=True)
        )
        return AcquiredRepository(
            resolved.repository,
            resolved.commit,
            files,
            sum(len(item.content) for item in files),
        )

    def _repository_path(self, repository: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", repository):
            raise GitAcquisitionError("reviewed repository name is unsafe")
        path = self._cache_root / f"{repository}.git"
        if path.parent != self._cache_root:
            raise GitAcquisitionError("reviewed repository escaped the Git cache")
        return path

    def _prepare_repository(self, path: Path, resolved: ResolvedRevision) -> None:
        self._cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.exists():
            if not path.is_dir() or path.is_symlink():
                raise GitAcquisitionError("Git cache entry is not a real directory")
            remote = self._run(path, ("remote", "get-url", "origin"), maximum_output_bytes=4096)
            if remote.decode("utf-8").strip().removesuffix(".git") != resolved.repository_url:
                raise GitAcquisitionError("Git cache remote does not match the reviewed source")
        else:
            clone_url = resolved.repository_url + ".git"
            self._run_git(
                (
                    "clone",
                    "--bare",
                    "--filter=blob:none",
                    "--no-tags",
                    clone_url,
                    str(path),
                ),
                input_bytes=None,
                maximum_output_bytes=1024 * 1024,
            )
        try:
            self._run(
                path,
                ("cat-file", "-e", f"{resolved.commit}^{{commit}}"),
                maximum_output_bytes=4096,
            )
        except GitAcquisitionError:
            self._run(
                path,
                ("fetch", "--no-tags", "origin", resolved.requested_ref),
                maximum_output_bytes=1024 * 1024,
            )
            self._run(
                path,
                ("cat-file", "-e", f"{resolved.commit}^{{commit}}"),
                maximum_output_bytes=4096,
            )

    def _run(
        self,
        repository_path: Path,
        arguments: tuple[str, ...],
        *,
        input_bytes: bytes | None = None,
        maximum_output_bytes: int,
    ) -> bytes:
        return self._run_git(
            (f"--git-dir={repository_path}", *arguments),
            input_bytes=input_bytes,
            maximum_output_bytes=maximum_output_bytes,
        )


def _run_git(
    arguments: tuple[str, ...],
    *,
    input_bytes: bytes | None,
    maximum_output_bytes: int,
) -> bytes:
    if (
        not isinstance(arguments, tuple)
        or not arguments
        or any(not isinstance(item, str) or not item or "\x00" in item for item in arguments)
        or type(maximum_output_bytes) is not int
        or maximum_output_bytes < 1
    ):
        raise GitAcquisitionError("Git command request is malformed")
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        # Arguments are constructed internally after repository/ref/path validation.
        result = subprocess.run(  # nosec B603
            ("git", *arguments),
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=600,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GitAcquisitionError(f"bounded Git command failed: {type(error).__name__}") from error
    if result.returncode != 0:
        raise GitAcquisitionError(f"Git command returned status {result.returncode}")
    if len(result.stdout) > maximum_output_bytes:
        raise GitAcquisitionError("Git command exceeded its output-byte bound")
    return result.stdout


def _selected_tree(
    source_inventory: Mapping[str, object],
    repository: str,
    document: bytes,
    limits: AcquisitionLimits,
) -> tuple[tuple[str, str, int], ...]:
    selected: list[tuple[str, str, int]] = []
    total_bytes = 0
    for raw in document.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, encoded_path = raw.split(b"\t", 1)
            mode, kind, encoded_sha, encoded_size = metadata.split(b" ", 3)
            path = encoded_path.decode("utf-8")
        except (ValueError, UnicodeError) as error:
            raise GitAcquisitionError("Git tree entry is malformed") from error
        if len(encoded_path) > limits.max_path_bytes or not path:
            raise GitAcquisitionError("Git tree path is outside its bound")
        included = classify_path(source_inventory, repository, path) == "include"
        if not included:
            continue
        try:
            sha = encoded_sha.decode("ascii")
            size = int(encoded_size)
            decoded_mode = mode.decode("ascii")
        except (ValueError, UnicodeError) as error:
            raise GitAcquisitionError(
                f"reviewed path {path!r} has malformed Git metadata"
            ) from error
        if decoded_mode not in _REGULAR_MODES or kind != b"blob":
            raise GitAcquisitionError(f"reviewed path {path!r} is not a regular Git blob")
        if _SHA.fullmatch(sha) is None or not 0 <= size <= limits.max_file_bytes:
            raise GitAcquisitionError(f"reviewed path {path!r} has invalid Git metadata")
        total_bytes += size
        if total_bytes > limits.max_total_bytes:
            raise GitAcquisitionError("Git acquisition exceeded its total-byte bound")
        selected.append((path, sha, size))
        if len(selected) > limits.max_files:
            raise GitAcquisitionError("Git acquisition exceeded its file-count bound")
    if not selected:
        raise GitAcquisitionError(f"reviewed source {repository!r} selected no files")
    if tuple(path for path, _, _ in selected) != tuple(sorted(path for path, _, _ in selected)):
        raise GitAcquisitionError("Git tree did not return lexical path order")
    return tuple(selected)


def _parse_batch(
    document: bytes,
    expected: tuple[tuple[str, str, int], ...],
    limits: AcquisitionLimits,
) -> tuple[bytes, ...]:
    offset = 0
    contents: list[bytes] = []
    for path, expected_sha, expected_size in expected:
        newline = document.find(b"\n", offset)
        if newline < 0:
            raise GitAcquisitionError("Git batch response ended before its header")
        try:
            sha, kind, encoded_size = document[offset:newline].decode("ascii").split(" ")
            size = int(encoded_size)
        except (ValueError, UnicodeError) as error:
            raise GitAcquisitionError("Git batch header is malformed") from error
        start = newline + 1
        end = start + size
        if (
            sha != expected_sha
            or kind != "blob"
            or size != expected_size
            or size > limits.max_file_bytes
            or end >= len(document)
            or document[end : end + 1] != b"\n"
        ):
            raise GitAcquisitionError(f"Git batch content conflicts for {path!r}")
        content = document[start:end]
        if b"\0" in content:
            raise GitAcquisitionError(f"reviewed path {path!r} contains binary NUL content")
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise GitAcquisitionError(f"reviewed path {path!r} is not UTF-8") from error
        contents.append(content)
        offset = end + 1
    if offset != len(document):
        raise GitAcquisitionError("Git batch response contains unexpected trailing bytes")
    return tuple(contents)


def _reviewed_entries(source_inventory: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    try:
        validate_source_inventory(source_inventory)
    except SourceInventoryError as error:
        raise GitAcquisitionError(f"source inventory is invalid: {error}") from error
    repositories = cast(list[dict[str, object]], source_inventory["repositories"])
    reviewed = tuple(
        sorted(
            (
                entry
                for entry in repositories
                if entry["classification"] in {"curated", "structured_exact"}
            ),
            key=lambda entry: cast(str, entry["name"]),
        )
    )
    if not reviewed:
        raise GitAcquisitionError("source inventory selects no reviewed static sources")
    return reviewed


def _validate_locked_revision(
    source_inventory: Mapping[str, object],
    entry: Mapping[str, object],
    revision: object,
) -> None:
    if not isinstance(revision, ResolvedRevision):
        raise GitAcquisitionError("source resolver returned the wrong runtime type")
    expected_kind = "commit" if _SHA.fullmatch(cast(str, entry["requested_ref"])) else None
    if (
        revision.repository != entry["name"]
        or revision.repository_url != entry["url"]
        or revision.requested_ref != entry["requested_ref"]
        or (expected_kind is not None and revision.ref_kind != expected_kind)
        or _SHA.fullmatch(revision.commit) is None
        or revision.authority != entry["authority"]
        or revision.version_scope != entry["version_scope"]
        or revision.source_policy_digest != _source_policy_digest(source_inventory)
    ):
        raise GitAcquisitionError(
            f"locked revision does not match reviewed source {entry['name']!r}"
        )


def _lock_preimage(revisions: tuple[ResolvedRevision, ...]) -> dict[str, object]:
    if not revisions:
        raise GitAcquisitionError("source lock cannot be empty")
    policy_digests = {item.source_policy_digest for item in revisions}
    if len(policy_digests) != 1:
        raise GitAcquisitionError("source lock revisions do not share one source policy")
    return {
        "api_version": _LOCK_API_VERSION,
        "kind": _LOCK_KIND,
        "source_policy_digest": next(iter(policy_digests)),
        "repositories": [_revision_value(item) for item in revisions],
    }


def _revision_value(value: ResolvedRevision) -> dict[str, str]:
    return {
        "repository": value.repository,
        "repository_url": value.repository_url,
        "requested_ref": value.requested_ref,
        "ref_kind": value.ref_kind,
        "commit": value.commit,
        "authority": value.authority,
        "version_scope": value.version_scope,
        "source_policy_digest": value.source_policy_digest,
    }


def _revision_from_value(value: object) -> ResolvedRevision:
    fields = {
        "repository",
        "repository_url",
        "requested_ref",
        "ref_kind",
        "commit",
        "authority",
        "version_scope",
        "source_policy_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise GitAcquisitionError("source lock revision has an unknown or missing field")
    if any(not isinstance(value[field], str) or not value[field] for field in fields):
        raise GitAcquisitionError("source lock revision contains malformed text")
    if value["ref_kind"] not in {"branch", "tag", "commit"}:
        raise GitAcquisitionError("source lock revision kind is invalid")
    if value["authority"] not in {"canonical", "secondary", "structured"}:
        raise GitAcquisitionError("source lock revision authority is invalid")
    return ResolvedRevision(
        repository=cast(str, value["repository"]),
        repository_url=cast(str, value["repository_url"]),
        requested_ref=cast(str, value["requested_ref"]),
        ref_kind=cast(object, value["ref_kind"]),  # type: ignore[arg-type]
        commit=cast(str, value["commit"]),
        authority=cast(object, value["authority"]),  # type: ignore[arg-type]
        version_scope=cast(str, value["version_scope"]),
        source_policy_digest=cast(str, value["source_policy_digest"]),
    )


def _source_policy_digest(source_inventory: Mapping[str, object]) -> str:
    try:
        validate_source_inventory(source_inventory)
    except SourceInventoryError as error:
        raise GitAcquisitionError(f"source inventory is invalid: {error}") from error
    return _digest(_canonical_json(source_inventory))


def _validate_limits(limits: object) -> None:
    if not isinstance(limits, AcquisitionLimits):
        raise GitAcquisitionError("Git acquisition limits have the wrong runtime type")
    integers = (
        limits.max_paths,
        limits.max_path_bytes,
        limits.max_files,
        limits.max_file_bytes,
        limits.max_total_bytes,
    )
    if any(type(value) is not int or value < 1 for value in integers):
        raise GitAcquisitionError("Git acquisition limits must be positive integers")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GitAcquisitionError("source lock content is not canonical JSON data") from error


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


class _StrictJsonError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise _StrictJsonError(f"non-finite JSON constant: {value}")
