"""Deterministic resolution of reviewed GitHub refs to immutable commits."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Literal, cast
from urllib.parse import quote

from valkeyrie.github import GitHubFetcher, GitHubReadError, HttpResponse, fetch_public_github
from valkeyrie.sources import SourceInventoryError, validate_source_inventory


class RevisionError(ValueError):
    """A reviewed source ref could not be resolved safely and deterministically."""


class RevisionChangedError(RevisionError):
    """A reviewed source or mutable ref changed before build verification."""


@dataclass(frozen=True)
class RevisionLimits:
    """Hard bounds for resolving or verifying one source revision."""

    max_requests: int = 8
    max_response_bytes: int = 1024 * 1024
    timeout_seconds: float = 15.0
    max_tag_depth: int = 4


RefKind = Literal["branch", "tag", "commit"]
Authority = Literal["canonical", "secondary", "structured"]


@dataclass(frozen=True)
class ResolvedRevision:
    """One reviewed source pinned to a validated immutable Git commit."""

    repository: str
    repository_url: str
    requested_ref: str
    ref_kind: RefKind
    commit: str
    authority: Authority
    version_scope: str
    source_policy_digest: str


@dataclass(frozen=True)
class _RefTarget:
    kind: Literal["commit", "tag"]
    sha: str


_DEFAULT_LIMITS = RevisionLimits()
_SHA = re.compile(r"^[0-9a-f]{40}$")
_API_ROOT = "https://api.github.com"


def resolve_source_revision(
    source_inventory: Mapping[str, object],
    repository_name: str,
    *,
    fetch: GitHubFetcher | None = None,
    limits: RevisionLimits = _DEFAULT_LIMITS,
    elapsed_clock: Callable[[], float] = monotonic,
) -> ResolvedRevision:
    """Resolve one reviewed static source ref to a full immutable commit."""
    entry = _source_entry(source_inventory, repository_name)
    policy_digest = _source_policy_digest(source_inventory)
    reader = _ReadBudget(fetch or fetch_public_github, limits, elapsed_clock)
    return _resolve_entry(entry, reader, policy_digest)


def verify_source_revision(
    source_inventory: Mapping[str, object],
    resolved: ResolvedRevision,
    *,
    fetch: GitHubFetcher | None = None,
    limits: RevisionLimits = _DEFAULT_LIMITS,
    elapsed_clock: Callable[[], float] = monotonic,
) -> None:
    """Fail if source policy or its selected mutable ref changed after content reads."""
    entry = _source_entry(source_inventory, resolved.repository)
    policy_digest = _source_policy_digest(source_inventory)
    if policy_digest != resolved.source_policy_digest:
        raise RevisionChangedError(f"source policy changed while building {resolved.repository}")

    reader = _ReadBudget(fetch or fetch_public_github, limits, elapsed_clock)
    try:
        current = _resolve_entry(entry, reader, policy_digest)
    except RevisionError as error:
        raise RevisionChangedError(
            f"source ref changed while building {resolved.repository}: {error}"
        ) from error
    if current.ref_kind != resolved.ref_kind or current.commit != resolved.commit:
        raise RevisionChangedError(
            f"source ref changed while building {resolved.repository}: "
            f"{resolved.commit} -> {current.commit}"
        )


class _ReadBudget:
    def __init__(
        self,
        fetch: GitHubFetcher,
        limits: RevisionLimits,
        elapsed_clock: Callable[[], float],
    ) -> None:
        _validate_limits(limits)
        self._fetch = fetch
        self._limits = limits
        self._clock = elapsed_clock
        self._started_at = elapsed_clock()
        self._requests = 0
        self._remaining_bytes = limits.max_response_bytes

    @property
    def max_tag_depth(self) -> int:
        """Return the validated annotated-tag dereference bound."""
        return self._limits.max_tag_depth

    def get_json(self, path: str) -> Mapping[str, object]:
        response = self._request(f"{_API_ROOT}/{path}")
        return _decode_json_object(response.body)

    def get_advertised_refs(self, repository: str) -> dict[str, str]:
        encoded_repository = quote(repository, safe="")
        url = (
            f"https://github.com/valkey-io/{encoded_repository}.git/info/refs"
            "?service=git-upload-pack"
        )
        response = self._request(url)
        content_type = next(
            (
                value
                for name, value in response.headers.items()
                if name.casefold() == "content-type"
            ),
            "",
        )
        if content_type.split(";", 1)[0].strip() != ("application/x-git-upload-pack-advertisement"):
            raise RevisionError("Git ref advertisement has an invalid content type")
        return _parse_ref_advertisement(response.body, repository)

    def _request(self, url: str) -> HttpResponse:
        if self._requests >= self._limits.max_requests:
            raise RevisionError("revision resolution exceeded its request bound")
        if self._remaining_bytes < 1:
            raise RevisionError("revision resolution exceeded its response-byte bound")
        remaining_time = self._limits.timeout_seconds - (self._clock() - self._started_at)
        if remaining_time <= 0:
            raise RevisionError("revision resolution exceeded its time bound")

        self._requests += 1
        try:
            response = self._fetch(url, remaining_time, self._remaining_bytes)
        except GitHubReadError as error:
            raise RevisionError(f"cannot read public GitHub revision: {error}") from error
        if self._clock() - self._started_at > self._limits.timeout_seconds:
            raise RevisionError("revision resolution exceeded its time bound")
        if len(response.body) > self._remaining_bytes:
            raise RevisionError("revision resolution exceeded its response-byte bound")
        self._remaining_bytes -= len(response.body)
        if response.status != 200:
            raise RevisionError(f"public GitHub revision read returned HTTP {response.status}")
        return response


def _validate_limits(limits: RevisionLimits) -> None:
    integers = (limits.max_requests, limits.max_response_bytes, limits.max_tag_depth)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in integers
    ):
        raise RevisionError(
            "revision request, byte, and tag-depth bounds must be positive integers"
        )
    if not isinstance(limits.timeout_seconds, (int, float)) or isinstance(
        limits.timeout_seconds, bool
    ):
        raise RevisionError("revision timeout bound must be numeric")
    if limits.timeout_seconds <= 0 or not math.isfinite(limits.timeout_seconds):
        raise RevisionError("revision timeout bound must be positive and finite")


def _source_entry(
    source_inventory: Mapping[str, object], repository_name: str
) -> dict[str, object]:
    try:
        validate_source_inventory(source_inventory)
    except SourceInventoryError as error:
        raise RevisionError(f"source inventory is invalid: {error}") from error
    repositories = cast(list[dict[str, object]], source_inventory["repositories"])
    entry = next(
        (repository for repository in repositories if repository["name"] == repository_name),
        None,
    )
    if entry is None:
        raise RevisionError(f"unknown reviewed source repository: {repository_name}")
    if entry["classification"] not in {"curated", "structured_exact"}:
        raise RevisionError(f"repository {repository_name} has no static source revision")
    return entry


def _source_policy_digest(source_inventory: Mapping[str, object]) -> str:
    canonical = json.dumps(
        source_inventory,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _resolve_entry(
    entry: Mapping[str, object],
    reader: _ReadBudget,
    source_policy_digest: str,
) -> ResolvedRevision:
    repository = cast(str, entry["name"])
    requested_ref = cast(str, entry["requested_ref"])
    direct_commit = requested_ref.lower()
    if _SHA.fullmatch(direct_commit):
        _validate_commit(reader, repository, direct_commit)
        ref_kind: RefKind = "commit"
        commit = direct_commit
    else:
        refs = reader.get_advertised_refs(repository)
        branch_name = f"refs/heads/{requested_ref}"
        tag_name = f"refs/tags/{requested_ref}"
        branch = refs.get(branch_name)
        tag = refs.get(tag_name)
        if branch is not None and tag is not None:
            raise RevisionError(
                f"repository {repository} ref {requested_ref!r} is both a branch and a tag"
            )
        if branch is None and tag is None:
            raise RevisionError(f"repository {repository} ref {requested_ref!r} is unavailable")
        if branch is not None:
            ref_kind = "branch"
            commit = branch
        else:
            if tag is None:  # pragma: no cover - guarded by the unavailable check
                raise RevisionError("tag resolution invariant failed")
            peeled = refs.get(f"{tag_name}^{{}}")
            if peeled is None:
                commit = tag
            else:
                target = _dereference_tag(reader, repository, _RefTarget("tag", tag))
                if target.sha != peeled:
                    raise RevisionError(
                        f"repository {repository} returned conflicting peeled tag identity"
                    )
                commit = target.sha
            ref_kind = "tag"
            _validate_commit(reader, repository, commit)

    return ResolvedRevision(
        repository=repository,
        repository_url=cast(str, entry["url"]),
        requested_ref=requested_ref,
        ref_kind=ref_kind,
        commit=commit,
        authority=cast(Authority, entry["authority"]),
        version_scope=cast(str, entry["version_scope"]),
        source_policy_digest=source_policy_digest,
    )


def _dereference_tag(reader: _ReadBudget, repository: str, target: _RefTarget) -> _RefTarget:
    seen: set[str] = set()
    depth = 0
    while target.kind == "tag":
        if target.sha in seen:
            raise RevisionError(f"repository {repository} contains an annotated-tag cycle")
        if depth >= reader.max_tag_depth:
            raise RevisionError(f"repository {repository} exceeded annotated-tag depth")
        seen.add(target.sha)
        path = f"repos/valkey-io/{quote(repository, safe='')}/git/tags/{target.sha}"
        value = reader.get_json(path)
        if value.get("sha") != target.sha:
            raise RevisionError(f"repository {repository} returned conflicting tag identity")
        _require_object_url(value.get("url"), repository, "tags", target.sha)
        target = _parse_target(value.get("object"), repository)
        _require_target_url(value.get("object"), repository, target)
        depth += 1
    return target


def _validate_commit(reader: _ReadBudget, repository: str, commit: str) -> None:
    path = f"repos/valkey-io/{quote(repository, safe='')}/git/commits/{commit}"
    value = reader.get_json(path)
    if value.get("sha") != commit:
        raise RevisionError(f"repository {repository} returned conflicting commit identity")
    _require_object_url(value.get("url"), repository, "commits", commit)


class _StrictJsonError(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _StrictJsonError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise _StrictJsonError(f"non-finite JSON constant: {value}")


def _decode_json_object(body: bytes) -> Mapping[str, object]:
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
        raise RevisionError("public GitHub revision response is not strict JSON") from error
    if not isinstance(value, dict):
        raise RevisionError("public GitHub revision response must be an object")
    return cast(Mapping[str, object], value)


def _parse_ref_advertisement(body: bytes, repository: str) -> dict[str, str]:
    offset = 0
    saw_service = False
    refs: dict[str, str] = {}
    while offset < len(body):
        if len(body) - offset < 4:
            raise RevisionError(f"repository {repository} returned a truncated Git packet")
        header = body[offset : offset + 4]
        offset += 4
        try:
            length = int(header, 16)
        except ValueError as error:
            raise RevisionError(
                f"repository {repository} returned an invalid Git packet length"
            ) from error
        if length == 0:
            continue
        if length < 4 or offset + length - 4 > len(body):
            raise RevisionError(f"repository {repository} returned an invalid Git packet")
        payload = body[offset : offset + length - 4]
        offset += length - 4
        line = payload.rstrip(b"\n")
        if line == b"# service=git-upload-pack":
            if saw_service or refs:
                raise RevisionError(
                    f"repository {repository} returned conflicting Git service packets"
                )
            saw_service = True
            continue
        if not saw_service:
            raise RevisionError(
                f"repository {repository} returned refs before the Git service packet"
            )
        if line == b"version 1":
            continue
        line = line.split(b"\0", 1)[0]
        try:
            decoded = line.decode("ascii")
        except UnicodeDecodeError as error:
            raise RevisionError(f"repository {repository} returned a non-ASCII Git ref") from error
        sha, separator, ref_name = decoded.partition(" ")
        if not separator or not _SHA.fullmatch(sha):
            raise RevisionError(f"repository {repository} returned an invalid Git ref")
        if ref_name == "HEAD":
            continue
        if not ref_name.startswith(("refs/heads/", "refs/tags/")):
            continue
        if ref_name in refs:
            raise RevisionError(f"repository {repository} returned a duplicate Git ref")
        refs[ref_name] = sha

    if not saw_service:
        raise RevisionError(f"repository {repository} returned no Git upload-pack service")
    for ref_name in refs:
        if ref_name.endswith("^{}") and ref_name[:-3] not in refs:
            raise RevisionError(f"repository {repository} returned an orphan peeled tag")
    return refs


def _parse_target(value: object, repository: str) -> _RefTarget:
    if not isinstance(value, dict):
        raise RevisionError(f"repository {repository} returned a malformed Git object")
    kind = value.get("type")
    sha = value.get("sha")
    if kind not in {"commit", "tag"} or not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise RevisionError(f"repository {repository} returned an unsupported Git object")
    return _RefTarget(cast(Literal["commit", "tag"], kind), sha)


def _require_target_url(value: object, repository: str, target: _RefTarget) -> None:
    if not isinstance(value, dict):
        raise RevisionError(f"repository {repository} returned a malformed Git object")
    collection = "commits" if target.kind == "commit" else "tags"
    _require_object_url(value.get("url"), repository, collection, target.sha)


def _require_object_url(value: object, repository: str, collection: str, sha: str) -> None:
    expected = f"{_API_ROOT}/repos/valkey-io/{repository}/git/{collection}/{sha}"
    if value != expected:
        raise RevisionError(f"repository {repository} returned a conflicting Git object URL")
