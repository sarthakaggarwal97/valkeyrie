from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from valkeyrie.github import HttpResponse
from valkeyrie.revisions import (
    RevisionChangedError,
    RevisionError,
    RevisionLimits,
    resolve_source_revision,
    verify_source_revision,
)
from valkeyrie.sources import load_source_inventory

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources.yaml"
COMMIT_1 = "1" * 40
COMMIT_2 = "2" * 40
TAG_1 = "a" * 40
TAG_2 = "b" * 40
ADVERTISEMENT_TYPE = "application/x-git-upload-pack-advertisement"


class StubFetcher:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, float, int]] = []

    def __call__(self, url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        self.calls.append((url, timeout_seconds, max_bytes))
        return self.responses[len(self.calls) - 1]


class SourceAdvertisementFetcher:
    def __init__(self, refs: dict[str, str]) -> None:
        self.refs = refs
        self.calls: list[str] = []

    def __call__(self, url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        del timeout_seconds, max_bytes
        self.calls.append(url)
        repository = url.split("/valkey-io/", 1)[1].split(".git/", 1)[0]
        return _advertisement([(f"refs/heads/{self.refs[repository]}", COMMIT_1)])


def _response(value: object, status: int = 200) -> HttpResponse:
    return HttpResponse(status, {"content-type": "application/json"}, json.dumps(value).encode())


def _object_url(repository: str, kind: str, sha: str) -> str:
    collection = "commits" if kind == "commit" else "tags"
    return f"https://api.github.com/repos/valkey-io/{repository}/git/{collection}/{sha}"


def _commit_response(
    sha: str, *, repository: str = "valkey", url: str | None = None
) -> HttpResponse:
    return _response({"sha": sha, "url": url or _object_url(repository, "commit", sha)})


def _tag_response(
    tag_sha: str,
    target_sha: str,
    *,
    target_kind: str = "commit",
    repository: str = "valkey",
    response_sha: str | None = None,
    target_url: str | None = None,
) -> HttpResponse:
    return _response(
        {
            "sha": response_sha or tag_sha,
            "url": _object_url(repository, "tag", tag_sha),
            "object": {
                "type": target_kind,
                "sha": target_sha,
                "url": target_url or _object_url(repository, target_kind, target_sha),
            },
        }
    )


def _packet(payload: bytes) -> bytes:
    return f"{len(payload) + 4:04x}".encode() + payload


def _advertisement(
    refs: list[tuple[str, str]],
    *,
    service: bool = True,
    content_type: str = ADVERTISEMENT_TYPE,
) -> HttpResponse:
    body = _packet(b"# service=git-upload-pack\n") + b"0000" if service else b""
    for index, (ref_name, sha) in enumerate(refs):
        capabilities = b"\0multi_ack" if index == 0 else b""
        body += _packet(f"{sha} {ref_name}".encode() + capabilities + b"\n")
    body += b"0000"
    return HttpResponse(200, {"Content-Type": content_type}, body)


def _raw_advertisement(body: bytes) -> HttpResponse:
    return HttpResponse(200, {"content-type": ADVERTISEMENT_TYPE}, body)


def _source(requested_ref: str = "unstable") -> dict[str, object]:
    document = deepcopy(load_source_inventory(SOURCES))
    _repository(document)["requested_ref"] = requested_ref
    return document


def _repository(document: dict[str, object], name: str = "valkey") -> dict[str, object]:
    repositories = cast(list[dict[str, object]], document["repositories"])
    return next(repository for repository in repositories if repository["name"] == name)


def _branch_advertisement(commit: str = COMMIT_1, requested_ref: str = "unstable") -> HttpResponse:
    return _advertisement([(f"refs/heads/{requested_ref}", commit)])


def test_resolves_reviewed_branch_with_one_credential_free_git_get() -> None:
    fetcher = StubFetcher([_branch_advertisement()])

    resolved = resolve_source_revision(_source(), "valkey", fetch=fetcher)

    assert resolved.repository == "valkey"
    assert resolved.repository_url == "https://github.com/valkey-io/valkey"
    assert resolved.requested_ref == "unstable"
    assert resolved.ref_kind == "branch"
    assert resolved.commit == COMMIT_1
    assert resolved.authority == "canonical"
    assert resolved.version_scope == "branches_tags_and_releases"
    assert resolved.source_policy_digest.startswith("sha256:")
    assert len(resolved.source_policy_digest) == 71
    assert [call[0] for call in fetcher.calls] == [
        "https://github.com/valkey-io/valkey.git/info/refs?service=git-upload-pack"
    ]
    assert fetcher.calls[0][1] <= 15.0
    assert fetcher.calls[0][2] == 1024 * 1024


def test_resolves_lightweight_and_nested_annotated_tags() -> None:
    lightweight = StubFetcher(
        [
            _advertisement([("refs/tags/v9.0", COMMIT_1)]),
            _commit_response(COMMIT_1),
        ]
    )
    resolved = resolve_source_revision(_source("v9.0"), "valkey", fetch=lightweight)
    assert resolved.ref_kind == "tag"
    assert resolved.commit == COMMIT_1

    annotated = StubFetcher(
        [
            _advertisement([("refs/tags/v9.0", TAG_1), ("refs/tags/v9.0^{}", COMMIT_2)]),
            _tag_response(TAG_1, TAG_2, target_kind="tag"),
            _tag_response(TAG_2, COMMIT_2),
            _commit_response(COMMIT_2),
        ]
    )
    resolved = resolve_source_revision(_source("v9.0"), "valkey", fetch=annotated)
    assert resolved.ref_kind == "tag"
    assert resolved.commit == COMMIT_2


def test_resolves_direct_full_sha_with_one_api_get() -> None:
    fetcher = StubFetcher([_commit_response(COMMIT_1)])

    resolved = resolve_source_revision(_source(COMMIT_1.upper()), "valkey", fetch=fetcher)

    assert resolved.ref_kind == "commit"
    assert resolved.commit == COMMIT_1
    assert fetcher.calls[0][0].endswith(f"/git/commits/{COMMIT_1}")


def test_branch_tag_version_ambiguity_fails_even_at_the_same_commit() -> None:
    response = _advertisement([("refs/heads/8.1", COMMIT_1), ("refs/tags/8.1", COMMIT_1)])
    with pytest.raises(RevisionError, match="both a branch and a tag"):
        resolve_source_revision(_source("8.1"), "valkey", fetch=StubFetcher([response]))


def test_unavailable_ref_fails_without_fallback() -> None:
    with pytest.raises(RevisionError, match="is unavailable"):
        resolve_source_revision(
            _source("missing"), "valkey", fetch=StubFetcher([_advertisement([])])
        )


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (HttpResponse(500, {}, b""), "HTTP 500"),
        (_advertisement([], content_type="application/json"), "invalid content type"),
        (_raw_advertisement(b"zzzz"), "invalid Git packet length"),
        (
            _raw_advertisement(_packet(f"{COMMIT_1} refs/heads/unstable\n".encode())),
            "refs before the Git service packet",
        ),
        (
            _advertisement([("refs/heads/unstable", "A" * 40)]),
            "invalid Git ref",
        ),
        (
            _advertisement(
                [
                    ("refs/heads/unstable", COMMIT_1),
                    ("refs/heads/unstable", COMMIT_2),
                ]
            ),
            "duplicate Git ref",
        ),
        (
            _advertisement([("refs/tags/v9.0^{}", COMMIT_1)]),
            "orphan peeled tag",
        ),
    ],
)
def test_rejects_http_content_type_and_malformed_ref_advertisements(
    response: HttpResponse, error: str
) -> None:
    with pytest.raises(RevisionError, match=error):
        resolve_source_revision(_source(), "valkey", fetch=StubFetcher([response]))


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        (
            b'{"sha":"2222222222222222222222222222222222222222",'
            b'"sha":"1111111111111111111111111111111111111111",'
            b'"url":"https://api.github.com/repos/valkey-io/valkey/git/commits/'
            + COMMIT_1.encode()
            + b'"}'
        ),
        (
            b'{"sha":"1111111111111111111111111111111111111111",'
            b'"url":"https://api.github.com/repos/valkey-io/valkey/git/commits/'
            + COMMIT_1.encode()
            + b'","unexpected":NaN}'
        ),
    ],
)
def test_rejects_non_strict_commit_json(body: bytes) -> None:
    with pytest.raises(RevisionError, match="strict JSON|must be an object"):
        resolve_source_revision(
            _source(COMMIT_1),
            "valkey",
            fetch=StubFetcher([HttpResponse(200, {}, body)]),
        )


def test_rejects_duplicate_keys_in_tag_json() -> None:
    duplicate_tag = (
        b'{"sha":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        b'"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"url":"https://api.github.com/repos/valkey-io/valkey/git/tags/'
        + TAG_1.encode()
        + b'","object":{"type":"commit","sha":"'
        + COMMIT_1.encode()
        + b'","url":"https://api.github.com/repos/valkey-io/valkey/git/commits/'
        + COMMIT_1.encode()
        + b'"}}'
    )
    responses = [
        _advertisement([("refs/tags/v9.0", TAG_1), ("refs/tags/v9.0^{}", COMMIT_1)]),
        HttpResponse(200, {}, duplicate_tag),
    ]
    with pytest.raises(RevisionError, match="strict JSON"):
        resolve_source_revision(_source("v9.0"), "valkey", fetch=StubFetcher(responses))


@pytest.mark.parametrize(
    ("commit_response", "error"),
    [
        (_commit_response(COMMIT_2), "conflicting commit identity"),
        (
            _commit_response(COMMIT_1, url=_object_url("other", "commit", COMMIT_1)),
            "conflicting Git object URL",
        ),
    ],
)
def test_rejects_conflicting_commit_identity_and_repository_url(
    commit_response: HttpResponse, error: str
) -> None:
    with pytest.raises(RevisionError, match=error):
        resolve_source_revision(_source(COMMIT_1), "valkey", fetch=StubFetcher([commit_response]))


def test_rejects_tag_cycles_depth_identity_urls_and_peeled_conflicts() -> None:
    advertisement = _advertisement([("refs/tags/v9.0", TAG_1), ("refs/tags/v9.0^{}", COMMIT_1)])
    cycle = [advertisement, _tag_response(TAG_1, TAG_1, target_kind="tag")]
    with pytest.raises(RevisionError, match="annotated-tag cycle"):
        resolve_source_revision(_source("v9.0"), "valkey", fetch=StubFetcher(cycle))

    depth = [advertisement, _tag_response(TAG_1, TAG_2, target_kind="tag")]
    with pytest.raises(RevisionError, match="annotated-tag depth"):
        resolve_source_revision(
            _source("v9.0"),
            "valkey",
            fetch=StubFetcher(depth),
            limits=RevisionLimits(max_tag_depth=1),
        )

    identity = [
        advertisement,
        _tag_response(TAG_1, COMMIT_1, response_sha=TAG_2),
    ]
    with pytest.raises(RevisionError, match="conflicting tag identity"):
        resolve_source_revision(_source("v9.0"), "valkey", fetch=StubFetcher(identity))

    wrong_url = [
        advertisement,
        _tag_response(
            TAG_1,
            COMMIT_1,
            target_url=_object_url("other", "commit", COMMIT_1),
        ),
    ]
    with pytest.raises(RevisionError, match="conflicting Git object URL"):
        resolve_source_revision(_source("v9.0"), "valkey", fetch=StubFetcher(wrong_url))

    peeled_conflict = [
        advertisement,
        _tag_response(TAG_1, COMMIT_2),
    ]
    with pytest.raises(RevisionError, match="conflicting peeled tag identity"):
        resolve_source_revision(_source("v9.0"), "valkey", fetch=StubFetcher(peeled_conflict))


def test_enforces_request_response_byte_and_elapsed_time_bounds() -> None:
    tag = _advertisement([("refs/tags/v9.0", COMMIT_1)])
    with pytest.raises(RevisionError, match="request bound"):
        resolve_source_revision(
            _source("v9.0"),
            "valkey",
            fetch=StubFetcher([tag]),
            limits=RevisionLimits(max_requests=1),
        )

    with pytest.raises(RevisionError, match="response-byte bound"):
        resolve_source_revision(
            _source(),
            "valkey",
            fetch=StubFetcher([_branch_advertisement()]),
            limits=RevisionLimits(max_response_bytes=1),
        )

    ticks = iter((0.0, 0.0, 2.0))
    with pytest.raises(RevisionError, match="time bound"):
        resolve_source_revision(
            _source(),
            "valkey",
            fetch=StubFetcher([_branch_advertisement()]),
            limits=RevisionLimits(timeout_seconds=1),
            elapsed_clock=lambda: next(ticks),
        )


@pytest.mark.parametrize(
    "limits",
    [
        RevisionLimits(max_requests=0),
        RevisionLimits(max_response_bytes=0),
        RevisionLimits(timeout_seconds=0),
        RevisionLimits(timeout_seconds=float("nan")),
        RevisionLimits(max_tag_depth=0),
        RevisionLimits(max_requests=cast(int, True)),
    ],
)
def test_rejects_non_bounding_limit_configuration(limits: RevisionLimits) -> None:
    with pytest.raises(RevisionError, match="bound"):
        resolve_source_revision(_source(), "valkey", fetch=StubFetcher([]), limits=limits)


def test_rejects_invalid_unknown_and_non_static_source_entries() -> None:
    with pytest.raises(RevisionError, match="source inventory is invalid"):
        resolve_source_revision({}, "valkey", fetch=StubFetcher([]))
    with pytest.raises(RevisionError, match="unknown reviewed"):
        resolve_source_revision(_source(), "unknown", fetch=StubFetcher([]))
    with pytest.raises(RevisionError, match="no static source revision"):
        resolve_source_revision(_source(), "assets", fetch=StubFetcher([]))


def test_all_43_static_sources_resolve_with_no_rest_api_quota_usage() -> None:
    source = load_source_inventory(SOURCES)
    repositories = cast(list[dict[str, object]], source["repositories"])
    static = [
        repository
        for repository in repositories
        if repository["classification"] in {"curated", "structured_exact"}
    ]
    refs = {
        cast(str, repository["name"]): cast(str, repository["requested_ref"])
        for repository in static
    }
    fetcher = SourceAdvertisementFetcher(refs)

    resolutions = [resolve_source_revision(source, name, fetch=fetcher) for name in sorted(refs)]

    assert len(resolutions) == len(fetcher.calls) == 43
    assert all(url.startswith("https://github.com/") for url in fetcher.calls)
    assert not any("api.github.com" in url for url in fetcher.calls)


def test_verification_accepts_an_unchanged_ref_and_complete_policy() -> None:
    source = _source()
    resolved = resolve_source_revision(
        source, "valkey", fetch=StubFetcher([_branch_advertisement()])
    )
    verify_source_revision(source, resolved, fetch=StubFetcher([_branch_advertisement()]))


def test_verification_rejects_ref_kind_commit_and_ambiguity_changes() -> None:
    source = _source()
    resolved = resolve_source_revision(
        source, "valkey", fetch=StubFetcher([_branch_advertisement()])
    )
    with pytest.raises(RevisionChangedError, match="source ref changed"):
        verify_source_revision(
            source,
            resolved,
            fetch=StubFetcher([_branch_advertisement(COMMIT_2)]),
        )

    changed_kind = _advertisement([("refs/tags/unstable", COMMIT_1)])
    with pytest.raises(RevisionChangedError, match="source ref changed"):
        verify_source_revision(
            source, resolved, fetch=StubFetcher([changed_kind, _commit_response(COMMIT_1)])
        )

    ambiguous = _advertisement(
        [("refs/heads/unstable", COMMIT_1), ("refs/tags/unstable", COMMIT_1)]
    )
    with pytest.raises(RevisionChangedError, match="source ref changed"):
        verify_source_revision(source, resolved, fetch=StubFetcher([ambiguous]))


def test_verification_binds_paths_limits_and_hard_exclusions() -> None:
    source = _source()
    resolved = resolve_source_revision(
        source, "valkey", fetch=StubFetcher([_branch_advertisement()])
    )

    path_changed = deepcopy(source)
    _repository(path_changed)["path_policy"] = "documentation"
    limit_changed = deepcopy(source)
    defaults = cast(dict[str, object], limit_changed["defaults"])
    defaults["max_file_bytes"] = cast(int, defaults["max_file_bytes"]) + 1
    exclusion_changed = deepcopy(source)
    groups = cast(list[dict[str, object]], exclusion_changed["hard_exclusions"])
    cast(list[str], groups[0]["patterns"]).append("new-generated/**")

    for changed in (path_changed, limit_changed, exclusion_changed):
        with pytest.raises(RevisionChangedError, match="source policy changed"):
            verify_source_revision(changed, resolved, fetch=StubFetcher([]))
