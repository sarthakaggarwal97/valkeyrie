from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from valkeyrie.github import GitHubReadError, HttpResponse
from valkeyrie.live_github import (
    ALLOWED_REPOSITORIES,
    MAX_COLLECTION_ITEMS,
    MAX_ENTITY_ID,
    MAX_RESPONSE_BYTES,
    MAX_SEARCH_BODY_BYTES,
    MAX_SEARCH_PER_PAGE,
    MAX_SEARCH_TERMS,
    OWNER,
    PROJECTS_GRAPHQL_QUERY,
    REQUEST_TIMEOUT_SECONDS,
    CheckRunQuery,
    IssueQuery,
    IssueSearchQuery,
    LatestReleaseQuery,
    LiveGitHubError,
    LiveGitHubQuery,
    ProjectQuery,
    PullRequestQuery,
    WorkflowRunQuery,
    infer_live_query,
    normalize_project_response,
    read_live_github,
)
from valkeyrie.request_audit import LiveObservation

SHA = "a" * 40
OTHER_SHA = "b" * 40
OBSERVED = datetime(2026, 8, 19, 23, 58, 28, tzinfo=UTC)
TIMESTAMP = "2026-08-19T23:00:00Z"


def _response(
    value: object,
    *,
    status: int = 200,
    content_type: str = "application/vnd.github+json; charset=utf-8",
) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers={"Content-Type": content_type},
        body=json.dumps(value, separators=(",", ":")).encode(),
    )


def _user() -> dict[str, object]:
    return {"login": "maintainer", "avatar_url": "https://ignored.example/avatar"}


def _pull_request(repository: str = "valkey", number: int = 7) -> dict[str, object]:
    return {
        "id": 1007,
        "number": number,
        "state": "open",
        "title": "Add a feature",
        "body": "Details",
        "draft": False,
        "locked": False,
        "user": _user(),
        "labels": [{"name": "enhancement", "color": "00ff00"}],
        "head": {"sha": SHA, "ref": "feature"},
        "base": {"sha": OTHER_SHA, "ref": "unstable"},
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
        "closed_at": None,
        "merged_at": None,
        "url": f"https://api.github.com/repos/{OWNER}/{repository}/pulls/{number}",
        "html_url": f"https://github.com/{OWNER}/{repository}/pull/{number}",
        "unknown": {"must": "not survive"},
    }


def _issue(repository: str = "valkey", number: int = 8) -> dict[str, object]:
    return {
        "id": 1008,
        "number": number,
        "state": "closed",
        "state_reason": "completed",
        "title": "Fix a bug",
        "body": "",
        "locked": False,
        "user": _user(),
        "labels": [],
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
        "closed_at": TIMESTAMP,
        "url": f"https://api.github.com/repos/{OWNER}/{repository}/issues/{number}",
        "html_url": f"https://github.com/{OWNER}/{repository}/issues/{number}",
    }


def _search_item(
    repository: str = "valkey", number: int = 8, *, pull_request: bool = False
) -> dict[str, object]:
    segment = "pull" if pull_request else "issues"
    value: dict[str, object] = {
        "repository_url": f"https://api.github.com/repos/{OWNER}/{repository}",
        "number": number,
        "title": "Release status",
        "body": "Current status",
        "state": "closed",
        "labels": [{"name": "release-tracker", "color": "00ff00"}],
        "milestone": {
            "number": 3,
            "title": "Valkey 9.1",
            "state": "open",
            "url": f"https://api.github.com/repos/{OWNER}/{repository}/milestones/3",
            "html_url": f"https://github.com/{OWNER}/{repository}/milestone/3",
        },
        "updated_at": TIMESTAMP,
        "closed_at": TIMESTAMP,
        "url": f"https://api.github.com/repos/{OWNER}/{repository}/issues/{number}",
        "html_url": f"https://github.com/{OWNER}/{repository}/{segment}/{number}",
        "score": 42.0,
    }
    if pull_request:
        value["pull_request"] = {
            "url": f"https://api.github.com/repos/{OWNER}/{repository}/pulls/{number}",
            "html_url": f"https://github.com/{OWNER}/{repository}/pull/{number}",
            "diff_url": f"https://github.com/{OWNER}/{repository}/pull/{number}.diff",
        }
    return value


def _issue_search(*items: dict[str, object], incomplete_results: bool = False) -> dict[str, object]:
    return {
        "total_count": len(items),
        "incomplete_results": incomplete_results,
        "items": list(items),
    }


def _release(repository: str = "valkey", release_id: int = 9) -> dict[str, object]:
    tag = "v9.0.0"
    return {
        "id": release_id,
        "tag_name": tag,
        "target_commitish": "unstable",
        "name": "Valkey 9.0.0",
        "body": "Release notes",
        "draft": False,
        "prerelease": False,
        "author": _user(),
        "created_at": TIMESTAMP,
        "published_at": TIMESTAMP,
        "url": f"https://api.github.com/repos/{OWNER}/{repository}/releases/{release_id}",
        "html_url": f"https://github.com/{OWNER}/{repository}/releases/tag/{tag}",
    }


def _workflow_run(repository: str = "valkey", run_id: int = 11) -> dict[str, object]:
    return {
        "id": run_id,
        "workflow_id": 21,
        "name": "CI",
        "display_title": "Run tests",
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "run_number": 5,
        "run_attempt": 1,
        "head_branch": "feature",
        "head_sha": SHA,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
        "url": f"https://api.github.com/repos/{OWNER}/{repository}/actions/runs/{run_id}",
        "html_url": f"https://github.com/{OWNER}/{repository}/actions/runs/{run_id}",
    }


def _check_run(
    repository: str = "valkey", check_run_id: int = 12, commit: str = SHA
) -> dict[str, object]:
    return {
        "id": check_run_id,
        "name": "test-ubuntu-latest",
        "status": "completed",
        "conclusion": "success",
        "head_sha": commit,
        "started_at": TIMESTAMP,
        "completed_at": TIMESTAMP,
        "url": f"https://api.github.com/repos/{OWNER}/{repository}/check-runs/{check_run_id}",
        "html_url": f"https://github.com/{OWNER}/{repository}/runs/{check_run_id}",
        "output": {"annotations_count": 1000000},
    }


def _project(number: int = 14) -> dict[str, object]:
    return {
        "data": {
            "organization": {
                "projectV2": {
                    "id": "PVT_project",
                    "number": number,
                    "title": "Valkey 9.2",
                    "shortDescription": "Release project",
                    "public": True,
                    "closed": False,
                    "url": f"https://github.com/orgs/{OWNER}/projects/{number}",
                    "items": {
                        "totalCount": 3,
                        "nodes": [
                            {
                                "id": "PVTI_issue",
                                "type": "ISSUE",
                                "content": {
                                    "__typename": "Issue",
                                    "number": 8,
                                    "title": "Fix a bug",
                                    "state": "OPEN",
                                    "url": f"https://github.com/{OWNER}/valkey/issues/8",
                                    "repository": {"nameWithOwner": f"{OWNER}/valkey"},
                                },
                            },
                            {
                                "id": "PVTI_pr",
                                "type": "PULL_REQUEST",
                                "content": {
                                    "__typename": "PullRequest",
                                    "number": 7,
                                    "title": "Add a feature",
                                    "state": "MERGED",
                                    "url": f"https://github.com/{OWNER}/valkey/pull/7",
                                    "repository": {"nameWithOwner": f"{OWNER}/valkey"},
                                },
                            },
                            {
                                "id": "PVTI_draft",
                                "type": "DRAFT_ISSUE",
                                "content": {
                                    "__typename": "DraftIssue",
                                    "title": "Draft task",
                                    "body": "",
                                },
                            },
                        ],
                    },
                    "unknown": "discarded",
                }
            }
        },
        "extensions": {"ignored": True},
    }


def _valid_payload(query: LiveGitHubQuery) -> dict[str, object]:
    if isinstance(query, PullRequestQuery):
        return _pull_request(query.repository, query.number)
    if isinstance(query, IssueQuery):
        return _issue(query.repository, query.number)
    if isinstance(query, LatestReleaseQuery):
        return _release(query.repository, 9)
    if isinstance(query, WorkflowRunQuery):
        return _workflow_run(query.repository, query.run_id)
    if isinstance(query, CheckRunQuery):
        return _check_run(query.repository, query.check_run_id)
    raise AssertionError("REST payload requested for non-REST query")


def _decoded(observation: LiveObservation) -> dict[str, object]:
    value = json.loads(observation.canonical_payload)
    assert type(value) is dict
    return cast(dict[str, object], value)


@pytest.mark.parametrize(
    ("query", "source_url", "object_type", "kind"),
    [
        (
            PullRequestQuery("valkey", 7),
            "https://api.github.com/repos/valkey-io/valkey/pulls/7",
            "pull_request",
            "pull_request",
        ),
        (
            IssueQuery("valkey", 8),
            "https://api.github.com/repos/valkey-io/valkey/issues/8",
            "issue",
            "issue",
        ),
        (
            LatestReleaseQuery("valkey"),
            "https://api.github.com/repos/valkey-io/valkey/releases/latest",
            "release",
            "release",
        ),
        (
            WorkflowRunQuery("valkey", 11),
            "https://api.github.com/repos/valkey-io/valkey/actions/runs/11",
            "workflow_run",
            "workflow_run",
        ),
        (
            CheckRunQuery("valkey", 12),
            "https://api.github.com/repos/valkey-io/valkey/check-runs/12",
            "check",
            "check_run",
        ),
    ],
)
def test_every_rest_query_is_one_fixed_bounded_get_and_normalized(
    query: LiveGitHubQuery,
    source_url: str,
    object_type: str,
    kind: str | None,
) -> None:
    calls: list[tuple[str, float, int]] = []

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        calls.append((url, timeout_seconds, max_bytes))
        return _response(_valid_payload(query))

    observation = read_live_github(query, fetch=fetch, observed_clock=lambda: OBSERVED)

    assert calls == [(source_url, REQUEST_TIMEOUT_SECONDS, MAX_RESPONSE_BYTES)]
    assert observation.source_url == source_url
    assert observation.object_type == object_type
    assert observation.observed_at == "2026-08-19T23:58:28Z"
    assert observation.complete is True
    assert observation.truncated is False
    assert observation.payload_digest == (
        f"sha256:{hashlib.sha256(observation.canonical_payload).hexdigest()}"
    )
    payload = _decoded(observation)
    if kind is None:
        assert payload["id"] == 12
        assert "output" not in payload
    else:
        assert payload["kind"] == kind
    assert "unknown" not in payload
    assert observation.observation_id.startswith("obs_")


def test_issue_search_is_one_fixed_encoded_get_and_normalizes_complete_items() -> None:
    calls: list[tuple[str, float, int]] = []
    value = _issue_search(_search_item(number=8), _search_item(number=9, pull_request=True))

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        calls.append((url, timeout_seconds, max_bytes))
        return _response(value)

    observation = read_live_github(
        IssueSearchQuery((" Release ", "STATUS"), repository="valkey"),
        fetch=fetch,
        observed_clock=lambda: OBSERVED,
    )

    expected_url = (
        "https://api.github.com/search/issues?"
        # repo: alone. Sending org: alongside it makes GitHub union the two scopes and
        # return sibling-repository items, which then fail the item repository check.
        # GitHub requires an explicit kind on authenticated search/issues requests and
        # returns 422 without one.
        "q=repo%3Avalkey-io%2Fvalkey+is%3Aissue+release+status"
        "&sort=updated&order=desc&per_page=20"
    )
    assert calls == [(expected_url, REQUEST_TIMEOUT_SECONDS, MAX_RESPONSE_BYTES)]
    assert observation.source_url == expected_url
    assert observation.object_type == "issue"
    payload = _decoded(observation)
    assert payload == {
        "api_version": "valkeyrie.io/live-github/1",
        "items": [
            {
                "api_url": "https://api.github.com/repos/valkey-io/valkey/issues/8",
                "body": "Current status",
                "body_truncated": False,
                "closed_at": TIMESTAMP,
                "kind": "issue",
                "labels": ["release-tracker"],
                "milestone": {
                    "api_url": "https://api.github.com/repos/valkey-io/valkey/milestones/3",
                    "number": 3,
                    "state": "open",
                    "title": "Valkey 9.1",
                    "url": "https://github.com/valkey-io/valkey/milestone/3",
                },
                "number": 8,
                "repository": "valkey",
                "state": "closed",
                "title": "Release status",
                "updated_at": TIMESTAMP,
                "url": "https://github.com/valkey-io/valkey/issues/8",
            },
            {
                "api_url": "https://api.github.com/repos/valkey-io/valkey/issues/9",
                "body": "Current status",
                "body_truncated": False,
                "closed_at": TIMESTAMP,
                "kind": "pull_request",
                "labels": ["release-tracker"],
                "milestone": {
                    "api_url": "https://api.github.com/repos/valkey-io/valkey/milestones/3",
                    "number": 3,
                    "state": "open",
                    "title": "Valkey 9.1",
                    "url": "https://github.com/valkey-io/valkey/milestone/3",
                },
                "number": 9,
                "repository": "valkey",
                "state": "closed",
                "title": "Release status",
                "updated_at": TIMESTAMP,
                "url": "https://github.com/valkey-io/valkey/pull/9",
            },
        ],
        "kind": "issue_search",
        "order": "desc",
        "owner": OWNER,
        "per_page": 20,
        "repository": "valkey",
        "sort": "updated",
        "terms": ["release", "status"],
        "total_count": 2,
    }


def test_issue_search_can_use_only_the_fixed_org_scope() -> None:
    value = _issue_search(
        _search_item("valkey", 8),
        _search_item("valkey-doc", 9),
    )
    calls: list[str] = []

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        calls.append(url)
        return _response(value)

    observation = read_live_github(
        IssueSearchQuery(("release", "timeline"), per_page=2),
        fetch=fetch,
        observed_clock=lambda: OBSERVED,
    )

    assert calls == [
        "https://api.github.com/search/issues?"
        "q=org%3Avalkey-io+is%3Aissue+release+timeline&sort=updated&order=desc&per_page=2"
    ]
    payload = _decoded(observation)
    assert payload["repository"] is None
    items = cast(list[dict[str, object]], payload["items"])
    assert [item["repository"] for item in items] == ["valkey", "valkey-doc"]


@pytest.mark.parametrize(
    "terms",
    [
        (),
        ("release",),
        tuple(f"term{number}" for number in range(MAX_SEARCH_TERMS + 1)),
        ("release", "release"),
        ("release", " RELEASE "),
        ("release", "is"),
        ("release", "repo:other"),
        ("release", "https://evil.example"),
        ("release", "x"),
        ("release", "1"),
        ("release", "é"),
        ("release", "x" * 65),
        cast(tuple[str, ...], ["release", "status"]),
        cast(tuple[str, ...], ("release", 3)),
    ],
)
def test_issue_search_terms_are_bounded_unique_safe_normalized_tokens(
    terms: tuple[str, ...],
) -> None:
    calls = 0

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        nonlocal calls
        calls += 1
        return _response({})

    with pytest.raises(LiveGitHubError, match="search term|search requires"):
        read_live_github(IssueSearchQuery(terms), fetch=fetch)

    assert calls == 0


@pytest.mark.parametrize("per_page", [0, MAX_SEARCH_PER_PAGE + 1, -1, cast(int, True)])
def test_issue_search_has_an_exact_twenty_item_page_ceiling(per_page: int) -> None:
    with pytest.raises(LiveGitHubError, match="per_page"):
        read_live_github(
            IssueSearchQuery(("release", "status"), per_page=per_page),
            fetch=lambda *args: _response({}),
        )


def test_issue_search_accepts_the_maximum_term_and_page_bounds() -> None:
    terms = tuple(f"term{number}" for number in range(MAX_SEARCH_TERMS))
    items = tuple(_search_item(number=number) for number in range(1, MAX_SEARCH_PER_PAGE + 1))

    observation = read_live_github(
        IssueSearchQuery(terms, per_page=MAX_SEARCH_PER_PAGE),
        fetch=lambda *args: _response(_issue_search(*items)),
        observed_clock=lambda: OBSERVED,
    )

    payload = _decoded(observation)
    assert payload["terms"] == list(terms)
    assert payload["total_count"] == MAX_SEARCH_PER_PAGE


@pytest.mark.parametrize(
    ("total_count", "incomplete_results", "item_count"),
    [
        (0, False, 1),
        (1, True, 1),
        (MAX_SEARCH_PER_PAGE + 1, False, MAX_SEARCH_PER_PAGE + 1),
    ],
)
def test_issue_search_rejects_partial_results_and_page_overruns(
    total_count: int, incomplete_results: bool, item_count: int
) -> None:
    items = [_search_item(number=number) for number in range(1, item_count + 1)]
    value = _issue_search(*items, incomplete_results=incomplete_results)
    value["total_count"] = total_count

    with pytest.raises(LiveGitHubError, match="incomplete|page bound"):
        read_live_github(
            IssueSearchQuery(("release", "status")),
            fetch=lambda *args: _response(value),
        )


def test_issue_search_accepts_a_bounded_page_of_a_larger_match_set() -> None:
    # total_count counts every match while items holds only the requested page, so a
    # match set larger than one page is ordinary pagination. Rejecting it reported an
    # app-layer refusal to users as an upstream GitHub outage.
    value = _issue_search(_search_item(number=1), _search_item(number=2))
    value["total_count"] = 137

    observation = read_live_github(
        IssueSearchQuery(("community", "meeting")),
        fetch=lambda *args: _response(value),
    )

    payload = _decoded(observation)
    assert payload["total_count"] == 137
    assert len(cast(list[object], payload["items"])) == 2
    assert payload["per_page"] == MAX_SEARCH_PER_PAGE


def test_issue_search_rejects_duplicate_items() -> None:
    item = _search_item()
    with pytest.raises(LiveGitHubError, match="duplicate"):
        read_live_github(
            IssueSearchQuery(("release", "status")),
            fetch=lambda *args: _response(_issue_search(item, item)),
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value.update(
                {"repository_url": "https://api.github.com/repos/other/valkey"}
            ),
            "fixed owner",
        ),
        (
            lambda value: value.update(
                {"repository_url": "https://api.github.com/repos/valkey-io/other"}
            ),
            "allowlist",
        ),
        (
            lambda value: value.update(
                {"url": "https://api.github.com/repos/valkey-io/valkey/issues/999"}
            ),
            "canonical URL",
        ),
        (
            lambda value: value.update({"html_url": "https://evil.example/valkey/issues/8"}),
            "canonical URL",
        ),
        (lambda value: value.update({"state": "merged"}), "unsupported value"),
        (lambda value: value.update({"labels": "release"}), "must be an array"),
        (lambda value: value.update({"pull_request": None}), "must be an object"),
        (lambda value: value.update({"updated_at": None}), "must be text"),
        (
            lambda value: cast(dict[str, object], value["milestone"]).update(
                {"html_url": "https://evil.example/milestone/3"}
            ),
            "canonical URL",
        ),
    ],
)
def test_issue_search_items_require_fixed_identity_and_strict_fields(
    mutator: Any, message: str
) -> None:
    item = _search_item()
    mutator(item)

    with pytest.raises(LiveGitHubError, match=message):
        read_live_github(
            IssueSearchQuery(("release", "status")),
            fetch=lambda *args: _response(_issue_search(item)),
        )


def test_issue_search_rejects_cross_repository_items_under_a_repository_scope() -> None:
    with pytest.raises(LiveGitHubError, match="conflicts with the query"):
        read_live_github(
            IssueSearchQuery(("release", "status"), repository="valkey"),
            fetch=lambda *args: _response(_issue_search(_search_item("valkey-doc"))),
        )


def test_issue_search_validates_pull_request_marker_urls() -> None:
    item = _search_item(pull_request=True)
    pull_request = cast(dict[str, object], item["pull_request"])
    pull_request["url"] = "https://api.github.com/repos/valkey-io/valkey/pulls/999"

    with pytest.raises(LiveGitHubError, match="canonical URL"):
        read_live_github(
            IssueSearchQuery(("release", "status")),
            fetch=lambda *args: _response(_issue_search(item)),
        )


def test_issue_search_body_is_nullable_but_strictly_bounded() -> None:
    item = _search_item()
    item["body"] = None
    result = read_live_github(
        IssueSearchQuery(("release", "status")),
        fetch=lambda *args: _response(_issue_search(item)),
        observed_clock=lambda: OBSERVED,
    )
    items = cast(list[dict[str, object]], _decoded(result)["items"])
    assert items[0]["body"] is None

    # A body over the bound must NOT discard the whole result set. A long issue body is
    # ordinary, and failing the search made every repository-scoped question return
    # "Live GitHub data is temporarily unavailable". It is truncated, and the item says so.
    item["body"] = "x" * (MAX_SEARCH_BODY_BYTES + 1)
    result = read_live_github(
        IssueSearchQuery(("release", "status")),
        fetch=lambda *args: _response(_issue_search(item)),
        observed_clock=lambda: OBSERVED,
    )
    items = cast(list[dict[str, object]], _decoded(result)["items"])
    assert items[0]["body_truncated"] is True
    body = cast(str, items[0]["body"])
    assert len(body.encode("utf-8")) == MAX_SEARCH_BODY_BYTES
    # The observation is still complete: every matching item is present.
    assert result.complete is True and result.truncated is False

    # A multi-byte character must not be cut mid-sequence.
    item["body"] = "\u00e9" * MAX_SEARCH_BODY_BYTES
    result = read_live_github(
        IssueSearchQuery(("release", "status")),
        fetch=lambda *args: _response(_issue_search(item)),
        observed_clock=lambda: OBSERVED,
    )
    items = cast(list[dict[str, object]], _decoded(result)["items"])
    truncated_body = cast(str, items[0]["body"])
    assert truncated_body == "\u00e9" * (MAX_SEARCH_BODY_BYTES // 2)
    assert len(truncated_body.encode("utf-8")) <= MAX_SEARCH_BODY_BYTES


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the status of PR #123?", PullRequestQuery("valkey", 123)),
        (
            "Show pull request number 42 in valkey-doc",
            PullRequestQuery("valkey-doc", 42),
        ),
        ("What happened to issue #77?", IssueQuery("valkey", 77)),
        ("What is the latest release?", LatestReleaseQuery("valkey")),
        (
            "What is the current release in valkey-search?",
            LatestReleaseQuery("valkey-search"),
        ),
        ("Show project 14", ProjectQuery(14)),
        ("What happened in workflow run 123?", WorkflowRunQuery("valkey", 123)),
        ("Show check run ID 456", CheckRunQuery("valkey", 456)),
    ],
)
def test_infer_live_query_handles_explicit_live_identifiers_deterministically(
    question: str, expected: LiveGitHubQuery
) -> None:
    assert infer_live_query(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What is the release status and timeline for 9.1?",
            IssueSearchQuery(("release", "status", "timeline", "9.1"), "valkey"),
        ),
        (
            "Backport progress for valkey-ci-agent",
            IssueSearchQuery(("backport", "progress"), "valkey-ci-agent"),
        ),
        ("Status?", IssueSearchQuery(("status", "release"), "valkey")),
        (
            "Which upcoming Valkey events are announced?",
            IssueSearchQuery(
                ("upcoming", "events", "announced"),
                "valkey-io.github.io",
            ),
        ),
        (
            "What happened in the most recent community meeting?",
            IssueSearchQuery(("happened", "most", "recent", "meeting"), "community"),
        ),
        (
            "What is the current status of the connection storm workstream?",
            IssueSearchQuery(
                ("current", "status", "connection", "storm", "workstream"),
                "valkey",
            ),
        ),
        ("How does SET work?", None),
    ],
)
def test_infer_live_query_returns_only_bounded_release_status_searches(
    question: str, expected: LiveGitHubQuery | None
) -> None:
    query = infer_live_query(question)
    assert query == expected
    if isinstance(query, IssueSearchQuery):
        assert 2 <= len(query.terms) <= MAX_SEARCH_TERMS
        assert query.per_page <= MAX_SEARCH_PER_PAGE


@pytest.mark.parametrize(
    "question",
    [
        "release status repo:other",
        "release status https://evil.example/search?q=secret",
        "release status org:other",
    ],
)
def test_infer_live_query_rejects_arbitrary_qualifiers_and_urls(question: str) -> None:
    with pytest.raises(LiveGitHubError, match="unsupported qualifier or URL"):
        infer_live_query(question)


def test_infer_live_query_rejects_ambiguous_identifiers_and_repository_scopes() -> None:
    with pytest.raises(LiveGitHubError, match="multiple pull request identifiers"):
        infer_live_query("Compare PR #1 with PR #2")
    with pytest.raises(LiveGitHubError, match="multiple reviewed repositories"):
        infer_live_query("release status for valkey-doc and valkey-search")


@pytest.mark.parametrize("question", ["", "x" * 4097, cast(str, 3)])
def test_infer_live_query_validates_question_type_and_size(question: str) -> None:
    with pytest.raises(LiveGitHubError, match="question"):
        infer_live_query(question)


def test_observed_at_is_sampled_only_after_the_fetch_and_normalization() -> None:
    events: list[str] = []

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        events.append("fetch")
        return _response(_pull_request())

    def clock() -> datetime:
        assert events == ["fetch"]
        events.append("clock")
        return OBSERVED

    read_live_github(PullRequestQuery("valkey", 7), fetch=fetch, observed_clock=clock)

    assert events == ["fetch", "clock"]


def test_transport_failure_is_not_retried_or_replaced_with_stale_data() -> None:
    calls = 0

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        nonlocal calls
        calls += 1
        raise GitHubReadError("network unavailable")

    with pytest.raises(LiveGitHubError, match="network unavailable"):
        read_live_github(PullRequestQuery("valkey", 7), fetch=fetch)

    assert calls == 1


@pytest.mark.parametrize(
    "repository",
    ["Valkey", "other", "valkey/../other", "valkey%2fother", ""],
)
def test_repository_must_be_an_exact_fixed_allowlisted_name(repository: str) -> None:
    with pytest.raises(LiveGitHubError, match="allowlist"):
        read_live_github(PullRequestQuery(repository, 7), fetch=lambda *args: _response({}))


def test_allowlist_is_fixed_to_the_reviewed_valkey_repositories() -> None:
    assert OWNER == "valkey-io"
    assert "valkey" in ALLOWED_REPOSITORIES
    assert "valkey-doc" in ALLOWED_REPOSITORIES
    assert "valkey-ci-agent" in ALLOWED_REPOSITORIES
    assert "redis" not in ALLOWED_REPOSITORIES


@pytest.mark.parametrize("number", [0, -1, MAX_ENTITY_ID + 1, cast(int, True)])
def test_numeric_query_identifiers_have_exact_positive_63_bit_bounds(number: int) -> None:
    with pytest.raises(LiveGitHubError, match="from 1 through"):
        read_live_github(PullRequestQuery("valkey", number), fetch=lambda *args: _response({}))


def test_maximum_numeric_identifier_is_accepted_by_query_validation() -> None:
    calls: list[str] = []

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        calls.append(url)
        return _response(_pull_request(number=MAX_ENTITY_ID))

    result = read_live_github(
        PullRequestQuery("valkey", MAX_ENTITY_ID), fetch=fetch, observed_clock=lambda: OBSERVED
    )

    assert _decoded(result)["number"] == MAX_ENTITY_ID
    assert calls[0].endswith(str(MAX_ENTITY_ID))


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (HttpResponse(404, {"content-type": "application/json"}, b"{}"), "HTTP 404"),
        (HttpResponse(200, {"content-type": "text/html"}, b"{}"), "content type"),
        (HttpResponse(200, {"content-type": "application/json"}, b"[]"), "must be an object"),
        (
            HttpResponse(200, {"content-type": "application/json"}, b'{"id":1,"id":2}'),
            "strict JSON",
        ),
        (HttpResponse(200, {"content-type": "application/json"}, b'{"id":NaN}'), "strict JSON"),
        (HttpResponse(200, {"content-type": "application/json"}, b"\xff"), "strict JSON"),
        (
            HttpResponse(
                200,
                {"content-type": "application/json"},
                b"x" * (MAX_RESPONSE_BYTES + 1),
            ),
            "byte bound",
        ),
    ],
)
def test_rest_response_requires_success_strict_json_object_and_exact_byte_bound(
    response: HttpResponse, message: str
) -> None:
    with pytest.raises(LiveGitHubError, match=message):
        read_live_github(PullRequestQuery("valkey", 7), fetch=lambda *args: response)


def test_fetcher_must_return_the_declared_response_type() -> None:
    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        return cast(HttpResponse, {"status": 200, "body": b"{}"})

    with pytest.raises(LiveGitHubError, match="wrong response type"):
        read_live_github(PullRequestQuery("valkey", 7), fetch=fetch)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("id", True, "bounded integer"),
        ("number", 99, "conflicts"),
        ("title", 3, "must be text"),
        ("body", [], "text or null"),
        ("draft", 1, "must be boolean"),
        ("created_at", "2026-02-30T00:00:00Z", "canonical UTC timestamp"),
        ("url", "https://api.github.com/repos/other/valkey/pulls/7", "canonical URL"),
        ("html_url", "https://evil.example/valkey/pull/7", "canonical URL"),
    ],
)
def test_pull_request_fields_have_strict_types_identity_and_canonical_urls(
    field: str, bad_value: object, message: str
) -> None:
    value = _pull_request()
    value[field] = bad_value

    with pytest.raises(LiveGitHubError, match=message):
        read_live_github(
            PullRequestQuery("valkey", 7),
            fetch=lambda *args: _response(value),
            observed_clock=lambda: OBSERVED,
        )


def test_issue_query_rejects_githubs_pull_request_shape() -> None:
    value = _issue()
    value["pull_request"] = {"url": "ignored"}

    with pytest.raises(LiveGitHubError, match="returned a pull request"):
        read_live_github(IssueQuery("valkey", 8), fetch=lambda *args: _response(value))


def test_duplicate_labels_are_rejected_instead_of_ambiguously_normalized() -> None:
    value = _pull_request()
    value["labels"] = [{"name": "bug"}, {"name": "bug"}]

    with pytest.raises(LiveGitHubError, match="duplicate"):
        read_live_github(PullRequestQuery("valkey", 7), fetch=lambda *args: _response(value))


def test_projects_fail_clearly_without_an_authenticated_graphql_fetcher() -> None:
    with pytest.raises(LiveGitHubError, match="injected authenticated GraphQL fetcher"):
        read_live_github(ProjectQuery(14))


def test_project_uses_only_the_fixed_query_variables_and_bounds() -> None:
    calls: list[tuple[str, Mapping[str, object], float, int]] = []

    def projects_fetch(
        query: str,
        variables: Mapping[str, object],
        timeout_seconds: float,
        max_bytes: int,
    ) -> HttpResponse:
        calls.append((query, variables, timeout_seconds, max_bytes))
        return _response(_project(), content_type="application/json")

    observation = read_live_github(
        ProjectQuery(14),
        projects_fetch=projects_fetch,
        observed_clock=lambda: OBSERVED,
    )

    assert calls == [
        (
            PROJECTS_GRAPHQL_QUERY,
            {"owner": OWNER, "number": 14, "itemCount": MAX_COLLECTION_ITEMS},
            REQUEST_TIMEOUT_SECONDS,
            MAX_RESPONSE_BYTES,
        )
    ]
    assert PROJECTS_GRAPHQL_QUERY.lstrip().startswith("query ValkeyrieProject")
    assert "mutation" not in PROJECTS_GRAPHQL_QUERY.casefold()
    assert "$owner" in PROJECTS_GRAPHQL_QUERY
    assert observation.source_url == "https://api.github.com/graphql"
    assert observation.object_type == "controller_status"
    payload = _decoded(observation)
    assert payload["kind"] == "project"
    assert payload["total_count"] == 3
    assert "unknown" not in payload
    items = cast(list[dict[str, object]], payload["items"])
    assert [item["type"] for item in items] == ["issue", "pull_request", "draft_issue"]


@pytest.mark.parametrize("status", [401, 403])
def test_projects_authentication_failures_are_clear(status: int) -> None:
    def projects_fetch(
        query: str,
        variables: Mapping[str, object],
        timeout_seconds: float,
        max_bytes: int,
    ) -> HttpResponse:
        return _response({"message": "forbidden"}, status=status)

    with pytest.raises(LiveGitHubError, match="requires authenticated GraphQL access"):
        read_live_github(ProjectQuery(14), projects_fetch=projects_fetch)


def test_projects_transport_failure_is_not_retried() -> None:
    calls = 0

    def projects_fetch(
        query: str,
        variables: Mapping[str, object],
        timeout_seconds: float,
        max_bytes: int,
    ) -> HttpResponse:
        nonlocal calls
        calls += 1
        raise GitHubReadError("timeout")

    with pytest.raises(LiveGitHubError, match="timeout"):
        read_live_github(ProjectQuery(14), projects_fetch=projects_fetch)

    assert calls == 1


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.update({"errors": [{"message": "partial"}]}), "returned errors"),
        (
            lambda value: cast(dict[str, Any], value["data"])["organization"]["projectV2"].update(
                {"number": 15}
            ),
            "conflicting project number",
        ),
        (
            lambda value: cast(dict[str, Any], value["data"])["organization"]["projectV2"].update(
                {"public": False}
            ),
            "non-public",
        ),
        (
            lambda value: cast(dict[str, Any], value["data"])["organization"]["projectV2"].update(
                {"url": "https://evil.example/project/14"}
            ),
            "canonical URL",
        ),
        (
            lambda value: cast(dict[str, Any], value["data"])["organization"]["projectV2"][
                "items"
            ].update({"totalCount": 4}),
            "incomplete",
        ),
        (
            lambda value: cast(dict[str, Any], value["data"])["organization"]["projectV2"]["items"][
                "nodes"
            ][0]["content"]["repository"].update({"nameWithOwner": "other/valkey"}),
            "fixed owner",
        ),
        (
            lambda value: cast(dict[str, Any], value["data"])["organization"]["projectV2"]["items"][
                "nodes"
            ][0]["content"].update({"url": "https://github.com/valkey-io/valkey/issues/999"}),
            "canonical URL",
        ),
        (
            lambda value: cast(dict[str, Any], value["data"])["organization"]["projectV2"]["items"][
                "nodes"
            ][0].update({"type": "REPOSITORY"}),
            "unsupported value",
        ),
    ],
)
def test_project_normalizer_rejects_partial_conflicting_or_unsupported_data(
    mutator: Any, message: str
) -> None:
    value = copy.deepcopy(_project())
    mutator(value)

    with pytest.raises(LiveGitHubError, match=message):
        normalize_project_response(value, number=14)


def test_project_item_count_has_the_same_exact_collection_bound() -> None:
    value = _project()
    project = cast(dict[str, Any], value["data"])["organization"]["projectV2"]
    project["items"]["totalCount"] = MAX_COLLECTION_ITEMS + 1

    with pytest.raises(LiveGitHubError, match="collection bound"):
        normalize_project_response(value, number=14)


def test_project_redacted_item_is_complete_only_with_null_content() -> None:
    value = _project()
    project = cast(dict[str, Any], value["data"])["organization"]["projectV2"]
    project["items"] = {
        "totalCount": 1,
        "nodes": [{"id": "PVTI_redacted", "type": "REDACTED", "content": None}],
    }

    normalized = normalize_project_response(value, number=14)

    assert normalized["items"] == [{"id": "PVTI_redacted", "type": "redacted", "content": None}]


def test_observation_clock_must_be_timezone_aware_and_is_not_called_on_failure() -> None:
    clock_calls = 0

    def clock() -> datetime:
        nonlocal clock_calls
        clock_calls += 1
        return datetime(2026, 8, 19)

    with pytest.raises(LiveGitHubError, match="aware datetime"):
        read_live_github(
            PullRequestQuery("valkey", 7),
            fetch=lambda *args: _response(_pull_request()),
            observed_clock=clock,
        )
    assert clock_calls == 1

    def failed_fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        raise GitHubReadError("failed")

    with pytest.raises(LiveGitHubError, match="failed"):
        read_live_github(PullRequestQuery("valkey", 7), fetch=failed_fetch, observed_clock=clock)
    assert clock_calls == 1


def test_unknown_runtime_query_type_fails_before_any_fetch() -> None:
    calls = 0

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        nonlocal calls
        calls += 1
        return _response({})

    with pytest.raises(LiveGitHubError, match="unsupported live GitHub query type"):
        read_live_github(cast(LiveGitHubQuery, object()), fetch=fetch)

    assert calls == 0


@pytest.mark.parametrize("kind", ["workflow", "check"])
def test_documented_waiting_and_startup_failure_states_are_supported(kind: str) -> None:
    if kind == "workflow":
        query: LiveGitHubQuery = WorkflowRunQuery("valkey", 11)
        value = _workflow_run()
        value["conclusion"] = "startup_failure"
    else:
        query = CheckRunQuery("valkey", 12)
        value = _check_run()
        value["status"] = "waiting"
        value["conclusion"] = None

    observation = read_live_github(
        query, fetch=lambda *args: _response(value), observed_clock=lambda: OBSERVED
    )

    assert _decoded(observation)["kind"] in {"workflow_run", "check_run"}


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            HttpResponse(cast(int, True), {"content-type": "application/json"}, b"{}"),
            "status type",
        ),
        (
            HttpResponse(
                200,
                cast(Mapping[str, str], {"content-type": 1}),
                b"{}",
            ),
            "invalid headers",
        ),
        (
            HttpResponse(
                200,
                {"content-type": "application/json"},
                cast(bytes, "{}"),
            ),
            "body must be bytes",
        ),
    ],
)
def test_injected_http_response_fields_keep_their_declared_runtime_types(
    response: HttpResponse, message: str
) -> None:
    with pytest.raises(LiveGitHubError, match=message):
        read_live_github(PullRequestQuery("valkey", 7), fetch=lambda *args: response)


def test_projects_fetcher_must_return_the_declared_response_type() -> None:
    def projects_fetch(
        query: str,
        variables: Mapping[str, object],
        timeout_seconds: float,
        max_bytes: int,
    ) -> HttpResponse:
        return cast(HttpResponse, {"status": 200, "body": b"{}"})

    with pytest.raises(LiveGitHubError, match="wrong response type"):
        read_live_github(ProjectQuery(14), projects_fetch=projects_fetch)
