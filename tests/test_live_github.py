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
    ReleaseListQuery,
    WorkflowRunQuery,
    infer_live_query,
    infer_supplementary_search,
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
        "user": {"login": "madolson"},
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
        "&per_page=20"
    )
    assert calls == [(expected_url, REQUEST_TIMEOUT_SECONDS, MAX_RESPONSE_BYTES)]
    assert observation.source_url == expected_url
    assert observation.object_type == "issue"
    payload = _decoded(observation)
    assert payload == {
        "api_version": "valkeyrie.io/live-github/1",
        "author": None,
        "authors_of_listed": {"madolson": 2},
        "finding": (
            'The search found 2 issues in valkey-io/valkey matching "release" and "status".'
        ),
        "items": [
            {
                "api_url": "https://api.github.com/repos/valkey-io/valkey/issues/8",
                "author": "madolson",
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
                "author": "madolson",
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
        "owner": OWNER,
        "per_page": 20,
        "query": "repo:valkey-io/valkey is:issue release status",
        "repositories": ["valkey"],
        "repository": "valkey",
        "since": None,
        "sort": "best-match",
        "until": None,
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
        "q=org%3Avalkey-io+is%3Aissue+release+timeline&per_page=2"
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
        # An array is now accepted by the transport for list endpoints, so an object query fed one
        # fails one step later, on the first field a single object must carry.
        (HttpResponse(200, {"content-type": "application/json"}, b"[]"), "bounded integer"),
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


def test_an_issue_number_that_names_a_pull_request_is_read_not_refused() -> None:
    """GitHub shares one number space between issues and pull requests.

    Its issues endpoint serves both, so refusing a pull request made "what is the status of issue
    #3853?" fail outright and report live data as unavailable, when the number does name a real
    object whose status is known. What it is is stated in the payload so an answer can say pull
    request rather than calling it an issue, and the canonical URL is the /pull/ one GitHub returns.
    """
    value = _issue()
    value["pull_request"] = {
        "url": "https://api.github.com/repos/valkey-io/valkey/pulls/8",
        "merged_at": "2026-09-15T21:15:53Z",
    }
    value["html_url"] = "https://github.com/valkey-io/valkey/pull/8"

    observation = read_live_github(IssueQuery("valkey", 8), fetch=lambda *args: _response(value))
    payload = json.loads(observation.canonical_payload)
    assert payload["is_pull_request"] is True
    assert payload["merged_at"] == "2026-09-15T21:15:53Z"
    assert payload["url"] == "https://github.com/valkey-io/valkey/pull/8"

    # A genuine issue still reports itself as one, with no merge instant.
    plain = read_live_github(IssueQuery("valkey", 8), fetch=lambda *args: _response(_issue()))
    plain_payload = json.loads(plain.canonical_payload)
    assert plain_payload["is_pull_request"] is False
    assert plain_payload["merged_at"] is None
    assert plain_payload["url"] == "https://github.com/valkey-io/valkey/issues/8"


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
            {"owner": OWNER, "number": 14, "itemCount": MAX_COLLECTION_ITEMS, "after": None},
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
    # Items with no Status are remaining work by definition, so all three are listed.
    items = cast(list[dict[str, object]], payload["remaining_items"])
    assert [item["type"] for item in items] == ["issue", "pull_request", "draft_issue"]
    assert payload["items_by_status"] == {"(no status)": 3}
    assert payload["remaining_truncated"] is False


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
            # More nodes than the board claims to hold is a contradiction. (Fewer is a partial
            # page now, which real release boards produce and which is reported, not refused.)
            lambda value: cast(dict[str, Any], value["data"])["organization"]["projectV2"][
                "items"
            ].update({"totalCount": 0}),
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


def test_a_board_larger_than_one_page_is_read_whole_and_totalled_by_status() -> None:
    """Release boards hold 150 to 300 items against a page of 100.

    GraphQL has no filter on item state, and "what is left" means the board's Status column, which
    only a complete read can total. The reader walks the pages and the normalizer merges them,
    refusing a walk that comes up short of the board's own count.
    """
    first = _project()
    project = cast(dict[str, Any], first["data"])["organization"]["projectV2"]
    project["items"]["totalCount"] = 6
    project["items"]["pageInfo"] = {"hasNextPage": True, "endCursor": "c1"}
    for node, status in zip(project["items"]["nodes"], ("Todo", "Done", "Done"), strict=True):
        node["status"] = {"name": status}
    second = copy.deepcopy(first)
    second_project = cast(dict[str, Any], second["data"])["organization"]["projectV2"]
    second_project["items"]["pageInfo"] = {"hasNextPage": False, "endCursor": None}
    for node in second_project["items"]["nodes"]:
        node["id"] = node["id"] + "-p2"
        node["status"] = {"name": "Needs Review"}

    pages = iter([first, second])
    seen_cursors: list[object] = []

    def fetch(
        query: str, variables: Mapping[str, object], timeout_seconds: float, max_bytes: int
    ) -> HttpResponse:
        seen_cursors.append(variables["after"])
        return _response(next(pages))

    observation = read_live_github(ProjectQuery(14), projects_fetch=fetch)
    payload = json.loads(observation.canonical_payload)
    assert seen_cursors == [None, "c1"]
    assert payload["items_total"] == 6
    assert payload["partial"] is False
    assert payload["items_by_status"] == {"Done": 2, "Needs Review": 3, "Todo": 1}
    # Done items are counted, not listed; the four remaining are listed in board order.
    assert [item["status"] for item in payload["remaining_items"]] == [
        "Todo",
        "Needs Review",
        "Needs Review",
        "Needs Review",
    ]
    assert payload["remaining_truncated"] is False

    # A walk that ends short of the board's count is refused, not presented as the board.
    short = _project()
    cast(dict[str, Any], short["data"])["organization"]["projectV2"]["items"]["totalCount"] = 9
    with pytest.raises(LiveGitHubError, match="incomplete"):
        normalize_project_response(short, number=14)


def test_project_redacted_item_is_complete_only_with_null_content() -> None:
    value = _project()
    project = cast(dict[str, Any], value["data"])["organization"]["projectV2"]
    project["items"] = {
        "totalCount": 1,
        "nodes": [{"id": "PVTI_redacted", "type": "REDACTED", "content": None}],
    }

    normalized = normalize_project_response(value, number=14)

    assert normalized["remaining_items"] == [
        {"id": "PVTI_redacted", "type": "redacted", "status": None, "content": None}
    ]


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


def test_supplementary_search_only_fires_on_design_or_state_intent() -> None:
    """A supplement is only worth its latency and quota where GitHub can actually answer.

    Issue and pull request search cannot see file contents, so a documentation or governance
    question returns unrelated bug reports: searching GitHub for "who leads the TSC" returns an
    issue about assigning PR owners, while the corpus answers it from MAINTAINERS.md. Those
    questions previously spent two searches each and capped throughput against the authenticated
    thirty-per-minute search budget.
    """
    # Answerable from indexed files. GitHub search cannot see files, so it must not be asked.
    for question in (
        "How do I use GET?",
        "What does SET do?",
        "Who leads the TSC?",
        "How do I contribute to Valkey?",
        "Where should documentation for a newly added command be written?",
    ):
        assert infer_supplementary_search(question) is None, question

    # Design, shipped-state, and live-state intent all still reach a search.
    for question in (
        "How does Valkey replication compression work?",
        "Is streaming compression merged yet?",
        "What is the proposed design for dual channel replication?",
        "What is the roadmap for 9.1?",
    ):
        assert infer_supplementary_search(question) is not None, question

    # The floor on subject terms still applies, so intent alone cannot force a search.
    assert infer_supplementary_search("does it work?") is None


def test_release_list_includes_prereleases_that_latest_omits() -> None:
    """/releases/latest omits prereleases, so it cannot say whether a release candidate exists.

    With 9.2.0-rc1 published, that endpoint still reported 9.1.2, and a question about the rc
    answered from it would have been answered wrongly. The list endpoint returns a bare JSON array,
    which the transport wraps so one response contract serves every normalizer.
    """
    rc = _release()
    rc.update({"id": 91, "tag_name": "9.2.0-rc1", "prerelease": True})
    rc["url"] = "https://api.github.com/repos/valkey-io/valkey/releases/91"
    rc["html_url"] = "https://github.com/valkey-io/valkey/releases/tag/9.2.0-rc1"
    stable = _release()

    observation = read_live_github(
        ReleaseListQuery("valkey", 4), fetch=lambda *args: _response([rc, stable])
    )
    payload = json.loads(observation.canonical_payload)
    assert payload["kind"] == "release_list"
    assert [item["tag"] for item in payload["releases"]] == ["9.2.0-rc1", "v9.0.0"]
    assert payload["releases"][0]["prerelease"] is True
    assert observation.source_url.endswith("/releases?per_page=4")

    # More elements than the requested page is a response that cannot be trusted.
    with pytest.raises(LiveGitHubError, match="exceeds the requested page bound"):
        read_live_github(ReleaseListQuery("valkey", 1), fetch=lambda *args: _response([rc, stable]))
    # And the page size itself is bounded.
    with pytest.raises(LiveGitHubError, match="outside its bound"):
        read_live_github(ReleaseListQuery("valkey", 500), fetch=lambda *args: _response([]))


def test_a_merged_pull_request_reports_which_recent_releases_contain_it() -> None:
    """Merged and released are different facts, and "has X shipped?" needs the second.

    The check lists the commits reachable from each recent tag in the one-second window around the
    merge instant: a tag that contains the merge commit returns it, one that does not returns
    nothing. Compare would answer too but embeds every file diff, over a megabyte for a diverged
    tag; GraphQL compare needs the repo scope the token deliberately lacks.
    """
    merge_sha = "a" * 40
    value = _pull_request(number=7)
    value["merged_at"] = "2026-09-15T21:15:53Z"
    value["merge_commit_sha"] = merge_sha
    seen: list[str] = []

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        seen.append(url)
        if url.endswith("/pulls/7"):
            return _response(value)
        if "/releases?per_page=" in url:
            return _response([{"tag_name": "9.2.0-rc1"}, {"tag_name": "9.1.2"}])
        if "/commits?sha=9.2.0-rc1&" in url:
            return _response([{"sha": merge_sha}])
        if "/commits?sha=9.1.2&" in url:
            return _response([])
        raise AssertionError(url)

    payload = json.loads(
        read_live_github(PullRequestQuery("valkey", 7), fetch=fetch).canonical_payload
    )
    assert payload["released_in"] == ["9.2.0-rc1"]
    assert payload["release_membership_checked"] == ["9.2.0-rc1", "9.1.2"]
    # The window is the merge instant plus or minus one second, on both sides.
    window = next(u for u in seen if "/commits?sha=9.2.0-rc1&" in u)
    assert "since=2026-09-15T21:15:52Z" in window and "until=2026-09-15T21:15:54Z" in window

    # An unmerged pull request is not checked at all: no releases call, no commits calls.
    seen.clear()
    open_value = _pull_request(number=7)
    open_value["merged_at"] = None
    open_value["merge_commit_sha"] = None
    payload = json.loads(
        read_live_github(
            PullRequestQuery("valkey", 7), fetch=lambda *a: _response(open_value)
        ).canonical_payload
    )
    assert "released_in" not in payload


def test_a_question_naming_two_repositories_is_searched_in_both() -> None:
    """Naming two repositories used to raise, which killed the supplement on exactly the
    cross-repository questions. A search carries one repo: qualifier per repository and GitHub ORs
    them; org: is never sent alongside, since it would union the whole organization back in."""
    from valkeyrie.live_github import _inferred_repositories

    assert _inferred_repositories("does valkey-glide support X from valkey 9.2?") == (
        "valkey",
        "valkey-glide",
    )
    assert _inferred_repositories("compare valkey-glide and valkey-py reconnects") == (
        "valkey-glide",
        "valkey-py",
    )
    # The core name inside a longer name does not count as the core repository.
    assert _inferred_repositories("what is new in valkey-glide") == ("valkey-glide",)
    assert _inferred_repositories("how does failover work") == ("valkey",)

    seen: list[str] = []

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        seen.append(url)
        return _response({"total_count": 0, "incomplete_results": False, "items": []})

    read_live_github(
        IssueSearchQuery(
            ("streaming", "compression"), repository="valkey", repositories=("valkey-glide",)
        ),
        fetch=fetch,
    )
    assert "q=repo%3Avalkey-io%2Fvalkey+repo%3Avalkey-io%2Fvalkey-glide+is%3Aissue" in seen[0]
    assert "org%3A" not in seen[0]
    # An empty result states its finding in words, derived only from the query's own fields:
    # a zero read as a gap made the model abstain when the absence WAS the answer.
    empty = json.loads(
        read_live_github(
            IssueSearchQuery(("streaming", "compression"), repository="valkey-glide"),
            fetch=fetch,
        ).canonical_payload
    )
    assert empty["finding"] == (
        "The search completed and found no issues in valkey-io/valkey-glide matching "
        '"streaming" and "compression". This establishes that no such issues existed '
        "there at observation time."
    )

    # An item from a repository outside the scope is still refused.
    def leaking(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        item = {
            "repository_url": "https://api.github.com/repos/valkey-io/valkey-py",
            "number": 1,
            "title": "t",
            "state": "open",
            "html_url": "https://github.com/valkey-io/valkey-py/issues/1",
            "updated_at": "2026-01-01T00:00:00Z",
            "body": "",
            "user": {"login": "x"},
            "labels": [],
        }
        return _response({"total_count": 1, "incomplete_results": False, "items": [item]})

    with pytest.raises(LiveGitHubError, match="conflicts with the query"):
        read_live_github(
            IssueSearchQuery(
                ("streaming", "compression"), repository="valkey", repositories=("valkey-glide",)
            ),
            fetch=leaking,
        )


def test_a_release_list_keeps_short_notes_for_the_newest_releases_only() -> None:
    """Eight releases with whole notes was 56 KB of a 64 KB evidence budget and starved every
    other record. The newest three keep their opening 3 KB (the headline features); the older
    ones matter as tags and dates, so their notes are dropped and the item says so."""
    from valkeyrie.live_github import MAX_RELEASE_LIST_BODY_BYTES, RELEASE_LIST_BODIES

    items = []
    for index in range(6):
        release = _release()
        release.update(
            {
                "id": 100 + index,
                "tag_name": f"9.{index}.0",
                "body": ("n" * (MAX_RELEASE_LIST_BODY_BYTES + 500)) if index != 1 else "short",
            }
        )
        release["url"] = f"https://api.github.com/repos/valkey-io/valkey/releases/{100 + index}"
        release["html_url"] = f"https://github.com/valkey-io/valkey/releases/tag/9.{index}.0"
        items.append(release)
    payload = json.loads(
        read_live_github(
            ReleaseListQuery("valkey", 8), fetch=lambda *a: _response(items)
        ).canonical_payload
    )
    releases = payload["releases"]
    assert len(releases[0]["body"].encode("utf-8")) == MAX_RELEASE_LIST_BODY_BYTES
    assert releases[0]["body_truncated"] is True
    assert releases[1]["body"] == "short" and "body_truncated" not in releases[1]
    for release in releases[RELEASE_LIST_BODIES:]:
        assert release["body"] is None and release["body_truncated"] is True
    assert len(json.dumps(payload).encode("utf-8")) < 16 * 1024


def test_a_date_window_search_lists_recent_items_with_short_bodies() -> None:
    """ "What merged this week" is a window with no terms: merged pull requests on or after the
    day, newest first, bodies cut short so twenty titles fit where five whole bodies did. An issue
    window is by creation date. The day must be a real calendar day."""
    from valkeyrie.live_github import MAX_WINDOW_BODY_BYTES

    seen: list[str] = []

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        seen.append(url)
        item = _search_item(number=4747, pull_request=True)
        item["body"] = "b" * 5000
        return _response({"total_count": 36, "incomplete_results": False, "items": [item]})

    payload = json.loads(
        read_live_github(
            IssueSearchQuery(
                (), repository="valkey", per_page=20, kind="pull-request", since="2026-09-15"
            ),
            fetch=fetch,
        ).canonical_payload
    )
    assert (
        "q=repo%3Avalkey-io%2Fvalkey+is%3Apull-request+is%3Amerged+merged%3A%3E%3D2026-09-15"
        in seen[0]
    )
    assert "sort=updated&order=desc&per_page=20" in seen[0]
    assert (
        payload["since"] == "2026-09-15" and payload["sort"] == "updated" and payload["terms"] == []
    )
    item = payload["items"][0]
    assert (
        len(item["body"].encode("utf-8")) == MAX_WINDOW_BODY_BYTES
        and item["body_truncated"] is True
    )
    assert payload["finding"] == (
        "The search found 36 pull requests merged on or after 2026-09-15 in valkey-io/valkey. "
        "The 1 most recently updated are listed."
    )

    seen.clear()
    read_live_github(
        IssueSearchQuery(("cluster",), repository="valkey", kind="issue", since="2026-09-01"),
        fetch=fetch,
    )
    assert "is%3Aissue+created%3A%3E%3D2026-09-01+cluster" in seen[0]

    for since in ("2026-13-01", "last week", "2026-9-1", "2026-02-30"):
        with pytest.raises(LiveGitHubError, match="search window"):
            read_live_github(
                IssueSearchQuery((), repository="valkey", kind="issue", since=since), fetch=fetch
            )
    # A closed window is inclusive on both ends and must be ordered.
    seen.clear()
    payload = json.loads(
        read_live_github(
            IssueSearchQuery(
                (), repository="valkey", kind="pull-request", since="2026-08-01", until="2026-08-31"
            ),
            fetch=fetch,
        ).canonical_payload
    )
    assert "merged%3A2026-08-01..2026-08-31" in seen[0]
    assert payload["finding"].startswith(
        "The search found 36 pull requests merged from 2026-08-01 through 2026-08-31"
    )
    with pytest.raises(LiveGitHubError, match="precedes"):
        read_live_github(
            IssueSearchQuery(
                (), repository="valkey", kind="issue", since="2026-08-31", until="2026-08-01"
            ),
            fetch=fetch,
        )
    with pytest.raises(LiveGitHubError, match="requires a start"):
        read_live_github(
            IssueSearchQuery(("a", "b"), repository="valkey", kind="issue", until="2026-08-01"),
            fetch=fetch,
        )
    # Without a window the term minimum still holds.
    with pytest.raises(LiveGitHubError, match="requires from 2"):
        read_live_github(IssueSearchQuery((), repository="valkey", kind="issue"), fetch=fetch)


def test_project_paging_fails_closed_on_a_stuck_cursor_a_repeated_item_or_an_unfinished_walk() -> (
    None
):
    """Three ways a bounded walk could present an incomplete or duplicated board as whole, each
    refused: a cursor that does not advance, an item id seen on two pages, and a final page that
    still says another follows."""
    from valkeyrie.live_github import MAX_PROJECT_PAGES

    def page(has_next: bool, cursor: str | None, total: int, suffix: str) -> dict[str, object]:
        value = _project()
        project = cast(dict[str, Any], value["data"])["organization"]["projectV2"]
        project["items"]["totalCount"] = total
        project["items"]["pageInfo"] = {"hasNextPage": has_next, "endCursor": cursor}
        for node in project["items"]["nodes"]:
            node["id"] = node["id"] + suffix
        return value

    # Stuck cursor: page two says "next is c1" again. Refused on the second page, not walked
    # to the cap and accepted because the counts happened to line up.
    stuck = iter([page(True, "c1", 9, "-a"), page(True, "c1", 9, "-b"), page(True, "c1", 9, "-c")])
    with pytest.raises(LiveGitHubError, match="did not advance"):
        read_live_github(ProjectQuery(14), projects_fetch=lambda *a: _response(next(stuck)))

    # A page whose items repeat the previous page's ids, with the total agreeing on the count.
    repeated = iter([page(True, "c1", 6, "-a"), page(False, None, 6, "-a")])
    with pytest.raises(LiveGitHubError, match="repeat an item"):
        read_live_github(ProjectQuery(14), projects_fetch=lambda *a: _response(next(repeated)))

    # Every page through the cap says another follows: the board is larger than the walk.
    endless = iter(
        [page(True, f"c{i}", 3 * MAX_PROJECT_PAGES, f"-{i}") for i in range(MAX_PROJECT_PAGES)]
    )
    with pytest.raises(LiveGitHubError, match="exceeds the page bound"):
        read_live_github(ProjectQuery(14), projects_fetch=lambda *a: _response(next(endless)))

    # Pages disagreeing on the total are refused whichever page is first.
    with pytest.raises(LiveGitHubError, match="disagree on the item total"):
        normalize_project_response(
            [page(True, "c1", 6, "-a"), page(False, None, 7, "-b")], number=14
        )


def test_release_membership_is_conclusive_only_within_one_commit_page_and_per_tag() -> None:
    """A tag is recorded as checked only when its commit window was read whole: a full page may
    hide the merge on the next one, so the tag is left unchecked rather than reported absent.
    A malformed listing for one tag leaves that tag unchecked and the pull request intact; a
    failed release list leaves membership unknown, never fails the read."""
    from valkeyrie.live_github import MEMBERSHIP_COMMIT_PAGE

    merge_sha = "a" * 40
    value = _pull_request(number=7)
    value["merged_at"] = "2026-09-15T21:15:53Z"
    value["merge_commit_sha"] = merge_sha

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        if url.endswith("/pulls/7"):
            return _response(value)
        if "/releases?per_page=" in url:
            return _response(
                [
                    {"tag_name": "full"},
                    {"tag_name": "found"},
                    {"tag_name": "broken"},
                    {"tag_name": "empty"},
                ]
            )
        if "sha=full&" in url:
            # A full page, none of them the merge: inconclusive.
            return _response([{"sha": f"{i:040x}"} for i in range(1, MEMBERSHIP_COMMIT_PAGE + 1)])
        if "sha=found&" in url:
            return _response([{"sha": merge_sha}])
        if "sha=broken&" in url:
            return _response({"not": "a list"})
        if "sha=empty&" in url:
            return _response([])
        raise AssertionError(url)

    payload = json.loads(
        read_live_github(PullRequestQuery("valkey", 7), fetch=fetch).canonical_payload
    )
    assert payload["release_membership_checked"] == ["found", "empty"]
    assert payload["released_in"] == ["found"]
    assert payload["number"] == 7, "the pull request read itself is intact"

    def no_releases(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        if url.endswith("/pulls/7"):
            return _response(value)
        raise GitHubReadError("GitHub returned HTTP 503")

    payload = json.loads(
        read_live_github(PullRequestQuery("valkey", 7), fetch=no_releases).canonical_payload
    )
    assert payload["released_in"] is None and payload["release_membership_checked"] == []


def test_release_by_tag_binds_the_endpoint_the_tag_and_the_notes_bound() -> None:
    from valkeyrie.live_github import MAX_RELEASE_NOTES_BYTES, ReleaseByTagQuery

    value = _release()
    value["tag_name"] = "9.1.0"
    value["body"] = "é" * (MAX_RELEASE_NOTES_BYTES // 2 + 10)
    value["html_url"] = "https://github.com/valkey-io/valkey/releases/tag/9.1.0"
    seen: list[str] = []

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        seen.append(url)
        return _response(value)

    payload = json.loads(
        read_live_github(ReleaseByTagQuery("valkey", "9.1.0"), fetch=fetch).canonical_payload
    )
    assert seen == ["https://api.github.com/repos/valkey-io/valkey/releases/tags/9.1.0"]
    assert payload["tag"] == "9.1.0"
    assert len(payload["body"].encode("utf-8")) <= MAX_RELEASE_NOTES_BYTES
    assert payload["body"].encode("utf-8").decode("utf-8") == payload["body"], "cut on a boundary"
    assert payload["body_truncated"] is True

    # A response for a different tag than asked is refused, and a tag is one path segment.
    other = dict(value)
    other["tag_name"] = "9.0.0"
    other["html_url"] = "https://github.com/valkey-io/valkey/releases/tag/9.0.0"
    with pytest.raises(LiveGitHubError, match="conflicts with the query"):
        read_live_github(ReleaseByTagQuery("valkey", "9.1.0"), fetch=lambda *a: _response(other))
    for bad in ("../latest", "9.1.0/notes", "-leading", "a" * 300):
        with pytest.raises(LiveGitHubError, match="malformed"):
            read_live_github(ReleaseByTagQuery("valkey", bad), fetch=fetch)


def test_an_author_search_carries_the_login_with_or_without_a_window() -> None:
    """ "What has madolson contributed" is an author search with no terms. The login must survive
    beside a window: the first version reassigned the qualifier list when a window was set and
    silently searched everyone's pull requests, reporting 137 for an author with 0."""
    seen: list[str] = []

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        seen.append(url)
        item = _search_item(number=4712, pull_request=True)
        return _response({"total_count": 161, "incomplete_results": False, "items": [item]})

    payload = json.loads(
        read_live_github(
            IssueSearchQuery(
                (), repository="valkey", per_page=20, kind="pull-request", author="madolson"
            ),
            fetch=fetch,
        ).canonical_payload
    )
    assert "q=repo%3Avalkey-io%2Fvalkey+is%3Apull-request+author%3Amadolson&sort=updated" in seen[0]
    assert payload["author"] == "madolson" and payload["terms"] == []
    assert payload["finding"].startswith("The search found 161 pull requests authored by madolson")

    seen.clear()
    read_live_github(
        IssueSearchQuery(
            (), repository="valkey", kind="pull-request", author="zuiderkwast", since="2026-08-22"
        ),
        fetch=fetch,
    )
    assert "author%3Azuiderkwast+is%3Amerged+merged%3A%3E%3D2026-08-22" in seen[0]

    for bad in ("-lead", "trailing-", "a--b", "x" * 40, "user name", "org/repo"):
        with pytest.raises(LiveGitHubError, match="GitHub login"):
            read_live_github(
                IssueSearchQuery((), repository="valkey", kind="issue", author=bad), fetch=fetch
            )
    # No author and no window: the term minimum still holds.
    with pytest.raises(LiveGitHubError, match="requires from 2"):
        read_live_github(IssueSearchQuery((), repository="valkey", kind="issue"), fetch=fetch)


def test_a_windowed_search_tallies_every_author_through_graphql_when_the_walk_is_complete() -> None:
    """ "Who contributed the most this month" is a tally over the whole set; the REST page lists
    twenty. The same qualifiers through GraphQL return logins only, walked in pages. The tally is
    present only when the walk covered the set the REST request described; a cut-short walk, a
    disagreeing count, or any failure omits it rather than counting part as whole."""
    rest_item = _search_item("valkey-glide", number=4712, pull_request=True)

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        if "valkey-glide" not in url:
            rest_item_core = _search_item(number=4712, pull_request=True)
            return _response(
                {"total_count": 3, "incomplete_results": False, "items": [rest_item_core]}
            )
        return _response({"total_count": 3, "incomplete_results": False, "items": [rest_item]})

    seen: list[Mapping[str, object]] = []

    def pages(*bodies: dict[str, object]) -> Any:
        it = iter(bodies)

        def graphql(
            query: str, variables: Mapping[str, object], timeout_seconds: float, max_bytes: int
        ) -> HttpResponse:
            seen.append(dict(variables))
            return _response(next(it))

        return graphql

    def page(
        logins: list[str | None], has_next: bool, cursor: str | None, total: int = 3
    ) -> dict[str, object]:
        return {
            "data": {
                "search": {
                    "issueCount": total,
                    "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                    "nodes": [{"author": {"login": x}} if x else {"author": None} for x in logins],
                }
            }
        }

    query = IssueSearchQuery((), repository="valkey-glide", kind="pull-request", since="2026-09-01")
    payload = json.loads(
        read_live_github(
            query,
            fetch=fetch,
            projects_fetch=pages(
                page(["dependabot", "currantw"], True, "c1"), page([None], False, None)
            ),
        ).canonical_payload
    )
    assert payload["authors_of_all"] == {"(deleted)": 1, "currantw": 1, "dependabot": 1}
    assert [v["after"] for v in seen] == [None, "c1"]
    assert (
        seen[0]["q"]
        == payload["query"]
        == ("repo:valkey-io/valkey-glide is:pull-request is:merged merged:>=2026-09-01")
    )

    # Incomplete: the count says 3, the walk delivered 2 and stopped. No tally.
    payload = json.loads(
        read_live_github(
            query, fetch=fetch, projects_fetch=pages(page(["a", "b"], False, None))
        ).canonical_payload
    )
    assert "authors_of_all" not in payload

    # A failing GraphQL read leaves the REST result intact.
    def broken(
        query: str, variables: Mapping[str, object], timeout_seconds: float, max_bytes: int
    ) -> HttpResponse:
        raise GitHubReadError("boom")

    payload = json.loads(
        read_live_github(query, fetch=fetch, projects_fetch=broken).canonical_payload
    )
    assert "authors_of_all" not in payload and payload["total_count"] == 3
    # A topic search without a window is not tallied at all.
    payload = json.loads(
        read_live_github(
            IssueSearchQuery(("release", "status"), repository="valkey"),
            fetch=fetch,
            projects_fetch=broken,
        ).canonical_payload
    )
    assert "authors_of_all" not in payload
