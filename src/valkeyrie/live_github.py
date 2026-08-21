"""Typed, bounded, read-only access to current public Valkey GitHub state."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol, TypeAlias, cast
from urllib.parse import quote, urlencode

from valkeyrie.github import GitHubFetcher, GitHubReadError, HttpResponse, fetch_public_github
from valkeyrie.request_audit import LiveObservation, create_live_observation


class LiveGitHubError(ValueError):
    """A live GitHub query or response failed closed."""


@dataclass(frozen=True)
class PullRequestQuery:
    repository: str
    number: int


@dataclass(frozen=True)
class IssueQuery:
    repository: str
    number: int


@dataclass(frozen=True)
class IssueSearchQuery:
    terms: tuple[str, ...]
    repository: str | None = None
    per_page: int = 20


@dataclass(frozen=True)
class LatestReleaseQuery:
    repository: str


@dataclass(frozen=True)
class WorkflowRunQuery:
    repository: str
    run_id: int


@dataclass(frozen=True)
class CheckRunQuery:
    repository: str
    check_run_id: int


@dataclass(frozen=True)
class ProjectQuery:
    number: int


LiveGitHubQuery: TypeAlias = (
    PullRequestQuery
    | IssueQuery
    | IssueSearchQuery
    | LatestReleaseQuery
    | WorkflowRunQuery
    | CheckRunQuery
    | ProjectQuery
)


class ProjectsGraphQLFetcher(Protocol):
    """Execute the fixed Projects query through separately authorized infrastructure."""

    def __call__(
        self,
        query: str,
        variables: Mapping[str, object],
        timeout_seconds: float,
        max_bytes: int,
    ) -> HttpResponse: ...


OWNER: Final = "valkey-io"
ALLOWED_REPOSITORIES: Final = frozenset(
    {
        ".github",
        "assets",
        "community",
        "iovalkey",
        "iovalkey-commands",
        "iovalkey-interface-generator",
        "libvalkey",
        "libvalkey-py",
        "one-time-for-planet",
        "planet",
        "spring-data-valkey",
        "valkey",
        "valkey-admin",
        "valkey-bloom",
        "valkey-bundle",
        "valkey-ci-agent",
        "valkey-container",
        "valkey-doc",
        "valkey-fuzzer",
        "valkey-glide",
        "valkey-glide-cpp",
        "valkey-glide-csharp",
        "valkey-glide-docs",
        "valkey-glide-php",
        "valkey-glide-ruby",
        "valkey-go",
        "valkey-hashes",
        "valkey-helm",
        "valkey-io.github.io",
        "valkey-java",
        "valkey-json",
        "valkey-ldap",
        "valkey-lua5.5",
        "valkey-luajit",
        "valkey-namespace",
        "valkey-operator",
        "valkey-perf-benchmark",
        "valkey-py",
        "valkey-release-automation",
        "valkey-search",
        "valkey-skills",
        "valkey-swift",
        "valkey-test-framework",
        "valkey-try-me",
        "valkeymodule-rs",
        "verify-provenance",
    }
)
REQUEST_TIMEOUT_SECONDS: Final = 8.0
MAX_RESPONSE_BYTES: Final = 256 * 1024
MAX_COLLECTION_ITEMS: Final = 100
MAX_ENTITY_ID: Final = 2**63 - 1
MAX_TAG_BYTES: Final = 255
MIN_SEARCH_TERMS: Final = 2
MAX_SEARCH_TERMS: Final = 8
MAX_SEARCH_TERM_BYTES: Final = 64
MAX_SEARCH_BODY_BYTES: Final = 16 * 1024
MAX_SEARCH_PER_PAGE: Final = 20
SEARCH_SORT: Final = "updated"
SEARCH_ORDER: Final = "desc"
PROJECTS_GRAPHQL_QUERY: Final = """\
query ValkeyrieProject($owner: String!, $number: Int!, $itemCount: Int!) {
  organization(login: $owner) {
    projectV2(number: $number) {
      id
      number
      title
      shortDescription
      public
      closed
      url
      items(first: $itemCount) {
        totalCount
        nodes {
          id
          type
          content {
            __typename
            ... on Issue {
              number
              title
              state
              url
              repository { nameWithOwner }
            }
            ... on PullRequest {
              number
              title
              state
              url
              repository { nameWithOwner }
            }
            ... on DraftIssue {
              title
              body
            }
          }
        }
      }
    }
  }
}
"""

_API_ROOT: Final = "https://api.github.com"
_WEB_ROOT: Final = "https://github.com"
_API_VERSION: Final = "valkeyrie.io/live-github/1"
_SHA: Final = re.compile(r"^[0-9a-f]{40}$")
_TAG: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_TIMESTAMP: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
_SEARCH_TERM: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_QUESTION_TOKEN: Final = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")
_UNSAFE_INFERRED_QUERY: Final = re.compile(
    r"https?://|\b(?:org|repo|user|is|type|state|label|milestone|sort|order|in|author|assignee):",
    re.IGNORECASE,
)
_PULL_REQUEST_NUMBER: Final = re.compile(
    r"\b(?:pr|pull[ -]?request)\s*(?:number\s*)?#?\s*([0-9]+)\b", re.IGNORECASE
)
_ISSUE_NUMBER: Final = re.compile(r"\bissue\s*(?:number\s*)?#?\s*([0-9]+)\b", re.IGNORECASE)
_PROJECT_NUMBER: Final = re.compile(
    r"\bproject(?:\s+v?2)?\s*(?:number\s*)?#?\s*([0-9]+)\b", re.IGNORECASE
)
_WORKFLOW_RUN_ID: Final = re.compile(
    r"\bworkflow(?:\s+run)?\s*(?:id|number)?\s*#?\s*([0-9]+)\b", re.IGNORECASE
)
_CHECK_RUN_ID: Final = re.compile(
    r"\bcheck(?:\s+run)?\s*(?:id|number)?\s*#?\s*([0-9]+)\b", re.IGNORECASE
)
_LATEST_RELEASE: Final = re.compile(
    r"\b(?:latest|newest|current)\s+(?:valkey\s+)?release\b", re.IGNORECASE
)
_LIVE_DISCOVERY_TERMS: Final = frozenset(
    {
        "backport",
        "event",
        "events",
        "meeting",
        "meetings",
        "progress",
        "release",
        "roadmap",
        "schedule",
        "status",
        "timeline",
        "upcoming",
        "workstream",
    }
)
_EVENT_DISCOVERY: Final = re.compile(r"\b(?:event|events|conference|meetup)\b", re.IGNORECASE)
_COMMUNITY_MEETING_DISCOVERY: Final = re.compile(
    r"\bcommunity\s+(?:meeting|meetings)\b", re.IGNORECASE
)
_SEARCH_STOP_WORDS: Final = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "do",
        "does",
        "for",
        "how",
        "i",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


def infer_live_query(question: str) -> LiveGitHubQuery | None:
    """Infer only the fixed live-query shapes supported by this adapter."""
    if type(question) is not str:
        raise LiveGitHubError("live query question must be text")
    try:
        encoded = question.encode("utf-8")
    except UnicodeEncodeError as error:
        raise LiveGitHubError("live query question must be valid UTF-8") from error
    if not 1 <= len(encoded) <= 4096:
        raise LiveGitHubError("live query question is outside its byte bound")
    if _UNSAFE_INFERRED_QUERY.search(question) is not None:
        raise LiveGitHubError("live query question contains an unsupported qualifier or URL")

    repository = _inferred_repository(question)
    pull_request_number = _single_inferred_id(question, _PULL_REQUEST_NUMBER, "pull request")
    if pull_request_number is not None:
        return PullRequestQuery(repository, pull_request_number)
    issue_number = _single_inferred_id(question, _ISSUE_NUMBER, "issue")
    if issue_number is not None:
        return IssueQuery(repository, issue_number)
    project_number = _single_inferred_id(question, _PROJECT_NUMBER, "project")
    if project_number is not None:
        return ProjectQuery(project_number)
    workflow_run_id = _single_inferred_id(question, _WORKFLOW_RUN_ID, "workflow run")
    if workflow_run_id is not None:
        return WorkflowRunQuery(repository, workflow_run_id)
    check_run_id = _single_inferred_id(question, _CHECK_RUN_ID, "check run")
    if check_run_id is not None:
        return CheckRunQuery(repository, check_run_id)
    if _LATEST_RELEASE.search(question) is not None:
        return LatestReleaseQuery(repository)

    terms = _inferred_search_terms(question, repository)
    if not any(term in _LIVE_DISCOVERY_TERMS for term in terms):
        return None
    if len(terms) == 1:
        terms = (*terms, "release" if terms[0] != "release" else "status")
    return IssueSearchQuery(terms=terms[:MAX_SEARCH_TERMS], repository=repository)


def _single_inferred_id(question: str, pattern: re.Pattern[str], label: str) -> int | None:
    matches = {int(value) for value in pattern.findall(question)}
    if len(matches) > 1:
        raise LiveGitHubError(f"live query question contains multiple {label} identifiers")
    if not matches:
        return None
    return _entity_id(matches.pop(), f"{label} number")


def _inferred_repository(question: str) -> str:
    lowered = question.casefold()
    matches = {
        repository
        for repository in ALLOWED_REPOSITORIES
        if repository != "valkey"
        and re.search(
            rf"(?<![a-z0-9._-])(?:{re.escape(OWNER)}/)?{re.escape(repository)}(?![a-z0-9._-])",
            lowered,
        )
        is not None
    }
    if re.search(rf"(?<![a-z0-9._-]){re.escape(OWNER)}/valkey(?![a-z0-9._-])", lowered):
        matches.add("valkey")
    if len(matches) > 1:
        raise LiveGitHubError("live query question names multiple reviewed repositories")
    if matches:
        return matches.pop()
    if _EVENT_DISCOVERY.search(question) is not None:
        return "valkey-io.github.io"
    if _COMMUNITY_MEETING_DISCOVERY.search(question) is not None:
        return "community"
    return "valkey"


def _inferred_search_terms(question: str, repository: str) -> tuple[str, ...]:
    ignored = {"valkey", OWNER, repository}
    terms: list[str] = []
    for token in _QUESTION_TOKEN.findall(question):
        normalized = token.casefold()
        if normalized in ignored or normalized in _SEARCH_STOP_WORDS:
            continue
        if not _is_meaningful_search_term(normalized) or normalized in terms:
            continue
        terms.append(normalized)
        if len(terms) == MAX_SEARCH_TERMS:
            break
    return tuple(terms)


def _is_meaningful_search_term(term: str) -> bool:
    try:
        encoded = term.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return (
        2 <= len(encoded) <= MAX_SEARCH_TERM_BYTES
        and _SEARCH_TERM.fullmatch(term) is not None
        and term not in _SEARCH_STOP_WORDS
        and (any(character.isalpha() for character in term) or "." in term)
    )


def read_live_github(
    query: LiveGitHubQuery,
    *,
    fetch: GitHubFetcher | None = None,
    projects_fetch: ProjectsGraphQLFetcher | None = None,
    observed_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> LiveObservation:
    """Execute one typed live query and return a complete content-addressed observation."""
    if isinstance(query, ProjectQuery):
        return _read_project(query, projects_fetch, observed_clock)

    source_url, object_type, normalizer = _rest_request(query)
    try:
        response = (fetch or fetch_public_github)(
            source_url, REQUEST_TIMEOUT_SECONDS, MAX_RESPONSE_BYTES
        )
    except GitHubReadError as error:
        raise LiveGitHubError(f"live GitHub read failed: {error}") from error
    value = _response_object(response, source="REST")
    payload = normalizer(value)
    return create_live_observation(
        observed_at=_observed_at(observed_clock),
        source_url=source_url,
        object_type=object_type,
        payload=payload,
    )


def normalize_project_response(value: object, *, number: int) -> dict[str, object]:
    """Normalize the fixed Projects GraphQL response without retaining unapproved fields."""
    project_number = _entity_id(number, "project number")
    root = _object(value, "Projects response")
    if "errors" in root:
        raise LiveGitHubError("GitHub Projects GraphQL returned errors")
    data = _object(root.get("data"), "Projects data")
    organization = _object(data.get("organization"), "Projects organization")
    project = _object(organization.get("projectV2"), "Projects project")
    if _integer(project, "number") != project_number:
        raise LiveGitHubError("GitHub Projects returned a conflicting project number")
    expected_url = f"{_WEB_ROOT}/orgs/{OWNER}/projects/{project_number}"
    _exact_url(project, "url", expected_url)
    if _boolean(project, "public") is not True:
        raise LiveGitHubError("GitHub Projects returned a non-public project")

    items_value = _object(project.get("items"), "project items")
    total_count = _bounded_count(items_value, "totalCount")
    nodes = _list(items_value, "nodes")
    if total_count != len(nodes):
        raise LiveGitHubError("GitHub Projects result is incomplete")
    items = [_project_item(item) for item in nodes]
    return {
        "api_version": _API_VERSION,
        "kind": "project",
        "owner": OWNER,
        "id": _text(project, "id", 256),
        "number": project_number,
        "title": _text(project, "title", 1024),
        "short_description": _nullable_text(project, "shortDescription", 4096),
        "closed": _boolean(project, "closed"),
        "url": expected_url,
        "total_count": total_count,
        "items": items,
    }


Normalizer: TypeAlias = Callable[[Mapping[str, object]], dict[str, object]]


def _rest_request(query: object) -> tuple[str, str, Normalizer]:
    if isinstance(query, PullRequestQuery):
        repository = _repository(query.repository)
        number = _entity_id(query.number, "pull request number")
        url = f"{_API_ROOT}/repos/{OWNER}/{repository}/pulls/{number}"
        return url, "pull_request", lambda value: _pull_request(value, repository, number)
    if isinstance(query, IssueQuery):
        repository = _repository(query.repository)
        number = _entity_id(query.number, "issue number")
        url = f"{_API_ROOT}/repos/{OWNER}/{repository}/issues/{number}"
        return url, "issue", lambda value: _issue(value, repository, number)
    if isinstance(query, IssueSearchQuery):
        terms = _search_terms(query.terms)
        search_repository = _optional_repository(query.repository)
        per_page = _search_per_page(query.per_page)
        qualifiers = [f"org:{OWNER}"]
        if search_repository is not None:
            qualifiers.append(f"repo:{OWNER}/{search_repository}")
        encoded_query = urlencode(
            {
                "q": " ".join((*qualifiers, *terms)),
                "sort": SEARCH_SORT,
                "order": SEARCH_ORDER,
                "per_page": str(per_page),
            }
        )
        url = f"{_API_ROOT}/search/issues?{encoded_query}"
        return (
            url,
            "issue",
            lambda value: _issue_search(value, terms, search_repository, per_page),
        )
    if isinstance(query, LatestReleaseQuery):
        repository = _repository(query.repository)
        url = f"{_API_ROOT}/repos/{OWNER}/{repository}/releases/latest"
        return url, "release", lambda value: _release(value, repository)
    if isinstance(query, WorkflowRunQuery):
        repository = _repository(query.repository)
        run_id = _entity_id(query.run_id, "workflow run ID")
        url = f"{_API_ROOT}/repos/{OWNER}/{repository}/actions/runs/{run_id}"
        return url, "workflow_run", lambda value: _workflow_run(value, repository, run_id)
    if isinstance(query, CheckRunQuery):
        repository = _repository(query.repository)
        check_run_id = _entity_id(query.check_run_id, "check run ID")
        url = f"{_API_ROOT}/repos/{OWNER}/{repository}/check-runs/{check_run_id}"
        return url, "check", lambda value: _single_check_run(value, repository, check_run_id)
    raise LiveGitHubError("unsupported live GitHub query type")


def _read_project(
    query: ProjectQuery,
    projects_fetch: ProjectsGraphQLFetcher | None,
    observed_clock: Callable[[], datetime],
) -> LiveObservation:
    number = _entity_id(query.number, "project number")
    if projects_fetch is None:
        raise LiveGitHubError("GitHub Projects requires an injected authenticated GraphQL fetcher")
    variables: Mapping[str, object] = {
        "owner": OWNER,
        "number": number,
        "itemCount": MAX_COLLECTION_ITEMS,
    }
    try:
        response = projects_fetch(
            PROJECTS_GRAPHQL_QUERY,
            variables,
            REQUEST_TIMEOUT_SECONDS,
            MAX_RESPONSE_BYTES,
        )
    except GitHubReadError as error:
        raise LiveGitHubError(f"GitHub Projects read failed: {error}") from error
    if (
        isinstance(response, HttpResponse)
        and type(response.status) is int
        and response.status in {401, 403}
    ):
        raise LiveGitHubError("GitHub Projects requires authenticated GraphQL access")
    value = _response_object(response, source="GraphQL")
    payload = normalize_project_response(value, number=number)
    return create_live_observation(
        observed_at=_observed_at(observed_clock),
        source_url=f"{_API_ROOT}/graphql",
        object_type="controller_status",
        payload=payload,
    )


def _pull_request(value: Mapping[str, object], repository: str, number: int) -> dict[str, object]:
    api_url = f"{_API_ROOT}/repos/{OWNER}/{repository}/pulls/{number}"
    web_url = f"{_WEB_ROOT}/{OWNER}/{repository}/pull/{number}"
    return {
        "api_version": _API_VERSION,
        "kind": "pull_request",
        "repository": repository,
        "id": _positive_integer(value, "id"),
        "number": _matching_integer(value, "number", number),
        "state": _choice(value, "state", {"open", "closed"}),
        "title": _text(value, "title", 1024),
        "body": _nullable_text(value, "body", 128 * 1024),
        "draft": _boolean(value, "draft"),
        "locked": _boolean(value, "locked"),
        "user": _login(value),
        "labels": _labels(value),
        "head_sha": _nested_sha(value, "head"),
        "base_sha": _nested_sha(value, "base"),
        "created_at": _timestamp(value, "created_at"),
        "updated_at": _timestamp(value, "updated_at"),
        "closed_at": _nullable_timestamp(value, "closed_at"),
        "merged_at": _nullable_timestamp(value, "merged_at"),
        "api_url": _exact_url(value, "url", api_url),
        "url": _exact_url(value, "html_url", web_url),
    }


def _issue(value: Mapping[str, object], repository: str, number: int) -> dict[str, object]:
    if "pull_request" in value:
        raise LiveGitHubError("issue query returned a pull request")
    api_url = f"{_API_ROOT}/repos/{OWNER}/{repository}/issues/{number}"
    web_url = f"{_WEB_ROOT}/{OWNER}/{repository}/issues/{number}"
    return {
        "api_version": _API_VERSION,
        "kind": "issue",
        "repository": repository,
        "id": _positive_integer(value, "id"),
        "number": _matching_integer(value, "number", number),
        "state": _choice(value, "state", {"open", "closed"}),
        "state_reason": _nullable_choice(
            value, "state_reason", {"completed", "not_planned", "reopened"}
        ),
        "title": _text(value, "title", 1024),
        "body": _nullable_text(value, "body", 128 * 1024),
        "locked": _boolean(value, "locked"),
        "user": _login(value),
        "labels": _labels(value),
        "created_at": _timestamp(value, "created_at"),
        "updated_at": _timestamp(value, "updated_at"),
        "closed_at": _nullable_timestamp(value, "closed_at"),
        "api_url": _exact_url(value, "url", api_url),
        "url": _exact_url(value, "html_url", web_url),
    }


def _issue_search(
    value: Mapping[str, object],
    terms: tuple[str, ...],
    repository: str | None,
    per_page: int,
) -> dict[str, object]:
    incomplete = _boolean(value, "incomplete_results")
    items = _list(value, "items")
    total_count = _integer(value, "total_count")
    if incomplete or total_count != len(items):
        raise LiveGitHubError("GitHub issue search result is incomplete")
    if len(items) > per_page:
        raise LiveGitHubError("GitHub issue search items exceed the requested page bound")
    normalized = [_search_item(_object(item, "search item"), repository) for item in items]
    identities = [(item["repository"], item["number"]) for item in normalized]
    if len(identities) != len(set(identities)):
        raise LiveGitHubError("GitHub issue search contains duplicate items")
    return {
        "api_version": _API_VERSION,
        "kind": "issue_search",
        "owner": OWNER,
        "repository": repository,
        "terms": list(terms),
        "sort": SEARCH_SORT,
        "order": SEARCH_ORDER,
        "per_page": per_page,
        "total_count": total_count,
        "items": normalized,
    }


def _search_item(value: Mapping[str, object], scoped_repository: str | None) -> dict[str, object]:
    repository_url = _text(value, "repository_url", 512)
    repository_prefix = f"{_API_ROOT}/repos/{OWNER}/"
    if not repository_url.startswith(repository_prefix):
        raise LiveGitHubError("GitHub search item repository is outside the fixed owner")
    repository = _repository(repository_url.removeprefix(repository_prefix))
    if scoped_repository is not None and repository != scoped_repository:
        raise LiveGitHubError("GitHub search item repository conflicts with the query")

    number = _entity_id(_integer(value, "number"), "GitHub search item number")
    api_url = f"{_API_ROOT}/repos/{OWNER}/{repository}/issues/{number}"
    if "pull_request" not in value:
        kind = "issue"
        web_url = f"{_WEB_ROOT}/{OWNER}/{repository}/issues/{number}"
    else:
        kind = "pull_request"
        web_url = f"{_WEB_ROOT}/{OWNER}/{repository}/pull/{number}"
        pull_request = _object(value.get("pull_request"), "GitHub search pull request")
        _exact_url(
            pull_request,
            "url",
            f"{_API_ROOT}/repos/{OWNER}/{repository}/pulls/{number}",
        )
        _exact_url(pull_request, "html_url", web_url)

    return {
        "kind": kind,
        "repository": repository,
        "number": number,
        "title": _text(value, "title", 1024),
        "body": _nullable_text(value, "body", MAX_SEARCH_BODY_BYTES),
        "state": _choice(value, "state", {"open", "closed"}),
        "labels": _labels(value),
        "milestone": _search_milestone(value.get("milestone"), repository),
        "updated_at": _timestamp(value, "updated_at"),
        "closed_at": _nullable_timestamp(value, "closed_at"),
        "api_url": _exact_url(value, "url", api_url),
        "url": _exact_url(value, "html_url", web_url),
    }


def _search_milestone(value: object, repository: str) -> dict[str, object] | None:
    if value is None:
        return None
    milestone = _object(value, "GitHub search milestone")
    number = _entity_id(_integer(milestone, "number"), "GitHub milestone number")
    return {
        "number": number,
        "title": _text(milestone, "title", 1024),
        "state": _choice(milestone, "state", {"open", "closed"}),
        "api_url": _exact_url(
            milestone,
            "url",
            f"{_API_ROOT}/repos/{OWNER}/{repository}/milestones/{number}",
        ),
        "url": _exact_url(
            milestone,
            "html_url",
            f"{_WEB_ROOT}/{OWNER}/{repository}/milestone/{number}",
        ),
    }


def _release(value: Mapping[str, object], repository: str) -> dict[str, object]:
    release_id = _positive_integer(value, "id")
    tag = _tag(_text(value, "tag_name", MAX_TAG_BYTES))
    api_url = f"{_API_ROOT}/repos/{OWNER}/{repository}/releases/{release_id}"
    web_url = f"{_WEB_ROOT}/{OWNER}/{repository}/releases/tag/{quote(tag, safe='/')}"
    return {
        "api_version": _API_VERSION,
        "kind": "release",
        "repository": repository,
        "id": release_id,
        "tag": tag,
        "target_commitish": _text(value, "target_commitish", 255),
        "name": _nullable_text(value, "name", 1024),
        "body": _nullable_text(value, "body", 128 * 1024),
        "draft": _boolean(value, "draft"),
        "prerelease": _boolean(value, "prerelease"),
        "author": _login(value, "author"),
        "created_at": _timestamp(value, "created_at"),
        "published_at": _nullable_timestamp(value, "published_at"),
        "api_url": _exact_url(value, "url", api_url),
        "url": _exact_url(value, "html_url", web_url),
    }


def _workflow_run(value: Mapping[str, object], repository: str, run_id: int) -> dict[str, object]:
    api_url = f"{_API_ROOT}/repos/{OWNER}/{repository}/actions/runs/{run_id}"
    web_url = f"{_WEB_ROOT}/{OWNER}/{repository}/actions/runs/{run_id}"
    return {
        "api_version": _API_VERSION,
        "kind": "workflow_run",
        "repository": repository,
        "id": _matching_integer(value, "id", run_id),
        "workflow_id": _positive_integer(value, "workflow_id"),
        "name": _nullable_text(value, "name", 1024),
        "display_title": _text(value, "display_title", 1024),
        "event": _text(value, "event", 255),
        "status": _choice(
            value,
            "status",
            {"queued", "in_progress", "completed", "requested", "waiting", "pending"},
        ),
        "conclusion": _nullable_choice(
            value,
            "conclusion",
            {
                "success",
                "failure",
                "neutral",
                "cancelled",
                "skipped",
                "timed_out",
                "action_required",
                "stale",
                "startup_failure",
            },
        ),
        "run_number": _positive_integer(value, "run_number"),
        "run_attempt": _positive_integer(value, "run_attempt"),
        "head_branch": _nullable_text(value, "head_branch", 255),
        "head_sha": _sha(value, "head_sha"),
        "created_at": _timestamp(value, "created_at"),
        "updated_at": _timestamp(value, "updated_at"),
        "api_url": _exact_url(value, "url", api_url),
        "url": _exact_url(value, "html_url", web_url),
    }


def _single_check_run(
    value: Mapping[str, object], repository: str, check_run_id: int
) -> dict[str, object]:
    return {
        "api_version": _API_VERSION,
        "kind": "check_run",
        "repository": repository,
        **_check_run(value, repository, check_run_id),
    }


def _check_run(
    value: Mapping[str, object], repository: str, expected_id: int | None = None
) -> dict[str, object]:
    check_run_id = _positive_integer(value, "id")
    if expected_id is not None and check_run_id != expected_id:
        raise LiveGitHubError("check run response returned a conflicting ID")
    api_url = f"{_API_ROOT}/repos/{OWNER}/{repository}/check-runs/{check_run_id}"
    web_url = f"{_WEB_ROOT}/{OWNER}/{repository}/runs/{check_run_id}"
    return {
        "id": check_run_id,
        "name": _text(value, "name", 1024),
        "status": _choice(
            value,
            "status",
            {"queued", "in_progress", "completed", "pending", "requested", "waiting"},
        ),
        "conclusion": _nullable_choice(
            value,
            "conclusion",
            {
                "success",
                "failure",
                "neutral",
                "cancelled",
                "skipped",
                "timed_out",
                "action_required",
                "stale",
                "startup_failure",
            },
        ),
        "head_sha": _sha(value, "head_sha"),
        "started_at": _nullable_timestamp(value, "started_at"),
        "completed_at": _nullable_timestamp(value, "completed_at"),
        "api_url": _exact_url(value, "url", api_url),
        "url": _exact_url(value, "html_url", web_url),
    }


def _project_item(value: object) -> dict[str, object]:
    item = _object(value, "project item")
    item_type = _choice(item, "type", {"ISSUE", "PULL_REQUEST", "DRAFT_ISSUE", "REDACTED"})
    content = item.get("content")
    normalized_content: dict[str, object] | None
    if item_type == "REDACTED":
        if content is not None:
            raise LiveGitHubError("redacted project item unexpectedly has content")
        normalized_content = None
    else:
        content_value = _object(content, "project item content")
        expected_typename = {
            "ISSUE": "Issue",
            "PULL_REQUEST": "PullRequest",
            "DRAFT_ISSUE": "DraftIssue",
        }[item_type]
        if _text(content_value, "__typename", 64) != expected_typename:
            raise LiveGitHubError("project item content has a conflicting type")
        if item_type == "DRAFT_ISSUE":
            normalized_content = {
                "kind": "draft_issue",
                "title": _text(content_value, "title", 1024),
                "body": _nullable_text(content_value, "body", 128 * 1024),
            }
        else:
            repository_value = _object(content_value.get("repository"), "project repository")
            full_name = _text(repository_value, "nameWithOwner", 256)
            prefix = f"{OWNER}/"
            if not full_name.startswith(prefix):
                raise LiveGitHubError("project item repository is outside the fixed owner")
            repository = _repository(full_name.removeprefix(prefix))
            number = _entity_id(_integer(content_value, "number"), "project item number")
            kind = "issue" if item_type == "ISSUE" else "pull_request"
            segment = "issues" if item_type == "ISSUE" else "pull"
            expected_url = f"{_WEB_ROOT}/{OWNER}/{repository}/{segment}/{number}"
            normalized_content = {
                "kind": kind,
                "repository": repository,
                "number": number,
                "title": _text(content_value, "title", 1024),
                "state": _choice(content_value, "state", {"OPEN", "CLOSED", "MERGED"}),
                "url": _exact_url(content_value, "url", expected_url),
            }
    return {
        "id": _text(item, "id", 256),
        "type": item_type.lower(),
        "content": normalized_content,
    }


def _response_object(response: object, *, source: str) -> Mapping[str, object]:
    if not isinstance(response, HttpResponse):
        raise LiveGitHubError(f"{source} fetcher returned the wrong response type")
    if type(response.status) is not int:
        raise LiveGitHubError(f"{source} response has an invalid status type")
    if response.status != 200:
        raise LiveGitHubError(f"{source} read returned HTTP {response.status}")
    if not isinstance(response.headers, Mapping):
        raise LiveGitHubError(f"{source} response has invalid headers")
    if type(response.body) is not bytes:
        raise LiveGitHubError(f"{source} response body must be bytes")
    if len(response.body) > MAX_RESPONSE_BYTES:
        raise LiveGitHubError(f"{source} response exceeded its byte bound")
    content_types: list[str] = []
    for name, header_value in response.headers.items():
        if type(name) is not str or type(header_value) is not str:
            raise LiveGitHubError(f"{source} response has invalid headers")
        if name.casefold() == "content-type":
            content_types.append(header_value)
    if len(content_types) != 1:
        raise LiveGitHubError(f"{source} response has an invalid content type")
    media_type = content_types[0].split(";", 1)[0].strip().casefold()
    if media_type not in {"application/json", "application/vnd.github+json"}:
        raise LiveGitHubError(f"{source} response has an invalid content type")
    try:
        value = cast(
            object,
            json.loads(
                response.body.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _StrictJsonError) as error:
        raise LiveGitHubError(f"{source} response is not strict JSON") from error
    return _object(value, f"{source} response")


class _StrictJsonError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _StrictJsonError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise _StrictJsonError(f"non-finite JSON constant: {value}")


def _repository(value: object) -> str:
    if type(value) is not str or value not in ALLOWED_REPOSITORIES:
        raise LiveGitHubError("repository is not in the fixed Valkey allowlist")
    return value


def _optional_repository(value: object) -> str | None:
    if value is None:
        return None
    return _repository(value)


def _search_terms(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise LiveGitHubError("search terms must be a tuple")
    raw_terms = cast(tuple[object, ...], value)
    if not MIN_SEARCH_TERMS <= len(raw_terms) <= MAX_SEARCH_TERMS:
        raise LiveGitHubError(
            f"search requires from {MIN_SEARCH_TERMS} through {MAX_SEARCH_TERMS} terms"
        )
    normalized: list[str] = []
    for raw_term in raw_terms:
        if type(raw_term) is not str:
            raise LiveGitHubError("search terms must be text")
        term = raw_term.strip().casefold()
        if not _is_meaningful_search_term(term):
            raise LiveGitHubError("search term is not a meaningful normalized token")
        normalized.append(term)
    if len(normalized) != len(set(normalized)):
        raise LiveGitHubError("search terms must be unique after normalization")
    return tuple(normalized)


def _search_per_page(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_SEARCH_PER_PAGE:
        raise LiveGitHubError(f"search per_page must be from 1 through {MAX_SEARCH_PER_PAGE}")
    return value


def _entity_id(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_ENTITY_ID:
        raise LiveGitHubError(f"{label} must be from 1 through {MAX_ENTITY_ID}")
    return value


def _tag(value: object) -> str:
    if type(value) is not str:
        raise LiveGitHubError("release tag is malformed")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise LiveGitHubError("release tag is malformed") from error
    if not 1 <= len(encoded) <= MAX_TAG_BYTES or _TAG.fullmatch(value) is None:
        raise LiveGitHubError("release tag is malformed")
    return value


def _object(value: object, label: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise LiveGitHubError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _list(value: Mapping[str, object], key: str) -> list[object]:
    result = value.get(key)
    if type(result) is not list:
        raise LiveGitHubError(f"GitHub field {key} must be an array")
    return cast(list[object], result)


def _integer(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if type(result) is not int or not 0 <= result <= MAX_ENTITY_ID:
        raise LiveGitHubError(f"GitHub field {key} must be a bounded integer")
    return result


def _matching_integer(value: Mapping[str, object], key: str, expected: int) -> int:
    result = _integer(value, key)
    if result != expected:
        raise LiveGitHubError(f"GitHub field {key} conflicts with the query")
    return result


def _bounded_count(value: Mapping[str, object], key: str) -> int:
    count = _integer(value, key)
    if count > MAX_COLLECTION_ITEMS:
        raise LiveGitHubError(f"GitHub field {key} exceeds the collection bound")
    return count


def _boolean(value: Mapping[str, object], key: str) -> bool:
    result = value.get(key)
    if type(result) is not bool:
        raise LiveGitHubError(f"GitHub field {key} must be boolean")
    return result


def _positive_integer(value: Mapping[str, object], key: str) -> int:
    return _entity_id(_integer(value, key), f"GitHub field {key}")


def _text(value: Mapping[str, object], key: str, max_bytes: int) -> str:
    result = value.get(key)
    if type(result) is not str:
        raise LiveGitHubError(f"GitHub field {key} must be text")
    try:
        encoded = result.encode("utf-8")
    except UnicodeEncodeError as error:
        raise LiveGitHubError(f"GitHub field {key} must be valid UTF-8") from error
    if not 1 <= len(encoded) <= max_bytes:
        raise LiveGitHubError(f"GitHub field {key} is outside its byte bound")
    return result


def _nullable_text(value: Mapping[str, object], key: str, max_bytes: int) -> str | None:
    result = value.get(key)
    if result is None:
        return None
    if type(result) is not str:
        raise LiveGitHubError(f"GitHub field {key} must be text or null")
    try:
        encoded = result.encode("utf-8")
    except UnicodeEncodeError as error:
        raise LiveGitHubError(f"GitHub field {key} must be valid UTF-8") from error
    if len(encoded) > max_bytes:
        raise LiveGitHubError(f"GitHub field {key} is outside its byte bound")
    return result


def _choice(value: Mapping[str, object], key: str, allowed: set[str]) -> str:
    result = _text(value, key, 255)
    if result not in allowed:
        raise LiveGitHubError(f"GitHub field {key} has an unsupported value")
    return result


def _nullable_choice(value: Mapping[str, object], key: str, allowed: set[str]) -> str | None:
    if value.get(key) is None:
        return None
    return _choice(value, key, allowed)


def _sha(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if type(result) is not str or _SHA.fullmatch(result) is None:
        raise LiveGitHubError(f"GitHub field {key} must be a full lowercase Git SHA")
    return result


def _nested_sha(value: Mapping[str, object], key: str) -> str:
    return _sha(_object(value.get(key), f"GitHub field {key}"), "sha")


def _timestamp(value: Mapping[str, object], key: str) -> str:
    result = _text(value, key, 64)
    if _TIMESTAMP.fullmatch(result) is None:
        raise LiveGitHubError(f"GitHub field {key} must be a canonical UTC timestamp")
    try:
        datetime.fromisoformat(result.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise LiveGitHubError(f"GitHub field {key} must be a canonical UTC timestamp") from error
    return result


def _nullable_timestamp(value: Mapping[str, object], key: str) -> str | None:
    if value.get(key) is None:
        return None
    return _timestamp(value, key)


def _exact_url(value: Mapping[str, object], key: str, expected: str) -> str:
    if value.get(key) != expected:
        raise LiveGitHubError(f"GitHub field {key} is not the canonical URL")
    return expected


def _login(value: Mapping[str, object], key: str = "user") -> str:
    user = _object(value.get(key), f"GitHub field {key}")
    return _text(user, "login", 255)


def _labels(value: Mapping[str, object]) -> list[str]:
    labels = _list(value, "labels")
    if len(labels) > MAX_COLLECTION_ITEMS:
        raise LiveGitHubError("GitHub labels exceed the collection bound")
    names = [_text(_object(label, "GitHub label"), "name", 255) for label in labels]
    if len(names) != len(set(names)):
        raise LiveGitHubError("GitHub labels contain duplicate names")
    return names


def _observed_at(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise LiveGitHubError("observation clock must return an aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
