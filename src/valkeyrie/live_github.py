"""Typed, bounded, read-only access to current public Valkey GitHub state."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
    # Further repositories to search alongside ``repository``. A cross-repository question
    # ("does valkey-glide support X from valkey 9.2?") is searched in every repository it names.
    repositories: tuple[str, ...] = ()
    # A date window: pull requests MERGED on or after this day (scope pull-request), or issues
    # CREATED on or after it (scope issue). "What merged this week" is a window with no terms, so
    # the term minimum is waived when a window is set; results are then ordered newest first and
    # bodies cut short, since a period summary wants many titles rather than a few whole bodies.
    since: str | None = None
    # The window's last day, inclusive; None means through today. "In August" is since and until.
    until: str | None = None
    # A GitHub login: only items this user authored. "What has madolson contributed" is an author
    # search with no terms; searching the login as a word finds mentions, not authorship.
    author: str | None = None
    # GitHub requires an explicit is:issue or is:pull-request on authenticated
    # search/issues requests and returns 422 without one. Anonymous requests are not yet
    # enforced, which is why this was invisible until the runtime started authenticating.
    kind: str = "issue"


@dataclass(frozen=True)
class LatestReleaseQuery:
    repository: str


@dataclass(frozen=True)
class ReleaseByTagQuery:
    """One release by its tag, with its notes whole.

    A release LIST keeps notes for the newest three only, so anything older ("what was new in
    9.1.0?") had nothing to answer from. This reads exactly the release the asker named.
    """

    repository: str
    tag: str


@dataclass(frozen=True)
class ReleaseListQuery:
    """The most recent releases INCLUDING prereleases.

    /releases/latest omits prereleases, so a question about a release candidate answered from it
    is answered wrongly: with 9.2.0-rc1 published, that endpoint still reports 9.1.2. The list
    endpoint is the only one that can say whether an rc exists.
    """

    repository: str
    per_page: int = 8


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
    | ReleaseByTagQuery
    | ReleaseListQuery
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
# Pages of MAX_COLLECTION_ITEMS a single board read may walk: 500 items covers every valkey-io
# release board with room, and bounds the work one question can cause.
MAX_PROJECT_PAGES: Final = 5
# Statuses that mean an item is finished, so it is counted but not listed. Everything else is
# remaining work and is listed in full (up to MAX_REMAINING_ITEMS).
_COMPLETED_STATUSES: Final[frozenset[str]] = frozenset({"Done", "Merged", "Closed", "Released"})
MAX_REMAINING_ITEMS: Final = 120
MAX_ENTITY_ID: Final = 2**63 - 1
MAX_TAG_BYTES: Final = 255
MIN_SEARCH_TERMS: Final = 2
MAX_SEARCH_TERMS: Final = 8
MAX_RELEASE_LIST: Final = 20
MAX_SEARCH_TERM_BYTES: Final = 64
# A topic search lists five items; at 16 KiB each the search alone could exceed the live share
# and be dropped whole by the budget. 6 KiB keeps the design section of a long issue and lets a
# five-item search fit beside a board or a release list.
MAX_SEARCH_BODY_BYTES: Final = 6 * 1024
# Per-release notes bound inside a release LIST; a single release keeps its full notes.
MAX_RELEASE_LIST_BODY_BYTES: Final = 3 * 1024
RELEASE_LIST_BODIES: Final = 3
# Notes of one release read by tag: whole for every release measured (largest 22 KB).
MAX_RELEASE_NOTES_BYTES: Final = 24 * 1024
# Body bound per item in a date-window search: many titles, short bodies.
MAX_WINDOW_BODY_BYTES: Final = 600
_SEARCH_SINCE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# GitHub's login rules: 1 to 39 alphanumerics or single hyphens, not at either end.
_GITHUB_LOGIN: Final = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
MAX_SEARCH_PER_PAGE: Final = 20
# Supplementary searches carry whole issue bodies, so they take a smaller page than
# a routed search: twenty bodies exceed MAX_RESPONSE_BYTES.
SUPPLEMENT_PER_PAGE: Final = 5
# A date-window search lists more, shorter items: bodies are cut at MAX_WINDOW_BODY_BYTES.
WINDOW_PER_PAGE: Final = 20
# Best match, GitHub's default when no sort is sent. Ordering by recency ranked any issue that
# mentioned both words anywhere in a long body above the one titled with them: "vector set"
# returned a radix tree proposal and an SSCAN bug first, and the vector sets datatype issue not
# at all in five. Best match puts the titled item first; recency is a bad proxy for relevance.
SEARCH_SORT: Final = "best-match"
PROJECTS_GRAPHQL_QUERY: Final = """\
query ValkeyrieProject($owner: String!, $number: Int!, $itemCount: Int!, $after: String) {
  organization(login: $owner) {
    projectV2(number: $number) {
      id
      number
      title
      shortDescription
      public
      closed
      url
      items(first: $itemCount, after: $after) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          type
          status: fieldValueByName(name: "Status") {
            ... on ProjectV2ItemFieldSingleSelectValue { name }
          }
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
        # Pronouns and demonstratives carry no subject, so a question made only of them would
        # otherwise clear the two-term floor and spend a search on "it work".
        "it",
        "its",
        "is",
        "of",
        "on",
        "or",
        "the",
        "them",
        "they",
        "this",
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


# Words that mark a question as being about how something was designed, whether it shipped, or
# what state it is in. A supplementary GitHub search is only worth its latency and its share of the
# authenticated 30-per-minute search budget for these: issue and pull request search cannot see
# file contents, so a documentation or governance question ("how do I use GET", "who leads the
# TSC") gets nothing back but unrelated bug reports, measured.
#
# This is deliberately a SUPERSET of what previously reached a search, judged by intent rather than
# by retrieval score: score does not separate these cases, and was measured inverted, with the
# compression question scoring HIGHER than the documentation questions that need no supplement.
_SUPPLEMENT_INTENT_TERMS: Final = frozenset(
    {
        "available",
        "design",
        "designed",
        "implement",
        "implemented",
        "implementation",
        "landed",
        "mechanism",
        "merged",
        "negotiate",
        "negotiated",
        "planned",
        "proposal",
        "proposed",
        "rfc",
        "supported",
        "supports",
        "upcoming",
        "work",
        "working",
        "works",
    }
)


def infer_supplementary_search(
    question: str, *, kind: str = "pull-request", force: bool = False
) -> IssueSearchQuery | None:
    """Infer a repository-scoped issue search to supplement corpus evidence.

    ``infer_live_query`` decides the live ROUTE and deliberately requires a discovery word
    such as "status" or "release", so "How does Valkey replication compression work?" infers
    nothing. That question is answerable: the design lives in open pull requests, and the
    corpus cannot document a feature that has not shipped. This inference drops the
    discovery-word gate so the corpus can be supplemented, and it never changes routing.

    Returns ``None`` when the question shows no design, shipped-state, or live-state intent, so a
    documentation or governance question spends no GitHub quota and adds no latency.

    The gate is intent, not precision. A documented mechanism phrased as "how does X work" still
    searches, because language cannot tell a documented feature from an unshipped one: only whether
    the corpus could answer can, and that is known after drafting rather than before. Erring toward
    searching keeps every question that works today working.
    """
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
    repositories = _inferred_repositories(question)
    terms = _inferred_search_terms(question, repositories)
    # Two terms is the floor for a search worth making: one term matches too much of the
    # repository to be evidence, and zero means the question carried no subject at all.
    if len(terms) < 2:
        return None
    # ``force`` bypasses the intent gate. It is used for one purpose: a second attempt after the
    # corpus abstained, where the question has already proven the gate wrong by needing more.
    lowered = {word.strip("?.,:;!()").lower() for word in question.split()}
    if not force and not lowered & (_SUPPLEMENT_INTENT_TERMS | _LIVE_DISCOVERY_TERMS):
        return None
    # Five items, not the default twenty. A search returns whole issue bodies, and twenty of
    # them exceeds MAX_RESPONSE_BYTES, which made every supplemented answer fail with
    # "REST response exceeded its byte bound". Five is enough to establish what a proposed
    # feature does and whether it has landed, which is all a supplement is for.
    return IssueSearchQuery(
        terms=terms[:MAX_SEARCH_TERMS],
        repository=repositories[0],
        repositories=repositories[1:],
        per_page=SUPPLEMENT_PER_PAGE,
        kind=kind,
    )


def _single_inferred_id(question: str, pattern: re.Pattern[str], label: str) -> int | None:
    matches = {int(value) for value in pattern.findall(question)}
    if len(matches) > 1:
        raise LiveGitHubError(f"live query question contains multiple {label} identifiers")
    if not matches:
        return None
    return _entity_id(matches.pop(), f"{label} number")


def _inferred_repositories(question: str) -> tuple[str, ...]:
    """Every reviewed repository the question names, or ("valkey",) when it names none.

    Naming two used to raise, which killed the supplement on exactly the cross-repository
    questions ("does valkey-glide support X from valkey 9.2?"). A search can carry several repo:
    qualifiers, so the question is searched in all of them.
    """
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
    # "Valkey" on its own usually means the project, not the core repository ("upcoming Valkey
    # events" is about the website). The core repository counts when named explicitly as
    # valkey-io/valkey, or when the bare word stands beside ANOTHER repository's name, where the
    # contrast is what makes it a repository ("does valkey-glide support X from valkey 9.2?").
    if re.search(rf"(?<![a-z0-9._-]){re.escape(OWNER)}/valkey(?![a-z0-9._-])", lowered):
        matches.add("valkey")
    elif matches:
        stripped = re.sub(r"valkey-[a-z0-9.-]+", " ", lowered)
        if re.search(r"(?<![a-z0-9._/-])valkey(?![a-z0-9._-])", stripped):
            matches.add("valkey")
    if not matches:
        # No repository named: the topic decides. Events live on the website repository and
        # community matters in the community repository, as before.
        if _EVENT_DISCOVERY.search(question) is not None:
            return ("valkey-io.github.io",)
        if _COMMUNITY_MEETING_DISCOVERY.search(question) is not None:
            return ("community",)
        return ("valkey",)
    if len(matches) > MAX_SEARCH_REPOSITORIES:
        raise LiveGitHubError("live query question names too many repositories")
    return tuple(sorted(matches))


def _inferred_repository(question: str) -> str:
    """The single repository a question names, for lookups that take exactly one."""
    repositories = _inferred_repositories(question)
    if len(repositories) > 1:
        raise LiveGitHubError("live query question names multiple reviewed repositories")
    return repositories[0]


def _inferred_search_terms(question: str, repository: str | tuple[str, ...]) -> tuple[str, ...]:
    named = (repository,) if isinstance(repository, str) else repository
    ignored = {"valkey", OWNER, *named}
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
    if isinstance(query, PullRequestQuery) and payload.get("merged_at") is not None:
        payload = _with_release_membership(payload, query.repository, fetch or fetch_public_github)
    return create_live_observation(
        observed_at=_observed_at(observed_clock),
        source_url=source_url,
        object_type=object_type,
        payload=payload,
    )


# How many recent releases a merged pull request is checked against. Every release board tracks
# at most the current line and two supported ones, so eight recent tags cover them with room.
MAX_RELEASE_MEMBERSHIP_CHECKS: Final = 8
# Commits listed per tag in the one-second window around the merge. A window that holds this many
# is inconclusive (the merge may be on the next page) and the tag is left unchecked; measured
# windows hold one commit.
MEMBERSHIP_COMMIT_PAGE: Final = 50
# Repositories one supplementary search may span; a question naming more is asking something else.
MAX_SEARCH_REPOSITORIES: Final = 4


def _with_release_membership(
    payload: dict[str, object], repository: str, fetch: GitHubFetcher
) -> dict[str, object]:
    """Add which recent releases contain a merged pull request's merge commit.

    "Has X shipped?" is the question release-watchers ask most, and merged and released are
    different facts: a change merged to unstable is in no release until a tag includes it.

    The test is exact and small. GitHub's compare endpoint answers it but embeds every file diff,
    over a megabyte for a diverged tag, past every bound here; GraphQL's compare needs the repo
    scope this token deliberately lacks. Instead, the commits reachable from a tag are listed for
    the one-second window around the merge instant: a tag that contains the merge commit returns
    it (about 8 KB), one that does not returns an empty list. merged_at is the merge commit's
    committer date, which is what that listing is indexed by.

    A failure to check any tag leaves it out of release_membership_checked rather than failing the
    pull request read: membership is an addition to an answer the pull request itself supports.
    """
    merge_commit = payload.get("merge_commit_sha")
    merged_at = payload.get("merged_at")
    if (
        not isinstance(merge_commit, str)
        or _SHA.fullmatch(merge_commit) is None
        or not isinstance(merged_at, str)
    ):
        return {**payload, "released_in": None, "release_membership_checked": []}
    repo = _repository(repository)
    try:
        merged = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
    except ValueError:
        return {**payload, "released_in": None, "release_membership_checked": []}
    since = (merged - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    until = (merged + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        releases = _response_object(
            fetch(
                f"{_API_ROOT}/repos/{OWNER}/{repo}/releases"
                f"?per_page={MAX_RELEASE_MEMBERSHIP_CHECKS}",
                REQUEST_TIMEOUT_SECONDS,
                MAX_RESPONSE_BYTES,
            ),
            source="REST",
        )
        tags = [
            _tag(_text(_object(item, "release"), "tag_name", MAX_TAG_BYTES))
            for item in _list(releases, "items")[:MAX_RELEASE_MEMBERSHIP_CHECKS]
        ]
    except (GitHubReadError, LiveGitHubError):
        return {**payload, "released_in": None, "release_membership_checked": []}

    def probe(tag: str) -> tuple[str, bool] | None:
        """(tag, contains) when the tag was checked conclusively, None when it was not.

        Everything about one tag sits inside this boundary: a malformed listing for one tag must
        leave that tag unchecked, not turn a valid pull request read into a failure. A listing
        that fills its page is inconclusive: the merge commit may be on a page not read, so the
        tag is NOT recorded as checked, since "checked and absent" is the claim an answer will
        make from it.
        """
        try:
            listing = _response_object(
                fetch(
                    f"{_API_ROOT}/repos/{OWNER}/{repo}/commits?sha={quote(tag, safe='')}"
                    f"&since={since}&until={until}&per_page={MEMBERSHIP_COMMIT_PAGE}",
                    REQUEST_TIMEOUT_SECONDS,
                    MAX_RESPONSE_BYTES,
                ),
                source="REST",
            )
            commits = _list(listing, "items")
        except (GitHubReadError, LiveGitHubError):
            return None
        if len(commits) >= MEMBERSHIP_COMMIT_PAGE:
            return None
        return tag, any(
            isinstance(commit, Mapping) and commit.get("sha") == merge_commit for commit in commits
        )

    # The probes are independent reads of 0.2 to 0.5 s each; eight in sequence measured three
    # seconds, which was the whole cost of the feature.
    with ThreadPoolExecutor(max_workers=min(len(tags), MAX_RELEASE_MEMBERSHIP_CHECKS) or 1) as pool:
        results = list(pool.map(probe, tags))
    checked = [tag for tag, _ in (r for r in results if r is not None)]
    contained = [tag for tag, inside in (r for r in results if r is not None) if inside]
    return {**payload, "released_in": contained, "release_membership_checked": checked}


def normalize_project_response(value: object, *, number: int) -> dict[str, object]:
    """Normalize one or more pages of the fixed Projects query into one complete board.

    ``value`` is a single response object, or a sequence of them for a board that spanned pages.
    Every page must describe the same project; the pages together must hold exactly totalCount
    items, so a truncated walk is refused rather than presented as the board.
    """
    project_number = _entity_id(number, "project number")
    pages = (
        list(value) if isinstance(value, Sequence) and not isinstance(value, Mapping) else [value]
    )
    if not 1 <= len(pages) <= MAX_PROJECT_PAGES:
        raise LiveGitHubError("GitHub Projects page count is outside its bound")
    expected_url = f"{_WEB_ROOT}/orgs/{OWNER}/projects/{project_number}"
    project: Mapping[str, object] | None = None
    nodes: list[object] = []
    total_count: int | None = None
    for page in pages:
        root = _object(page, "Projects response")
        if "errors" in root:
            raise LiveGitHubError("GitHub Projects GraphQL returned errors")
        data = _object(root.get("data"), "Projects data")
        organization = _object(data.get("organization"), "Projects organization")
        current = _object(organization.get("projectV2"), "Projects project")
        if _integer(current, "number") != project_number:
            raise LiveGitHubError("GitHub Projects returned a conflicting project number")
        _exact_url(current, "url", expected_url)
        if _boolean(current, "public") is not True:
            raise LiveGitHubError("GitHub Projects returned a non-public project")
        if project is not None and _text(current, "id", 256) != _text(project, "id", 256):
            raise LiveGitHubError("GitHub Projects pages describe different projects")
        project = current
        items_value = _object(current.get("items"), "project items")
        page_total = _positive_or_zero_integer(items_value, "totalCount")
        if total_count is None:
            total_count = page_total
        elif page_total != total_count:
            raise LiveGitHubError("GitHub Projects pages disagree on the item total")
        page_nodes = _list(items_value, "nodes")
        if len(page_nodes) > MAX_COLLECTION_ITEMS:
            raise LiveGitHubError("GitHub project items exceed the requested page bound")
        nodes.extend(page_nodes)
    if project is None:  # pragma: no cover - at least one page is required above
        raise LiveGitHubError("GitHub Projects returned no pages")
    if len(nodes) != total_count:
        raise LiveGitHubError("GitHub Projects result is incomplete")
    items = [_project_item(item) for item in nodes]
    if len({item["id"] for item in items}) != len(items):
        raise LiveGitHubError("GitHub Projects pages repeat an item")
    # The board's Status column is what "what is left" means to a release manager: Todo, Needs
    # Review and To be backported are remaining work; Done and Merged are not. Every status is
    # totalled, and only items in a remaining-work status are listed in full. A 268-item board
    # listed whole is 70 KB, past the evidence bound, and the 243 merged rows carry nothing an
    # answer about remaining work needs; the totals say the board is 91% done, the open items say
    # what the 9% is.
    by_status: dict[str, int] = {}
    for item in items:
        label = cast(str | None, item.get("status")) or "(no status)"
        by_status[label] = by_status.get(label, 0) + 1
    remaining = [
        item
        for item in items
        if (cast(str | None, item.get("status")) or "(no status)") not in _COMPLETED_STATUSES
    ]
    # Bounded for the evidence budget; a board with more open work than this is summarised by its
    # totals and the newest items, and the payload says so.
    remaining_truncated = len(remaining) > MAX_REMAINING_ITEMS
    remaining = remaining[:MAX_REMAINING_ITEMS]
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
        "items_total": total_count,
        "items_by_status": dict(sorted(by_status.items())),
        "remaining_items": remaining,
        "remaining_truncated": remaining_truncated,
        "partial": False,
    }


def _project_page_info(value: Mapping[str, object]) -> tuple[bool, str | None]:
    """Return (has_next_page, end_cursor) for one page, fail-closed on shape."""
    try:
        items = cast(
            Mapping[str, object],
            cast(
                Mapping[str, object],
                cast(
                    Mapping[str, object], cast(Mapping[str, object], value["data"])["organization"]
                )["projectV2"],
            )["items"],
        )
    except (KeyError, TypeError) as error:
        raise LiveGitHubError("GitHub Projects response lacks page information") from error
    if items.get("pageInfo") is None:
        return False, None
    info = _object(items.get("pageInfo"), "project pageInfo")
    has_next = _boolean(info, "hasNextPage")
    if not has_next:
        return False, None
    cursor = info.get("endCursor")
    if not isinstance(cursor, str) or not cursor or len(cursor) > 512:
        raise LiveGitHubError("GitHub Projects page cursor is malformed")
    return True, cursor


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
        since = _search_since(query.since)
        until = _search_since(query.until)
        author = _search_author(query.author)
        if until is not None and since is None:
            raise LiveGitHubError("search window end requires a start")
        if until is not None and since is not None and until < since:
            raise LiveGitHubError("search window end precedes its start")
        terms = _search_terms(query.terms, minimum=0 if (since or author) else MIN_SEARCH_TERMS)
        search_repository = _optional_repository(query.repository)
        per_page = _search_per_page(query.per_page)
        scope: tuple[str, ...] = tuple(
            dict.fromkeys(
                r
                for r in ((search_repository,) if search_repository else ())
                + tuple(query.repositories)
                if r is not None
            )
        )
        if len(scope) > MAX_SEARCH_REPOSITORIES:
            raise LiveGitHubError("issue search names too many repositories")
        for r in scope:
            _repository(r)
        # A repo: qualifier scopes to one repository, and several of them are OR-ed by GitHub.
        # org: must NOT be sent alongside: it unions the whole organization back in, and
        # _search_item then correctly rejects the sibling-repository items as out of scope.
        qualifiers = [f"repo:{OWNER}/{r}" for r in scope] or [f"org:{OWNER}"]
        if query.kind not in {"issue", "pull-request"}:
            raise LiveGitHubError("issue search kind must be issue or pull-request")
        window: list[str] = []
        parameters: dict[str, str] = {}
        if author is not None:
            window.append(f"author:{author}")
            # Authorship alone has nothing to rank by; newest first is what "their work" wants.
            parameters = {"sort": "updated", "order": "desc"}
        if since is not None:
            # A merged window needs is:merged, or unmerged pull requests with a merged date of
            # nothing would be excluded silently and the count would mislead. Newest first: a
            # period summary wants the recent end, and best match has nothing to match on.
            span = f"{since}..{until}" if until else f">={since}"
            window.extend(
                ["is:merged", f"merged:{span}"]
                if query.kind == "pull-request"
                else [f"created:{span}"]
            )
            parameters = {"sort": "updated", "order": "desc"}
        # Without a window, no sort parameter: GitHub has no value that names best match, it is
        # what you get by not asking for a sort. The payload records the ordering by name.
        encoded_query = urlencode(
            {
                "q": " ".join((*qualifiers, f"is:{query.kind}", *window, *terms)),
                **parameters,
                "per_page": str(per_page),
            }
        )
        url = f"{_API_ROOT}/search/issues?{encoded_query}"
        return (
            url,
            "issue",
            lambda value: _issue_search(
                value,
                terms,
                scope or None,
                per_page,
                query.kind,
                since=since,
                until=until,
                author=author,
            ),
        )
    if isinstance(query, LatestReleaseQuery):
        repository = _repository(query.repository)
        url = f"{_API_ROOT}/repos/{OWNER}/{repository}/releases/latest"
        return url, "release", lambda value: _release(value, repository)
    if isinstance(query, ReleaseByTagQuery):
        repository = _repository(query.repository)
        tag = _tag(query.tag)
        if "/" in tag or ".." in tag:
            raise LiveGitHubError("release tag is malformed")
        url = f"{_API_ROOT}/repos/{OWNER}/{repository}/releases/tags/{quote(tag, safe='')}"
        return url, "release", lambda value: _release_by_tag(value, repository, tag)
    if isinstance(query, ReleaseListQuery):
        repository = _repository(query.repository)
        per_page = query.per_page
        if type(per_page) is not int or not 1 <= per_page <= MAX_RELEASE_LIST:
            raise LiveGitHubError("release list page size is outside its bound")
        url = f"{_API_ROOT}/repos/{OWNER}/{repository}/releases?per_page={per_page}"
        return url, "release", lambda value: _release_list(value, repository, per_page)
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
    # A board is read whole. GraphQL offers no filter on item state, and "what is left" is
    # answered by the board's Status column, which only a complete read can total: the release
    # boards hold 150 to 300 items, so two or three pages. The page count is bounded so a runaway
    # board cannot turn one question into an unbounded walk.
    pages: list[Mapping[str, object]] = []
    seen_cursors: set[str] = set()
    after: str | None = None
    for _ in range(MAX_PROJECT_PAGES):
        variables: Mapping[str, object] = {
            "owner": OWNER,
            "number": number,
            "itemCount": MAX_COLLECTION_ITEMS,
            "after": after,
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
        pages.append(value)
        has_next, cursor = _project_page_info(value)
        if not has_next:
            break
        # A cursor that does not advance would walk the same page until the cap and then, if
        # the node count happened to equal the total, present a duplicated board as complete.
        if cursor is None or cursor in seen_cursors:
            raise LiveGitHubError("GitHub Projects paging did not advance")
        seen_cursors.add(cursor)
        after = cursor
    else:
        # Every page said another followed. The walk is bounded, so the board is not whole.
        raise LiveGitHubError("GitHub Projects board exceeds the page bound")
    payload = normalize_project_response(pages, number=number)
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
        "merge_commit_sha": _nullable_sha(value, "merge_commit_sha"),
        "api_url": _exact_url(value, "url", api_url),
        "url": _exact_url(value, "html_url", web_url),
    }


def _issue(value: Mapping[str, object], repository: str, number: int) -> dict[str, object]:
    # GitHub shares one number space between issues and pull requests, and its issues endpoint
    # serves both. Refusing a pull request here meant "what is the status of issue #3853?" failed
    # outright, reported to the asker as live data being unavailable, when the number does name a
    # real object whose status is known. What it is is stated in the payload instead of guessed at,
    # so an answer can say "pull request" rather than calling it an issue.
    nested = value.get("pull_request")
    is_pull_request = isinstance(nested, Mapping)
    merged_at = (
        _nullable_timestamp(nested, "merged_at")
        if isinstance(nested, Mapping) and "merged_at" in nested
        else None
    )
    api_url = f"{_API_ROOT}/repos/{OWNER}/{repository}/issues/{number}"
    leaf = "pull" if is_pull_request else "issues"
    web_url = f"{_WEB_ROOT}/{OWNER}/{repository}/{leaf}/{number}"
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
        "is_pull_request": is_pull_request,
        "merged_at": merged_at,
        "api_url": _exact_url(value, "url", api_url),
        "url": _exact_url(value, "html_url", web_url),
    }


def _issue_search(
    value: Mapping[str, object],
    terms: tuple[str, ...],
    repositories: tuple[str, ...] | None,
    per_page: int,
    kind: str = "issue",
    *,
    since: str | None = None,
    until: str | None = None,
    author: str | None = None,
) -> dict[str, object]:
    incomplete = _boolean(value, "incomplete_results")
    items = _list(value, "items")
    total_count = _integer(value, "total_count")
    # total_count counts matches across every page, while items holds only the requested
    # page, so total_count > len(items) is ordinary pagination rather than a failure. The
    # bounded page is still an honest observation: the payload reports both numbers, so a
    # consumer can see it is a partial view. Only GitHub's own incomplete_results flag, or
    # more items than matches, means the result cannot be trusted.
    if incomplete or total_count < len(items):
        raise LiveGitHubError("GitHub issue search result is incomplete")
    if len(items) > per_page:
        raise LiveGitHubError("GitHub issue search items exceed the requested page bound")
    normalized = [
        _search_item(
            _object(item, "search item"),
            repositories,
            compact=since is not None or author is not None,
        )
        for item in items
    ]
    identities = [(item["repository"], item["number"]) for item in normalized]
    if len(identities) != len(set(identities)):
        raise LiveGitHubError("GitHub issue search contains duplicate items")
    return {
        "api_version": _API_VERSION,
        "kind": "issue_search",
        "owner": OWNER,
        "repository": repositories[0] if repositories and len(repositories) == 1 else None,
        "repositories": list(repositories) if repositories else None,
        "terms": list(terms),
        "sort": "updated" if (since or author) else SEARCH_SORT,
        "since": since,
        "until": until,
        "author": author,
        "per_page": per_page,
        "total_count": total_count,
        "items": normalized,
        # The result in words. An empty list read as a gap rather than a finding: on a question
        # about a client library's support for a server feature, the search that showed the
        # library has nothing on it was the answer, and the model abstained for want of it two
        # times in five. Every word here is derived from fields above; nothing is added.
        "finding": _search_finding(
            kind, terms, repositories, total_count, since, len(normalized), until, author
        ),
    }


def _search_finding(
    kind: str,
    terms: tuple[str, ...],
    repositories: tuple[str, ...] | None,
    total_count: int,
    since: str | None = None,
    shown: int = 0,
    until: str | None = None,
    author: str | None = None,
) -> str:
    span = f"from {since} through {until}" if until else f"on or after {since}"
    if since is None:
        what = "pull requests" if kind == "pull-request" else "issues"
    elif kind == "pull-request":
        what = f"pull requests merged {span}"
    else:
        what = f"issues opened {span}"
    if author is not None:
        what = f"{what} authored by {author}"
    where = (
        " or ".join(f"{OWNER}/{r}" for r in repositories)
        if repositories
        else f"the {OWNER} organization"
    )
    matching = " matching " + " and ".join(f'"{term}"' for term in terms) if terms else ""
    if total_count == 0:
        return (
            f"The search completed and found no {what} in {where}{matching}. This establishes "
            f"that no such {what} existed there at observation time."
        )
    # The listing order is what the request asked for: a date window orders by recency, and a
    # topic search takes GitHub's best match. The finding must not describe one as the other.
    order = "most recently updated" if (since or author) else "best matching"
    listed = f" The {shown} {order} are listed." if shown < total_count else ""
    return f"The search found {total_count} {what} in {where}{matching}.{listed}"


def _search_body(value: Mapping[str, object], bound: int | None = None) -> tuple[str | None, bool]:
    """Return a search item body bounded to MAX_SEARCH_BODY_BYTES, and whether it was cut."""
    limit = MAX_SEARCH_BODY_BYTES if bound is None else bound
    raw = value.get("body")
    if raw is None:
        return None, False
    if type(raw) is not str:
        raise LiveGitHubError("GitHub field body must be a string")
    encoded = raw.encode("utf-8")
    if len(encoded) <= limit:
        return raw, False
    # Cut on a character boundary so the retained text is always valid UTF-8.
    return encoded[:limit].decode("utf-8", "ignore"), True


def _search_item(
    value: Mapping[str, object],
    scoped_repositories: tuple[str, ...] | None,
    *,
    compact: bool = False,
) -> dict[str, object]:
    repository_url = _text(value, "repository_url", 512)
    repository_prefix = f"{_API_ROOT}/repos/{OWNER}/"
    if not repository_url.startswith(repository_prefix):
        raise LiveGitHubError("GitHub search item repository is outside the fixed owner")
    repository = _repository(repository_url.removeprefix(repository_prefix))
    if scoped_repositories is not None and repository not in scoped_repositories:
        raise LiveGitHubError("GitHub search item repository conflicts with the query")

    # A search returns up to 20 items, so its per-item body bound is tighter than the
    # 128 KiB the single-issue paths allow. A body over that bound is normal on a long
    # issue and must not discard the whole result set, so it is truncated and the item
    # says so. The observation stays complete: every matching item is still present.
    body, body_truncated = _search_body(value, MAX_WINDOW_BODY_BYTES if compact else None)

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
        "body": body,
        "body_truncated": body_truncated,
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


def _release_by_tag(value: Mapping[str, object], repository: str, tag: str) -> dict[str, object]:
    release = _release(value, repository)
    if release["tag"] != tag:
        raise LiveGitHubError("GitHub release tag conflicts with the query")
    # Notes whole, within the evidence bound: a major rc's notes measured 22 KB, and the whole
    # record must still leave room for the other releases a comparison names.
    body = release.get("body")
    if isinstance(body, str):
        encoded = body.encode("utf-8")
        if len(encoded) > MAX_RELEASE_NOTES_BYTES:
            release["body"] = encoded[:MAX_RELEASE_NOTES_BYTES].decode("utf-8", "ignore")
            release["body_truncated"] = True
    return release


def _release_list(value: Mapping[str, object], repository: str, per_page: int) -> dict[str, object]:
    # The endpoint returns a bare JSON array. The transport requires an object, and that contract
    # is worth keeping, so the array is wrapped under "items" before it arrives here (see
    # _response_object). Each element is validated exactly as a single release is.
    items = _list(value, "items")
    if len(items) > per_page:
        raise LiveGitHubError("GitHub release list exceeds the requested page bound")
    releases = [_release(_object(item, "release list item"), repository) for item in items]
    # A release list carries eight releases, and a release's notes run to 22 KB for a major rc.
    # Whole, the list was 56 KB of the 64 KB evidence budget and starved every other record;
    # measured, that turned a mixed plan into an abstention. The notes' opening section (the
    # headline features) is what a "what shipped" question needs, so each body is cut there.
    # Only the newest few keep notes at all: the older ones matter as tags and dates (is this
    # version out, does it contain that commit), which the notes add nothing to.
    for position, release in enumerate(releases):
        body = release.get("body")
        if not isinstance(body, str):
            continue
        if position >= RELEASE_LIST_BODIES:
            release["body"] = None
            release["body_truncated"] = True
            continue
        encoded = body.encode("utf-8")
        if len(encoded) > MAX_RELEASE_LIST_BODY_BYTES:
            release["body"] = encoded[:MAX_RELEASE_LIST_BODY_BYTES].decode("utf-8", "ignore")
            release["body_truncated"] = True
    return {
        "api_version": _API_VERSION,
        "kind": "release_list",
        "repository": repository,
        "per_page": per_page,
        # Newest first, as GitHub orders them, and prereleases included: that is the point.
        "releases": releases,
        "url": f"{_WEB_ROOT}/{OWNER}/{repository}/releases",
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
    # The board's Status column (a single-select field); absent when the item has none set.
    status_value = item.get("status")
    status: str | None = None
    if isinstance(status_value, Mapping) and status_value:
        status = _text(status_value, "name", 128)
    elif status_value not in (None, {}):
        raise LiveGitHubError("project item status is malformed")
    return {
        "id": _text(item, "id", 256),
        "type": item_type.lower(),
        "status": status,
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
    if type(value) is list:
        # A list endpoint (releases) returns a bare array. Wrapping it keeps one response
        # contract for every normalizer instead of two, and bounds the element count here so an
        # unexpectedly large page is refused before any element is parsed.
        if len(value) > MAX_RELEASE_LIST:
            raise LiveGitHubError(f"{source} response array exceeds its bound")
        return {"items": value}
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


def _search_since(value: object) -> str | None:
    """A calendar day as YYYY-MM-DD, or None. It is placed in a search qualifier verbatim."""
    if value is None:
        return None
    if type(value) is not str or _SEARCH_SINCE.fullmatch(value) is None:
        raise LiveGitHubError("search window must be a YYYY-MM-DD day")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise LiveGitHubError("search window must be a real calendar day") from error
    return value


def _search_author(value: object) -> str | None:
    """A GitHub login, or None. Placed in a search qualifier verbatim, so its shape is exact."""
    if value is None:
        return None
    if type(value) is not str or _GITHUB_LOGIN.fullmatch(value) is None:
        raise LiveGitHubError("search author must be a GitHub login")
    return value


def _search_terms(value: object, *, minimum: int = MIN_SEARCH_TERMS) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise LiveGitHubError("search terms must be a tuple")
    raw_terms = cast(tuple[object, ...], value)
    if not minimum <= len(raw_terms) <= MAX_SEARCH_TERMS:
        raise LiveGitHubError(f"search requires from {minimum} through {MAX_SEARCH_TERMS} terms")
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


def _positive_or_zero_integer(value: Mapping[str, object], key: str) -> int:
    count = value.get(key)
    if type(count) is not int or count < 0 or count > 1_000_000:
        raise LiveGitHubError(f"GitHub field {key} must be a bounded non-negative integer")
    return count


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


def _is_calendar_timestamp(value: str) -> bool:
    """Reject impossible dates and times the shape regex admits.

    The regex pins digit layout only, so 2026-99-99T99:99:99Z matches it. Parsing is what
    establishes the value names a real instant.
    """
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _timestamp(value: Mapping[str, object], key: str) -> str:
    result = _text(value, key, 64)
    if _TIMESTAMP.fullmatch(result) is None or not _is_calendar_timestamp(result):
        raise LiveGitHubError(f"GitHub field {key} must be a canonical UTC timestamp")
    try:
        datetime.fromisoformat(result.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise LiveGitHubError(f"GitHub field {key} must be a canonical UTC timestamp") from error
    return result


def _nullable_sha(value: Mapping[str, object], key: str) -> str | None:
    sha = value.get(key)
    if sha is None:
        return None
    if not isinstance(sha, str) or _SHA.fullmatch(sha) is None:
        raise LiveGitHubError(f"GitHub field {key} is not a commit id")
    return sha


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
