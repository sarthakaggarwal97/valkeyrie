"""Mandatory request-pinned, intent-scoped static retrieval."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, cast

from valkeyrie.evaluations import EvaluationSuite
from valkeyrie.retrieval_config import (
    CandidateConfiguration,
    FrozenRetrievalConfiguration,
    RetrievalConfigError,
    validate_frozen_retrieval_configuration,
)


class RetrievalError(RuntimeError):
    """A static retrieval request or result cannot be trusted."""


@dataclass(frozen=True)
class GenerationAvailability:
    """Strongly consistent lifecycle state for one immutable generation."""

    generation_id: str
    revision: int
    sealed: bool
    available: bool
    ingested: bool
    retrievable: bool
    evaluation_passed: bool
    retained: bool
    evaluation_report_id: str | None = None


@dataclass(frozen=True)
class PinnedGeneration:
    """Generation identity captured once for a request or candidate evaluation."""

    generation_id: str
    state_revision: int


@dataclass(frozen=True)
class RetrievedChunk:
    """One verified Bedrock result whose metadata matches the request pin."""

    text: str
    score: float | None
    metadata: Mapping[str, object]
    location: Mapping[str, object]


@dataclass(frozen=True)
class RetrievalIntent:
    """Immutable query expansion and reviewed repository scope for one request."""

    query: str
    repositories: tuple[str, ...]


class GenerationRegistry(Protocol):
    def get_generation(self, generation_id: str) -> GenerationAvailability | None: ...


class BedrockRetrievalClient(Protocol):
    def retrieve(self, **kwargs: object) -> Mapping[str, object]: ...


_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_KB_ID: Final = re.compile(r"^[A-Z0-9]{10}$")
_MAX_QUERY_CHARACTERS: Final = 8_000
_CORE_REPOSITORIES: Final = ("valkey", "valkey-doc")
_MODULE_REPOSITORIES: Final = (
    "valkey-bloom",
    "valkey-json",
    "valkey-search",
    "valkey-ldap",
    "valkey-lua5.5",
    "valkey-luajit",
    "valkey-bundle",
    "valkeymodule-rs",
)
_GLIDE_REPOSITORIES: Final = (
    "valkey-glide",
    "valkey-glide-docs",
    "valkey-glide-cpp",
    "valkey-glide-csharp",
    "valkey-glide-php",
    "valkey-glide-ruby",
)
_CLIENT_REPOSITORIES: Final = _GLIDE_REPOSITORIES + (
    "valkey-go",
    "valkey-py",
    "valkey-java",
    "valkey-swift",
    "libvalkey",
    "libvalkey-py",
    "iovalkey",
    "iovalkey-commands",
    "spring-data-valkey",
    "valkey-namespace",
)
_OTHER_CANONICAL_REPOSITORIES: Final = (
    "valkey-io.github.io",
    "community",
    ".github",
    "planet",
    "valkey-container",
    "valkey-helm",
    "valkey-operator",
    "valkey-admin",
    "valkey-try-me",
    "valkey-fuzzer",
    "valkey-test-framework",
    "valkey-perf-benchmark",
    "verify-provenance",
    "valkey-ci-agent",
    "valkey-release-automation",
)
_REVIEWED_AUTHORITIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        **{
            repository: "canonical"
            for repository in (
                _CORE_REPOSITORIES
                + _MODULE_REPOSITORIES
                + _CLIENT_REPOSITORIES
                + _OTHER_CANONICAL_REPOSITORIES
            )
        },
        "valkey-skills": "secondary",
    }
)
# Content and marketing wording. "Content" alone is ambiguous (it can mean stored data), so it
# counts only alongside something that makes it editorial.
_CONTENT: Final = re.compile(
    r"\b(?:blog|blogs|blogging|blogpost|social\s+media|twitter|linkedin|mastodon|newsletter"
    r"|announcement|announcements|press|marketing|brand|branding|logo|planet|podcast"
    r"|website\s+content|content\s+(?:policy|calendar|guidelines|strategy|review))\b"
    r"|\bcontent\b(?=[^.]*\b(?:blog|social|media|marketing|website|post|posts|publish)\b)"
    r"|\b(?:blog|social|marketing|website|publish\w*)\b(?=[^.]*\bcontent\b)",
    re.IGNORECASE,
)


_REPOSITORY_NAMES: Final = tuple(_REVIEWED_AUTHORITIES)
_FEATURE_ALIASES: Final = (
    (
        re.compile(r"\bhash(?:[ -]+field)?[ -]+(?:expiration|expiry|ttl)\b", re.IGNORECASE),
        ("HEXPIRE", "HPEXPIRE", "HTTL", "HPERSIST"),
    ),
    (
        re.compile(r"\b(?:compressed[ -]+replication|replication[ -]+compressed)\b", re.IGNORECASE),
        ("replication compression",),
    ),
    (re.compile(r"\bfork[ -]+less\b", re.IGNORECASE), ("forkless",)),
    (
        re.compile(
            r"\b(?:govern(?:ance|s|ed|ing)?|technical steering committee|tsc)\b", re.IGNORECASE
        ),
        ("GOVERNANCE", "MAINTAINERS", "Technical Steering Committee", "voting"),
    ),
    (
        re.compile(r"\b(?:get(?:ting)? started|quick[ -]?start)\b", re.IGNORECASE),
        ("quick start", "installation"),
    ),
    (
        re.compile(r"\b(?:contributor|contribute|contributing)\b", re.IGNORECASE),
        ("CONTRIBUTING", "Developer Certificate of Origin", "DCO", "pull request"),
    ),
    (
        re.compile(
            # Not the event loop, the events API or a keyspace event: those are core internals.
            r"\b(?:event|events)\b(?!\s*(?:loop|handler|handlers|api|notification|notifications))"
            r"(?![- ]driven)|\b(?:conference|meetup)\b",
            re.IGNORECASE,
        ),
        ("events calendar",),
    ),
    (
        # The project's own content work: blog posts, the website, social media, the Planet feed
        # that aggregates community writing, and brand assets. Asked as "how does the project
        # handle content", every one of these scoped to valkey and valkey-doc, the two most
        # technical repositories, which is why the bot read "content" as stored data.
        _CONTENT,
        ("blog", "post", "social media", "announcement", "planet", "brand", "logo"),
    ),
    (
        re.compile(
            r"\b(?:workstream|working group"
            # A meeting body and the meeting word only have to share a clause: "what did the TSC
            # discuss in its September meeting?" put four words between them and lost the
            # community repository entirely. Minutes are not published as files or issues
            # anywhere in the org, so the valkey repository is in scope for GOVERNANCE.md and the
            # website for the events calendar: a useful answer beats a bare negative finding.
            r"|(?:community|tsc|technical steering committee|governance)\b[^.;:!?]*"
            r"\b(?:meeting|meetings|minutes|agenda)"
            r"|meeting\s+(?:minutes|notes|agenda))\b",
            re.IGNORECASE,
        ),
        ("working groups", "meeting notes", "roadmap", "progress"),
    ),
    (
        re.compile(r"\breplication(?: and |/|[ -]+)?failover\b|\bfailover\b", re.IGNORECASE),
        ("replication", "PSYNC", "partial resynchronization", "Sentinel", "Cluster failover"),
    ),
    (
        re.compile(r"\bleader[ -]?board\b", re.IGNORECASE),
        ("sorted sets", "ZADD", "ZINCRBY", "ZRANGE", "ZREVRANK"),
    ),
)
# An error string or a log line pasted into Slack. These are the server's own words, so they are
# unambiguous: the person is troubleshooting, and the answer is in the core sources and the
# documentation rather than in a client library or the website. The live supplement adds known
# issues for the same token, because "is this a known bug" is half the question.
_TROUBLESHOOTING: Final = re.compile(
    # NOT case-insensitive overall. Some error names are ordinary English words: ASK and MOVED are
    # cluster redirects, LOADING and READONLY are error prefixes, and "who should I ask about
    # cluster failover?" matched ASK and was read as a pasted error. Those need their upper-case
    # error form. The rest are unmistakable in any case, so they carry a scoped (?i:) group.
    r"(?i:wrongtype|crossslot|clusterdown|masterdown|misconf|noauth|noperm|busygroup|noscript"
    r"|noreplicas|execabort|sigsegv|sigbus|sigabrt|segmentation\s+fault|stack\s+trace"
    r"|protocol\s+error|bad\s+message\s+length|maxmemory\s+limit)"
    r"|\b(?:ASK|MOVED|LOADING|READONLY|ERR|OOM|=== VALKEY BUG REPORT)\b"
)
# "Show me how to do X in <language>". A code example lives in a client library's own examples and
# tests, not in the server documentation, and the language decides which library.
_CODE_EXAMPLE: Final = re.compile(
    r"\b(?:code\s+)?(?:example|examples|sample|samples|snippet|snippets|boilerplate)\b"
    r"|\bhow\s+(?:do|would)\s+(?:i|you|we)\s+(?:write|code|call|connect|use)\b"
    r"|\bshow\s+me\s+(?:the\s+)?(?:code|how)\b",
    re.IGNORECASE,
)
# "Where does this belong, who owns it, where do I file it." Routing a person to the right place is
# its own question, and the answer is the repository taxonomy and the ownership files.
_ROUTING: Final = re.compile(
    r"\b(?:where\s+(?:do|should|can)\s+(?:i|we)\s+(?:file|report|open|raise|ask|post|start)"
    r"|which\s+repo(?:sitory)?\s+(?:does|should|do)"
    # Not "who reviews X": "who reviews blog content?" is an editorial question, and the
    # ownership words below already cover the routing sense.
    r"|who\s+(?:owns|maintains|should\s+(?:i|we)\s+(?:ask|contact))"
    r"|who\s+do\s+(?:i|we)\s+(?:ask|contact|talk\s+to)"
    r"|where\s+(?:does|should)\s+(?:this|that|it)\s+(?:go|belong)"
    r"|how\s+do\s+(?:i|we)\s+(?:get|find)\s+help)\b",
    re.IGNORECASE,
)
# Redis to Valkey, or one Valkey version to another. Compatibility statements live in the core
# repository and the documentation, and the website carries the migration guidance.
_MIGRATION: Final = re.compile(
    r"\bmigrat(?:e|ing|ion)\b|\bupgrad(?:e|ing)\b|\bdowngrad(?:e|ing)\b"
    r"|\b(?:moving|switch(?:ing)?|coming|port(?:ing)?)\s+(?:from|to)\s+(?:redis|valkey)\b"
    r"|\bdrop[- ]in\s+replacement\b|\bbackward[s]?\s+compatib\w*\b",
    re.IGNORECASE,
)
# The language a code example is wanted in, and where that library's own examples live.
_EXAMPLE_LANGUAGES: Final[tuple[tuple[re.Pattern[str], tuple[str, ...]], ...]] = (
    (re.compile(r"\bpython\b", re.IGNORECASE), ("valkey-py", "valkey-glide", "valkey-doc")),
    (
        re.compile(r"\b(?:java|kotlin)\b", re.IGNORECASE),
        ("valkey-java", "valkey-glide", "valkey-doc"),
    ),
    (
        re.compile(r"\b(?:node|nodejs|node\.js|javascript|typescript|ts)\b", re.IGNORECASE),
        ("iovalkey", "valkey-glide", "valkey-doc"),
    ),
    # Case-sensitive "Go", because lowercase "go" is a verb in half of these questions.
    (
        re.compile(r"\bGo\b|\bgolang\b|\bgo\s+(?:example|client|code|snippet)\b"),
        ("valkey-go", "valkey-glide", "valkey-doc"),
    ),
    # "C++" and "C#" END in non-word characters, so a trailing \b after them never matches.
    (
        re.compile(r"\bc\+\+|\bcpp\b", re.IGNORECASE),
        ("valkey-glide-cpp", "libvalkey", "valkey-doc"),
    ),
    (
        re.compile(r"\bc#|\bcsharp\b|\.net\b|\bdotnet\b", re.IGNORECASE),
        ("valkey-glide-csharp", "valkey-doc"),
    ),
    (re.compile(r"\bphp\b", re.IGNORECASE), ("valkey-glide-php", "valkey-doc")),
    (re.compile(r"\bruby\b", re.IGNORECASE), ("valkey-glide-ruby", "valkey-doc")),
    (re.compile(r"\bswift\b", re.IGNORECASE), ("valkey-swift", "valkey-doc")),
    (re.compile(r"\brust\b", re.IGNORECASE), ("valkey-glide", "valkeymodule-rs", "valkey-doc")),
    (
        re.compile(r"\b(?:bare\s+)?c\b(?!\+\+|#)", re.IGNORECASE),
        ("libvalkey", "valkey", "valkey-doc"),
    ),
)
_CATEGORY_SCOPES: Final = (
    (
        re.compile(r"\b(?:skills summary|valkey-skills)\b", re.IGNORECASE),
        ("valkey-doc", "valkey-skills"),
    ),
    # Troubleshooting first: a pasted error outranks every other word in the message, including a
    # client library name, because the error text is the subject.
    (_TROUBLESHOOTING, ("valkey", "valkey-doc")),
    (_ROUTING, ("community", "valkey", "valkey-doc")),
    (_MIGRATION, ("valkey", "valkey-doc", "valkey-io.github.io")),
    (
        # The TSC itself is documented in the core repository, but its MEETINGS are minuted in
        # valkey-io/community, so that wording is left to the meeting scope below.
        re.compile(
            # Defers whenever the clause mentions a meeting at all, not only when the next word
            # does: "what did the TSC discuss in its September meeting?" is a meeting question.
            r"\b(?:technical steering committee|tsc)\b"
            r"(?![^.;:!?]*\b(?:meeting|meetings|minutes|agenda)\b)",
            re.IGNORECASE,
        ),
        ("valkey",),
    ),
    (
        # Same deferral: a governance MEETING is a meeting question first.
        re.compile(
            r"\bgovern(?:ance|s|ed|ing)?\b"
            r"(?![^.;:!?]*\b(?:meeting|meetings|minutes|agenda)\b)",
            re.IGNORECASE,
        ),
        ("valkey", "valkey-io.github.io"),
    ),
    (
        re.compile(r"\b(?:contributor|contribute|contributing)\b", re.IGNORECASE),
        ("valkey", "community", "valkey-io.github.io"),
    ),
    (
        re.compile(
            # Not the event loop, the events API or a keyspace event: those are core internals.
            r"\b(?:event|events)\b(?!\s*(?:loop|handler|handlers|api|notification|notifications))"
            r"(?![- ]driven)|\b(?:conference|meetup)\b",
            re.IGNORECASE,
        ),
        ("valkey-io.github.io", "community"),
    ),
    (
        _CONTENT,
        ("valkey-io.github.io", "planet", "one-time-for-planet", "community", "assets"),
    ),
    (
        re.compile(
            r"\b(?:workstream|working group"
            # A meeting body and the meeting word only have to share a clause: "what did the TSC
            # discuss in its September meeting?" put four words between them and lost the
            # community repository entirely. Minutes are not published as files or issues
            # anywhere in the org, so the valkey repository is in scope for GOVERNANCE.md and the
            # website for the events calendar: a useful answer beats a bare negative finding.
            r"|(?:community|tsc|technical steering committee|governance)\b[^.;:!?]*"
            r"\b(?:meeting|meetings|minutes|agenda)"
            r"|meeting\s+(?:minutes|notes|agenda))\b",
            re.IGNORECASE,
        ),
        ("community", "valkey", "valkey-io.github.io"),
    ),
    (
        re.compile(r"\bwho\s+is\b", re.IGNORECASE),
        ("valkey", "valkey-io.github.io"),
    ),
)
# Core server commands whose first word also names an ecosystem: MODULE LOAD/LIST/UNLOAD and the
# CLIENT subcommands are implemented in the server, not in a module or a client library.
_CORE_COMMAND: Final = re.compile(
    r"\bMODULE\s+(?:LOAD|LOADEX|UNLOAD|LIST|HELP)\b"
    r"|\bCLIENT\s+(?:LIST|INFO|ID|KILL|NO-EVICT|NO-TOUCH|PAUSE|UNPAUSE|REPLY|SETNAME|GETNAME"
    r"|SETINFO|UNBLOCK|TRACKING|TRACKINGINFO|CACHING|HELP)\b"
)


_MODULE_ALIASES: Final = MappingProxyType(
    {
        "valkey-bloom": ("valkey bloom", "bloom module", "bloom filter module"),
        "valkey-json": ("valkey json", "json module"),
        "valkey-search": ("valkey search", "search module"),
        "valkey-ldap": ("valkey ldap", "ldap module"),
        "valkey-lua5.5": ("valkey lua 5.5", "lua 5.5 module", "lua5.5 module"),
        "valkey-luajit": ("valkey luajit", "luajit module"),
        "valkey-bundle": ("valkey bundle", "bundle module"),
        "valkeymodule-rs": ("rust module", "rust module sdk", "module rust sdk"),
    }
)
_CLIENT_ALIASES: Final = MappingProxyType(
    {
        "valkey-glide-docs": ("glide docs", "glide documentation"),
        "valkey-glide-cpp": ("glide c++", "glide cpp"),
        "valkey-glide-csharp": ("glide c#", "glide csharp", "glide .net"),
        "valkey-glide-php": ("glide php", "php glide"),
        "valkey-glide-ruby": ("glide ruby", "ruby glide"),
        "valkey-go": ("go client", "golang client"),
        "valkey-py": ("python client", "official python client"),
        "libvalkey-py": ("libvalkey python", "python binding for libvalkey"),
        "valkey-java": ("java client",),
        "valkey-swift": ("swift client",),
    }
)
# What people call these repositories when they do not type the repository name. Every one of them
# is indexed, and without an alias the question fell through to the core/docs default and abstained
# with "no evidence about the Valkey Helm chart" while valkey-helm sat in the corpus unsearched.
_SIBLING_ALIASES: Final[tuple[tuple[re.Pattern[str], tuple[str, ...]], ...]] = (
    (re.compile(r"\bhelm\b", re.IGNORECASE), ("valkey-helm", "valkey-doc")),
    (re.compile(r"\boperator\b", re.IGNORECASE), ("valkey-operator", "valkey-doc")),
    (
        re.compile(r"\b(?:docker|container|image|podman)\b", re.IGNORECASE),
        ("valkey-container", "valkey-doc"),
    ),
    (re.compile(r"\bfuzz(?:er|ing)?\b", re.IGNORECASE), ("valkey-fuzzer",)),
    (
        re.compile(r"\b(?:test framework|tcl test|test suite)\b", re.IGNORECASE),
        ("valkey-test-framework", "valkey"),
    ),
    (
        re.compile(
            r"\b(?:benchmark|benchmarking|valkey-benchmark|perf(?:ormance)? test)\b", re.IGNORECASE
        ),
        ("valkey-perf-benchmark", "valkey", "valkey-doc"),
    ),
    (re.compile(r"\bspring(?:\s+(?:boot|data))?\b", re.IGNORECASE), ("spring-data-valkey",)),
    (re.compile(r"\bnamespac(?:e|ing|es)\b", re.IGNORECASE), ("valkey-namespace", "valkey")),
    (re.compile(r"\bprovenance\b", re.IGNORECASE), ("verify-provenance", "valkey")),
)
_CORE_PRIORITY: Final = MappingProxyType({"valkey": 0, "valkey-doc": 1})


def derive_retrieval_intent(query: str) -> RetrievalIntent:
    """Derive a closed, deterministic repository scope and fixed query aliases."""
    _validate_query(query)
    repositories = _requested_repositories(query)
    aliases: list[str] = []
    for pattern, additions in _FEATURE_ALIASES:
        if pattern.search(query):
            aliases.extend(alias for alias in additions if not _contains_term(query, alias))
    expanded = query if not aliases else f"{query} {' '.join(aliases)}"
    _validate_query(expanded)
    return RetrievalIntent(expanded, repositories)


def _requested_repositories(query: str) -> tuple[str, ...]:
    explicit = tuple(
        repository
        for repository in _REPOSITORY_NAMES
        if repository != "valkey" and _contains_term(query, repository)
    )
    if explicit:
        if explicit == ("valkey-skills",):
            return ("valkey-doc", "valkey-skills")
        # A question naming another repository AND the core one ("does valkey-glide support the
        # compression in valkey 9.2?") spans both. The core name is excluded from the explicit
        # scan because it is a substring of every other repository; it is admitted here only when
        # it appears as a whole word on its own, not as part of a longer name.
        # Only beside another repository's name does the bare word mean the core repository;
        # on its own it means the project, and the category scopes below handle that.
        if _contains_term(re.sub(r"valkey-[a-z0-9.-]+", " ", query, flags=re.IGNORECASE), "valkey"):
            return (*explicit, "valkey")
        # "valkey-py throws WRONGTYPE" is two questions in one: what the client did, and what the
        # error means. The error is defined by the server, so both are in scope.
        if _TROUBLESHOOTING.search(query) is not None:
            return (*explicit, "valkey", "valkey-doc")
        return explicit

    for pattern, repositories in _SIBLING_ALIASES:
        if pattern.search(query):
            return repositories

    if re.search(
        r"\bstart release\b|\brelease control\b|\b(?:build|package) publication\b",
        query,
        re.IGNORECASE,
    ):
        return ("valkey", "valkey-ci-agent", "valkey-release-automation")

    for pattern, repositories in _CATEGORY_SCOPES:
        if pattern.search(query):
            return repositories

    # A code example belongs to a client library, chosen by the language named. Without a language
    # the core documentation's own examples are the honest answer.
    if _CODE_EXAMPLE.search(query) is not None:
        for_language = _repositories_for_aliases(query, _CLIENT_ALIASES, ())
        if for_language:
            return (*for_language, "valkey-doc")
        if _contains_term(query, "glide"):
            return ("valkey-glide", "valkey-glide-docs")
        # A bare language name is how people ask: "an example for HSET in Python". The client
        # aliases need "python client", which nobody writes when they are asking for code.
        for pattern, repositories in _EXAMPLE_LANGUAGES:
            if pattern.search(query):
                return repositories

    # MODULE LOAD and CLIENT LIST are core server commands. Matching the bare words "module" and
    # "client" sent them to the eight module repositories and the sixteen client libraries, where
    # the answer is not.
    if _CORE_COMMAND.search(query) is not None:
        return ("valkey", "valkey-doc")
    if _contains_term(query, "glide"):
        specific = _repositories_for_aliases(query, _CLIENT_ALIASES, _GLIDE_REPOSITORIES)
        return specific or _GLIDE_REPOSITORIES
    if _contains_term(query, "module"):
        specific = _repositories_for_aliases(query, _MODULE_ALIASES, _MODULE_REPOSITORIES)
        selected = specific or _MODULE_REPOSITORIES
        if _contains_term(query, "core"):
            return ("valkey", *selected, "valkey-doc")
        return selected
    if _contains_term(query, "client"):
        specific = _repositories_for_aliases(query, _CLIENT_ALIASES, _CLIENT_REPOSITORIES)
        selected = specific or _CLIENT_REPOSITORIES
        if re.search(r"\b(?:specified\s+)?valkey release\b", query, re.IGNORECASE):
            return ("valkey", "valkey-doc", *selected)
        return selected
    return _CORE_REPOSITORIES


def _repositories_for_aliases(
    query: str,
    aliases: Mapping[str, tuple[str, ...]],
    allowed: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        repository
        for repository in allowed
        if any(_contains_term(query, alias) for alias in aliases.get(repository, ()))
    )


def _contains_term(value: str, term: str) -> bool:
    return (
        re.search(
            rf"(?<![a-z0-9._-]){re.escape(term)}(?![a-z0-9._-])",
            value,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _validate_query(query: object) -> None:
    if (
        not isinstance(query, str)
        or not query.strip()
        or len(query) > _MAX_QUERY_CHARACTERS
        or "\x00" in query
    ):
        raise RetrievalError("retrieval query is outside its bound")


def pin_generation(registry: GenerationRegistry, generation_id: str) -> PinnedGeneration:
    """Validate and pin one sealed, ingested, currently retrievable generation."""
    record = _generation_record(registry, generation_id)
    _require_retrievable(record)
    return PinnedGeneration(record.generation_id, record.revision)


def retrieve_generation(
    registry: GenerationRegistry,
    client: BedrockRetrievalClient,
    config: FrozenRetrievalConfiguration,
    *,
    knowledge_base_id: str,
    generation: PinnedGeneration,
    query: str,
) -> tuple[RetrievedChunk, ...]:
    """Retrieve through the sole static path and reject any untrusted result."""
    if not isinstance(generation, PinnedGeneration):
        raise RetrievalError("static retrieval requires an immutable generation pin")
    if not isinstance(generation.state_revision, int) or generation.state_revision < 1:
        raise RetrievalError("generation pin revision is malformed")
    record = _generation_record(registry, generation.generation_id)
    _require_retrievable(record)
    if record.revision != generation.state_revision:
        raise RetrievalError("generation pin revision no longer matches registry state")
    _validate_client(client)
    selected = _selected_configuration(config)
    if not isinstance(knowledge_base_id, str) or not _KB_ID.fullmatch(knowledge_base_id):
        raise RetrievalError("knowledge base ID is malformed")
    intent = derive_retrieval_intent(query)

    filter_field = selected.retrieval.generation_filter_field
    generation_filter = {
        "equals": {
            "key": filter_field,
            "value": generation.generation_id,
        }
    }
    repository_filters = [
        {"equals": {"key": "repository", "value": repository}} for repository in intent.repositories
    ]
    repository_filter = (
        repository_filters[0] if len(repository_filters) == 1 else {"orAll": repository_filters}
    )
    scoped_filter = {"andAll": [generation_filter, repository_filter]}
    response = _retrieve(
        client,
        knowledge_base_id,
        intent.query,
        scoped_filter,
        selected,
    )
    results = verify_retrieval_results(
        response,
        generation.generation_id,
        filter_field,
        selected.retrieval.number_of_results,
        intent.repositories,
    )
    if results:
        return results

    fallback = _retrieve(
        client,
        knowledge_base_id,
        intent.query,
        generation_filter,
        selected,
    )
    return verify_retrieval_results(
        fallback,
        generation.generation_id,
        filter_field,
        selected.retrieval.number_of_results,
        intent.repositories,
    )


def _retrieve(
    client: BedrockRetrievalClient,
    knowledge_base_id: str,
    query: str,
    retrieval_filter: Mapping[str, object],
    selected: CandidateConfiguration,
) -> Mapping[str, object]:
    return client.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "filter": retrieval_filter,
                "numberOfResults": selected.retrieval.number_of_results,
                "overrideSearchType": selected.retrieval.search_type,
            }
        },
    )


def run_candidate_retrieval_smoke(
    registry: GenerationRegistry,
    client: BedrockRetrievalClient,
    config: FrozenRetrievalConfiguration,
    suite: EvaluationSuite,
    *,
    knowledge_base_id: str,
    generation_id: str,
) -> PinnedGeneration:
    """Run every available candidate fixture through the mandatory pinned retrieval path."""
    if not isinstance(suite, EvaluationSuite):
        raise RetrievalError("candidate smoke suite is malformed")
    pin = pin_generation(registry, generation_id)
    fixtures = tuple(item for item in suite.retrieval_fixtures if item.generation_available)
    if not fixtures:
        raise RetrievalError("candidate smoke suite has no available-generation fixtures")
    for fixture in fixtures:
        retrieve_generation(
            registry,
            client,
            config,
            knowledge_base_id=knowledge_base_id,
            generation=pin,
            query=fixture.query_text,
        )
    return pin


def _generation_record(
    registry: GenerationRegistry,
    generation_id: object,
) -> GenerationAvailability:
    if registry is None or not callable(getattr(registry, "get_generation", None)):
        raise RetrievalError("generation registry does not implement lookup")
    if not isinstance(generation_id, str) or not _DIGEST.fullmatch(generation_id):
        raise RetrievalError("generation ID is malformed")
    record = registry.get_generation(generation_id)
    if record is None:
        raise RetrievalError("generation is unknown")
    _validate_generation_record(record)
    if record.generation_id != generation_id:
        raise RetrievalError("generation registry returned a mismatched identity")
    return record


def _validate_generation_record(record: object) -> None:
    if not isinstance(record, GenerationAvailability):
        raise RetrievalError("generation registry returned a malformed record")
    if not _DIGEST.fullmatch(record.generation_id):
        raise RetrievalError("generation registry contains a malformed identity")
    if (
        not isinstance(record.revision, int)
        or isinstance(record.revision, bool)
        or record.revision < 1
    ):
        raise RetrievalError("generation registry revision is malformed")
    flags = (
        record.sealed,
        record.available,
        record.ingested,
        record.retrievable,
        record.evaluation_passed,
        record.retained,
    )
    if any(not isinstance(value, bool) for value in flags):
        raise RetrievalError("generation registry flags are malformed")
    if record.evaluation_report_id is not None and (
        not isinstance(record.evaluation_report_id, str)
        or not re.fullmatch(r"eval_[0-9a-f]{64}", record.evaluation_report_id)
    ):
        raise RetrievalError("generation evaluation report ID is malformed")


def _require_retrievable(record: GenerationAvailability) -> None:
    if not record.sealed:
        raise RetrievalError("generation is not sealed")
    if not record.available:
        raise RetrievalError("generation is unavailable")
    if not record.ingested:
        raise RetrievalError("generation has not completed ingestion")
    if not record.retrievable:
        raise RetrievalError("generation is not retrievable")


def _selected_configuration(
    config: FrozenRetrievalConfiguration,
) -> CandidateConfiguration:
    try:
        verified = validate_frozen_retrieval_configuration(config)
    except RetrievalConfigError as error:
        raise RetrievalError(f"retrieval configuration is not authoritative: {error}") from error
    selected = verified.selected
    if (
        selected.retrieval.generation_filter_field != "generation_id"
        or selected.retrieval.unavailable_generation_behavior != "fail_closed"
        or selected.retrieval.reranking
    ):
        raise RetrievalError("retrieval configuration violates the frozen generation contract")
    return selected


def _validate_client(client: object) -> None:
    if client is None or not callable(getattr(client, "retrieve", None)):
        raise RetrievalError("Bedrock client does not implement retrieval")


def reviewed_source_authority(repository: object) -> str:
    """Return the declared authority for one reviewed repository or fail closed."""
    if not isinstance(repository, str) or repository not in _REVIEWED_AUTHORITIES:
        raise RetrievalError("retrieval result has invalid repository metadata")
    return _REVIEWED_AUTHORITIES[repository]


def verify_retrieval_results(
    response: object,
    generation_id: str,
    filter_field: str,
    limit: int,
    requested_repositories: tuple[str, ...],
) -> tuple[RetrievedChunk, ...]:
    if not isinstance(response, Mapping):
        raise RetrievalError("Bedrock retrieval response is malformed")
    values = response.get("retrievalResults")
    if not isinstance(values, list):
        raise RetrievalError("Bedrock retrieval response has no result list")
    if len(values) > limit:
        raise RetrievalError("Bedrock returned more results than requested")

    results: list[RetrievedChunk] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise RetrievalError(f"retrieval result {index} is malformed")
        metadata_value = value.get("metadata")
        content_value = value.get("content")
        location_value = value.get("location")
        if not isinstance(metadata_value, Mapping):
            raise RetrievalError(f"retrieval result {index} has malformed metadata")
        metadata = dict(cast(Mapping[str, object], metadata_value))
        observed_generation = metadata.get(filter_field)
        if not isinstance(observed_generation, str) or not _DIGEST.fullmatch(observed_generation):
            raise RetrievalError(f"retrieval result {index} has no valid generation metadata")
        if observed_generation != generation_id:
            raise RetrievalError(f"retrieval result {index} crossed the pinned generation")
        if not isinstance(content_value, Mapping) or content_value.get("type") != "TEXT":
            raise RetrievalError(f"retrieval result {index} has malformed content")
        text = content_value.get("text")
        if not isinstance(text, str) or not text:
            raise RetrievalError(f"retrieval result {index} has empty content")
        if not isinstance(location_value, Mapping) or not location_value:
            raise RetrievalError(f"retrieval result {index} has malformed location")
        repository = metadata.get("repository")
        authority = metadata.get("authority")
        try:
            expected_authority = reviewed_source_authority(repository)
        except RetrievalError as error:
            raise RetrievalError(
                f"retrieval result {index} has invalid repository metadata"
            ) from error
        if authority != expected_authority:
            raise RetrievalError(f"retrieval result {index} has invalid authority metadata")
        score_value = value.get("score")
        if score_value is not None and (
            not isinstance(score_value, (int, float))
            or isinstance(score_value, bool)
            or not 0 <= score_value <= 1
        ):
            raise RetrievalError(f"retrieval result {index} has malformed score")
        results.append(
            RetrievedChunk(
                text,
                None if score_value is None else float(score_value),
                MappingProxyType(metadata),
                MappingProxyType(dict(cast(Mapping[str, object], location_value))),
            )
        )
    requested = frozenset(requested_repositories)
    return tuple(
        result
        for _, result in sorted(
            enumerate(results),
            key=lambda indexed: _result_order(indexed, requested),
        )
    )


def _result_order(
    indexed: tuple[int, RetrievedChunk],
    requested_repositories: frozenset[str],
) -> tuple[int, int, int, int, float, int]:
    index, result = indexed
    repository = cast(str, result.metadata["repository"])
    authority = cast(str, result.metadata["authority"])
    return (
        0 if authority == "canonical" else 1,
        0 if repository in requested_repositories else 1,
        _CORE_PRIORITY.get(repository, 2),
        0 if result.score is not None else 1,
        -(result.score or 0.0),
        index,
    )
