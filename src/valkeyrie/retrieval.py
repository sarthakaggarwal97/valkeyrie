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
        re.compile(r"\b(?:event|events|conference|meetup)\b", re.IGNORECASE),
        ("events calendar",),
    ),
    (
        re.compile(r"\b(?:workstream|working group|community meeting)\b", re.IGNORECASE),
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
_CATEGORY_SCOPES: Final = (
    (
        re.compile(r"\b(?:skills summary|valkey-skills)\b", re.IGNORECASE),
        ("valkey-doc", "valkey-skills"),
    ),
    (
        re.compile(r"\b(?:technical steering committee|tsc)\b", re.IGNORECASE),
        ("valkey",),
    ),
    (
        re.compile(r"\bgovern(?:ance|s|ed|ing)?\b", re.IGNORECASE),
        ("valkey", "valkey-io.github.io"),
    ),
    (
        re.compile(r"\b(?:contributor|contribute|contributing)\b", re.IGNORECASE),
        ("valkey", "community", "valkey-io.github.io"),
    ),
    (
        re.compile(r"\b(?:event|events|conference|meetup)\b", re.IGNORECASE),
        ("valkey-io.github.io", "community"),
    ),
    (
        re.compile(r"\b(?:workstream|working group|community meeting)\b", re.IGNORECASE),
        ("community", "valkey", "valkey-io.github.io"),
    ),
    (
        re.compile(r"\bwho\s+is\b", re.IGNORECASE),
        ("valkey", "valkey-io.github.io"),
    ),
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
        return explicit

    if re.search(
        r"\bstart release\b|\brelease control\b|\b(?:build|package) publication\b",
        query,
        re.IGNORECASE,
    ):
        return ("valkey", "valkey-ci-agent", "valkey-release-automation")

    for pattern, repositories in _CATEGORY_SCOPES:
        if pattern.search(query):
            return repositories

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
