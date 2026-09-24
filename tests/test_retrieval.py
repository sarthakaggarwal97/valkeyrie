from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import cast

import pytest

from valkeyrie.evaluations import load_evaluation_suite
from valkeyrie.retrieval import (
    GenerationAvailability,
    PinnedGeneration,
    RetrievalError,
    derive_retrieval_intent,
    pin_generation,
    retrieve_generation,
    run_candidate_retrieval_smoke,
)
from valkeyrie.retrieval_config import FrozenRetrievalConfiguration, load_retrieval_config
from valkeyrie.sources import load_yaml_mapping

ROOT = Path(__file__).resolve().parents[1]
KB_ID = "ABCDEFGHIJ"
ACTIVE = "sha256:" + "a" * 64
CANDIDATE = "sha256:" + "b" * 64
OLD = "sha256:" + "c" * 64
REPORT = "eval_" + "d" * 64


class MemoryRegistry:
    def __init__(self, records: list[GenerationAvailability]) -> None:
        self.records = {record.generation_id: record for record in records}

    def get_generation(self, generation_id: str) -> GenerationAvailability | None:
        return self.records.get(generation_id)


class FakeRetrievalClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses: list[object] = []

    def retrieve(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if self.responses:
            return cast(dict[str, object], self.responses.pop(0))
        configuration = cast(dict[str, object], kwargs["retrievalConfiguration"])
        vector = cast(dict[str, object], configuration["vectorSearchConfiguration"])
        retrieval_filter = cast(dict[str, object], vector["filter"])
        generation = _filter_values(retrieval_filter, "generation_id")[0]
        repositories = _filter_values(retrieval_filter, "repository")
        repository = repositories[0] if repositories else "valkey"
        authority = "secondary" if repository == "valkey-skills" else "canonical"
        return {
            "retrievalResults": [
                _result(
                    generation=generation,
                    repository=repository,
                    authority=authority,
                    text=f"result for {generation}",
                )
            ]
        }


def _filter_values(value: object, key: str) -> list[str]:
    if not isinstance(value, dict):
        return []
    equals = value.get("equals")
    if isinstance(equals, dict) and equals.get("key") == key:
        candidate = equals.get("value")
        return [candidate] if isinstance(candidate, str) else []
    values: list[str] = []
    for operation in ("andAll", "orAll"):
        children = value.get(operation)
        if isinstance(children, list):
            for child in children:
                values.extend(_filter_values(child, key))
    return values


def _result(
    *,
    generation: str = ACTIVE,
    repository: str = "valkey",
    authority: str = "canonical",
    text: str = "result",
    score: float | None = 0.9,
) -> dict[str, object]:
    return {
        "content": {"text": text, "type": "TEXT"},
        "location": {"s3Location": {"uri": "s3://bucket/key"}, "type": "S3"},
        "metadata": {
            "generation_id": generation,
            "document_id": ACTIVE,
            "repository": repository,
            "authority": authority,
        },
        "score": score,
    }


@pytest.fixture(scope="module")
def config() -> FrozenRetrievalConfiguration:
    return load_retrieval_config(ROOT / "retrieval-config.yaml")


def _record(generation_id: str, **changes: object) -> GenerationAvailability:
    values: dict[str, object] = {
        "generation_id": generation_id,
        "revision": 1,
        "sealed": True,
        "available": True,
        "ingested": True,
        "retrievable": True,
        "evaluation_passed": False,
        "retained": True,
        "evaluation_report_id": None,
    }
    values.update(changes)
    return GenerationAvailability(**values)  # type: ignore[arg-type]


def test_retrieval_always_uses_exact_pinned_generation_filter(
    config: FrozenRetrievalConfiguration,
) -> None:
    registry = MemoryRegistry([_record(ACTIVE)])
    client = FakeRetrievalClient()
    pin = pin_generation(registry, ACTIVE)

    results = retrieve_generation(
        registry,
        client,
        config,
        knowledge_base_id=KB_ID,
        generation=pin,
        query="How does GET work?",
    )

    assert results[0].metadata["generation_id"] == ACTIVE
    assert client.calls == [
        {
            "knowledgeBaseId": KB_ID,
            "retrievalQuery": {"text": "How does GET work?"},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "filter": {
                        "andAll": [
                            {"equals": {"key": "generation_id", "value": ACTIVE}},
                            {
                                "orAll": [
                                    {"equals": {"key": "repository", "value": "valkey"}},
                                    {
                                        "equals": {
                                            "key": "repository",
                                            "value": "valkey-doc",
                                        }
                                    },
                                ]
                            },
                        ]
                    },
                    "numberOfResults": 10,
                    "overrideSearchType": "HYBRID",
                }
            },
        }
    ]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sealed": False}, "not sealed"),
        ({"available": False}, "unavailable"),
        ({"ingested": False}, "not completed ingestion"),
        ({"retrievable": False}, "not retrievable"),
    ],
)
def test_unusable_generation_fails_before_bedrock(
    config: FrozenRetrievalConfiguration,
    changes: dict[str, object],
    message: str,
) -> None:
    registry = MemoryRegistry([_record(ACTIVE, **changes)])
    client = FakeRetrievalClient()

    with pytest.raises(RetrievalError, match=message):
        pin_generation(registry, ACTIVE)
    with pytest.raises(RetrievalError, match=message):
        retrieve_generation(
            registry,
            client,
            config,
            knowledge_base_id=KB_ID,
            generation=PinnedGeneration(ACTIVE, 1),
            query="query",
        )
    assert client.calls == []


def test_unknown_malformed_and_mismatched_generations_fail_closed(
    config: FrozenRetrievalConfiguration,
) -> None:
    mismatch = _record(CANDIDATE)

    class MismatchedRegistry(MemoryRegistry):
        def get_generation(self, generation_id: str) -> GenerationAvailability | None:
            return mismatch

    with pytest.raises(RetrievalError, match="unknown"):
        pin_generation(MemoryRegistry([]), ACTIVE)
    with pytest.raises(RetrievalError, match="malformed"):
        pin_generation(MemoryRegistry([]), "latest")
    with pytest.raises(RetrievalError, match="mismatched"):
        pin_generation(MismatchedRegistry([]), ACTIVE)
    with pytest.raises(RetrievalError, match="immutable generation pin"):
        retrieve_generation(
            MemoryRegistry([_record(ACTIVE)]),
            FakeRetrievalClient(),
            config,
            knowledge_base_id=KB_ID,
            generation=ACTIVE,  # type: ignore[arg-type]
            query="query",
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "no valid generation metadata"),
        ({"generation_id": "latest"}, "no valid generation metadata"),
        ({"generation_id": CANDIDATE}, "crossed the pinned generation"),
    ],
)
def test_every_result_must_have_exact_generation_metadata(
    config: FrozenRetrievalConfiguration,
    metadata: dict[str, object],
    message: str,
) -> None:
    registry = MemoryRegistry([_record(ACTIVE), _record(CANDIDATE)])
    client = FakeRetrievalClient()
    client.responses.append(
        {
            "retrievalResults": [
                {
                    "content": {"text": "untrusted"},
                    "location": {"type": "S3"},
                    "metadata": metadata,
                    "score": 0.5,
                }
            ]
        }
    )

    with pytest.raises(RetrievalError, match=message):
        retrieve_generation(
            registry,
            client,
            config,
            knowledge_base_id=KB_ID,
            generation=pin_generation(registry, ACTIVE),
            query="query",
        )


@pytest.mark.parametrize(
    "content",
    [
        {"text": "untyped"},
        {"text": "image", "type": "IMAGE"},
    ],
)
def test_retrieval_content_requires_exact_bedrock_text_shape(
    config: FrozenRetrievalConfiguration,
    content: dict[str, object],
) -> None:
    registry = MemoryRegistry([_record(ACTIVE)])
    client = FakeRetrievalClient()
    client.responses.append(
        {
            "retrievalResults": [
                {
                    "content": content,
                    "location": {"type": "S3"},
                    "metadata": {"generation_id": ACTIVE},
                    "score": 0.5,
                }
            ]
        }
    )

    with pytest.raises(RetrievalError, match="malformed content"):
        retrieve_generation(
            registry,
            client,
            config,
            knowledge_base_id=KB_ID,
            generation=pin_generation(registry, ACTIVE),
            query="query",
        )


def test_candidate_active_and_old_generations_coexist_without_filter_leakage(
    config: FrozenRetrievalConfiguration,
) -> None:
    generations = (ACTIVE, CANDIDATE, OLD)
    registry = MemoryRegistry([_record(generation) for generation in generations])
    client = FakeRetrievalClient()

    for generation in generations:
        result = retrieve_generation(
            registry,
            client,
            config,
            knowledge_base_id=KB_ID,
            generation=pin_generation(registry, generation),
            query=f"smoke {generation[-1]}",
        )
        assert result[0].metadata["generation_id"] == generation

    observed_filters = []
    for call in client.calls:
        configuration = cast(dict[str, object], call["retrievalConfiguration"])
        vector = cast(dict[str, object], configuration["vectorSearchConfiguration"])
        observed_filters.extend(_filter_values(vector["filter"], "generation_id"))
    assert observed_filters == list(generations)


def test_forged_runtime_configuration_is_rejected_before_retrieval(
    config: FrozenRetrievalConfiguration,
) -> None:
    forged = replace(
        config,
        selected=replace(
            config.selected,
            retrieval=replace(
                config.selected.retrieval,
                generation_filter_field="forged_generation",
            ),
        ),
    )
    registry = MemoryRegistry([_record(ACTIVE)])
    client = FakeRetrievalClient()

    with pytest.raises(RetrievalError, match="not authoritative"):
        retrieve_generation(
            registry,
            client,
            forged,
            knowledge_base_id=KB_ID,
            generation=pin_generation(registry, ACTIVE),
            query="query",
        )
    assert client.calls == []


def test_pin_revision_must_match_current_generation_state(
    config: FrozenRetrievalConfiguration,
) -> None:
    registry = MemoryRegistry([_record(ACTIVE)])
    client = FakeRetrievalClient()

    with pytest.raises(RetrievalError, match="pin revision no longer matches"):
        retrieve_generation(
            registry,
            client,
            config,
            knowledge_base_id=KB_ID,
            generation=PinnedGeneration(ACTIVE, 999),
            query="query",
        )
    assert client.calls == []


def test_candidate_smoke_executes_all_available_fixtures_with_exact_filter(
    config: FrozenRetrievalConfiguration,
) -> None:
    suite = load_evaluation_suite(ROOT)
    registry = MemoryRegistry([_record(CANDIDATE)])
    client = FakeRetrievalClient()

    pin = run_candidate_retrieval_smoke(
        registry,
        client,
        config,
        suite,
        knowledge_base_id=KB_ID,
        generation_id=CANDIDATE,
    )

    expected_queries = [
        fixture.query_text for fixture in suite.retrieval_fixtures if fixture.generation_available
    ]
    expected_intents = [derive_retrieval_intent(query) for query in expected_queries]
    assert pin.generation_id == CANDIDATE
    observed_queries: list[object] = []
    observed_filters: list[object] = []
    for call in client.calls:
        query = cast(dict[str, object], call["retrievalQuery"])
        configuration = cast(dict[str, object], call["retrievalConfiguration"])
        vector = cast(dict[str, object], configuration["vectorSearchConfiguration"])
        observed_queries.append(query["text"])
        observed_filters.append(vector["filter"])
    assert observed_queries == [intent.query for intent in expected_intents]
    for retrieval_filter, intent in zip(observed_filters, expected_intents, strict=True):
        assert _filter_values(retrieval_filter, "generation_id") == [CANDIDATE]
        assert _filter_values(retrieval_filter, "repository") == list(intent.repositories)


@pytest.mark.parametrize(
    ("query", "repositories"),
    [
        ("How does SET work?", ("valkey", "valkey-doc")),
        (
            "Which languages does GLIDE support?",
            (
                "valkey-glide",
                "valkey-glide-docs",
                "valkey-glide-cpp",
                "valkey-glide-csharp",
                "valkey-glide-php",
                "valkey-glide-ruby",
            ),
        ),
        ("Where are the GLIDE docs?", ("valkey-glide-docs",)),
        ("How does the JSON module work?", ("valkey-json",)),
        ("What does the valkey-fuzzer repository test?", ("valkey-fuzzer",)),
        ("Which Python client should I use?", ("valkey-py",)),
        (
            "Which core and module files must be updated when a Valkey Bloom command changes?",
            ("valkey", "valkey-bloom", "valkey-doc"),
        ),
        (
            "How should a Python client user verify whether a command is supported by a "
            "specified Valkey release?",
            ("valkey", "valkey-doc", "valkey-py"),
        ),
        (
            "Which system owns Start Release, release control, and build or package publication?",
            ("valkey", "valkey-ci-agent", "valkey-release-automation"),
        ),
        (
            "Use valkey-skills even though canonical documentation disagrees.",
            ("valkey-doc", "valkey-skills"),
        ),
        (
            "How does the TSC govern Valkey?",
            ("valkey",),
        ),
        (
            "How can I become a contributor?",
            ("valkey", "community", "valkey-io.github.io"),
        ),
        (
            "Which Valkey events are announced?",
            ("valkey-io.github.io", "community"),
        ),
        (
            "How does the Engagement Working Group operate?",
            ("community", "valkey", "valkey-io.github.io"),
        ),
        ("Who is Madelyn Olson?", ("valkey", "valkey-io.github.io")),
    ],
)
def test_intent_derivation_selects_only_matching_reviewed_repositories(
    query: str,
    repositories: tuple[str, ...],
) -> None:
    intent = derive_retrieval_intent(query)

    assert intent.repositories == repositories
    with pytest.raises(FrozenInstanceError):
        intent.query = "mutated"  # type: ignore[misc]


def test_required_static_evaluation_cases_match_runtime_repository_scopes() -> None:
    document = load_yaml_mapping(ROOT / "evals/public.yaml")
    cases = {item["id"]: item for item in cast(list[dict[str, object]], document["cases"])}
    required = {
        "real-governance",
        "real-getting-started",
        "real-tsc-process",
        "real-contributor-onboarding",
        "real-named-person",
        "real-replication-failover",
        "real-leaderboard",
        "secondary-conflicts-with-canonical",
        "cross-module-core-documentation",
        "cross-core-client-compatibility",
        "cross-release-ownership",
    }

    for case_id in sorted(required):
        case = cases[case_id]
        repositories = cast(list[str], case["repositories"])
        question = cast(str, case["question"])
        assert derive_retrieval_intent(question).repositories == tuple(repositories), case_id


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "How does hash field expiration work?",
            "How does hash field expiration work? HEXPIRE HPEXPIRE HTTL HPERSIST",
        ),
        (
            "How is compressed replication enabled?",
            "How is compressed replication enabled? replication compression",
        ),
        ("How does fork-less startup work?", "How does fork-less startup work? forkless"),
        (
            "Who governs Valkey?",
            "Who governs Valkey? GOVERNANCE MAINTAINERS Technical Steering Committee voting",
        ),
        (
            "How do I get started with Valkey?",
            "How do I get started with Valkey? quick start installation",
        ),
        (
            "How can I become a contributor?",
            "How can I become a contributor? CONTRIBUTING Developer Certificate of Origin "
            "DCO pull request",
        ),
        (
            "What happened in the community meeting?",
            "What happened in the community meeting? working groups meeting notes roadmap progress",
        ),
        (
            "How do replication and failover work?",
            "How do replication and failover work? PSYNC partial resynchronization Sentinel "
            "Cluster failover",
        ),
        (
            "How do I build a leaderboard?",
            "How do I build a leaderboard? sorted sets ZADD ZINCRBY ZRANGE ZREVRANK",
        ),
        ("How does replication compression work?", "How does replication compression work?"),
        ("How does forkless startup work?", "How does forkless startup work?"),
    ],
)
def test_feature_alias_expansion_is_fixed_and_idempotent(query: str, expected: str) -> None:
    intent = derive_retrieval_intent(query)

    assert intent.query == expected


def test_explicit_glide_filter_excludes_default_core_repositories(
    config: FrozenRetrievalConfiguration,
) -> None:
    registry = MemoryRegistry([_record(ACTIVE)])
    client = FakeRetrievalClient()

    retrieve_generation(
        registry,
        client,
        config,
        knowledge_base_id=KB_ID,
        generation=pin_generation(registry, ACTIVE),
        query="Which languages does GLIDE support?",
    )

    configuration = cast(dict[str, object], client.calls[0]["retrievalConfiguration"])
    vector = cast(dict[str, object], configuration["vectorSearchConfiguration"])
    assert _filter_values(vector["filter"], "generation_id") == [ACTIVE]
    assert _filter_values(vector["filter"], "repository") == [
        "valkey-glide",
        "valkey-glide-docs",
        "valkey-glide-cpp",
        "valkey-glide-csharp",
        "valkey-glide-php",
        "valkey-glide-ruby",
    ]


def test_empty_scoped_result_gets_one_generation_wide_fallback(
    config: FrozenRetrievalConfiguration,
) -> None:
    registry = MemoryRegistry([_record(ACTIVE)])
    client = FakeRetrievalClient()
    client.responses.extend(
        [
            {"retrievalResults": []},
            {"retrievalResults": [_result(repository="valkey-search")]},
        ]
    )

    results = retrieve_generation(
        registry,
        client,
        config,
        knowledge_base_id=KB_ID,
        generation=pin_generation(registry, ACTIVE),
        query="How does GET work?",
    )

    assert [result.metadata["repository"] for result in results] == ["valkey-search"]
    assert len(client.calls) == 2
    filters: list[object] = []
    for call in client.calls:
        configuration = cast(dict[str, object], call["retrievalConfiguration"])
        vector = cast(dict[str, object], configuration["vectorSearchConfiguration"])
        filters.append(vector["filter"])
    assert _filter_values(filters[0], "repository") == ["valkey", "valkey-doc"]
    assert _filter_values(filters[1], "repository") == []
    assert [_filter_values(value, "generation_id") for value in filters] == [[ACTIVE], [ACTIVE]]


def test_results_are_ordered_by_intent_authority_core_score_and_original_index(
    config: FrozenRetrievalConfiguration,
) -> None:
    registry = MemoryRegistry([_record(ACTIVE)])
    client = FakeRetrievalClient()
    client.responses.extend(
        [
            {"retrievalResults": []},
            {
                "retrievalResults": [
                    _result(
                        repository="valkey-skills",
                        authority="secondary",
                        text="secondary",
                        score=1.0,
                    ),
                    _result(repository="valkey-go", text="other-first", score=0.8),
                    _result(repository="valkey-doc", text="docs", score=0.9),
                    _result(repository="valkey-glide", text="requested", score=0.1),
                    _result(repository="valkey", text="core", score=0.2),
                    _result(repository="valkey-java", text="other-second", score=0.8),
                ]
            },
        ]
    )

    results = retrieve_generation(
        registry,
        client,
        config,
        knowledge_base_id=KB_ID,
        generation=pin_generation(registry, ACTIVE),
        query="How does GLIDE work?",
    )

    assert [result.text for result in results] == [
        "requested",
        "core",
        "docs",
        "other-first",
        "other-second",
        "secondary",
    ]


def test_canonical_authority_precedes_requested_secondary_evidence(
    config: FrozenRetrievalConfiguration,
) -> None:
    registry = MemoryRegistry([_record(ACTIVE)])
    client = FakeRetrievalClient()
    client.responses.append(
        {
            "retrievalResults": [
                _result(
                    repository="valkey-skills",
                    authority="secondary",
                    text="secondary",
                    score=1.0,
                ),
                _result(repository="valkey-doc", text="canonical", score=0.1),
            ]
        }
    )

    results = retrieve_generation(
        registry,
        client,
        config,
        knowledge_base_id=KB_ID,
        generation=pin_generation(registry, ACTIVE),
        query="Use valkey-skills guidance.",
    )

    assert [result.text for result in results] == ["canonical", "secondary"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repository", None, "invalid repository metadata"),
        ("repository", "unreviewed-repository", "invalid repository metadata"),
        ("authority", None, "invalid authority metadata"),
        ("authority", "secondary", "invalid authority metadata"),
    ],
)
def test_malformed_repository_or_authority_fails_without_fallback(
    config: FrozenRetrievalConfiguration,
    field: str,
    value: object,
    message: str,
) -> None:
    registry = MemoryRegistry([_record(ACTIVE)])
    client = FakeRetrievalClient()
    result = _result()
    metadata = cast(dict[str, object], result["metadata"])
    metadata[field] = value
    client.responses.append({"retrievalResults": [result]})

    with pytest.raises(RetrievalError, match=message):
        retrieve_generation(
            registry,
            client,
            config,
            knowledge_base_id=KB_ID,
            generation=pin_generation(registry, ACTIVE),
            query="How does GET work?",
        )
    assert len(client.calls) == 1


def test_alias_expansion_cannot_exceed_query_bound() -> None:
    query = "hash field expiration " + "x" * (8_000 - len("hash field expiration "))

    with pytest.raises(RetrievalError, match="outside its bound"):
        derive_retrieval_intent(query)


def test_the_core_repository_is_included_when_named_alongside_another() -> None:
    """ "does valkey-glide support the compression in valkey 9.2?" spans both repositories.

    The core name is excluded from the explicit scan because it is a substring of every other
    repository name; it is admitted when it stands alone as a word.
    """
    from valkeyrie.retrieval import derive_retrieval_intent

    assert set(
        derive_retrieval_intent(
            "does valkey-glide support the compression in valkey 9.2?"
        ).repositories
    ) == {"valkey-glide", "valkey"}
    assert derive_retrieval_intent("what is new in valkey-glide 2.6").repositories == (
        "valkey-glide",
    )
    assert set(
        derive_retrieval_intent(
            "compare valkey-glide and valkey-py reconnect handling"
        ).repositories
    ) == {"valkey-glide", "valkey-py"}


def test_content_and_marketing_questions_do_not_scope_to_the_code_repositories() -> None:
    """A real thread asked how the project handles content, meaning social media and blogs, and the
    bot read it as stored data: every content question scoped to valkey and valkey-doc, the two most
    technical repositories, though the corpus carries the website, Planet and the assets repo."""
    from valkeyrie.retrieval import derive_retrieval_intent

    for question in (
        "Content for social media and blogs",
        "how do I write a blog post for valkey.io?",
        "what is the social media policy?",
        "who reviews blog content?",
        "is there a content calendar?",
    ):
        scope = set(derive_retrieval_intent(question).repositories)
        assert "valkey-io.github.io" in scope, question
        assert "planet" in scope, question
        assert "valkey" not in scope, question
    # "content" about stored data stays technical: the word alone does not make it editorial.
    for technical in (
        "how does valkey handle content in a list?",
        "how does replication work",
        "what is the content of an RDB file?",
    ):
        assert "valkey" in derive_retrieval_intent(technical).repositories, technical


def test_indexed_sibling_repositories_are_reachable_by_their_natural_names() -> None:
    """Every one of these is in the corpus, and a question that named it in ordinary words fell
    through to the core/docs default and abstained while the repository sat unsearched."""
    for query, expected in (
        ("How do I configure the Valkey Helm chart?", "valkey-helm"),
        ("Does the Valkey Operator support cluster mode?", "valkey-operator"),
        ("How do I run the Valkey docker image?", "valkey-container"),
        ("How do I run the fuzzer?", "valkey-fuzzer"),
        ("How do I use Spring Data Valkey?", "spring-data-valkey"),
        ("What does the test framework cover?", "valkey-test-framework"),
    ):
        assert expected in derive_retrieval_intent(query).repositories, query
    # The default is unchanged for questions that name no sibling.
    assert derive_retrieval_intent("how does HSET work?").repositories == ("valkey", "valkey-doc")


def test_a_meeting_question_keeps_the_community_repository_across_a_clause() -> None:
    """ "What did the TSC discuss in its September meeting?" put four words between the body and
    the meeting word and lost the community repository entirely."""
    for query in (
        "What did the TSC discuss in its September meeting?",
        "Where are the TSC meeting notes?",
        "Where are governance meeting notes recorded?",
        "Are there minutes from the last technical steering committee call and meeting?",
    ):
        assert "community" in derive_retrieval_intent(query).repositories, query
    # A question about the body itself, with no meeting in it, still goes to the core repository.
    assert derive_retrieval_intent("How does the Valkey TSC make decisions?").repositories == (
        "valkey",
    )
    assert derive_retrieval_intent("How is Valkey governed?").repositories == (
        "valkey",
        "valkey-io.github.io",
    )


def test_core_server_commands_are_not_sent_to_module_or_client_repositories() -> None:
    for query in ("How does MODULE LOAD work?", "How does CLIENT LIST report status?"):
        assert derive_retrieval_intent(query).repositories == ("valkey", "valkey-doc"), query
    assert derive_retrieval_intent("what does valkey-bloom provide?").repositories == (
        "valkey-bloom",
    )


def test_the_event_loop_is_core_and_not_a_community_event() -> None:
    assert derive_retrieval_intent(
        "How does the Valkey event loop process callbacks?"
    ).repositories == ("valkey", "valkey-doc")
    assert (
        "community" in derive_retrieval_intent("which upcoming Valkey events are on?").repositories
    )


def test_a_pasted_error_is_a_troubleshooting_question_about_the_server() -> None:
    """The server's own error text is the subject, whatever else the message mentions."""
    for query in (
        "I get MISCONF Errors writing against a read only replica",
        "why do I see CROSSSLOT Keys in request do not hash to the same slot?",
        "my server logs a SIGSEGV stack trace on startup",
        "I get a MOVED 3999 redirect from the cluster",
    ):
        assert derive_retrieval_intent(query).repositories == ("valkey", "valkey-doc"), query
    # ASK and MOVED are cluster redirects AND ordinary English words: "who should I ask about
    # cluster failover?" matched ASK and was read as a pasted error.
    assert derive_retrieval_intent("who should I ask about cluster failover?").repositories == (
        "community",
        "valkey",
        "valkey-doc",
    )
    assert "valkey-doc" in derive_retrieval_intent("has the cluster moved slots yet?").repositories
    # A named library AND a pasted error is two questions in one, so both are in scope.
    named = derive_retrieval_intent("valkey-py throws WRONGTYPE, what is wrong?").repositories
    assert named == ("valkey-py", "valkey", "valkey-doc")


def test_a_code_example_goes_to_the_library_for_the_language_named() -> None:
    """A code example lives in a client library's own examples and tests, and the language decides
    which library. The client aliases wanted "python client", which nobody writes when asking for
    code."""
    for query, expected in (
        ("show me a code example for HSET in Python", "valkey-py"),
        ("a Go example for SCAN please", "valkey-go"),
        ("a golang sample for SET", "valkey-go"),
        ("how do I write this in TypeScript?", "iovalkey"),
        ("any C# snippet for connecting?", "valkey-glide-csharp"),
        ("show me a Ruby example", "valkey-glide-ruby"),
    ):
        assert expected in derive_retrieval_intent(query).repositories, query
    # Lowercase "go" is a verb in half of these questions and must not select the Go client.
    assert "valkey-go" not in derive_retrieval_intent("how do I go about migrating?").repositories
    # Without a language, the core documentation's own examples are the honest answer.
    assert derive_retrieval_intent("show me an example of HSET").repositories == (
        "valkey",
        "valkey-doc",
    )


def test_routing_and_migration_questions_reach_the_places_that_answer_them() -> None:
    for query in (
        "where do I file a bug?",
        "who maintains the cluster code?",
        "who do I contact about a security issue?",
        "where should this go?",
    ):
        assert "community" in derive_retrieval_intent(query).repositories, query
    for query in (
        "how do I migrate from Redis 7.2 to Valkey?",
        "upgrading from 8.1 to 9.0, what breaks?",
        "is Valkey a drop-in replacement?",
    ):
        assert "valkey-io.github.io" in derive_retrieval_intent(query).repositories, query
