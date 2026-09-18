"""The router's parser is the boundary between untrusted model text and real requests."""

from __future__ import annotations

import json

import pytest

from valkeyrie.live_github import IssueQuery, PullRequestQuery, ReleaseListQuery
from valkeyrie.lookup_router import (
    ROUTER_SYSTEM,
    LookupRouterError,
    parse_lookup_plan,
    route_lookups,
)


def test_accepts_every_catalog_entry_with_bounded_arguments() -> None:
    plan = parse_lookup_plan(
        json.dumps(
            {
                "lookups": [
                    {"kind": "corpus_search"},
                    {"kind": "pull_request", "repository": "valkey", "number": 3853},
                    {"kind": "issue", "repository": "valkey-glide", "number": 12},
                    {"kind": "releases", "repository": "valkey"},
                ]
            }
        )
    )
    assert plan.corpus_search is True
    assert plan.live == (
        PullRequestQuery("valkey", 3853),
        IssueQuery("valkey-glide", 12),
        ReleaseListQuery("valkey"),
    )


def test_repository_defaults_to_valkey_and_duplicates_collapse() -> None:
    plan = parse_lookup_plan(
        '{"lookups":[{"kind":"issue","number":7},{"kind":"issue","number":7},'
        '{"kind":"corpus_search"},{"kind":"corpus_search"}]}'
    )
    assert plan.corpus_search is True
    assert plan.live == (IssueQuery("valkey", 7),)


def test_a_whole_reply_fence_is_tolerated_but_nothing_else_is() -> None:
    """Models wrap JSON in a fence despite instructions; that alone is forgiven."""
    fenced = '```json\n{"lookups":[{"kind":"releases"}]}\n```'
    assert parse_lookup_plan(fenced).live == (ReleaseListQuery("valkey"),)
    with pytest.raises(LookupRouterError, match="not JSON"):
        parse_lookup_plan('Sure! {"lookups":[]}')


def test_empty_lookups_is_a_valid_no_op_plan() -> None:
    plan = parse_lookup_plan('{"lookups":[]}')
    assert plan.corpus_search is False
    assert plan.live == ()


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        # A capability outside the catalog is refused, never coerced to a near match.
        ('{"lookups":[{"kind":"shell","cmd":"id"}]}', "not in the catalog"),
        ('{"lookups":[{"kind":"project_board","repository":"valkey"}]}', "not in the catalog"),
        # A repository outside the organization's naming, or a traversal, is refused.
        ('{"lookups":[{"kind":"issue","repository":"../../etc","number":1}]}', "malformed"),
        ('{"lookups":[{"kind":"issue","repository":"valkey-io/valkey","number":1}]}', "malformed"),
        ('{"lookups":[{"kind":"issue","repository":"","number":1}]}', "malformed"),
        # Numbers are bounded and typed; a string number is not coerced.
        ('{"lookups":[{"kind":"issue","number":"3853"}]}', "outside its bound"),
        ('{"lookups":[{"kind":"issue","number":0}]}', "outside its bound"),
        ('{"lookups":[{"kind":"issue","number":99999999999}]}', "outside its bound"),
        ('{"lookups":[{"kind":"issue","number":true}]}', "outside its bound"),
        # Extra keys are refused: they are how an output would smuggle an argument.
        ('{"lookups":[{"kind":"releases","per_page":9999}]}', "unsupported keys"),
        ('{"lookups":[{"kind":"corpus_search","query":"x"}]}', "unsupported keys"),
        # Shape violations.
        ('{"lookups":[], "answer":"yes"}', "lookups and optionally question"),
        ('{"lookups":"corpus_search"}', "must be an array"),
        ('{"lookups":["corpus_search"]}', "must be an object"),
        ("[]", "lookups and optionally question"),
        ("", "not JSON"),
    ],
)
def test_hostile_or_malformed_replies_are_refused(raw: str, reason: str) -> None:
    with pytest.raises(LookupRouterError, match=reason):
        parse_lookup_plan(raw)


def test_too_many_lookups_and_oversized_replies_are_refused() -> None:
    many = json.dumps({"lookups": [{"kind": "issue", "number": n} for n in range(1, 6)]})
    with pytest.raises(LookupRouterError, match="too many"):
        parse_lookup_plan(many)
    with pytest.raises(LookupRouterError, match="byte bound"):
        parse_lookup_plan('{"lookups":[]}' + " " * 5000)


def test_route_lookups_returns_none_on_every_failure_so_keywords_remain() -> None:
    """A routing failure must never remove a capability the keyword path already has."""

    def broken(system: str, question: str) -> str:
        raise TimeoutError("model unavailable")

    assert route_lookups("is 9.2 rc1 released?", broken) is None
    assert route_lookups("is 9.2 rc1 released?", lambda s, q: "I think releases") is None
    assert route_lookups("   ", lambda s, q: '{"lookups":[]}') is None

    def ok(system: str, question: str) -> str:
        assert system == ROUTER_SYSTEM
        assert question == "is 9.2 rc1 released?"
        return '{"lookups":[{"kind":"releases"}]}'

    plan = route_lookups("is 9.2 rc1 released?", ok)
    assert plan is not None
    assert plan.live == (ReleaseListQuery("valkey"),)


def test_a_follow_up_is_resolved_only_when_history_is_supplied() -> None:
    """The resolved question is the memory. Without history there is nothing to resolve."""
    from valkeyrie.lookup_router import ConversationTurn

    history = (
        ConversationTurn("user", "How does Valkey replication work?"),
        ConversationTurn("assistant", "A replica connects to a primary and receives a stream..."),
    )
    seen: dict[str, str] = {}

    def converse(system: str, prompt: str) -> str:
        seen["prompt"] = prompt
        return (
            '{"question":"How does Valkey replication failover work?",'
            '"lookups":[{"kind":"corpus_search"}]}'
        )

    plan = route_lookups("and what about failover?", converse, history)
    assert plan is not None
    assert plan.question == "How does Valkey replication failover work?"
    assert plan.corpus_search is True
    # The model saw the history and the fragment, in order, labelled by speaker.
    assert "Asker: How does Valkey replication work?" in seen["prompt"]
    assert "Assistant: A replica connects" in seen["prompt"]
    assert seen["prompt"].rstrip().endswith("Current question: and what about failover?")

    # Same reply with NO history: a rewritten question is the model changing what was asked.
    plan = route_lookups("and what about failover?", converse)
    assert plan is not None
    assert plan.question is None
    assert seen["prompt"] == "and what about failover?"


def test_resolved_question_is_bounded_and_single_line() -> None:
    with pytest.raises(LookupRouterError, match="non-blank"):
        parse_lookup_plan('{"question":"   ","lookups":[]}')
    with pytest.raises(LookupRouterError, match="single line"):
        parse_lookup_plan('{"question":"a\\nb","lookups":[]}')
    with pytest.raises(LookupRouterError, match="byte bound"):
        parse_lookup_plan(json.dumps({"question": "x" * 3000, "lookups": []}))
    with pytest.raises(LookupRouterError, match="lookups and optionally question"):
        parse_lookup_plan('{"question":"q","lookups":[],"answer":"no"}')


def test_conversation_is_bounded_and_oversize_history_degrades_to_none() -> None:
    from valkeyrie.lookup_router import (
        MAX_CONVERSATION_TURNS,
        ConversationTurn,
        validate_conversation,
    )

    many = tuple(ConversationTurn("user", f"turn {i}") for i in range(20))
    kept = validate_conversation(many)
    # Oldest turns fall away; the most recent six remain, in order.
    assert len(kept) == MAX_CONVERSATION_TURNS
    assert [t.text for t in kept] == [f"turn {i}" for i in range(14, 20)]

    with pytest.raises(LookupRouterError, match="byte bound"):
        validate_conversation((ConversationTurn("user", "x" * 5000),))
    with pytest.raises(LookupRouterError, match="role"):
        validate_conversation((ConversationTurn("system", "x"),))  # type: ignore[arg-type]

    # route_lookups itself never fails on bad history: it routes the question alone.
    seen: dict[str, str] = {}

    def converse(system: str, prompt: str) -> str:
        seen["prompt"] = prompt
        return '{"lookups":[{"kind":"corpus_search"}]}'

    plan = route_lookups("what is AOF", converse, (ConversationTurn("user", "x" * 5000),))
    assert plan is not None and plan.corpus_search
    assert seen["prompt"] == "what is AOF"
