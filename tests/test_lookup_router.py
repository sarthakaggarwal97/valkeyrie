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
        ('{"lookups":[], "answer":"yes"}', "exactly one key"),
        ('{"lookups":"corpus_search"}', "must be an array"),
        ('{"lookups":["corpus_search"]}', "must be an object"),
        ("[]", "exactly one key"),
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
